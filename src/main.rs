//! rag-mcp: hybrid RAG (SQLite FTS5 + candle embeddings) over project docs.
//!
//! Usage:
//!   rag-mcp index [--force]  full rebuild from config (default ./rag.json)
//!   rag-mcp sync             incremental update (new/changed/deleted files)
//!   rag-mcp serve            run the MCP server on stdio
//!   rag-mcp stats            print index stats and exit
//!
//! Env: RAG_CONFIG (default rag.json), RAG_DOCS / RAG_DB / RAG_MODEL_DIR overrides.

mod chunk;
mod config;
mod embed;
mod extract;
mod search;
mod server;
mod store;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn config_path() -> PathBuf {
    std::env::var("RAG_CONFIG").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("rag.json"))
}

/// Collect files from config `docs` dirs filtered by include/exclude globs.
/// `docs` dirs resolve against the config file's directory.
/// Returns (display_name relative to that dir, absolute path).
fn collect_files(cfg: &config::Config, cfg_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out: Vec<(String, PathBuf)> = vec![];
    for dir in &cfg.docs {
        let base = cfg_dir.join(dir);
        if !base.is_dir() {
            eprintln!("warning: docs dir missing: {}", base.display());
            continue;
        }
        for e in walkdir::WalkDir::new(&base).into_iter().filter_map(|e| e.ok()) {
            if !e.file_type().is_file() {
                continue;
            }
            let rel = e
                .path()
                .strip_prefix(cfg_dir)
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| e.path().display().to_string());
            if !cfg.include.iter().any(|g| glob::Pattern::new(g).is_ok_and(|p| p.matches(&rel))) {
                continue;
            }
            if cfg.exclude.iter().any(|g| glob::Pattern::new(g).is_ok_and(|p| p.matches(&rel))) {
                continue;
            }
            out.push((rel, e.path().to_path_buf()));
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn file_sig(path: &Path) -> Result<(i64, i64)> {
    let m = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let mtime = m
        .modified()
        .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64)
        .unwrap_or(0);
    Ok((mtime, m.len() as i64))
}

fn chunk_for(path: &Path, text: &str, cfg: &config::Config) -> Vec<String> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "md" | "markdown" => chunk::chunk_markdown(text, cfg.chunk_chars, cfg.overlap_chars),
        _ => chunk::chunk_text(text, cfg.chunk_chars, cfg.overlap_chars),
    }
}

fn index_one(
    conn: &rusqlite::Connection,
    emb: &embed::Embedder,
    cfg: &config::Config,
    name: &str,
    path: &Path,
    i: usize,
    n: usize,
) -> Result<&'static str> {
    let (mtime, size) = file_sig(path)?;
    if store::doc_fresh(conn, name, mtime, size)? {
        return Ok("fresh");
    }
    store::delete_doc(conn, name).ok();
    let ext = path.to_path_buf();
    let Some(extracted) = extract::extract(path, cfg)? else {
        // Reason already printed by the extractor.
        return Ok("skipped");
    };
    if extracted.text.trim().is_empty() {
        eprintln!("[{i}/{n}] SKIP {name} (empty)");
        return Ok("skipped");
    }
    let chunks = chunk_for(&ext, &extracted.text, cfg);
    if chunks.is_empty() {
        eprintln!("[{i}/{n}] SKIP {name} (no chunks)");
        return Ok("skipped");
    }
    let vecs = emb.embed(&chunks)?;
    let pairs: Vec<(String, Vec<f32>)> = chunks.into_iter().zip(vecs.into_iter()).collect();
    let chars = extracted.text.chars().count() as i64;
    store::insert_doc(conn, name, chars, mtime, size, &pairs)?;
    let src = if extracted.source == extract::Source::Ocr { " [ocr]" } else { "" };
    eprintln!("[{i}/{n}] {name} ({chars} chars, {} chunks{src})", pairs.len());
    Ok("indexed")
}

fn cmd_index(force: bool) -> Result<()> {
    let cfg = config::load()?;
    let cfg_dir = config::dir();
    let files = collect_files(&cfg, &cfg_dir)?;
    eprintln!("indexing {} files (force={force})", files.len());
    if files.is_empty() {
        anyhow::bail!("no files matched include globs");
    }
    eprintln!("loading embedding model...");
    let emb = embed::Embedder::load(&cfg)?;
    let db = cfg_dir.join(&cfg.db);
    if force && db.exists() {
        std::fs::remove_file(&db).context("remove old db")?;
    }
    let conn = store::open_db(&db)?;
    let n = files.len();
    let (mut fresh, mut indexed, mut skipped) = (0, 0, 0);
    for (i, (name, path)) in files.iter().enumerate() {
        match index_one(&conn, &emb, &cfg, name, path, i + 1, n)? {
            "fresh" => fresh += 1,
            "skipped" => skipped += 1,
            _ => indexed += 1,
        }
    }
    // Drop docs whose files vanished.
    let known: std::collections::HashSet<&str> =
        files.iter().map(|(name, _)| name.as_str()).collect();
    for d in store::list_docs(&conn)? {
        if !known.contains(d.name.as_str()) {
            store::delete_doc(&conn, &d.name)?;
            eprintln!("forgot {}", d.name);
        }
    }
    let docs = store::list_docs(&conn)?;
    let n_chunks: i64 = docs.iter().map(|d| d.n_chunks).sum();
    eprintln!(
        "done: {} docs, {} chunks -> {} (fresh={fresh} indexed={indexed} skipped={skipped})",
        docs.len(),
        n_chunks,
        db.display()
    );
    Ok(())
}

async fn cmd_serve() -> Result<()> {
    let cfg = config::load()?;
    let cfg_dir = config::dir();
    let db = cfg_dir.join(&cfg.db);
    if !db.exists() {
        anyhow::bail!("index db not found at {} — run `rag-mcp sync` first", db.display());
    }
    // Staleness check: file count/size vs index.
    let files = collect_files(&cfg, &cfg_dir).unwrap_or_default();
    let total_size: i64 =
        files.iter().filter_map(|(_, p)| file_sig(p).ok().map(|(_, s)| s)).sum();
    let docs = store::open_db(&db).and_then(|c| store::list_docs(&c)).unwrap_or_default();
    if docs.len() != files.len() {
        eprintln!(
            "warning: index has {} docs but {} files on disk — run `rag-mcp sync`",
            docs.len(),
            files.len()
        );
    }
    let _ = total_size;
    let state = Arc::new(server::AppState::new(cfg, db));
    let rag = server::Rag::new(state);
    let transport = rmcp::transport::stdio();
    let service = {
        use rmcp::ServiceExt;
        rag.serve(transport).await.map_err(|e| anyhow::anyhow!("{e}"))?
    };
    service.waiting().await.map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

fn cmd_stats() -> Result<()> {
    let cfg = config::load()?;
    let db = config::dir().join(&cfg.db);
    let conn = store::open_db(&db)?;
    let docs = store::list_docs(&conn)?;
    let n_chunks: i64 = docs.iter().map(|d| d.n_chunks).sum();
    println!("collection: {} ({})", cfg.name, cfg.description);
    println!("db: {}\ndocs: {}\nchunks: {}", db.display(), docs.len(), n_chunks);
    for d in docs.iter().take(60) {
        println!("  {} ({} chars, {} chunks)", d.name, d.chars, d.n_chunks);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // MCP speaks on stdout — keep logs on stderr.
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    match cmd.as_str() {
        "index" => cmd_index(args.any(|a| a == "--force" || a == "-f")),
        "sync" => cmd_index(false),
        "serve" => cmd_serve().await,
        "stats" => cmd_stats(),
        _ => {
            eprintln!("usage: rag-mcp <index [--force]|sync|serve|stats>");
            eprintln!("config: RAG_CONFIG (default rag.json); overrides RAG_DOCS RAG_DB RAG_MODEL_DIR");
            std::process::exit(2);
        }
    }
}
