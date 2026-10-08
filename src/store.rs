//! SQLite storage: docs, chunks (+f32 embedding blobs), FTS5 index.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

pub const SCHEMA_VERSION: i32 = 2;

pub fn open_db(path: &std::path::Path) -> Result<Connection> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).context("create db dir")?;
        }
    }
    let conn = Connection::open(path).context("open sqlite db")?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS docs(
           id INTEGER PRIMARY KEY,
           name TEXT NOT NULL UNIQUE,
           chars INTEGER NOT NULL,
           mtime INTEGER NOT NULL DEFAULT 0,
           size INTEGER NOT NULL DEFAULT 0
         );
         CREATE TABLE IF NOT EXISTS chunks(
           id INTEGER PRIMARY KEY,
           doc_id INTEGER NOT NULL REFERENCES docs(id),
           idx INTEGER NOT NULL,
           text TEXT NOT NULL,
           embedding BLOB NOT NULL
         );
         CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
           text, content='chunks', content_rowid='id', tokenize='unicode61'
         );
         CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
           INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
         END;
         CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
           INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
         END;",
    )
    .context("init schema")?;
    // Migrate v1 -> v2: track mtime/size for incremental sync.
    for col in ["mtime", "size"] {
        let sql = format!("ALTER TABLE docs ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0");
        if let Err(e) = conn.execute(&sql, []) {
            let msg = e.to_string();
            if !msg.contains("duplicate column") {
                anyhow::bail!("migrate docs.{col}: {msg}");
            }
        }
    }
    conn.execute(
        "INSERT INTO meta(k, v) VALUES ('schema_version', ?1)
         ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        params![SCHEMA_VERSION.to_string()],
    )?;
    Ok(conn)
}

pub struct DocRow {
    pub id: i64,
    pub name: String,
    pub chars: i64,
    pub n_chunks: i64,
    pub mtime: i64,
    pub size: i64,
}

pub struct ChunkRow {
    pub id: i64,
    pub doc: String,
    pub idx: i64,
    pub text: String,
    pub embedding: Vec<f32>,
}

pub fn wipe(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DELETE FROM chunks; DELETE FROM docs;
         INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild');",
    )
    .context("wipe")?;
    Ok(())
}

pub fn insert_doc(
    conn: &Connection,
    name: &str,
    chars: i64,
    mtime: i64,
    size: i64,
    chunks: &[(String, Vec<f32>)],
) -> Result<()> {
    conn.execute(
        "INSERT INTO docs(name, chars, mtime, size) VALUES (?1, ?2, ?3, ?4)",
        params![name, chars, mtime, size],
    )?;
    let doc_id = conn.last_insert_rowid();
    let mut stmt =
        conn.prepare("INSERT INTO chunks(doc_id, idx, text, embedding) VALUES (?1, ?2, ?3, ?4)")?;
    for (i, (text, emb)) in chunks.iter().enumerate() {
        let blob = crate::embed::to_blob(emb);
        stmt.execute(params![doc_id, i as i64, text, blob])?;
    }
    Ok(())
}

pub fn list_docs(conn: &Connection) -> Result<Vec<DocRow>> {
    let mut stmt = conn.prepare(
        "SELECT d.id, d.name, d.chars, COUNT(c.id), d.mtime, d.size
         FROM docs d LEFT JOIN chunks c ON c.doc_id = d.id
         GROUP BY d.id ORDER BY d.name",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok(DocRow {
                id: r.get(0)?,
                name: r.get(1)?,
                chars: r.get(2)?,
                n_chunks: r.get(3)?,
                mtime: r.get(4)?,
                size: r.get(5)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// True if the stored doc matches the file on disk (for incremental sync).
pub fn doc_fresh(conn: &Connection, name: &str, mtime: i64, size: i64) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM docs WHERE name = ?1 AND mtime = ?2 AND size = ?3",
        params![name, mtime, size],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

pub fn delete_doc(conn: &Connection, name: &str) -> Result<()> {
    conn.execute("DELETE FROM chunks WHERE doc_id = (SELECT id FROM docs WHERE name = ?1)", params![name])?;
    conn.execute("DELETE FROM docs WHERE name = ?1", params![name])?;
    Ok(())
}

/// Resolve a user-supplied name to the stored document name.
/// Accepts the exact stored name, with or without `.txt`, or a unique
/// filename suffix (e.g. just the basename from list_documents).
pub fn resolve_name(conn: &Connection, name: &str) -> Result<Resolve> {
    let mut exact = || -> Result<Option<String>> {
        let mut stmt = conn.prepare("SELECT name FROM docs WHERE name = ?1")?;
        let mut rows = stmt.query_map(params![name], |r| r.get(0))?;
        Ok(rows.next().transpose()?)
    };
    if let Some(n) = exact()? {
        return Ok(Resolve::One(n));
    }
    if !name.ends_with(".txt") {
        let with_ext = format!("{name}.txt");
        let mut stmt = conn.prepare("SELECT name FROM docs WHERE name = ?1")?;
        let mut rows = stmt.query_map(params![with_ext], |r| r.get::<_, String>(0))?;
        if let Some(n) = rows.next().transpose()? {
            return Ok(Resolve::One(n));
        }
    }
    // Suffix match on basename: docs whose name ends with "/<name[.txt]>".
    let base = name.rsplit('/').next().unwrap_or(name);
    let mut stmt =
        conn.prepare("SELECT name FROM docs WHERE name = ?1 OR name LIKE '%/' || ?1 ORDER BY name")?;
    let mut cands: Vec<String> = stmt
        .query_map(params![base], |r| r.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !base.ends_with(".txt") {
        let with_ext = format!("{base}.txt");
        let mut stmt2 = conn.prepare(
            "SELECT name FROM docs WHERE name = ?1 OR name LIKE '%/' || ?1 ORDER BY name",
        )?;
        let mut more: Vec<String> = stmt2
            .query_map(params![with_ext], |r| r.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        cands.append(&mut more);
        cands.sort();
        cands.dedup();
    }
    match cands.len() {
        0 => Ok(Resolve::None),
        1 => Ok(Resolve::One(cands.pop().unwrap())),
        _ => Ok(Resolve::Many(cands)),
    }
}

pub enum Resolve {
    One(String),
    Many(Vec<String>),
    None,
}

pub fn read_doc(    conn: &Connection,
    name: &str,
    offset: i64,
    limit: i64,
) -> Result<Option<(String, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT c.text FROM chunks c JOIN docs d ON d.id = c.doc_id
         WHERE d.name = ?1 ORDER BY c.idx",
    )?;
    let parts: Vec<String> = stmt
        .query_map(params![name], |r| r.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parts.is_empty() {
        // Distinguish missing doc.
        let n: i64 =
            conn.query_row("SELECT COUNT(*) FROM docs WHERE name = ?1", params![name], |r| {
                r.get(0)
            })?;
        if n == 0 {
            return Ok(None);
        }
    }
    let text = parts.join(" ");
    let total = text.chars().count() as i64;
    let off = offset.max(0).min(total);
    let lim = if limit <= 0 { total - off } else { limit.min(total - off) };
    let slice: String = text.chars().skip(off as usize).take(lim as usize).collect();
    Ok(Some((slice, total)))
}

/// FTS5 BM25 search. Returns (chunk_id, bm25) with *lower = better*
/// (bm25() returns negative values, more negative is more relevant).
pub fn fts_search(conn: &Connection, query: &str, limit: i64) -> Result<Vec<(i64, f64)>> {
    let match_q = to_match_query(query);
    if match_q.is_empty() {
        return Ok(vec![]);
    }
    let mut stmt = conn.prepare(
        "SELECT rowid, bm25(chunk_fts) FROM chunk_fts
         WHERE chunk_fts MATCH ?1 ORDER BY bm25(chunk_fts) LIMIT ?2",
    )?;
    let rows = stmt
        .query_map(params![match_q, limit], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn to_match_query(q: &str) -> String {
    let toks: Vec<String> = q
        .split(|c: char| !(c.is_alphanumeric() || c == '+' || c == '#' || c == '_'))
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    toks.join(" OR ")
}

pub fn chunk_by_id(conn: &Connection, id: i64) -> Result<Option<ChunkRow>> {
    let mut stmt = conn.prepare(
        "SELECT c.id, d.name, c.idx, c.text, c.embedding
         FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.id = ?1",
    )?;
    let mut rows = stmt.query_map(params![id], |r| {
        let blob: Vec<u8> = r.get(4)?;
        Ok(ChunkRow {
            id: r.get(0)?,
            doc: r.get(1)?,
            idx: r.get(2)?,
            text: r.get(3)?,
            embedding: crate::embed::from_blob(&blob),
        })
    })?;
    Ok(rows.next().transpose()?)
}

pub fn all_chunk_embeddings(conn: &Connection) -> Result<Vec<(i64, Vec<f32>)>> {
    let mut stmt = conn.prepare("SELECT id, embedding FROM chunks")?;
    let rows = stmt
        .query_map([], |r| {
            let blob: Vec<u8> = r.get(1)?;
            Ok((r.get::<_, i64>(0)?, crate::embed::from_blob(&blob)))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}
