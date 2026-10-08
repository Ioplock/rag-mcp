//! Text extraction per file type.
//!
//! - `.txt` / `.md`: read as-is (chunking decides structure later).
//! - `.pdf`: `pdftotext -layout` (poppler) from PATH. If the result is
//!   suspiciously short and `ocr` is enabled, fall back to per-page OCR:
//!   `pdftoppm -png` + `tesseract -l <langs>` (both from PATH).
//!   Needs `rus.traineddata` visible via TESSDATA_PREFIX — download once:
//!   `curl -o ~/.cache/rag-mcp/tessdata/rus.traineddata
//!    https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata`
//!   (same for `eng.traineddata`).

use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Source {
    Text,
    Ocr,
}

pub struct Extracted {
    pub text: String,
    pub source: Source,
}

pub fn extract(path: &std::path::Path, cfg: &crate::config::Config) -> Result<Option<Extracted>> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "txt" | "md" | "markdown" => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            Ok(Some(Extracted { text, source: Source::Text }))
        }
        "pdf" => extract_pdf(path, cfg),
        _ => Ok(None), // unreachable: include globs decide
    }
}

fn run(cmd: &str, args: &[&str]) -> Result<String> {
    let out = std::process::Command::new(cmd)
        .args(args)
        .output()
        .with_context(|| format!("run {cmd} (is it on PATH?)"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("{cmd} failed: {err}");
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn extract_pdf(path: &std::path::Path, cfg: &crate::config::Config) -> Result<Option<Extracted>> {
    let p = path.to_string_lossy();
    let text = run("pdftotext", &["-layout", "-enc", "UTF-8", &p, "-"])?;
    // Per-page heuristic: a digital page holds hundreds of chars, a scanned
    // one (or a failed extraction) almost none.
    let pages = run("pdfinfo", &[&p])?
        .lines()
        .find_map(|l| {
            let n = l.strip_prefix("Pages:")?.trim();
            n.parse::<usize>().ok()
        })
        .unwrap_or(1)
        .max(1);
    let per_page = text.chars().count() / pages;
    if per_page >= cfg.ocr_min_chars {
        return Ok(Some(Extracted { text, source: Source::Text }));
    }
    if !cfg.ocr {
        eprintln!(
            "SKIP {} ({per_page} chars/page over {pages} pages, no digital text layer; enable `ocr` for scans)",
            path.display()
        );
        return Ok(None);
    }
    eprintln!("OCR fallback for {} ...", path.display());
    let tmp = std::env::temp_dir().join(format!("rag-ocr-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).context("ocr tmpdir")?;
    let prefix = tmp.join("page");
    let prefix_s = prefix.to_string_lossy().into_owned();
    run("pdftoppm", &["-r", "200", "-png", &p, &prefix_s])?;
    let mut pages: Vec<std::path::PathBuf> = std::fs::read_dir(&tmp)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|e| e.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    pages.sort();
    let mut out = String::new();
    for (i, img) in pages.iter().enumerate() {
        let t = run(
            "tesseract",
            &[&img.to_string_lossy(), "stdout", "-l", &cfg.ocr_langs, "--psm", "6"],
        )?;
        out.push_str(&format!("\n\n[page {}]\n{t}", i + 1));
    }
    std::fs::remove_dir_all(&tmp).ok();
    Ok(Some(Extracted { text: out, source: Source::Ocr }))
}
