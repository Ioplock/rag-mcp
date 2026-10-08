//! Project configuration (`rag.json`).
//!
//! Env overrides (all optional):
//!   RAG_CONFIG — path to config file (default `rag.json` in cwd)
//!   RAG_DOCS   — single docs dir, overrides `docs`
//!   RAG_DB     — index db path, overrides `db`

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_docs")]
    pub docs: Vec<String>,
    #[serde(default = "default_include")]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default = "default_db")]
    pub db: String,
    #[serde(default = "default_chunk_chars")]
    pub chunk_chars: usize,
    #[serde(default = "default_overlap_chars")]
    pub overlap_chars: usize,
    /// `false` (default): skip pages/files with no digital text.
    /// `true`: fall back to `tesseract` OCR for them.
    #[serde(default)]
    pub ocr: bool,
    #[serde(default = "default_ocr_langs")]
    pub ocr_langs: String,
    /// Chars per page below which the pdftotext output counts as "no text".
    #[serde(default = "default_ocr_min_chars")]
    pub ocr_min_chars: usize,
    #[serde(default = "default_model_dir")]
    pub model_dir: String,
}

fn default_name() -> String {
    "docs".to_string()
}
fn default_docs() -> Vec<String> {
    vec!["materials".to_string()]
}
fn default_include() -> Vec<String> {
    vec!["**/*.txt".to_string(), "**/*.md".to_string(), "**/*.pdf".to_string()]
}
fn default_db() -> String {
    ".rag/index.db".to_string()
}
fn default_chunk_chars() -> usize {
    1000
}
fn default_overlap_chars() -> usize {
    180
}
fn default_ocr_langs() -> String {
    "rus+eng".to_string()
}
fn default_ocr_min_chars() -> usize {
    30
}
fn default_model_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{home}/.cache/rag-mcp/model")
}

impl Default for Config {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

pub fn load() -> Result<Config> {
    let path =
        std::env::var("RAG_CONFIG").unwrap_or_else(|_| "rag.json".to_string());
    let mut cfg: Config = if std::path::Path::new(&path).exists() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read {path}"))?;
        serde_json::from_str(&raw).with_context(|| format!("parse {path}"))?
    } else {
        eprintln!("no {path}, using defaults (materials/**/*.txt)");
        Config::default()
    };
    if let Ok(d) = std::env::var("RAG_DOCS") {
        cfg.docs = vec![d];
    }
    if let Ok(d) = std::env::var("RAG_DB") {
        cfg.db = d;
    }
    if let Ok(d) = std::env::var("RAG_MODEL_DIR") {
        cfg.model_dir = d;
    }
    if cfg.chunk_chars < 100 {
        anyhow::bail!("chunk_chars too small (<100)");
    }
    if cfg.overlap_chars * 2 > cfg.chunk_chars {
        anyhow::bail!("overlap_chars must be < chunk_chars/2");
    }
    Ok(cfg)
}

/// Directory the config file lives in — `docs`/`db` resolve against it.
pub fn dir() -> std::path::PathBuf {
    let p = std::env::var("RAG_CONFIG").unwrap_or_else(|_| "rag.json".to_string());
    let p = std::path::PathBuf::from(p);
    if p.is_absolute() {
        p.parent().map(|x| x.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("/"))
    } else {
        let cwd =
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let abs = cwd.join(&p);
        abs.parent().map(|x| x.to_path_buf()).unwrap_or(cwd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_parse() {
        let c: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(c.chunk_chars, 1000);
        assert!(!c.ocr);
        assert!(c.include.iter().any(|g| g.ends_with(".pdf")));
    }
}
