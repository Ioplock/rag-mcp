//! MCP server (stdio) exposing the transcript RAG as three tools.

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

pub struct AppState {
    pub db_path: PathBuf,
    pub cfg: crate::config::Config,
    /// Loaded on first `search` — keeps server startup instant.
    pub embedder: std::sync::OnceLock<crate::embed::Embedder>,
}

impl AppState {
    pub fn new(cfg: crate::config::Config, db_path: PathBuf) -> Self {
        Self { db_path, cfg, embedder: std::sync::OnceLock::new() }
    }

    pub fn embedder(&self) -> anyhow::Result<&crate::embed::Embedder> {
        if let Some(e) = self.embedder.get() {
            return Ok(e);
        }
        // Rare race: two threads may both load; the loser is dropped. Harmless.
        let fresh = crate::embed::Embedder::load(&self.cfg)?;
        Ok(self.embedder.get_or_init(|| fresh))
    }
}

#[derive(Clone)]
pub struct Rag {
    state: Arc<AppState>,
    tool_router: ToolRouter<Self>,
}

impl Rag {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state, tool_router: Self::tool_router() }
    }

    fn open_db(&self) -> anyhow::Result<rusqlite::Connection> {
        crate::store::open_db(&self.state.db_path)
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// User question or keywords, Russian or English.
    pub query: String,
    /// How many chunks to return (1-50, default 5).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}
fn default_top_k() -> usize {
    5
}

#[derive(Serialize, JsonSchema)]
pub struct SearchHit {
    pub document: String,
    pub chunk: i64,
    pub score: f32,
    pub text: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct ReadArgs {
    /// Exact document name from list_documents (with or without .txt).
    pub name: String,
    /// Character offset into the document text (default 0).
    #[serde(default)]
    pub offset_chars: i64,
    /// Max characters to return (default 8000, <=0 means rest of doc).
    #[serde(default = "default_limit")]
    pub limit_chars: i64,
}
fn default_limit() -> i64 {
    8000
}

#[derive(Serialize, JsonSchema)]
pub struct ReadOut {
    pub document: String,
    pub total_chars: i64,
    pub offset_chars: i64,
    pub text: String,
}

#[derive(Serialize, JsonSchema)]
pub struct DocInfo {
    pub name: String,
    pub chars: i64,
    pub chunks: i64,
}

#[tool_router]
impl Rag {
    #[tool(
        name = "search",
        description = "Hybrid full-text + semantic search over Android lecture transcripts (Kotlin, Clean Architecture, MVVM/MVP/MVI, DI, testing). Returns the most relevant passages with document names. Use this first for any question about the lecture content."
    )]
    async fn search(&self, p: rmcp::handler::server::wrapper::Parameters<SearchArgs>) -> Result<String, String> {
        let top_k = p.0.top_k.clamp(1, 50);
        let conn = self.open_db().map_err(|e| e.to_string())?;
        let emb = self.state.embedder().map_err(|e| e.to_string())?;
        let hits = crate::search::hybrid_search(&conn, emb, &p.0.query, top_k)
            .map_err(|e| e.to_string())?;
        let out: Vec<SearchHit> = hits
            .into_iter()
            .map(|h| SearchHit {
                document: h.doc,
                chunk: h.chunk_idx,
                score: (h.score * 1000.0).round() / 1000.0,
                text: h.text,
            })
            .collect();
        serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
    }

    #[tool(
        name = "read_document",
        description = "Read a full transcript (or a slice of it) verbatim by exact document name from list_documents. Use after search to get complete context."
    )]
    async fn read_document(&self, p: rmcp::handler::server::wrapper::Parameters<ReadArgs>) -> Result<String, String> {
        let name = p.0.name.trim().to_string();
        let conn = self.open_db().map_err(|e| e.to_string())?;
        let resolved = match crate::store::resolve_name(&conn, &name).map_err(|e| e.to_string())? {
            crate::store::Resolve::One(n) => n,
            crate::store::Resolve::None => {
                return Err(format!("document not found: {name} (see list_documents)"));
            }
            crate::store::Resolve::Many(cands) => {
                return Err(format!(
                    "ambiguous name, pick one:\n{}",
                    cands.join("\n")
                ));
            }
        };
        match crate::store::read_doc(&conn, &resolved, p.0.offset_chars, p.0.limit_chars)
            .map_err(|e| e.to_string())?
        {
            None => Err(format!("document not found: {name}")),
            Some((text, total)) => {
                let out = ReadOut {
                    document: resolved,
                    total_chars: total,
                    offset_chars: p.0.offset_chars.max(0),
                    text,
                };
                serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
            }
        }
    }

    #[tool(
        name = "list_documents",
        description = "List all lecture transcripts and the Kotlin cheatsheet available in the RAG index with sizes. Use to discover coverage before searching."
    )]
    async fn list_documents(&self) -> Result<String, String> {
        let conn = self.open_db().map_err(|e| e.to_string())?;
        let docs = crate::store::list_docs(&conn).map_err(|e| e.to_string())?;
        let out: Vec<DocInfo> = docs
            .into_iter()
            .map(|d| DocInfo { name: d.name, chars: d.chars, chunks: d.n_chunks })
            .collect();
        serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
    }
}

#[tool_handler]
impl ServerHandler for Rag {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        let mut info = rmcp::model::ServerInfo::default();
        info.instructions = Some(format!(
            "Collection '{}' ({}). Start with `search`, then `read_document` for full context.",
            self.state.cfg.name, self.state.cfg.description
        ));
        info
    }
}
