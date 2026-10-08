# rag-mcp

*[Читать на русском](README.ru.md)*

Local hybrid RAG (SQLite FTS5 + on-device embeddings) exposed as an MCP server.
One ~16 MB Rust binary, no services, no cloud, no API keys. Built for weak
laptops and Nix flakes.

## How it works

Documents are split into chunks (~1000 chars, sentence-aware) and stored in a
single SQLite file together with:

- an **FTS5 BM25 index** — exact matches for code identifiers, API names, terms;
- a **384-dim embedding** per chunk from `paraphrase-multilingual-MiniLM-L12-v2`
  (multilingual incl. Russian, runs on CPU via [candle](https://github.com/huggingface/candle),
  no ONNX runtime, no torch) — meaning, paraphrases, ASR/OCR typos.

At query time both signals are min-max normalized and combined
`0.5·BM25 + 0.5·cosine`. The model always gets the source document name and
chunk id with every hit.

Supported formats:

| Format | How |
|---|---|
| `.txt` | Sentence packing (handles single-line ASR transcripts with no line breaks) |
| `.md` | Header-aware splitting, fenced code blocks are never cut |
| `.pdf` | `pdftotext -layout` (poppler) from `PATH` |
| scans / images inside PDFs | Opt-in `tesseract` OCR fallback (`rus+eng`), per-page heuristic |

## MCP tools

| Tool | Args | What it returns |
|---|---|---|
| `search` | `query`, `top_k` (1–50, default 5) | Ranked chunks: `document`, `chunk`, `score`, `text` |
| `read_document` | `name`, `offset_chars`, `limit_chars` | Full document verbatim (or a slice). Accepts the exact stored name, with/without extension, or a unique filename suffix; ambiguous names return candidates |
| `list_documents` | — | All indexed documents with sizes |

Server `instructions` (from `rag.json` `name`/`description`) tell the model what
the collection contains. Tool descriptions in `tools/list` are generated from
the same fields, so every collection advertises itself correctly.

## Use in your project (flake input)

```nix
# flake.nix
inputs.rag-mcp.url = "github:Ioplock/rag-mcp";
# ...
devShells.default = pkgs.mkShell {
  packages = [
    inputs.rag-mcp.packages.${system}.default
    pkgs.poppler-utils              # iff you index PDFs
    # pkgs.tesseract                # iff rag.json has "ocr": true
  ];
};
```

```jsonc
// rag.json (see schema.json for all fields)
{ "name": "my-docs", "description": "What is inside, for the model",
  "docs": ["docs"], "include": ["**/*.md", "**/*.pdf"], "db": ".rag/index.db" }
```

```jsonc
// opencode.json — run opencode from inside `nix develop` (or direnv),
// so `rag-mcp` is on PATH
{ "mcp": { "docs": {
  "type": "local", "command": ["rag-mcp", "serve"], "timeout": 60000 } } }
```

```bash
rag-mcp sync    # index / incremental update (mtime+size, drops deleted files)
rag-mcp index --force   # full rebuild
rag-mcp stats   # collection overview
```

Env overrides: `RAG_CONFIG` (default `rag.json`), `RAG_DOCS`, `RAG_DB`,
`RAG_MODEL_DIR`.

## One-time downloads

Embedding model (~450 MB, ru+en) into `~/.cache/rag-mcp/model`:

```bash
mkdir -p ~/.cache/rag-mcp/model && cd $_
for f in config.json tokenizer.json model.safetensors; do
  curl -L -o $f https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/$f
done
```

For OCR you also need traineddata once:

```bash
mkdir -p ~/.cache/rag-mcp/tessdata && cd $_
for l in rus eng; do
  curl -L -o $l.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata
done
export TESSDATA_PREFIX=~/.cache/rag-mcp/tessdata
```

## Notes

- Server startup is instant: the model loads lazily on the first `search`
  (`list_documents`/`read_document` never need it).
- `serve` warns on stderr when the index is stale — rerun `sync`.
- One binary serves all projects; each project has its own `rag.json` + index db.
- OCR toolchain (`nix shell nixpkgs#tesseract`) is a heavy one-time download
  (~0.5 GB); enable `ocr` only where scans actually exist.
- Hack on it: `nix develop` here gives the full Rust toolchain
  (`cargo test`, `cargo build --release`).

## License

MIT — see [LICENSE](LICENSE).
