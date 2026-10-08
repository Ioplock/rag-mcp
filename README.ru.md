# rag-mcp

*[Read in English](README.md)*

Локальный гибридный RAG (SQLite FTS5 + эмбеддинги на устройстве) в виде
MCP-сервера. Один Rust-бинарь ~16 МБ, без сервисов, облаков и API-ключей.
Заточен под слабые ноутбуки и Nix-флейки.

## Как устроено

Документы режутся на чанки (~1000 символов, с учётом предложений) и кладутся в
один SQLite-файл вместе с:

- **BM25-индексом FTS5** — точные совпадения: идентификаторы кода, имена API, термины;
- **эмбеддингом 384 dim** на чанк от `paraphrase-multilingual-MiniLM-L12-v2`
  (мультиязычная, включая русский; работает на CPU через
  [candle](https://github.com/huggingface/candle), без ONNX-рантайма и torch) —
  смысл, перефразы, опечатки ASR/OCR.

На запрос оба сигнала нормируются (min-max) и складываются:
`0.5·BM25 + 0.5·cosine`. К каждому хиту модель получает имя документа и номер чанка.

Форматы:

| Формат | Как обрабатывается |
|---|---|
| `.txt` | Упаковка по предложениям (переваривает даже однострочные ASR-простыни без переносов) |
| `.md` | Рез по заголовкам, код-блоки ``` никогда не рвутся |
| `.pdf` | `pdftotext -layout` (poppler) из `PATH` |
| сканы / картинки в PDF | Opt-in fallback на `tesseract` OCR (`rus+eng`), эвристика по страницам |

## MCP-инструменты

| Инструмент | Аргументы | Что возвращает |
|---|---|---|
| `search` | `query`, `top_k` (1–50, default 5) | Ранжированные чанки: `document`, `chunk`, `score`, `text` |
| `read_document` | `name`, `offset_chars`, `limit_chars` | Документ целиком дословно (или кусок). Понимает точное имя, имя без расширения и уникальный суффикс; при неоднозначности возвращает кандидатов |
| `list_documents` | — | Все документы коллекции с размерами |

Поля `name`/`description` из `rag.json` отдаются в MCP `instructions`, чтобы
модель знала, что лежит в коллекции.

## Подключение к проекту (flake input)

```nix
# flake.nix
inputs.rag-mcp.url = "github:Ioplock/rag-mcp";
# ...
devShells.default = pkgs.mkShell {
  packages = [
    inputs.rag-mcp.packages.${system}.default
    pkgs.poppler-utils              # если индексируешь PDF
    # pkgs.tesseract                # если в rag.json "ocr": true
  ];
};
```

```jsonc
// rag.json (все поля — в schema.json)
{ "name": "my-docs", "description": "Что внутри, для модели",
  "docs": ["docs"], "include": ["**/*.md", "**/*.pdf"], "db": ".rag/index.db" }
```

```jsonc
// opencode.json — сам opencode запускай изнутри `nix develop` (или direnv),
// чтобы `rag-mcp` был в PATH
{ "mcp": { "docs": {
  "type": "local", "command": ["rag-mcp", "serve"], "timeout": 60000 } } }
```

```bash
rag-mcp sync    # индексация / инкрементальное обновление (mtime+size, удалённые забывает)
rag-mcp index --force   # полная пересборка
rag-mcp stats   # обзор коллекции
```

Переменные окружения: `RAG_CONFIG` (default `rag.json`), `RAG_DOCS`, `RAG_DB`,
`RAG_MODEL_DIR`.

## Разовые скачивания

Модель эмбеддингов (~450 МБ, ru+en) в `~/.cache/rag-mcp/model`:

```bash
mkdir -p ~/.cache/rag-mcp/model && cd $_
for f in config.json tokenizer.json model.safetensors; do
  curl -L -o $f https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/$f
done
```

Для OCR один раз нужны traineddata:

```bash
mkdir -p ~/.cache/rag-mcp/tessdata && cd $_
for l in rus eng; do
  curl -L -o $l.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata
done
export TESSDATA_PREFIX=~/.cache/rag-mcp/tessdata
```

## Заметки

- Старт сервера мгновенный: модель грузится лениво на первом `search`
  (`list_documents`/`read_document` её вообще не трогают).
- `serve` пишет в stderr предупреждение, если индекс протух — лечится `sync`.
- Бинарь один на все проекты; у каждого проекта свои `rag.json` + база индекса.
- Тулчейн OCR (`nix shell nixpkgs#tesseract`) — тяжёлая разовая загрузка
  (~0.5 ГБ); включай `ocr` только там, где реально есть сканы.
- Разработка: `nix develop` здесь даёт полный Rust-тулчейн
  (`cargo test`, `cargo build --release`).

## Лицензия

MIT — см. [LICENSE](LICENSE).
