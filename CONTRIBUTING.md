# Contributing to RepoQA

This document is the contributor guide and development log for RepoQA. It explains what’s implemented in this repository today, and how it maps onto the system described in `docs/pdflatex/RepoQA_Report.pdf` (the “RepoQA paper/report”).

---

## Table of Contents

1. [Project Setup](#1-project-setup)
  - [RepoQA paper alignment (high level)](#repoqa-paper-alignment-high-level)
2. [Shared Data Models](#2-shared-data-models)
3. [Configuration](#3-configuration)
4. [Phase 1 — Repository Ingestion](#4-phase-1--repository-ingestion)
  - [Repo Crawler](#41-repo-crawler)
  - [AST Parser Registry](#42-ast-parser-registry)
  - [AST-Aware Chunker](#43-ast-aware-chunker)
  - [Non-Code Chunker](#44-non-code-chunker)
  - [Ingestion Pipeline Orchestrator](#45-ingestion-pipeline-orchestrator)
5. [Phase 2 — Hierarchical Summarization](#5-phase-2--hierarchical-summarization)
  - [Prompt Templates](#51-prompt-templates)
  - [File Summarizer](#52-file-summarizer)
  - [Directory Summarizer](#53-directory-summarizer)
  - [Project Summarizer](#54-project-summarizer)
  - [Summarization Pipeline Orchestrator](#55-summarization-pipeline-orchestrator)
6. [Phase 3 — Embedding & Vector Storage](#6-phase-3--embedding--vector-storage)
  - [Metadata Schema](#61-metadata-schema)
  - [Embedder](#62-embedder)
  - [ChromaDB Store](#63-chromadb-store)
  - [Embedding Pipeline Orchestrator](#64-embedding-pipeline-orchestrator)
7. [CLI Scripts](#7-cli-scripts)
8. [Tests](#8-tests)
9. [Environment Setup](#9-environment-setup)
10. [What Comes Next](#10-what-comes-next)

---

## RepoQA paper alignment (high level)

The report (`docs/pdflatex/RepoQA_Report.pdf`, Figure 1) describes RepoQA as a **four-stage system**:

- **Stage 1 (offline, once per repo)**: clone/filter → Tree-sitter parsing → AST-aware chunking → non-code indexing → hierarchical summaries (project → dir → file) → embeddings → storage in **ChromaDB + BM25**, plus an **import dependency graph**
- **Stage 2 (online, per question)**: question-type classification (architecture/logic/deployment) → retrieval strategy routing → hybrid retrieval (semantic + BM25 + dependency-graph traversal) → fuse results with **Reciprocal Rank Fusion (RRF)** and optional **query expansion**
- **Stage 3 (online)**: structured prompt (summaries → code → question) → answer generation → **citation verification**
- **Stage 4 (online)**: conversational UI (report proposes Streamlit) with short conversation memory and clickable source references

**What this repo currently contains (as of this doc):**

- **Implemented (Stage 1)**: ingestion + chunking, hierarchical summarization, embedding + ChromaDB persistence (see sections 4–6 and scripts in section 7).
- **Partially implemented (Stage 2/3)**: a lightweight question-type classifier exists at `repoqa/evaluation/question_classifier.py` (regex-based, for dataset labeling and experimentation), but the full hybrid retrieval + routing + RRF + citation verifier + UI are tracked in “What Comes Next” (section 10).
- **Implemented (evaluation utilities)**: dataset mining tooling (`scripts/build_dataset.py`, `repoqa/evaluation/`*) to build a CoReQA-style dataset from GitHub issues.

This section is intentionally “paper-first”; the rest of the document is “code-first” and points at the concrete modules in this repository.

## 1. Project Setup

**Files created:** `requirements.txt`, `.env`, `.gitignore`, `pyproject.toml`

**Why:**

- An empty directory existed at `/repoqa`. The project needed a dependency manifest before any code could be written or tested.
- `requirements.txt` lists every library used across all three pipeline phases (tree-sitter grammar packages, chromadb, langchain, openai, anthropic, tiktoken, pydantic-settings, etc.) so contributors can reproduce the exact environment.
- `.env.example` documents every environment variable the system reads (API keys, model names, ChromaDB path, token budgets) without committing real secrets. Contributors copy it to `.env` and fill in their own values.
- `.gitignore` prevents `.env`, `__pycache__`, `data/`, and the virtual environment from being committed.
- `pyproject.toml` sets the `pytest` test path and `asyncio_mode` so tests can be run with a single `pytest` command.

**Directory structure created:**

```
repoqa/
├── config/
├── repoqa/
│   ├── ingestion/
│   ├── summarization/
│   └── embedding/
├── scripts/
├── tests/
└── data/           ← gitignored; holds chunks.json, summaries.json, chroma/
```

---

## 2. Shared Data Models

**File:** `repoqa/models.py`

**Why first:** Every other module imports these dataclasses. Defining them once in a single file avoids circular imports and makes the data contracts explicit.

**What was defined:**


| Class        | Purpose                                                                                                                                                                                                                                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileRecord` | Represents one file discovered during crawling: relative repo path, absolute path, language classification, size, and raw text content.                                                                                                                                                                                      |
| `Chunk`      | A semantically bounded piece of content (a function, a class, a config section, a prose paragraph). Carries its source `FileRecord`, line range, chunk type, symbol name, and token count. Its `id` is a deterministic 16-character SHA-256 hash of `repo_path:start_line` so re-running the pipeline produces the same IDs. |
| `Summary`    | An LLM-generated summary at file, directory, or project level. Carries the list of source chunk/summary IDs that produced it, enabling traceability.                                                                                                                                                                         |


**Key design decision — deterministic IDs:** Using `sha256(repo_path + start_line)` means upserts into ChromaDB are idempotent. Re-ingesting a repository that hasn't changed does not create duplicate documents.

---

## 3. Configuration

**File:** `config/settings.py`

**Why:** All three pipeline phases need API keys, model names, file paths, and token budgets. Rather than reading `os.environ` in scattered places, a single `Settings` class (backed by `pydantic-settings`) reads from the `.env` file and validates types at startup. `@lru_cache` on `get_settings()` means the `.env` file is parsed exactly once per process.

**Fields:**


| Field                  | Default                  | Purpose                                                                    |
| ---------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `llm_provider`         | `openai`                 | Switch between OpenAI and Anthropic for summarization                      |
| `chat_model`           | `gpt-4o-mini`            | LLM used for summarization (cheaper during development)                    |
| `embedding_model`      | `text-embedding-3-small` | OpenAI embedding model                                                     |
| `chroma_persist_dir`   | `./data/chroma`          | Where ChromaDB writes its on-disk index                                    |
| `chunk_max_tokens`     | `512`                    | Maximum tokens per chunk (balances context size vs. retrieval granularity) |
| `chunk_overlap_tokens` | `64`                     | Overlap between adjacent chunks when splitting large nodes                 |
| `summary_max_tokens`   | `4096`                   | Maximum tokens the LLM may generate per summary                            |


---

## 4. Phase 1 — Repository Ingestion

### 4.1 Repo Crawler

**File:** `repoqa/ingestion/repo_crawler.py`

**Why:** Before any parsing or chunking can happen, the system needs to know which files in the repository are relevant. A naive `glob("**/*")` would include binary files, build artifacts, generated code, and dependency directories — all of which add noise without value.

**What it does:**

- Walks the filesystem with `os.walk`, pruning skip-directories in-place (`.git`, `node_modules`, `__pycache__`, `build`, `dist`, `vendor`, etc.) so they are never descended into.
- Classifies each file by extension using `_EXT_TO_LANGUAGE` and by filename using `_SPECIAL_NAMES` (e.g., `Dockerfile`, `Makefile`). Files with `language="unknown"` are skipped.
- Probes for binary content by reading the first 8 KB and checking for null bytes. Binary files are skipped.
- Reads accepted files as UTF-8 text (with `errors="replace"` to handle encoding edge cases).
- Returns a list of `FileRecord` objects.

**Why skip unknown languages:** The downstream chunkers and parsers only understand specific languages. Feeding them random binary or generated files would produce meaningless chunks.

---

### 4.2 AST Parser Registry

**File:** `repoqa/ingestion/ast_parser.py`

**Why:** Tree-sitter is the standard library for fast, error-tolerant parsing across many languages. The new tree-sitter ≥0.21 Python API requires importing each grammar as a separate Python package (`tree-sitter-python`, `tree-sitter-go`, etc.) and calling `Language(capsule)` to construct the language object. The registry wraps this boilerplate and handles missing grammar packages gracefully via lazy imports — if `tree-sitter-java` is not installed, Java files fall back to line-based splitting rather than crashing.

**Supported languages:** Python, JavaScript, TypeScript (tsx variant), Java, Go, Rust, C, C++

**Key method:** `parse(language, source) -> tree_sitter.Tree | None` — returns `None` for unsupported or parse-error cases, so callers always have a clean fallback path.

---

### 4.3 AST-Aware Chunker

**File:** `repoqa/ingestion/ast_chunker.py`

**Why this is the most important component in Phase 1:** The quality of retrieval depends entirely on the quality of chunks. A naive character-window splitter will cut a function in half, making it impossible for the LLM to understand. AST-aware chunking guarantees every chunk is a complete syntactic unit.

**Algorithm:**

1. Parse the file with `ASTParserRegistry`.
2. Walk the root node's direct children, collecting nodes whose type appears in `_TOP_LEVEL_TYPES` for that language (e.g., `function_definition`, `class_definition` for Python).
3. For each structural node: if its token count ≤ `max_tokens`, emit it as one chunk. If it exceeds `max_tokens` (e.g., a very large class), recursively split by its direct children with `overlap_tokens` of shared context prepended.
4. Collect all source lines not covered by any structural node (imports, global constants, module-level code) and emit them as `"module"` chunks.
5. If the parser is unavailable or fails, fall back to a simple sliding-window line splitter.

**Token counting:** Uses `tiktoken cl100k_base` throughout. This tokenizer is used by GPT-4o and `text-embedding-3-small`, so a chunk that fits within `max_tokens` here will also fit in the embedding model's 8,191-token input limit.

**Why overlap:** When a large node is split into multiple sub-chunks, the last few lines of the previous chunk are prepended to the next. This ensures the LLM has enough surrounding context to understand a chunk even when it is retrieved in isolation.

---

### 4.4 Non-Code Chunker

**File:** `repoqa/ingestion/noncode_chunker.py`

**Why:** Architecture and deployment questions ("How is this service configured?", "How do I build this project?") require context from README files, Dockerfiles, CI configs, and `package.json` — not from source code. These files need their own chunking strategies because they are not valid ASTs.

**Strategies per file type:**


| Language       | Strategy                                                                                                           | Chunk type |
| -------------- | ------------------------------------------------------------------------------------------------------------------ | ---------- |
| `markdown`     | Split on heading boundaries (`#`, `##`, `###`). Each section becomes one chunk.                                    | `prose`    |
| `json`         | Parse with `json.loads`. If the whole file fits in token budget, one chunk. Otherwise one chunk per top-level key. | `config`   |
| `yaml`         | Same as JSON using `yaml.safe_load`.                                                                               | `config`   |
| `toml`         | Same as JSON using stdlib `tomllib`.                                                                               | `config`   |
| `dockerfile`   | Treat as text with sliding window.                                                                                 | `config`   |
| `text` / other | Sliding-window line splitter.                                                                                      | `prose`    |


All strategies fall back to the text splitter if parsing fails.

---

### 4.5 Ingestion Pipeline Orchestrator

**File:** `repoqa/ingestion/pipeline.py`

**Why:** The crawler, AST chunker, and non-code chunker need to be wired together with progress reporting, error isolation (one bad file should not abort the entire run), and JSON serialization for caching.

**What it does:**

- Calls `crawl_repo` to get all `FileRecord` objects.
- Dispatches each record to the appropriate chunker based on `language` (code languages → `ASTChunker`, others → `NonCodeChunker`).
- Wraps each file in a `try/except` so a single malformed file is logged and skipped rather than crashing the pipeline.
- Shows a `tqdm` progress bar and a `rich` summary table after completion.
- `run_and_save` serializes the full chunk list to JSON via `dataclasses.asdict`, enabling later pipeline stages to resume without re-running ingestion.
- `load_chunks` is the inverse: deserializes the JSON back into `Chunk` objects.

**Why cache to JSON:** Ingestion is CPU-bound (parsing) and can take minutes on large repositories. Summarization is API-bound (LLM calls) and costs money. Caching the output of each stage independently means you can re-run summarization with a different prompt without re-parsing the repo.

---

## 5. Phase 2 — Hierarchical Summarization

### 5.1 Prompt Templates

**File:** `repoqa/summarization/prompts.py`

**Why a separate file for prompts:** Prompt engineering is iterative. Keeping all templates in one place means they can be refined without touching any logic files. Each template is a `ChatPromptTemplate` from LangChain.

**Four templates defined:**


| Template                 | Input                                                         | Output                                                                    |
| ------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `FILE_SUMMARY_PROMPT`    | File path, language, code excerpts                            | 3–6 sentence summary of the file's purpose, exports, and patterns         |
| `DIR_SUMMARY_PROMPT`     | Directory path, bullet list of child file summaries           | 2–4 sentence summary of the directory's role                              |
| `PROJECT_SUMMARY_PROMPT` | Repo name, README excerpt, bullet list of directory summaries | 1–2 paragraph project overview covering architecture and tech stack       |
| `SYMBOL_SUMMARY_PROMPT`  | Symbol type, code snippet                                     | One-sentence docstring-style description (for future metadata enrichment) |


**Why top-down ordering in prompts:** The structured prompts mirror how a senior developer explains code: big picture first, then details. This mirrors the retrieval strategy and helps the LLM stay grounded.

---

### 5.2 File Summarizer

**File:** `repoqa/summarization/file_summarizer.py`

**Why:** Each file needs a summary that can later be used as retrieval context (stored in ChromaDB) and as input to the directory summarizer.

**Key design decisions:**

- **Representative chunk selection:** Not all chunks from a file fit in the LLM's context window. Chunks are selected greedily by priority (`function` > `class` > `module` > `config` > `prose`) until the token budget is exhausted. This ensures the most structurally important code is always included.
- **Retry with `tenacity`:** LLM API calls can fail with transient errors (rate limits, timeouts). `wait_exponential` retry with 3 attempts prevents a single API hiccup from failing the entire summarization run.

---

### 5.3 Directory Summarizer

**File:** `repoqa/summarization/dir_summarizer.py`

**Why:** File summaries alone do not capture how files relate to each other within a module. A directory summary answers questions like "What does the `auth/` directory do overall?" without requiring the LLM to read every file.

**Token management:** The bullet list of child file summaries is built greedily. If a child summary is too long, it is replaced with `<summary omitted for length>` rather than truncating mid-sentence.

---

### 5.4 Project Summarizer

**File:** `repoqa/summarization/project_summarizer.py`

**Why:** The project summary is the top of the retrieval hierarchy. It is always included in the context for architecture-type questions. It answers "What is this project?" in 1–2 paragraphs, giving the LLM the big-picture context needed to answer high-level questions.

**README handling:** The README (if present) is extracted from the chunk list and capped at 500 tokens for the project prompt. READMEs are often the best human-written description of a project's purpose and are prioritized accordingly.

---

### 5.5 Summarization Pipeline Orchestrator

**File:** `repoqa/summarization/pipeline.py`

**Why:** The three summarizers must run in a specific order (file → directory → project) and the directory structure must be inferred from file paths. The pipeline handles this bottom-up tree traversal automatically.

**Algorithm:**

1. Group chunks by `file_record.repo_path` → run `FileSummarizer` for each file.
2. Group file summaries by their parent directory (`Path(repo_path).parent`) → run `DirSummarizer` for each directory.
3. Propagate directory summaries upward (a directory summary is also a child input for its parent directory summarizer).
4. Collect all top-level directory summaries → run `ProjectSummarizer` once.

**LLM selection:** The `_build_llm` helper instantiates either `ChatOpenAI` or `ChatAnthropic` based on `settings.llm_provider`. Both expose the same LangChain interface, so all summarizers are model-agnostic.

---

## 6. Phase 3 — Embedding & Vector Storage

### 6.1 Metadata Schema

**File:** `repoqa/embedding/metadata.py`

**Why:** ChromaDB metadata values must be flat scalars (`str | int | float | bool`). Defining a `ChunkMetadata` TypedDict enforces this constraint at development time and documents exactly what fields are available for `where` filters during retrieval.

**Key metadata fields:**


| Field           | Used for                                                                        |
| --------------- | ------------------------------------------------------------------------------- |
| `doc_type`      | Filter to retrieve only chunks (`"chunk"`) or only summaries (`"summary"`)      |
| `summary_level` | Filter to retrieve only file-level, directory-level, or project-level summaries |
| `file_path`     | Associate retrieved chunks back to their source file                            |
| `language`      | Filter retrieval to a specific language                                         |
| `repo_name`     | Isolate one repository's documents from another in the shared collection        |
| `chunk_type`    | Filter to retrieve only functions, classes, etc.                                |


**Why a single collection:** Using one ChromaDB collection with metadata filtering (rather than per-level collections) allows the retrieval layer to issue hybrid queries that return both raw code chunks and summaries in a single call — which is needed for the architecture question routing strategy.

---

### 6.2 Embedder

**File:** `repoqa/embedding/embedder.py`

**Why:** Embedding is the most expensive step (API cost and latency). Pre-computing embeddings outside of ChromaDB gives full control over batching, retry logic, and caching. If ChromaDB upsert fails partway, the embeddings do not need to be recomputed.

**Note on the report vs this repository:** The report’s architecture diagram calls out `all-mpnet-base-v2` as the embedding model (a local SentenceTransformers-style embedding). This repository’s current implementation uses the OpenAI embeddings API (default `text-embedding-3-small`, configured via `EMBEDDING_MODEL`). If you update the implementation to match the report exactly, this is the main swap to make (local encoder instead of OpenAI).

**Key decisions:**

- **Batch size of 100:** The OpenAI embeddings API accepts up to 2,048 inputs per request. Batches of 100 balance throughput against request size.
- **Token truncation:** `text-embedding-3-small` accepts at most 8,191 tokens. Any chunk longer than this (rare but possible with very large generated files) is truncated before embedding rather than causing an API error.
- **Tenacity retry:** 5 attempts with exponential backoff (2s → 60s). Rate limit errors are transient and always recoverable with a wait.

---

### 6.3 ChromaDB Store

**File:** `repoqa/embedding/chroma_store.py`

**Why ChromaDB:** It is the only vector store in the stack that runs entirely in-process with no external service dependency, persists to disk, and supports metadata-filtered queries. For a research project, this eliminates infrastructure complexity.

**Key methods:**

- `upsert_chunks` / `upsert_summaries` — use ChromaDB's `upsert` (not `add`) so re-running the pipeline is safe. Documents with the same ID are updated, not duplicated.
- `query` — accepts a pre-computed embedding vector and an optional `where` filter. Returns documents, metadata, and cosine distances.
- `delete_repo` — removes all documents for a given `repo_name`. Used when re-ingesting a repository after code changes.
- Collection is configured with `hnsw:space=cosine` to use cosine similarity (standard for text embeddings).

---

### 6.4 Embedding Pipeline Orchestrator

**File:** `repoqa/embedding/pipeline.py`

**Why:** Chunks and summaries need to be embedded together in a single batched API call (to minimize round trips and cost), then split back and upserted separately with their respective metadata.

**What it does:**

1. Concatenates all chunk texts and summary texts into one list.
2. Calls `Embedder.embed_batch` once on the combined list.
3. Splits the resulting embedding list back at the boundary between chunks and summaries.
4. Calls `ChromaStore.upsert_chunks` and `ChromaStore.upsert_summaries`.
5. Logs final document count from ChromaDB.

`run_from_cache` loads chunks and summaries from the JSON files produced by the upstream pipelines — the typical production entry point when all three stages are run as separate CLI commands.

---

## 7. CLI Scripts

**Files:** `scripts/ingest_repo.py`, `scripts/build_summaries.py`, `scripts/build_index.py`

**Why three separate scripts:** Each pipeline stage has a different cost profile and failure mode:

- Ingestion is fast and free (local parsing only).
- Summarization is slow and costs LLM API credits.
- Embedding costs embedding API credits.

Separating them means you can re-run only the stage that failed without repeating the earlier expensive work. Each script saves its output to a JSON file, which the next script reads as input.

**Usage:**

```bash
# Step 1 — ingest (fast, free, always safe to re-run)
python scripts/ingest_repo.py --repo-path /path/to/repo --output data/chunks.json

# Step 2 — summarize (slow, costs LLM API calls)
python scripts/build_summaries.py --chunks data/chunks.json --repo-root /path/to/repo --output data/summaries.json

# Step 3 — embed and index (costs embedding API calls)
python scripts/build_index.py --chunks data/chunks.json --summaries data/summaries.json --repo-name my-repo

```

Using uv

```bash
uv run scripts/ingest_repo.py --repo-path /repoqa/flask --output data/flask/transformers_chunks.json

uv run scripts/build_summaries.py --chunks data/flask/transformers_chunks.json --repo-root /repoqa/flask --output data/flask/summaries.json

```

---

## 8. Tests

**File:** `tests/test_ingestion.py`

**Why tests written immediately after Phase 1:** The ingestion pipeline (crawler, chunker) is pure logic — no API calls required. Writing tests at this point validates correctness before any money is spent on LLM or embedding calls.

**10 tests covering:**


| Test                                       | What it validates                                                                  |
| ------------------------------------------ | ---------------------------------------------------------------------------------- |
| `test_classify_file_python`                | `.py` extension → `"python"`                                                       |
| `test_classify_file_dockerfile`            | `Dockerfile` and `Dockerfile.prod` → `"dockerfile"`                                |
| `test_classify_file_unknown`               | `.bin` → `"unknown"`                                                               |
| `test_crawl_repo_on_self`                  | `node_modules/` is excluded; Python and Markdown files are included                |
| `test_ast_chunker_python_functions`        | AST chunker produces chunks containing `add` and `Calculator` from sample Python   |
| `test_ast_chunker_fallback_text`           | Unsupported language falls back to sliding-window; chunks stay within token budget |
| `test_chunk_id_is_deterministic`           | Same file → same chunk IDs across two runs                                         |
| `test_noncode_markdown_splits_on_headings` | Markdown is split into ≥2 chunks at heading boundaries                             |
| `test_noncode_json_single_chunk`           | Small JSON produces one `config` chunk                                             |
| `test_noncode_dockerfile`                  | Dockerfile produces `config`-type chunks                                           |


**Run tests:**

```bash
uv run pytest tests/ -v
```

** Run test with coverage:**

```bash
# install pytest coverage first - if not installed. uv sync should already add pytest-cov
# uv add --dev pytest pytest-cov

```

---

## 9. Environment Setup

`[uv](https://docs.astral.sh/uv/)` is the recommended package manager. It is significantly faster than `pip` and manages the virtual environment automatically.

### Install uv (once, globally)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Set up the project

```bash
# Create venv + install all dependencies declared in pyproject.toml in one step
uv sync

# Install dev dependencies (pytest) as well
uv sync --extra dev

# Copy and fill in API keys
cp .env.example .env
# Edit .env: set OPENAI_API_KEY (required for summarization + embedding)
#            set ANTHROPIC_API_KEY (optional, if using Claude for summarization)
```

`uv sync` reads `pyproject.toml`, creates `.venv/` automatically, and pins exact versions in `uv.lock`. Commit `uv.lock` so every contributor installs the exact same dependency tree.

### Run commands

```bash
# Run tests
uv run pytest tests/ -v

# Run a CLI script
uv run python scripts/ingest_repo.py --repo-path /path/to/repo --output data/chunks.json
uv run python scripts/build_summaries.py --chunks data/chunks.json --repo-root /path/to/repo --output data/summaries.json
uv run python scripts/build_index.py --chunks data/chunks.json --summaries data/summaries.json --repo-name my-repo
```


`uv run` executes the command inside the managed virtual environment without needing to activate it first.

### Add / remove a dependency

```bash
# Add a runtime dependency (updates pyproject.toml and uv.lock)
uv add some-package

# Add a dev-only dependency
uv add --optional dev some-dev-package

# Remove a dependency
uv remove some-package
```

### pip fallback (no uv)

If `uv` is not available, the classic approach still works:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest pytest-asyncio
cp .env.example .env
```

---

## 10. What Comes Next

The three pipeline phases above constitute the **offline indexing layer** (report Stage 1; run once per repository). The remaining work is the **online QA layer** (report Stages 2–4), plus the **benchmark evaluation** described in the report:


| Component                    | Description                                                                                                                                       | Paper section |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Question classifier          | Classify each developer question as `architecture`, `logic`, or `deployment` using an LLM prompt                                                  | §3.2          |
| Hybrid retrieval engine      | Fuse semantic search (ChromaDB), BM25 keyword matching, and dependency-graph traversal using Reciprocal Rank Fusion (RRF)                         | §3.2          |
| Type-aware routing           | Route each question type to the appropriate retrieval strategy (top-down for architecture, function-level for logic, config-first for deployment) | §3.2          |
| Query expansion              | Expand the developer's question into additional search terms before retrieval                                                                     | §3.2          |
| Structured prompt generation | Assemble retrieved context top-down (project summary → directory summaries → code chunks) with citation requirements                              | §3.3          |
| Citation verifier            | Post-generation check that every cited file path and function name exists in the repository                                                       | §3.4          |
| Streamlit web UI             | Chat interface with source reference links                                                                                                        | §8.3          |
| CoReQA evaluation            | Run the CoReQA benchmark and ablations; report Accuracy/Completeness/Relevance/Clarity, Retrieval Recall@k, and citation accuracy                 | §4–§6         |


**Evaluation support already in this repo:**

- **Dataset mining**: `scripts/build_dataset.py` builds a CoReQA-style dataset from GitHub issues/comments (requires `GITHUB_TOKEN`) into `data/evaluation/dataset.json`.
- **Question-type labeling**: `repoqa/evaluation/question_classifier.py` provides a simple rule-based baseline classifier for categorizing questions during dataset construction/analysis.

