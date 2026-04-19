# RepoQA

Repository-level question answering over source code. RepoQA ingests a codebase, builds hierarchical summaries, embeds everything into ChromaDB, and answers developer questions using hybrid retrieval (semantic + BM25 + RRF fusion) followed by citation-verified LLM generation. A Streamlit UI and an ablation-experiment harness are included.

Primary evaluation target: `pallets/flask`. Default generator/embedder run locally via Ollama (Qwen 2.5 7B + nomic-embed-text); an OpenAI path is also supported.

## Steps followed:

### 1. Flask codebase taken as 1 repo to evaluate

```
git clone git@github.com:pallets/flask.git
```

### 2. Check Lines of code of flask codebase

Total lines of code = **~19K**

```sh
# cloc comand to read line of code under flask and list ignored files in ignore_cloc.txt

$ cloc flask -ignored=ignore_cloc.txt 
     227 text files.
     219 unique files.                                          
Wrote ignore_cloc.txt
      14 files ignored.

github.com/AlDanial/cloc v 2.08  T=0.08 s (2609.6 files/s, 411762.6 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          80           4262           3665          10435
reStructuredText                79           3828           3213           7609
HTML                            20             36              0            326
TOML                             5             48              0            322
YAML                             8              1              4            241
CSS                              2             25              1            109
Text                             8              5              0             94
Markdown                         6             36             42             75
SVG                              3              0              0             55
DOS Batch                        1              8              1             26
SQL                              2              4              2             22
JSON                             2              0              0             21
INI                              1              2              0             11
make                             1              4              7              9
Bourne Shell                     1              0              0              7
-------------------------------------------------------------------------------
SUM:                           219           8259           6935          19362
-------------------------------------------------------------------------------

```

Ignored files list

```sh
$ cat ignore_cloc.txt 
flask:  --exclude-dir=1
flask/.gitignore:  listed in $Not_Code_Extension{gitignore}
flask/LICENSE.txt:  duplicate of flask/examples/tutorial/LICENSE.txt
flask/docs/_static/debugger.png:  binary file
flask/docs/_static/pycharm-run-config.png:  binary file
flask/docs/tutorial/flaskr_edit.png:  binary file
flask/docs/tutorial/flaskr_index.png:  binary file
flask/docs/tutorial/flaskr_login.png:  binary file
flask/examples/javascript/.gitignore:  listed in $Not_Code_Extension{gitignore}
flask/examples/javascript/LICENSE.txt:  duplicate of flask/examples/tutorial/LICENSE.txt
flask/examples/tutorial/.gitignore:  listed in $Not_Code_Extension{gitignore}
flask/tests/test_apps/.env:  language unknown (#3)
flask/tests/test_apps/.flaskenv:  language unknown (#3)
flask/uv.lock:  language unknown (#3)
```

### Exclude noise

When traversing the repo, skip generated artifacts and non-source/binary content before parsing/chunking

- Skip entire directories early: prune known "arftifacts" dirs so the crawler never reads files eg. `.git`, `node_modules`, `__pychache__`, `build`, etc.
- Filter by language/extension: Keep only files whose extension or name maps to supported language (source and key non-code like README and config). Unkown extension are dropped.
- Drop binary files: read a small prefix (e.g., first ~8KB) and skip if it looks binary (null bytes). Refer [Why binary files are checked separately](NOTES.md)
- Optional size/fixture controls (paper mentions): the paper suggests excluding oversized fixtures (e.g., “test fixtures exceeding 1000 lines”)—if you want that behavior, it should be implemented as an additional filter (line-count or byte-size threshold) in the crawler step.

## Architecture

**Offline (once per repo):**

```
repo_crawler -> ast_chunker / noncode_chunker -> dep_graph_builder
             -> file_summarizer -> dir_summarizer -> project_summarizer
             -> embedder -> chroma_store
```

**Online (per question):**

```
question -> question_classifier -> strategy_router
         -> query_expander -> hybrid_retriever (semantic + BM25 + RRF)
         -> context_builder -> answer_generator -> citation_verifier
```

Phases:

1. **Ingestion** — clone/crawl repo, parse ASTs (tree-sitter), chunk code into semantic units (functions, classes, modules), chunk non-code files (markdown, config, dockerfiles), build import dependency graph.
2. **Summarization** — bottom-up LLM summarization: file → directory → project level.
3. **Embedding** — embed chunks + summaries with Ollama (default) or OpenAI, store in ChromaDB (cosine).
4. **Retrieval** — classify question type (architecture/logic/deployment), route to type-specific strategy, expand query, fuse semantic + BM25 via Reciprocal Rank Fusion.
5. **Generation** — build token-budgeted context, generate structured `ANSWER/CITATIONS/CONFIDENCE` output, verify each citation against retrieved chunks.
6. **Evaluation** — deterministic metrics (`retrieval_recall@k`, `citation_accuracy`) + LLM-as-Judge (Gemini 2.5 Flash or local llama3.1:8b) across an ablation grid.

> **Known gap:** the dependency graph is built at ingestion but **not yet wired into retrieval**. Full `hybrid_retriever` currently fuses semantic + BM25 only.

## Project Structure

```
repoqa/
  models.py                  # Shared dataclasses: FileRecord, Chunk, Summary
  tokenizer.py               # tiktoken helpers (cl100k_base)
  ingestion/
    repo_crawler.py          # Clone & walk repository files
    ast_parser.py            # tree-sitter AST parsing (8 languages)
    ast_chunker.py           # Split code into semantic chunks (function/class/module)
    noncode_chunker.py       # Chunk markdown, config, dockerfiles, etc.
    dep_graph_builder.py     # Build import/call dep graph (not yet used at retrieval time)
    pipeline.py              # IngestionPipeline orchestrator
  summarization/
    file_summarizer.py       # Summarize individual files from their chunks
    dir_summarizer.py        # Summarize directories from child summaries
    project_summarizer.py    # Summarize entire project
    prompts.py               # LLM prompt templates
    pipeline.py              # SummarizationPipeline orchestrator
  embedding/
    embedder.py              # Ollama (default) / OpenAI embedder
    chroma_store.py          # ChromaDB persistence layer (cosine space)
    metadata.py              # Metadata schema for vector store
    pipeline.py              # EmbeddingPipeline orchestrator
  retrieval/
    question_classifier.py   # LLM classifier: architecture | logic | deployment
    strategy_router.py       # Per-question-type retrieval strategy (k, weights, filters)
    query_expander.py        # LLM-driven keyword expansion
    bm25_retriever.py        # BM25 keyword search over chunks
    rrf_fusion.py            # Reciprocal Rank Fusion
    hybrid_retriever.py      # Orchestrates classify → expand → semantic + BM25 → RRF
  qa/
    context_builder.py       # Pack retrieved chunks into token-budgeted context
    prompts.py               # Type-specific ANSWER/CITATIONS/CONFIDENCE templates
    answer_generator.py      # LLM invocation + structured-output parsing
    citation_verifier.py     # Check each citation against retrieved chunks
    pipeline.py              # End-to-end QAPipeline (retrieve → generate → verify)
  evaluation/
    models.py                # QAPair, EvaluationDataset dataclasses
    github_miner.py          # Mine QA pairs from GitHub issues (REST API)
    question_classifier.py   # Offline dataset-labeling classifier
    dataset_builder.py       # Orchestrate mining + classification + save
    metrics.py               # retrieval_recall@k, citation_accuracy (no LLM)
    judge.py                 # LLM-as-Judge (Gemini 2.5 Flash / local llama3.1:8b)
  ui/
    app.py                   # Streamlit chat UI
config/
  settings.py                # Pydantic-settings config (reads .env) + ablation flags
  target_repos.json          # Target repos for dataset mining
scripts/
  ingest_repo.py             # CLI: run ingestion pipeline on a repo
  build_summaries.py         # CLI: run summarization pipeline
  build_index.py             # CLI: run embedding pipeline
  build_dataset.py           # CLI: mine CoReQA-style eval dataset from GitHub
  curate_flask.py            # Curate a flask-specific eval subset
  run_experiments.py         # Run full ablation grid → JSONL
  generate_report.py         # JSONL → markdown report + CSV
  validate_summaries.py      # Tier-1 identifier grounding check on summaries
data/
  flask/                     # chunks, summaries, dep graph for pallets/flask
  chroma/                    # persistent vector store
  evaluation/
    seed_dataset.json        # curated QA pairs
    dataset.json             # mined dataset (from build_dataset.py)
  experiments/               # JSONL runs + markdown reports
tests/
  test_ingestion.py          # Ingestion pipeline tests
docs/
  pdflatex/
    RepoQA_Report.tex        # ACL-style LaTeX report
    acl_style.sty            # ACL style package
```

## Core Data Models

**FileRecord** -- a file discovered during crawling

- `repo_path`, `abs_path`, `language`, `size_bytes`, `raw_content`

**Chunk** -- a semantically bounded piece of content ready for embedding

- `id` (sha256 of `repo_path:start_line`), `content`, `chunk_type` (function|class|module|prose|config), `symbol_name`, `start_line`, `end_line`, `token_count`, `language`

**Summary** -- an LLM-generated summary at file/directory/project level

- `id` (sha256 of `summary:level:path`), `level`, `path`, `content`, `source_ids`, `token_count`

**QAPair** -- evaluation dataset entry

- `id` (sha256 of `repo_name:issue_number`), `repo_name`, `question`, `answer`, `question_type` (architecture|logic|deployment), `issue_url`, `answer_score`, `reference_contexts`

**EvaluationDataset** -- collection of QAPairs with `save()`, `load()`, `summary()` methods

## Supported Languages

tree-sitter parsers: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++

## Configuration

All settings are in `config/settings.py` (pydantic-settings), loaded from `.env`:

Variables are listed under `.env.example`. Rename it to `.env` to exclude it during git commit. Update keys and token accordingly in `.env` file.


| Variable                 | Default              | Description                                                  |
| ------------------------ | -------------------- | ------------------------------------------------------------ |
| `LLM_PROVIDER`           | `ollama`             | `ollama` (default, local) or `openai`                        |
| `CHAT_MODEL`             | `qwen2.5:7b`         | Generator model — `qwen2.5:7b` locally, or `gpt-4o-mini`     |
| `OLLAMA_BASE_URL`        | `http://localhost:11434` | Ollama REST endpoint                                     |
| `EMBEDDING_PROVIDER`     | `ollama`             | `ollama` or `openai`                                         |
| `EMBEDDING_MODEL`        | `nomic-embed-text`   | 768-dim embedder (or `text-embedding-3-small` for OpenAI)    |
| `OPENAI_API_KEY`         | --                   | OpenAI API key (only if using OpenAI providers)              |
| `ANTHROPIC_API_KEY`      | --                   | Anthropic API key (optional)                                 |
| `CHROMA_PERSIST_DIR`     | `./data/chroma`      | ChromaDB storage path                                        |
| `CHUNK_MAX_TOKENS`       | `512`                | Max tokens per chunk                                         |
| `CHUNK_OVERLAP_TOKENS`   | `64`                 | Overlap between adjacent chunks                              |
| `SUMMARY_MAX_TOKENS`     | `4096`               | Max tokens for summarization prompts                         |
| `GITHUB_TOKEN`           | --                   | GitHub PAT for dataset mining                                |
| `ENABLE_SUMMARIES`       | `true`               | Ablation: include summaries in retrieval                     |
| `ENABLE_HYBRID`          | `true`               | Ablation: BM25 + semantic RRF fusion                         |
| `ENABLE_QUERY_EXPANSION` | `true`               | Ablation: LLM query expansion                                |
| `ENABLE_ROUTING`         | `true`               | Ablation: question-type-aware strategy routing               |
| `ENABLE_RETRIEVAL`       | `true`               | Ablation: retrieval enabled (else generator answers from question only) |
| `JUDGE_PROVIDER`         | `gemini`             | `gemini` (20 req/day free) or `ollama` (local)               |
| `JUDGE_MODEL`            | `gemini-2.5-flash`   | Gemini judge model                                           |
| `JUDGE_MODEL_LOCAL`      | `llama3.1:8b`        | Fallback judge model when `JUDGE_PROVIDER=ollama`            |
| `GEMINI_API_KEY`         | --                   | Gemini API key (only if `JUDGE_PROVIDER=gemini`)             |
| `JUDGE_RUNS_PER_QUESTION`| `3`                  | Judge invocations per answer for mean ± std                  |


## Quick Start

```bash
# Install (uv manages venv + deps)
uv sync

# Copy and fill .env (only needed if using OpenAI/Anthropic/Gemini)
cp .env.example .env

# Pull local models (if using Ollama defaults)
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 1. Ingest a repo
uv run scripts/ingest_repo.py --repo-path ./flask --output data/flask/transformers_chunks.json

# 2. Build summaries
uv run scripts/build_summaries.py --chunks data/flask/transformers_chunks.json --repo-root ./flask --output data/flask/summaries.json

# 3. Embed + index
uv run scripts/build_index.py --chunks data/flask/transformers_chunks.json --summaries data/flask/summaries.json --repo-name pallets/flask

# 4. Mine evaluation dataset from GitHub (optional)
uv run scripts/build_dataset.py --output data/evaluation/dataset.json

# 5. Validate summaries — Tier-1 identifier-grounding check
uv run scripts/validate_summaries.py \
  --summaries data/flask/summaries.json \
  --repo-root ./flask \
  --output data/flask/summary_validation.json

# 6. Run ablation experiments
uv run scripts/run_experiments.py \
  --chunks data/flask/transformers_chunks.json \
  --repo-root ./flask \
  --dataset data/evaluation/seed_dataset.json \
  --output data/experiments/run_$(date +%Y-%m-%d).jsonl

# 7. Generate markdown report from JSONL
uv run scripts/generate_report.py --jsonl data/experiments/run_*.jsonl

# 8. Launch the chat UI
uv run streamlit run repoqa/ui/app.py
```

## Requirements

- Python >= 3.11 (project is tested with 3.13/3.14)
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com) if using local generator/embedder (default)
- See `pyproject.toml` for full dependency list
- Key deps: tree-sitter + 8 grammar packages, chromadb, langchain, openai/anthropic, tiktoken, pydantic-settings, rank-bm25, streamlit

