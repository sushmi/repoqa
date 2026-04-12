# RepoQA

Repository-level question answering over source code. RepoQA ingests a codebase, builds hierarchical summaries, embeds everything into ChromaDB, and answers developer questions using hybrid retrieval (semantic + BM25 + structural).

## Steps followed:

### 1. Flask codebase taken as 1 repo to evaluate
```
git clone git@github.com:pallets/flask.git
```
### 2. Check Lines of code of flask codebase
Total lines of code = <strong>~19K </strong>

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

```
repo_crawler -> ast_chunker/noncode_indexer -> file_summarizer -> dir_summarizer -> project_summarizer -> embedder -> chroma_store
     |                  |                              |                                                      |
  Phase 1: Ingestion    |                    Phase 2: Summarization                                  Phase 3: Embedding
                        |
                    Chunk(id, content, chunk_type, symbol_name, language, ...)
```

Three-phase pipeline:

1. **Ingestion** -- clone/crawl repo, parse ASTs (tree-sitter), chunk code into semantic units (functions, classes, modules), index non-code files (markdown, config, dockerfiles)
2. **Summarization** -- bottom-up LLM summarization: file -> directory -> project level
3. **Embedding** -- embed chunks + summaries with OpenAI embeddings, store in ChromaDB

## Project Structure

```
repoqa/
  models.py                  # Shared dataclasses: FileRecord, Chunk, Summary
  ingestion/
    repo_crawler.py          # Clone & walk repository files
    ast_parser.py            # tree-sitter AST parsing (8 languages)
    ast_chunker.py           # Split code into semantic chunks (function/class/module)
    noncode_indexer.py       # Index markdown, config, dockerfiles, etc.
    pipeline.py              # IngestionPipeline orchestrator
  summarization/
    file_summarizer.py       # Summarize individual files from their chunks
    dir_summarizer.py        # Summarize directories from child summaries
    project_summarizer.py    # Summarize entire project
    prompts.py               # LLM prompt templates
    pipeline.py              # SummarizationPipeline orchestrator
  embedding/
    embedder.py              # OpenAI embedding wrapper
    chroma_store.py          # ChromaDB persistence layer
    metadata.py              # Metadata extraction for vector store
    pipeline.py              # EmbeddingPipeline orchestrator
  evaluation/
    models.py                # QAPair, EvaluationDataset dataclasses
    github_miner.py          # Mine QA pairs from GitHub issues (REST API)
    question_classifier.py   # Regex classifier: architecture | logic | deployment
    dataset_builder.py       # Orchestrate mining + classification + save
config/
  settings.py                # Pydantic-settings config (reads .env)
  target_repos.json          # 18 target repos for dataset mining (flask, fastapi, spring-boot, gin, nest, ...)
scripts/
  ingest_repo.py             # CLI: run ingestion pipeline on a repo
  build_summaries.py         # CLI: run summarization pipeline
  build_index.py             # CLI: run embedding pipeline
  build_dataset.py           # CLI: mine CoReQA-style eval dataset from GitHub
data/
  evaluation/
    seed_dataset.json        # 30 curated QA pairs (10 Python, 5 Java, 8 Go, 7 TypeScript)
    dataset.json             # Full mined dataset (from build_dataset.py)
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

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | -- | OpenAI API key |
| `ANTHROPIC_API_KEY` | -- | Anthropic API key |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM for summarization |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `64` | Overlap between adjacent chunks |
| `SUMMARY_MAX_TOKENS` | `4096` | Max tokens for summarization prompts |
| `GITHUB_TOKEN` | -- | GitHub PAT for dataset mining |

## Quick Start

```bash
# Install
pip install -e .

# Copy and fill .env
cp .env.example .env

# Ingest a repo
python scripts/ingest_repo.py --repo /path/to/repo

# Build summaries
python scripts/build_summaries.py --chunks data/chunks.json

# Build embeddings index
python scripts/build_index.py --chunks data/chunks.json --summaries data/summaries.json

# Mine evaluation dataset from GitHub
python scripts/build_dataset.py --output data/evaluation/dataset.json
```

## Requirements

- Python >= 3.11
- See `pyproject.toml` for full dependency list
- Key deps: tree-sitter, chromadb, langchain, openai/anthropic, tiktoken, pydantic-settings, gitpython
