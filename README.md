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
  tokenizer.py               # TokenCounter singleton (Qwen2.5 tokenizer + LLM-call logging)
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

## Sample Questions — How They Are Generated

RepoQA does **not** synthesize questions with an LLM. Instead, it follows the [CoReQA](https://arxiv.org/abs/2501.03447) methodology and mines real developer Q&A from GitHub issues, so every (question, gold answer) pair is grounded in something a human actually asked and another human up-voted.

### Pipeline

```
config/target_repos.json
        │
        ▼
GitHubMiner.mine_repo()        ── REST API: closed issues, sorted by comments desc, max 100/repo
        │                         (skips PRs; skips bodies < 30 chars)
        ▼
_best_answer()                 ── pick top comment with ≥ min_reactions positive
        │                         reactions (+1, heart, hooray, rocket); default = 3
        ▼
QAPair                         ── question = issue body, answer = top comment
        │                         id = sha256(repo:issue#), language detected from repo
        ▼
classify_question()            ── regex classifier → architecture | logic | deployment
        │
        ▼
EvaluationDataset.save()       ── data/evaluation/dataset.json
        │
        ▼ (optional)
scripts/curate_flask.py        ── filter to pallets/flask, answer_score ≥ 5,
                                  question length ≥ 80, drop link-only questions
                               → data/evaluation/flask_curated.json
```

Code entry points: [github_miner.py](repoqa/evaluation/github_miner.py), [dataset_builder.py](repoqa/evaluation/dataset_builder.py), [question_classifier.py](repoqa/evaluation/question_classifier.py), [build_dataset.py](scripts/build_dataset.py), [curate_flask.py](scripts/curate_flask.py).

### Quality filters


| Filter                                        | Where                                                              | Default                   |
| --------------------------------------------- | ------------------------------------------------------------------ | ------------------------- |
| Closed issues only, sorted by comment count   | [github_miner.py:55-80](repoqa/evaluation/github_miner.py#L55-L80) | hard-coded                |
| Skip pull requests                            | [github_miner.py:73](repoqa/evaluation/github_miner.py#L73)        | hard-coded                |
| Issue body ≥ 30 chars                         | [github_miner.py:84](repoqa/evaluation/github_miner.py#L84)        | hard-coded                |
| Top answer must have ≥ N positive reactions   | [github_miner.py:111](repoqa/evaluation/github_miner.py#L111)      | `min_reactions=3`         |
| Max issues per repo                           | [github_miner.py:25](repoqa/evaluation/github_miner.py#L25)        | `max_issues_per_repo=100` |
| Curated answer score ≥ 5, question ≥ 80 chars | [curate_flask.py:4-8](scripts/curate_flask.py#L4-L8)               | flask subset only         |


### Question-type classification

`[classify_question](repoqa/evaluation/question_classifier.py)` is a deterministic regex scorer. Each question is matched against three pattern lists; the highest-scoring bucket wins, with `logic` as the conservative default when no patterns fire.


| Type             | Triggers (excerpt)                                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **architecture** | `architect`, `module`, `component`, `pipeline`, `middleware`, `routing`, `directory`, `flow`, `how does the project/system…` |
| **logic**        | `function`, `method`, `class`, `bug`, `error`, `return`, `parameter`, `exception`, `async/await`, `recurs`                   |
| **deployment**   | `deploy`, `docker`, `ci/cd`, `kubernetes`, `install`, `setup`, `pip`, `venv`, `requirements`, `migrat`, `database`, `.env`   |


### Current datasets


| File                                                                     | Pairs | Repos               | Notes                                                                  |
| ------------------------------------------------------------------------ | ----- | ------------------- | ---------------------------------------------------------------------- |
| [data/evaluation/dataset.json](data/evaluation/dataset.json)             | 976   | 18                  | full mined set across Python/TS/Java/Go (see `target_repos.json`)      |
| [data/evaluation/flask_curated.json](data/evaluation/flask_curated.json) | 23    | 1 (`pallets/flask`) | high-confidence flask subset; 13 logic / 6 architecture / 4 deployment |
| [data/evaluation/seed_dataset.json](data/evaluation/seed_dataset.json)   | small | 1                   | hand-picked seed used by ablation runs                                 |


### Sample questions (from `flask_curated.json`)


| #   | Type         | Issue                                                      | Question (truncated)                                                                                                                                       |
| --- | ------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | architecture | [flask#1045](https://github.com/pallets/flask/issues/1045) | "I am trying to include css files from the static directory, the browser downloads the content of the CSS file but it says the server returned text/html…" |
| 2   | architecture | [flask#2626](https://github.com/pallets/flask/issues/2626) | "I'd like to create an official guide on how to structure a Flask application — MVC pattern, factory pattern, folder layout…"                              |
| 3   | architecture | [flask#1902](https://github.com/pallets/flask/issues/1902) | "I'm trying to run the *flaskr* and *minitwit* examples but get `ImportError: No module named 'flaskr'` when running `flask initdb`…"                      |
| 4   | logic        | [flask#510](https://github.com/pallets/flask/issues/510)   | "Can I use `jsonify` as `jsonify([{'a':1,'b':2},{'c':3,'d':4}])`?"                                                                                         |
| 5   | logic        | [flask#593](https://github.com/pallets/flask/issues/593)   | "I'd like to register sub-blueprints using `Blueprint.register_blueprint(...)` so nested blueprints register with the app when the parent does…"           |
| 6   | logic        | [flask#1602](https://github.com/pallets/flask/issues/1602) | "`flask.json` does not allow the user to specify an alternative json library (such as rapidjson). Would you consider giving the user some control…?"       |
| 7   | logic        | [flask#941](https://github.com/pallets/flask/issues/941)   | "When registering a handler for `werkzeug.exceptions.HTTPException`, it has no effect when an HTTP error is raised…"                                       |
| 8   | logic        | [flask#2086](https://github.com/pallets/flask/issues/2086) | "When handling large file uploads (+300MB) the files seem to be cached in memory until the upload is finished — is this expected?"                         |
| 9   | deployment   | [flask#2023](https://github.com/pallets/flask/issues/2023) | "Flask ships with a default, hardcoded logging handler. This makes it harder to install custom logging setups…"                                            |
| 10  | deployment   | [flask#4027](https://github.com/pallets/flask/issues/4027) | "About the pinned `install_requires` versions (`Werkzeug>=0.15`, `Jinja2>=2.10.1`, …) — should these floors be raised?"                                    |
| 11  | deployment   | [flask#1847](https://github.com/pallets/flask/issues/1847) | "Flask CLI issue with Docker — `flask run` inside the container behaves differently than expected. Reproducer at `sean-lynch/flask-cli-issue`…"            |
| 12  | deployment   | [flask#3168](https://github.com/pallets/flask/issues/3168) | "Preparing for a PyCon tutorial — the output of `flask run` is confusing for beginners; can the dev-server messages be cleaned up?"                        |


### Example record (one `QAPair` from `flask_curated.json`)

```json
{
  "id": "5121dd64ce4608e0",
  "repo_name": "pallets/flask",
  "repo_url": "https://github.com/pallets/flask",
  "language": "python",
  "question": "It looks like `flask.json` does not allow the user to specify an alternative json library (such as rapidjson). Instead it just does this:\n\n    try:\n        from itsdangerous import simplejson as _json\n    except ImportError:\n        from itsdangerous import json as _json\n\nThis un-overridable behavior also means that, due to differences in when `default(..)` is called between `simplejson` and `json`, bugs can mysteriously appear when `simplejson` is installed that didn't appear without it.\n\nWould you consider giving the user some control over which json library is used?",
  "answer": "I think @dsully's plan summarizes it well: (1) simplejson should be a direct dependency and the default; (2) being able to swap it for rapidjson/ujson is desired but the behavior would be undefined; (3) jsonify() should accept all args that can be passed to the encoder so custom encoder classes can be used. I would just default to stdlib's `json` instead — it's stable and it works. Switching to an alternative implementation should be programmatically specified (no auto-detection), so it's clear to users which implementation they are actually using.",
  "question_type": "logic",
  "issue_url": "https://github.com/pallets/flask/issues/1602",
  "issue_number": 1602,
  "answer_score": 5,
  "reference_contexts": []
}
```

Field meanings: `id` is `sha256(repo_name:issue_number)[:16]`; `question_type` is filled in by the regex classifier after mining; `answer_score` is the count of positive reactions on the chosen top comment; `reference_contexts` is reserved for gold file/line references and is empty for issue-mined pairs.

### Reproducing the dataset

```bash
# 1. mine from default repos in config/target_repos.json
export GITHUB_TOKEN=github_pat_...
uv run scripts/build_dataset.py --output data/evaluation/dataset.json

# 2. (optional) build the high-confidence flask subset
uv run scripts/curate_flask.py
```

## Context Window Size as a Cost Proxy

We don't have an API meter on the local Ollama path, and re-running the full ablation grid against a paid model just to get a dollar figure is wasteful. Instead, we use **prompt token count** — measured with the same Qwen tokenizer used during chunking — as a *deterministic, provider-agnostic* proxy for cost and latency. Two facts make this a faithful proxy:

1. Almost every commercial LLM bills **per input token** (Claude, GPT-4o, Gemini, etc.). Cost is linear in prompt size.
2. Decoder latency on a fixed model scales near-linearly with prompt length once attention KV-cache fills, so prompt tokens also predict wall-clock time within a single backend.

A configuration that retrieves more, expands queries more aggressively, or stuffs longer summaries into the prompt will have a higher token count — and would cost proportionally more on any paid backend, even if its accuracy looks similar.

### What gets counted

The "context window size" we report is the **total prompt tokens fed to the generator** for a single question:

```
prompt_tokens
  = count_tokens(system_prompt)            # ~70-90 tokens, type-specific
  + count_tokens(question)                 # 10-300 tokens
  + count_tokens(formatted_context)        # 0 .. max_context_tokens
  + scaffolding (≈30 tokens)               # "ANSWER:", "CITATIONS:", "CONFIDENCE:" markers
```

`formatted_context` is the output of [build_context()](repoqa/qa/context_builder.py#L18) — retrieved chunks rendered as

```
[path/to/file.py:10-42 | function: my_func]
```python
<chunk content>
```

```

…packed greedily in relevance order until adding the next chunk would exceed `max_context_tokens` (default `4000`, set in [answer_generator.py:47](repoqa/qa/answer_generator.py#L47)).

### How to measure it

All token counting is centralized in the **`TokenCounter`** singleton in [tokenizer.py](repoqa/tokenizer.py), exposed as the module-level `TOKEN_COUNTER`. It wraps the Qwen2.5-7B tokenizer (the same one the chunker uses, so per-chunk `token_count` values are directly additive) and provides four methods used everywhere in the codebase:

| Method | Purpose |
| --- | --- |
| `TOKEN_COUNTER.count(text)` | Token count for a string |
| `TOKEN_COUNTER.count_messages(prompt)` | Token count for a string or list of LangChain messages/dicts |
| `TOKEN_COUNTER.truncate(text, n)` | Truncate to ≤ n tokens |
| `TOKEN_COUNTER.log_llm_call(label, prompt, response)` | Log input/output/total for one LLM call; returns the counts as a dict |

```python
from repoqa.tokenizer import TOKEN_COUNTER
from repoqa.qa.context_builder import build_context

context, included = build_context(retrieved_chunks, max_tokens=4000)

prompt_tokens = (
    TOKEN_COUNTER.count(system_prompt)
    + TOKEN_COUNTER.count(question)
    + TOKEN_COUNTER.count(context)
    + 30                                  # scaffolding overhead
)
print(f"context tokens used: {TOKEN_COUNTER.count(context)}")
print(f"total prompt tokens: {prompt_tokens}")
```

Because every `Chunk` already stores `token_count` (filled in at ingestion via `TOKEN_COUNTER.count`), the per-question cost can also be estimated **without re-tokenizing** — just sum the token counts of the chunks that were actually included:

```python
context_tokens ≈ sum(c.token_count for c in included)        # tight lower bound
prompt_tokens  ≈ context_tokens + 100                         # + system + question + scaffolding
```

### Translating tokens to dollars — offline vs online

The pipeline charges tokens in two distinct phases, each priced differently. Splitting them out matters when switching from local Ollama to a paid backend, because the **offline** phase is paid once per repo while the **online** phase is paid per question.

**Phase A — Offline (once per repository).** Two LLM-driven steps fire at indexing time:

1. **Hierarchical summarization** — `FileSummarizer` → `DirSummarizer` → `ProjectSummarizer` (chat-model billing: input + output).
2. **Embedding** — every chunk and every summary is embedded once (embedder billing: input only, no output).

**Phase B — Online (per question).** Three LLM calls per question (chat-model billing):

1. `question_classifier` (~180 input tokens, 1 output)
2. `query_expander` (~80 input, ~20 output)
3. `answer_generator` (~3.9k input, ~200 output)

Total ≈ 4.5k tokens per question on the default `max_context_tokens=4000`.

#### Provider price reference (2026-04)

| Provider | Chat in ($/1M) | Chat out ($/1M) | Embedding ($/1M) |
| --- | --- | --- | --- |
| OpenAI GPT-4o-mini + `text-embedding-3-small` | 0.15 | 0.60 | 0.02 |
| OpenAI GPT-4o + `text-embedding-3-large` | 2.50 | 10.00 | 0.13 |
| Anthropic Claude Haiku 4.5 + Voyage `voyage-3-lite` | 1.00 | 5.00 | 0.02 |
| Anthropic Claude Sonnet 4.6 + Voyage `voyage-3` | 3.00 | 15.00 | 0.06 |
| Google Gemini 2.5 Flash + `text-embedding-004` | 0.30 | 2.50 | 0.025 |
| Local Qwen 2.5 7B + nomic-embed-text (Ollama) | 0 | 0 | 0 |

> Anthropic doesn't ship a first-party text embedder; pair Claude with Voyage AI (an Anthropic-recommended provider) for the embedding side.

#### Cost formulas

```
$ offline = embedding_tokens   × emb_price
          + summ_input_tokens  × chat_input_price
          + summ_output_tokens × chat_output_price

$ online_per_question = per_q_input_tokens  × chat_input_price
                      + per_q_output_tokens × chat_output_price

$ total = $ offline  +  N_questions × $ online_per_question
```

#### Worked example — flask (`pallets/flask`, ~80 files, ~5k chunks, ~110 summaries, 23 curated questions)

Token figures below come from running the recipe in **Step 1** of the "How to count tokens" section against `data/flask/transformers_chunks.json` and `data/flask/summaries.json`. Substitute your own measurements for non-flask repos.

| Phase | Token type | flask total | × Claude Haiku 4.5 + Voyage | × GPT-4o-mini | × Gemini 2.5 Flash |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Offline — embedding** | input | ~350,000 | $0.0070 | $0.0070 | $0.0088 |
| **Offline — summarization** | input | ~280,000 | $0.2800 | $0.0420 | $0.0840 |
| **Offline — summarization** | output | ~20,000 | $0.1000 | $0.0120 | $0.0500 |
| **Offline subtotal (one-time)** | | | **$0.39** | **$0.06** | **$0.14** |
| **Online — per question** | input | ~4,300 | $0.0043 | $0.0006 | $0.0013 |
| **Online — per question** | output | ~230 | $0.0012 | $0.0001 | $0.0006 |
| **Per-question total** | | | **$0.0055** | **$0.0008** | **$0.0019** |
| Online for 23 questions | | | $0.13 | $0.02 | $0.04 |
| Online for 1,000 questions | | | $5.50 | $0.80 | $1.90 |

#### Reading the table

- **Switching to Claude Haiku** is dominated by *summarization*, not by per-question generation: the one-time $0.39 amortizes after ~70 questions.
- **GPT-4o-mini** is the cheapest paid path on both axes — useful when paid embeddings + paid generation are required for reproducibility.
- **Gemini 2.5 Flash** sits in the middle on chat but is essentially free on embeddings (free tier covers small repos), making it attractive when the offline budget matters most.
- **Local Qwen + nomic-embed-text** stays at $0 across the board — the cost shows up as GPU/RAM time instead of dollars.

Output tokens are reported separately because they (a) follow a different price column and (b) are roughly constant across ablation configs at the answer-generator stage, so they cancel out when **comparing** retrieval configs even though they matter when **comparing** providers.

### Why this is the right metric for ablations

It separates two confounds that retrieval-quality metrics alone can't:


| Config          | recall@8 | prompt_tokens | tokens / recall-pp |
| --------------- | -------- | ------------- | ------------------ |
| `bm25_baseline` | 56.5%    | ~2,800        | 49.6               |
| `full_repoqa`   | 58.3%    | ~3,900        | 66.9               |
| `semantic_only` | 59.4%    | ~3,400        | 57.2               |


A config that gains 2pp recall by quadrupling context size is paying a steep premium — visible only when prompt tokens are reported alongside accuracy. This is the same intuition behind cost-aware leaderboards like HELM and Stanford's "tokens-per-correct-answer".

### Built-in per-call token logging

Every LLM invocation in the pipeline emits a token-count log line via [`TOKEN_COUNTER.log_llm_call()`](repoqa/tokenizer.py) (logger name `repoqa.llm`, level `INFO`). Wired call sites:

| Stage | Call site | Log label |
| --- | --- | --- |
| Summarization | [file_summarizer.py](repoqa/summarization/file_summarizer.py) | `file_summary:<path>` |
| Summarization | [dir_summarizer.py](repoqa/summarization/dir_summarizer.py) | `dir_summary:<path>` |
| Summarization | [project_summarizer.py](repoqa/summarization/project_summarizer.py) | `project_summary:<repo>` |
| Retrieval | [question_classifier.py](repoqa/retrieval/question_classifier.py) | `question_classifier` |
| Retrieval | [query_expander.py](repoqa/retrieval/query_expander.py) | `query_expander` |
| QA | [answer_generator.py](repoqa/qa/answer_generator.py) | `answer_generator` |

Sample output:

```
repoqa.llm INFO: LLM call [question_classifier]: input=178 output=1 total=179 tokens
repoqa.llm INFO: LLM call [query_expander]: input=82 output=24 total=106 tokens
repoqa.llm INFO: LLM call [answer_generator]: input=3914 output=212 total=4126 tokens
```

`TOKEN_COUNTER.log_llm_call` also returns the counts as a dict (`input_tokens` / `output_tokens` / `total_tokens`) so callers can persist them into experiment JSONL.

### Reporting in experiment runs

To start tracking this, log the per-question prompt size from `AnswerGenerator.generate` into the JSONL emitted by [run_experiments.py](scripts/run_experiments.py), then aggregate in [generate_report.py](scripts/generate_report.py):

```python
# inside generate(), after build_context(...)
messages = prompt.format_messages(question=question, context=context, num_chunks=len(included_chunks))
result.prompt_tokens = TOKEN_COUNTER.count_messages(messages)
```

Aggregate as `mean ± std` across the eval set per config; the resulting column makes the recall-vs-cost trade-off explicit in the ablation report.

## Supported Languages

tree-sitter parsers: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++

## Configuration

All settings are in `config/settings.py` (pydantic-settings), loaded from `.env`:

Variables are listed under `.env.example`. Rename it to `.env` to exclude it during git commit. Update keys and token accordingly in `.env` file.


| Variable                  | Default                  | Description                                                             |
| ------------------------- | ------------------------ | ----------------------------------------------------------------------- |
| `LLM_PROVIDER`            | `ollama`                 | `ollama` (default, local) or `openai`                                   |
| `CHAT_MODEL`              | `qwen2.5:7b`             | Generator model — `qwen2.5:7b` locally, or `gpt-4o-mini`                |
| `OLLAMA_BASE_URL`         | `http://localhost:11434` | Ollama REST endpoint                                                    |
| `EMBEDDING_PROVIDER`      | `ollama`                 | `ollama` or `openai`                                                    |
| `EMBEDDING_MODEL`         | `nomic-embed-text`       | 768-dim embedder (or `text-embedding-3-small` for OpenAI)               |
| `OPENAI_API_KEY`          | --                       | OpenAI API key (only if using OpenAI providers)                         |
| `ANTHROPIC_API_KEY`       | --                       | Anthropic API key (optional)                                            |
| `CHROMA_PERSIST_DIR`      | `./data/chroma`          | ChromaDB storage path                                                   |
| `CHUNK_MAX_TOKENS`        | `512`                    | Max tokens per chunk                                                    |
| `CHUNK_OVERLAP_TOKENS`    | `64`                     | Overlap between adjacent chunks                                         |
| `SUMMARY_MAX_TOKENS`      | `4096`                   | Max tokens for summarization prompts                                    |
| `GITHUB_TOKEN`            | --                       | GitHub PAT for dataset mining                                           |
| `ENABLE_SUMMARIES`        | `true`                   | Ablation: include summaries in retrieval                                |
| `ENABLE_HYBRID`           | `true`                   | Ablation: BM25 + semantic RRF fusion                                    |
| `ENABLE_QUERY_EXPANSION`  | `true`                   | Ablation: LLM query expansion                                           |
| `ENABLE_ROUTING`          | `true`                   | Ablation: question-type-aware strategy routing                          |
| `ENABLE_RETRIEVAL`        | `true`                   | Ablation: retrieval enabled (else generator answers from question only) |
| `JUDGE_PROVIDER`          | `gemini`                 | `gemini` (20 req/day free) or `ollama` (local)                          |
| `JUDGE_MODEL`             | `gemini-2.5-flash`       | Gemini judge model                                                      |
| `JUDGE_MODEL_LOCAL`       | `llama3.1:8b`            | Fallback judge model when `JUDGE_PROVIDER=ollama`                       |
| `GEMINI_API_KEY`          | --                       | Gemini API key (only if `JUDGE_PROVIDER=gemini`)                        |
| `JUDGE_RUNS_PER_QUESTION` | `3`                      | Judge invocations per answer for mean ± std                             |


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

