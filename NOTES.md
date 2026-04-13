# NOTES

## Why binary files need a separate check (prefix + null bytes)

Filtering by extension/name isn’t enough, because “binary vs text” is about file contents, not file suffix.

- Extensions lie / are missing: repos often contain files with no extension (e.g., LICENSE, Makefile), misleading extensions, or generated assets that still look “texty” by name.
- Some “text” extensions can still be binary: e.g., a .txt that’s actually a compressed blob, a mis-encoded file, or accidentally committed binary content.
- Pipeline assumes UTF-8 text downstream: Tree-sitter parsing, tokenization, chunking, summarization prompts, and embeddings all expect text. Feeding binary bytes can:
  - explode token counts with garbage characters,
  - produce meaningless chunks/embeddings,
  - or cause decode/parsing errors.
- 8KB prefix is a fast, safe heuristic: checking for \x00 (null bytes) in a small prefix catches most binaries cheaply without reading whole large files.  
So the binary-prefix probe is a content-based guardrail that complements (but can’t be replaced by) extension-based filtering.

## NLP Pipeline vs Workflow

An NLP pipeline is a specific, sequential series of data processing steps (e.g., tokenization, stemming, model inference) that transforms raw text into structured insights. 

An NLP workflow is broader, encompassing the entire project life cycle—including data acquisition, training, model evaluation, deployment, and monitoring

## Tokenizers Options (comparison) - for running Qwen locally


| Option                        | Library                                         | Typical use                                | Speed/Size | Token couning accuracy for Qwen                                   |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------ | ---------- | ----------------------------------------------------------------- |
| Qwen tokenizer (recommended)  | `transformers` (Qwen model tokenizer)           | Exact token count + ecnode/decode for Qwen | Medium     | Eact (matches Qwen vocab/merges)                                  |
| Fast tokenizer (Rust-backend) | `tokenizer` via `transformers` "fast tokenizers | Samw as above                              | Fast       | Exact (if a fast tokenizer exists for that model)                 |
| SentencePiece                 | `sentencepiecet`                                | Many LLaMA/T5-style models                 | Medium     | Usually not exact for Qwen(unless Qwen uses it)                   |
| BPE (standalone)              | `tokenizers` (custom BPE)                       | Build/use your own BPE                     | Fast       | Depends (often wrong uness you load Qwen's exact tokenizer files) |
| OpenAI tokenizer              | `tiktoken`                                      | OpenAAI GPT-family token budgeting         | Fast       | Not exact for Qwen                                                |
| Heuristic (chars/words)       | none                                            | Quick estimates                            | Fastest    | Approximate                                                       |


## Tree-sitter

Tree-sitter is an incremental parsing system. Unlike the built-in ast module in Python (which only parses Python code), Tree-sitter provides a uniform way to parse any language (Java, C++, Go, Python, etc.) into a concrete syntax tree (CST).

Fast: It is written in C and is extremely efficient.

Incremental: If you change one line of code, it only re-parses that specific part of the tree.

Error-Tolerant: If a file has a syntax error (common in real-world repos), it doesn't crash; it parses what it can and marks the error node. This is vital for a robust AI assistant.

### Why it’s better than Python's ast module

The standard `ast` module is limited.

1. Language Locked: ast **only works for Python**. If your repo has a .js or .sh file, ast is useless.
2. Detail: Tree-sitter keeps track of comments and exact whitespace coordinates, which are usually stripped by standard ASTs but are crucial for providing context to an LLM.

## Troubleshoot

### Unknown config option: ayncio_mode

```sh
_pytest/config/__init__.py:1428: PytestConfigWarning: Unknown config option: asyncio_mode
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

That warning means your pyproject.toml sets asyncio_mode = "auto", but your current pytest run doesn’t have a pytest plugin that understands it (typically pytest-asyncio), or it’s an older version. I’ll check what pytest/plugins are installed in your .venv and then give the minimal fix.

Fix (recommended):

Install the dev test deps into your project environment, then run pytest via that environment

Use uv:

```
uv sync --extra dev
uv run pytest -q tests/test_ingestion.py
```

## uv failure to sync setuptool.backend

```sh
$ uv sync --extra dev                
Resolved 135 packages in 4ms
  × Failed to build `repoqa @ file:///xxxx`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta.build_editable` failed (exit status: 1)

      [stderr]
      error: Multiple top-level packages discovered in a flat-layout: ['data', 'flask', 'preps', 'config', 'images', 'repoqa'].
```

Root cause: setuptools refused to build because it auto-discovered multiple top-level "packages" (data, flask, etc).

Fix : update `pyproject.toml` to explictly package only the repoqa* and config* others are excluded.

Verify again using

```sh
uv sync --extra dev 
uv run pytest -q tests/test_ingestion.py
```



### what does this mean PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.

While executing;

```
$ uv run scripts/ingest_repo.py --repo-path flask/ --output data/flask/transformer_chunks.json
```

That warning is harmless for your use case. It means:

- `transformers` has **optional** backends: PyTorch, TensorFlow, JAX/Flax
- You don't have PyTorch installed, so transformers can't load **model weights** (i.e., the actual neural network for inference)
- But **tokenizers, configs, and file utilities still work fine** — which is all you need

Since RepoQA only uses `AutoTokenizer` for token counting (not running Qwen inference), you don't need PyTorch installed. The tokenizer loads the vocab/merge files from the Hub and runs pure BPE — no GPU or model weights involved.

You can safely ignore the warning. If it's noisy, you can suppress it by setting the environment variable:

```bash
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

```

You'd only need to install PyTorch if you later want to run Qwen locally for inference (e.g., the generation/QA phase).