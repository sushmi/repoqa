## What is qwen-agent?

`qwen-agent` is Qwen's official Python SDK. It can run Qwen locally **and** handle tool use, RAG, and agents — all in one package. No Ollama needed.

## The current setup (4 layers)

```
RepoQA Code → LangChain → langchain-ollama → Ollama → Qwen 2.5

```

## With qwen-agent (2 layers)

```
RepoQA code → qwen-agent → Qwen 2.5

```

`qwen-agent` bundles its own inference using `transformers` + `torch` under the hood. It loads the model directly in Python — no separate server.

## Comparison


|                            | LangChain + Ollama                      | qwen-agent                                       |
| -------------------------- | --------------------------------------- | ------------------------------------------------ |
| Dependencies               | langchain, langchain-ollama, Ollama app | qwen-agent, torch, transformers                  |
| Needs Ollama installed?    | Yes                                     | No                                               |
| Needs PyTorch?             | No                                      | **Yes** (~2GB install)                           |
| Needs GPU?                 | No (Ollama handles CPU/GPU)             | Strongly recommended for 7B                      |
| Swap models easily?        | Yes (OpenAI/Claude/any Ollama model)    | Qwen models only                                 |
| Built-in RAG/tools?        | No (you build it)                       | Yes (built-in)                                   |
| Your existing code changes | Small (just the connector)              | **Big** (rip out LangChain from all summarizers) |


## The trade-off

**qwen-agent** is simpler (fewer moving parts) and purpose-built for Qwen. But:

1. **Your summarizers all use LangChain's** `prompt | llm` **chain pattern.** Switching to qwen-agent means rewriting [file_summarizer.py](vscode-webview://0pmhp2m8fcs290b7n99h427qojo82fqf8gsmq53eh8ulmq34rr14/repoqa/summarization/file_summarizer.py), [dir_summarizer.py](vscode-webview://0pmhp2m8fcs290b7n99h427qojo82fqf8gsmq53eh8ulmq34rr14/repoqa/summarization/dir_summarizer.py), and [project_summarizer.py](vscode-webview://0pmhp2m8fcs290b7n99h427qojo82fqf8gsmq53eh8ulmq34rr14/repoqa/summarization/project_summarizer.py) — every `chain.invoke()` call changes.
2. **You lose the ability to swap providers.** Right now you can switch between OpenAI, Anthropic, and Ollama with one env var (`LLM_PROVIDER`). With qwen-agent, you're locked to Qwen.
3. **PyTorch is heavy.** ~2GB download, and running 7B parameters without a GPU will be very slow.

## Recommendation

Stick with **LangChain + Ollama** because:

- Minimal code changes (already done)
- Ollama handles CPU/GPU optimization for you
- You keep provider flexibility for the project
- No PyTorch needed

But if you later want to use qwen-agent's **built-in RAG and tool-use features** for the retrieval/QA phase (not summarization), that could make sense as a separate component.

## Size comparison:

### Dependency sizes


| Package                 | Install size | What you get                                  |
| ----------------------- | ------------ | --------------------------------------------- |
| **langchain + plugins** | ~50-80 MB    | Framework + all the adapters you already have |
| **langchain-ollama**    | ~1 MB        | Tiny connector (just HTTP calls)              |
| **Ollama app**          | ~150-200 MB  | Full inference engine (bundles llama.cpp)     |
| **Qwen 2.5:7B model**   | ~4.7 GB      | The actual model weights                      |
| **Total new**           | **~5 GB**    | Ollama + Qwen model download                  |


vs. the alternative:


| Package                 | Install size                         |
| ----------------------- | ------------------------------------ |
| **qwen-agent**          | ~5 MB                                |
| **PyTorch**             | **~2-3 GB**                          |
| **transformers**        | ~500 MB (already installed)          |
| **Qwen 2.5:7B weights** | ~14 GB (unquantized via HuggingFace) |
| **Total new**           | **~17 GB**                           |


## Why Ollama is actually smaller

Ollama downloads **quantized** models (4-bit GGUF format) — that's why Qwen is 4.7 GB instead of 14 GB. It also uses llama.cpp (C++) instead of PyTorch (Python + CUDA), so no 2-3 GB PyTorch install.

LangChain is already in your project, so the only **new** cost is:

- `langchain-ollama`: 1 MB
- Ollama app: ~200 MB
- `ollama pull qwen2.5:7b`: ~4.7 GB

## The real cost

LangChain is heavy for what it does — it's essentially a fancy wrapper around HTTP calls. But you already have it and your entire summarization pipeline is built on it. Removing it would mean rewriting 4+ files for minimal size savings.

The biggest thing by far is the **model weights** (~5 GB), and you need those regardless of which approach you use.



## Pytorch and why it's big

- Training a CNN for image classification in a college ML course
- Fine-tuning BERT for sentiment analysis
- Running Qwen 2.5 inference locally

It's all the same `pip install torch` → same 2-3 GB package.

## Why it's so big

PyTorch ships with:

- **CUDA libraries** (~1.5 GB) — GPU computation code from NVIDIA
- **CPU math kernels** (~500 MB) — optimized matrix multiplication
- **The Python framework** (~200 MB) — tensors, autograd, nn modules

You need all of that to run a neural network — whethre training or inference.

## Why Ollama avoids it

Ollama uses **llama.cpp** instead — a C++ reimplementation of the inference-only parts. It doesn't need:

- Python at all
- CUDA libraries (it has its own lightweight GPU backend)
- Training code, autograd, optimizer logic

That's why Ollama is ~200 MB vs PyTorch's ~2-3 GB. It does less — just inference — but that's all you need for running Qwen.

## In your project


| Component                                     | Uses PyTorch?                       |
| --------------------------------------------- | ----------------------------------- |
| `transformers.AutoTokenizer` (token counting) | No — tokenizer is pure Python/Rust  |
| Ollama running Qwen 2.5                       | No — uses llama.cpp                 |
| `qwen-agent` running Qwen 2.5                 | **Yes** — loads weights via PyTorch |
| Training/fine-tuning any model                | **Yes**                             |


That "PyTorch not found" warning you saw earlier was just `transformers` checking if PyTorch is available. Since you only use the tokenizer, it doesn't matter.