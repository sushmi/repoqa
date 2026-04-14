## Why LangChain is used

LangChain is used here as a **convenience wrapper** for calling LLM APIs during summarization. It provides:

- `ChatOpenAI` / `ChatAnthropic` — unified interface to swap providers with one config change
- Prompt templates (`FILE_SUMMARY_PROMPT | self.llm`) — the pipe syntax chains prompts to models
- Built-in retries, streaming, token tracking

It's only used in the **summarization phase** (generating file/dir/project summaries), not in ingestion or embedding.

## Should the LLM provider be changed to Qwen?

These are **two separate concerns**:


| Component        | What it does                           | Currently uses        |
| ---------------- | -------------------------------------- | --------------------- |
| **Tokenizer**    | Counts tokens to size chunks correctly | Qwen (just changed)   |
| **LLM provider** | Generates summaries, answers questions | OpenAI/Anthropic APIs |


The tokenizer was switched to Qwen because the **CoReQA benchmark** evaluates with Qwen-7B as the generator — so your chunks need to fit Qwen's token boundaries.

But the **summarization LLM** (which writes file/directory summaries during indexing) doesn't need to be Qwen. That's an offline preprocessing step — using GPT-4o or Claude there is fine and arguably better quality.

**If you want to switch the QA generation phase to Qwen** (to match the benchmark), you'd either:

- Run Qwen-7B locally (needs PyTorch + GPU) and add a LangChain `ChatHuggingFace` or `ChatOllama` wrapper
- Use a hosted Qwen API (e.g., Together AI, Fireworks) via `ChatOpenAI` with a custom `base_url`

But that's a retrieval/generation phase decision — not needed for the current ingestion pipeline.

## How langchain used in code

`langchain-ollama` **is NOT a model.** It's a LangChain connector/adapter package that lets LangChain talk to Ollama (a local LLM runner).

Here's how the pieces fit together:

```
RepoQA Code (LangChain) → langchain-ollama (connector) → Ollama (server) → Qwen 2.5 (the actual model)

```

- **Qwen 2.5** — the actual LLM that generates summaries
- **Ollama** — a local server that loads and runs Qwen 2.5 on your machine
- **langchain-ollama** — a Python package that tells LangChain how to send requests to Ollama
- **LangChain** — the framework your summarization pipeline already uses

So we **are** using Qwen 2.5. The `chat_model: "qwen2.5:7b"` in settings tells Ollama which model to load. `langchain-ollama` is just the glue so LangChain can call it.

Without `langchain-ollama`, you'd have to rip out LangChain from the summarizers and write raw HTTP calls to Ollama yourself — that's unnecessary since the pipeline already uses LangChain's `prompt | llm` chain pattern.

## How to run it

1. **Install Ollama** — [https://ollama.com](https://ollama.com) (one download, like installing any app)
2. **Pull Qwen 2.5** — `ollama pull qwen2.5:7b` (downloads the model ~4.7GB)
3. **Ollama runs automatically** on `localhost:11434`
4. **Run your pipeline** — it connects to Ollama, which runs Qwen 2.5 for summarization

