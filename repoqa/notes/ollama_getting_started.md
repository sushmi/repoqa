## Ollama Crash Course for RepoQA

### 1. What is Ollama?

A local LLM server. One app that downloads, loads, and serves models via a REST API on `localhost:11434`. Think of it as "Docker for LLMs."

### 2. Install

```bash
# macOS
brew install ollama

# or download from https://ollama.com

```

### 3. Pull Qwen 2.5

```bash
# Downloads the 4-bit quantized model (~4.7 GB, one-time)
ollama pull qwen2.5:7b

```

### 4. Test it works

```bash
# Interactive chat (like ChatGPT in your terminal)
ollama run qwen2.5:7b
>>> What is a REST API?
# Type /bye to exit

```

### 5. How your project talks to it

Ollama runs as a background server. Your code sends HTTP requests to it:

```bash
# This is what langchain-ollama does under the hood
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [{"role": "user", "content": "Summarize this Python function..."}]
}'

```

### 6. Key commands

```bash
ollama list              # Show downloaded models
ollama pull qwen2.5:7b   # Download a model
ollama rm qwen2.5:7b     # Delete a model (frees ~4.7 GB)
ollama run qwen2.5:7b    # Interactive chat
ollama serve             # Start the server (auto-starts on macOS)
ollama ps                # Show running models (loaded in memory)

```

### 7. How it connects to your project

```
settings.py                    pipeline.py
┌─────────────────────┐        ┌──────────────────────────┐
│ llm_provider: ollama │───────▶│ _build_llm()             │
│ chat_model: qwen2.5:7b│      │   → ChatOllama(          │
│ ollama_base_url:     │       │       model="qwen2.5:7b", │
│  localhost:11434     │       │       base_url=...,       │
└─────────────────────┘        │     )                     │
                               └───────────┬──────────────┘
                                           │ HTTP POST
                                           ▼
                               ┌──────────────────────────┐
                               │ Ollama server (:11434)    │
                               │   → loads qwen2.5:7b      │
                               │   → runs inference        │
                               │   → returns text          │
                               └──────────────────────────┘

```

### 8. Running your pipeline

```bash
# Terminal 1: Ollama starts automatically on macOS, but if needed:
ollama serve

# Terminal 2: Run your project
cd /Users/sushmi/dev/nlp/repoqa
.venv/bin/python -m repoqa.summarization.pipeline  # or however you run it

```

### 9. Model sizes (pick based on your RAM)

```bash
ollama pull qwen2.5:3b    # ~2 GB, fast, lower quality
ollama pull qwen2.5:7b    # ~4.7 GB, good balance (recommended)
ollama pull qwen2.5:14b   # ~9 GB, better quality, needs 16GB+ RAM

```

**Rule of thumb**: you need ~2x the model file size in free RAM. So 7B needs ~10 GB free.

### 10. Switching providers

Your config makes switching easy via `.env` or environment variables:

```bash
# Local Qwen (default, free)
LLM_PROVIDER=ollama
CHAT_MODEL=qwen2.5:7b

# OpenAI (paid)
LLM_PROVIDER=openai
CHAT_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Anthropic (paid)
LLM_PROVIDER=anthropic
CHAT_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...

```

No code changes needed — just change the env vars.

### Next step

Install Ollama and pull the model:

```bash
brew install ollama && ollama pull qwen2.5:7b

```

Then install the Python connector:

```bash
uv pip install langchain-ollama
```

