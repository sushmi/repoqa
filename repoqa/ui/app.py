"""Streamlit conversational interface for RepoQA.

Chat UI that wraps the Stage 2 + Stage 3 pipeline:
  - User asks a question in natural language
  - Backend: HybridRetriever → AnswerGenerator
  - UI renders: answer, citations (with verification status),
    retrieved chunks, and confidence

Run:
    uv run streamlit run repoqa/ui/app.py -- \
        --chunks ./data/flask/transformers_chunks.json \
        --repo-name flask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from repoqa.qa.pipeline import QAPipeline
from repoqa.qa.answer_generator import QAResult


# ── CLI args passed through Streamlit ─────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON.")
    parser.add_argument("--repo-name", default="repo")
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=4000)
    # Streamlit passes its own args; ignore unknown ones
    args, _ = parser.parse_known_args()
    return args


# ── Pipeline initialization (cached — loads once per session) ─────

@st.cache_resource(show_spinner="Loading repository and building index...")
def load_pipeline(chunks_path: str, max_context_tokens: int) -> QAPipeline:
    """Load the QA pipeline once. Cached across reruns."""
    return QAPipeline.from_chunks_file(
        chunks_path=chunks_path,
        max_context_tokens=max_context_tokens,
    )


# ── Rendering helpers ─────────────────────────────────────────────

_TYPE_EMOJI = {
    "architecture": "🏛️",
    "logic": "⚙️",
    "deployment": "🚀",
}

_CONFIDENCE_COLOR = {
    "high": "green",
    "medium": "orange",
    "low": "red",
}


def render_answer(result: QAResult) -> None:
    """Render a full QA result inside the chat."""
    qtype = result.question_type
    emoji = _TYPE_EMOJI.get(qtype, "❓")
    conf_color = _CONFIDENCE_COLOR.get(result.confidence, "gray")

    # Header badges
    cols = st.columns([1, 1, 1, 3])
    cols[0].markdown(f"**{emoji} {qtype}**")
    cols[1].markdown(f"Confidence: :{conf_color}[**{result.confidence}**]")
    cols[2].markdown(
        f"Citations: **{result.verified.verified_count}/"
        f"{len(result.verified.citations)}** ✓"
    )

    # The answer
    st.markdown(result.answer)

    # Citations panel
    if result.verified.citations:
        with st.expander(
            f"📎 Citations ({result.verified.verified_count} verified, "
            f"{result.verified.unverified_count} unverified)"
        ):
            for cite in result.verified.citations:
                loc = cite.file_path
                if cite.start_line:
                    loc += f":{cite.start_line}"
                    if cite.end_line and cite.end_line != cite.start_line:
                        loc += f"-{cite.end_line}"
                if cite.verified:
                    st.markdown(f"- ✅ `{loc}`")
                else:
                    st.markdown(f"- ⚠️ `{loc}` _(not found in retrieved context)_")

    # Retrieved chunks
    chunks_used = result.verified.used_chunk_ids
    with st.expander(
        f"🔎 Retrieved context ({len(result.retrieved_chunk_ids)} chunks, "
        f"{len(chunks_used)} cited)"
    ):
        st.caption(
            "These chunks were retrieved by Stage 2 (hybrid retrieval). "
            "Ones marked ★ were cited in the answer."
        )
        for chunk_id in result.retrieved_chunk_ids:
            mark = "★ " if chunk_id in chunks_used else "  "
            st.code(chunk_id, language=None)


def render_unverified_warning(result: QAResult) -> None:
    """Show a warning if many citations failed verification."""
    if (
        result.verified.unverified_count > 0
        and result.verified.coverage_ratio < 0.5
    ):
        st.warning(
            f"⚠️ Low citation coverage: only "
            f"{result.verified.coverage_ratio:.0%} of citations could be "
            f"verified against the retrieved context. Answer may contain "
            f"unsupported claims."
        )


# ── Main app ──────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    st.set_page_config(
        page_title="RepoQA",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar: config + stats ──
    with st.sidebar:
        st.title("RepoQA")
        st.caption("Retrieval-augmented QA for code repositories")

        st.markdown(f"**Repo:** `{args.repo_name}`")
        st.markdown(f"**Chunks file:** `{Path(args.chunks).name}`")

        st.divider()

        st.subheader("Settings")
        n_results = st.slider(
            "Chunks to retrieve", min_value=3, max_value=20,
            value=args.n_results,
            help="Top-K chunks fetched by hybrid retrieval per question.",
        )
        show_raw = st.checkbox(
            "Show raw LLM output", value=False,
            help="Display the unparsed model response for debugging.",
        )

        st.divider()

        if st.button("🔄 New conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption("Pipeline: classify → expand → semantic + BM25 → RRF → Qwen")

    # ── Initialize state ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        pipeline = load_pipeline(args.chunks, args.max_context_tokens)
    except Exception as exc:
        st.error(f"Failed to load pipeline: {exc}")
        st.stop()

    # ── Main chat area ──
    st.header("💬 Ask about the repository")
    st.caption(
        "Ask architecture, logic, or deployment questions. "
        "Answers cite source files so you can verify claims."
    )

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                result: QAResult = msg["result"]
                render_answer(result)
                if show_raw:
                    with st.expander("🐞 Raw LLM output"):
                        st.text(result.raw_llm_output)
                render_unverified_warning(result)

    # New question input
    if question := st.chat_input("Ask a question about the codebase..."):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = pipeline.ask(question, n_results=n_results)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()
                render_answer(result)
                if show_raw:
                    with st.expander("🐞 Raw LLM output"):
                        st.text(result.raw_llm_output)
                render_unverified_warning(result)

        st.session_state.messages.append(
            {"role": "assistant", "result": result}
        )


if __name__ == "__main__":
    main()
