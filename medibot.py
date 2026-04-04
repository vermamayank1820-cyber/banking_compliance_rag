"""
medibot.py — Banking Compliance RAG Assistant
═══════════════════════════════════════════════
Entry point: streamlit run medibot.py

Architecture:
  services/ingestion.py  — PDF loading, chunking, FAISS build/update
  services/retrieval.py  — score-aware chunk retrieval + confidence filter
  services/audit.py      — JSONL governance / audit log
  config.py              — all tuneable constants live here
"""

import os
import shutil
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import (
    AUDIT_LOG_PATH, DATA_PATH, DB_FAISS_PATH,
    LLM_MODEL, MULTI_QUERY_COUNT,
    REGISTRY_PATH, SIMILARITY_THRESHOLD, USE_HYDE, USE_MULTI_QUERY,
)
from services import audit, ingestion, retrieval

# ─── API key ──────────────────────────────────────────────────────────────────
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Add it to Streamlit secrets or a .env file.")
    st.stop()

# ─── Prompts ──────────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """You are a banking compliance assistant.
Use the retrieved document context below as your PRIMARY source.
Use the conversation history as SECONDARY context for continuity and follow-up awareness.
Do NOT default to not knowing — first try the context, then the conversation history.
If the answer genuinely cannot be found in either, say exactly: "Hey, sorry I don't know the answer to this."
Do not invent requirements, regulations, or guidance.
Be concise and compliance-focused.

Conversation History:
{history}

Context:
{context}

Question: {question}

Answer directly. No preamble."""

# HyDE: generate a hypothetical passage the document *would* contain.
# Embedding this passage (rather than the short user question) finds chunks
# that share the document's vocabulary even when the user's phrasing differs.
_HYDE_PROMPT_TEMPLATE = """You are a banking regulatory document. Write a concise technical paragraph (3–5 sentences) that would appear verbatim in a Basel III / capital adequacy / banking compliance document and directly answers the question below. Use precise regulatory terminology — ratios, article numbers, buffer names — exactly as they appear in such documents.

Question: {question}

Compliance document excerpt:"""

# Multi-query: rephrase with alternative regulatory synonyms to widen recall.
_MULTI_QUERY_PROMPT_TEMPLATE = """You are a banking compliance expert. Rephrase the question below in exactly {n} different ways, each using alternative regulatory terminology or synonyms (e.g. "CET1" ↔ "common equity tier 1", "leverage" ↔ "gearing", "SREP" ↔ "supervisory review"). Return only the rephrased questions, one per line, no numbering or bullets.

Original question: {question}

Rephrased questions:"""


# ─── Modern RAG helpers ───────────────────────────────────────────────────────

def _build_llm():
    return ChatGroq(model_name=LLM_MODEL, temperature=0.0, groq_api_key=groq_api_key)


def _generate_hyde_query(question: str) -> str:
    """
    HyDE — Hypothetical Document Embeddings.
    Ask the LLM to write a short passage that *would* answer the question
    in a banking compliance document.  This passage uses the document's own
    vocabulary, so its embedding lands close to the real chunks in vector space
    even when the user's phrasing uses completely different words.
    """
    try:
        prompt = PromptTemplate(
            template=_HYDE_PROMPT_TEMPLATE, input_variables=["question"]
        )
        chain = prompt | _build_llm() | StrOutputParser()
        return chain.invoke({"question": question})
    except Exception:
        return ""


def _expand_query(question: str, n: int = MULTI_QUERY_COUNT) -> list[str]:
    """
    Multi-query expansion — generate n alternative phrasings with different
    regulatory synonyms so that vocabulary gaps in any single phrasing are
    covered by at least one variant.
    """
    try:
        prompt = PromptTemplate(
            template=_MULTI_QUERY_PROMPT_TEMPLATE, input_variables=["question", "n"]
        )
        chain = prompt | _build_llm() | StrOutputParser()
        result = chain.invoke({"question": question, "n": n})
        lines = [ln.strip() for ln in result.strip().splitlines() if ln.strip()]
        return lines[:n]
    except Exception:
        return []


# ─── Vector store (cached across reruns) ──────────────────────────────────────
@st.cache_resource
def get_vectorstore():
    if not Path(DB_FAISS_PATH).exists():
        return None
    embedding_model = ingestion.get_embedding_model()
    return FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )


# ─── Core QA function ─────────────────────────────────────────────────────────
def answer_question(
    question: str,
    filter_filename: str | None = None,
    conversation_history: list | None = None,
) -> tuple:
    """
    Retrieve relevant chunks then generate a grounded answer.

    Returns: (answer_text, sources_list, max_relevance_score)

    sources_list entries:
      filename, page (1-indexed), score, preview (first 200 chars of chunk)
    """
    db = get_vectorstore()
    if db is None:
        audit.log_no_answer(question, "no_vectorstore")
        return "Hey, sorry I don't know the answer to this.", [], 0.0

    audit.log_question(question, filter_filename)

    # ── Modern RAG: build a richer set of query strings ────────────────────
    # Start with the original question, then optionally add a HyDE hypothetical
    # passage and multi-query variants.  retrieve_multi_query() deduplicates and
    # returns the best-scoring unique chunks across all query embeddings.
    queries: list[str] = [question]

    if USE_HYDE:
        hyde_passage = _generate_hyde_query(question)
        if hyde_passage:
            queries.append(hyde_passage)

    if USE_MULTI_QUERY:
        variants = _expand_query(question)
        queries.extend(variants)

    if len(queries) > 1:
        relevant, above_threshold, max_score = retrieval.retrieve_multi_query(
            db, queries, filter_filename=filter_filename
        )
    else:
        relevant, above_threshold, max_score = retrieval.retrieve(
            db, question, filter_filename=filter_filename
        )

    if not above_threshold or not relevant:
        audit.log_no_answer(question, f"below_threshold (max={max_score:.3f})")
        return "Hey, sorry I don't know the answer to this.", [], max_score

    context = "\n\n---\n\n".join(doc.page_content for doc, _ in relevant)

    # Format the last 6 turns of conversation history for the prompt
    history_lines: list[str] = []
    if conversation_history:
        for msg in conversation_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
    history_text = "\n".join(history_lines) if history_lines else "No prior conversation."

    prompt = PromptTemplate(
        template=_PROMPT_TEMPLATE, input_variables=["context", "question", "history"]
    )
    chain = prompt | _build_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question, "history": history_text})

    # Build structured source references
    sources = []
    for doc, score in relevant:
        raw_page = doc.metadata.get("page", None)
        sources.append({
            "filename": doc.metadata.get("filename", doc.metadata.get("source", "Unknown")),
            "page": (raw_page + 1) if isinstance(raw_page, int) else "n/a",
            "score": round(score, 3),
            "preview": doc.page_content[:200].replace("\n", " "),
            "has_table": doc.metadata.get("has_table", False),
        })

    audit.log_answer(
        question,
        [f"{s['filename']} p.{s['page']}" for s in sources],
        max_score,
        low_confidence=False,
    )

    return answer, sources, max_score


# ─── Upload / rebuild helpers ─────────────────────────────────────────────────
def _data_path() -> Path:
    p = Path(DATA_PATH)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_uploaded_files(uploaded_files, replace_existing: bool) -> dict:
    data_dir = _data_path()
    filenames = [f.name for f in uploaded_files]
    audit.log_upload(filenames, replace_existing)

    if replace_existing:
        for pdf in data_dir.glob("*.pdf"):
            pdf.unlink()
        if Path(DB_FAISS_PATH).exists():
            shutil.rmtree(DB_FAISS_PATH)
        if Path(REGISTRY_PATH).exists():
            Path(REGISTRY_PATH).unlink()

    for uf in uploaded_files:
        (data_dir / uf.name).write_bytes(uf.getbuffer())

    all_pdfs = sorted(data_dir.glob("*.pdf"))
    audit.log_indexing_start(filenames)

    try:
        result = ingestion.build_vectorstore(all_pdfs, force_rebuild=replace_existing)
        audit.log_indexing_complete(
            result["indexed"], result["total_new_chunks"], result["skipped"]
        )
        get_vectorstore.clear()
        return result
    except Exception as exc:
        audit.log_indexing_failed(filenames, str(exc))
        raise


def rebuild_all() -> dict:
    """Force a full reindex of all PDFs currently in data/."""
    all_pdfs = sorted(_data_path().glob("*.pdf"))
    if not all_pdfs:
        raise ValueError("No PDFs in the data directory.")
    names = [p.name for p in all_pdfs]
    audit.log_indexing_start(names)
    try:
        result = ingestion.build_vectorstore(all_pdfs, force_rebuild=True)
        audit.log_indexing_complete(
            result["indexed"], result["total_new_chunks"], result["skipped"]
        )
        get_vectorstore.clear()
        return result
    except Exception as exc:
        audit.log_indexing_failed(names, str(exc))
        raise


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def _render_sidebar() -> str | None:
    """Render the full sidebar. Returns the active filename filter (or None)."""
    with st.sidebar:
        st.markdown("## Banking Compliance")
        st.caption("RAG Assistant")
        st.divider()

        # ── System status ──────────────────────────────────────────────────
        st.markdown("### System Status")
        registry = ingestion.load_registry()
        db_ready = Path(DB_FAISS_PATH).exists()

        col1, col2 = st.columns(2)
        with col1:
            if db_ready:
                st.success("Store: Ready")
            else:
                st.error("Store: Not built")
        with col2:
            st.metric("Docs", len(registry), label_visibility="visible")

        st.caption(f"Model: `{LLM_MODEL}`")

        if registry:
            last_ts = max(v.get("indexed_at", "") for v in registry.values())
            if last_ts:
                st.caption(f"Last indexed: {last_ts[:19].replace('T', ' ')} UTC")

        st.divider()

        # ── Document filter ────────────────────────────────────────────────
        st.markdown("### Filter")
        options = ["All Documents"] + sorted(registry.keys())
        selection = st.selectbox(
            "Search within document",
            options,
            label_visibility="collapsed",
            key="doc_filter_select",
        )
        filter_filename = None if selection == "All Documents" else selection

        st.divider()

        # ── Knowledge base management ──────────────────────────────────────
        st.markdown("### Knowledge Base")

        if registry:
            for fname, meta in sorted(registry.items()):
                with st.expander(fname, expanded=False):
                    st.caption(f"Pages: {meta.get('page_count', 'n/a')}")
                    st.caption(f"Chunks: {meta.get('chunk_count', 'n/a')}")
                    ts = meta.get("indexed_at", "")[:19].replace("T", " ")
                    st.caption(f"Indexed: {ts} UTC")
                    st.caption(f"Status: {meta.get('status', 'n/a')}")
        else:
            st.caption("No documents indexed yet.")

        # Upload
        with st.expander("Upload PDFs", expanded=not bool(registry)):
            uploaded = st.file_uploader(
                "Select one or more PDF files",
                type="pdf",
                accept_multiple_files=True,
                key="sidebar_uploader",
            )
            replace_all = st.checkbox(
                "Replace all existing PDFs",
                value=False,
                key="sidebar_replace",
            )
            if st.button("Save & Index", use_container_width=True, key="sidebar_save"):
                if not uploaded:
                    st.warning("Select at least one PDF first.")
                else:
                    with st.spinner("Indexing — this may take a minute…"):
                        try:
                            result = save_uploaded_files(uploaded, replace_all)
                            indexed = result["indexed"]
                            skipped = result["skipped"]
                            if indexed:
                                st.success(f"Indexed: {', '.join(indexed)}")
                            if skipped:
                                st.info(f"Skipped (unchanged): {', '.join(skipped)}")
                            st.session_state.messages = []
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Indexing failed: {exc}")

        if registry:
            if st.button(
                "Rebuild Knowledge Base",
                use_container_width=True,
                key="sidebar_rebuild",
                help="Re-embed all documents from scratch",
            ):
                with st.spinner("Rebuilding…"):
                    try:
                        result = rebuild_all()
                        st.success(
                            f"Rebuilt. {len(result['indexed'])} document(s) re-indexed."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Rebuild failed: {exc}")

    return filter_filename


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Banking Compliance RAG",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS: hide the default Streamlit chat input widget; style the custom form
    st.markdown(
        """
        <style>
        div[data-testid="stChatInput"] { display: none; }
        section.main > div.block-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding-bottom: 2rem;
        }
        div[data-testid="stForm"] {
            background: #1e2030;
            border: 1px solid #3a3d4d;
            border-radius: 999px;
            padding: 0.4rem 0.8rem;
        }
        div[data-testid="stForm"] input {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
            border-radius: 999px;
            min-width: 2.6rem;
            min-height: 2.6rem;
            padding: 0;
        }
        .push-to-bottom { margin-top: auto; min-height: 8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    filter_filename = _render_sidebar()

    # ── Page header ────────────────────────────────────────────────────────
    st.title("Banking Compliance RAG Assistant")
    st.caption(
        "Answers are grounded strictly in your uploaded compliance documents. "
        "If the context does not contain the answer, the assistant will say so."
    )

    # ── Session state init ─────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── No-KB warning ──────────────────────────────────────────────────────
    if not Path(DB_FAISS_PATH).exists():
        st.info(
            "No knowledge base found. "
            "Upload one or more compliance PDFs using the sidebar to get started.",
            icon="ℹ️",
        )

    # ── Render chat history ────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources")
            if sources:
                max_score = msg.get("max_score", 0.0)
                label = (
                    f"{len(sources)} source chunk(s) — "
                    f"top relevance: {max_score:.2f}"
                )
                with st.expander(label, expanded=False):
                    for src in sources:
                        badge = " `[table]`" if src.get("has_table") else ""
                        st.markdown(
                            f"**{src['filename']}** — Page {src['page']}"
                            f"  |  relevance: `{src['score']}`{badge}"
                        )
                        st.caption(f"> {src['preview']}…")

    # ── Push composer to bottom ────────────────────────────────────────────
    st.markdown('<div class="push-to-bottom"></div>', unsafe_allow_html=True)

    # ── Chat composer ──────────────────────────────────────────────────────
    with st.form("chat_composer", clear_on_submit=True):
        input_col, send_col = st.columns([11, 1], vertical_alignment="center")
        with input_col:
            prompt = st.text_input(
                "question",
                placeholder="Ask a banking compliance question…",
                label_visibility="collapsed",
            )
        with send_col:
            submitted = st.form_submit_button("➤", use_container_width=True)

    # ── Handle submission ──────────────────────────────────────────────────
    if submitted and prompt.strip():
        question = prompt.strip()

        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                ans, sources, max_score = answer_question(
                    question, filter_filename, st.session_state.messages
                )
            st.markdown(ans)
            if sources:
                label = (
                    f"{len(sources)} source chunk(s) — "
                    f"top relevance: {max_score:.2f}"
                )
                with st.expander(label, expanded=True):
                    for src in sources:
                        badge = " `[table]`" if src.get("has_table") else ""
                        st.markdown(
                            f"**{src['filename']}** — Page {src['page']}"
                            f"  |  relevance: `{src['score']}`{badge}"
                        )
                        st.caption(f"> {src['preview']}…")

        st.session_state.messages.append({
            "role": "assistant",
            "content": ans,
            "sources": sources,
            "max_score": max_score,
        })


if __name__ == "__main__":
    main()
