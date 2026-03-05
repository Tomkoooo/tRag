"""
NIS2 RAG System -- Local Audit Compliance Tool
Runs entirely on a MacBook Air M4 (16GB) using Ollama + LlamaIndex + HuggingFace + Streamlit.
"""

import os
import io
import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("./data")
STORAGE_DIR = Path("./storage")

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 6

LLM_MODEL = "llama3.1:8b"
EMBED_MODEL_NAME = "BAAI/bge-m3"
OLLAMA_TIMEOUT = 180  # seconds – generous for 8B on M4

QA_PROMPT_TMPL = PromptTemplate(
    "You are an expert NIS2 compliance auditor. Use ONLY the context below to "
    "answer the question. For every claim, cite the source document file name "
    "and page number (if available). If the context does not contain enough "
    "information, say so explicitly.\n\n"
    "Answer in the SAME LANGUAGE as the question.\n\n"
    "-----\n"
    "Context:\n{context_str}\n"
    "-----\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# ---------------------------------------------------------------------------
# Cached model initialisation
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~2.3 GB)…")
def get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to Ollama…")
def get_llm() -> Ollama:
    return Ollama(model=LLM_MODEL, request_timeout=OLLAMA_TIMEOUT)


def init_settings() -> None:
    """Configure LlamaIndex global settings once per session."""
    Settings.embed_model = get_embed_model()
    Settings.llm = get_llm()
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    Settings.transformations = [SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)]


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def save_uploaded_files(uploaded_files: list) -> list[Path]:
    """Persist Streamlit UploadedFile objects into DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in uploaded_files:
        dest = DATA_DIR / f.name
        dest.write_bytes(f.getvalue())
        saved.append(dest)
    return saved


def load_documents():
    """Load all supported documents from DATA_DIR with file-level metadata."""
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        return []

    def _file_metadata(file_path: str) -> dict:
        return {"file_name": Path(file_path).name}

    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".pdf", ".docx", ".txt"],
        file_metadata=_file_metadata,
        recursive=True,
    )
    return reader.load_data()


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


def _storage_exists() -> bool:
    if not STORAGE_DIR.exists():
        return False
    required = {"docstore.json", "index_store.json", "default__vector_store.json"}
    existing = {p.name for p in STORAGE_DIR.iterdir()}
    return required.issubset(existing)


def load_index():
    """Return a persisted VectorStoreIndex or None."""
    if not _storage_exists():
        return None
    storage_ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_ctx)


def build_index(documents) -> VectorStoreIndex:
    """Build a fresh VectorStoreIndex from documents and persist it."""
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index


def get_or_build_index(force_rebuild: bool = False):
    """Load persisted index or build from DATA_DIR documents."""
    if not force_rebuild:
        idx = load_index()
        if idx is not None:
            return idx

    docs = load_documents()
    if not docs:
        return None
    return build_index(docs)


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------


def get_query_engine(index: VectorStoreIndex):
    return index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="tree_summarize",
        text_qa_template=QA_PROMPT_TMPL,
    )


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


def format_evidence(source_nodes) -> list[dict]:
    """Extract structured evidence from retrieval source nodes."""
    evidence: list[dict] = []
    for i, node in enumerate(source_nodes, 1):
        meta = node.metadata or {}
        file_name = meta.get("file_name", "unknown")
        page = meta.get("page_label", meta.get("page_number", "n/a"))
        excerpt = node.get_content()[:300].replace("\n", " ").strip()
        score = round(node.score, 4) if node.score is not None else None
        evidence.append(
            {
                "rank": i,
                "file_name": file_name,
                "page": page,
                "score": score,
                "excerpt": excerpt,
            }
        )
    return evidence


def render_evidence(evidence: list[dict]) -> None:
    """Render evidence blocks inside the Streamlit chat."""
    if not evidence:
        return
    with st.expander(f"📎 Evidence ({len(evidence)} chunks)", expanded=False):
        for e in evidence:
            score_str = f" | score {e['score']}" if e['score'] is not None else ""
            st.markdown(
                f"**[{e['rank']}] {e['file_name']}** — page {e['page']}{score_str}"
            )
            st.caption(e["excerpt"])
            st.divider()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_batch(
    questions: list[str],
    query_engine,
    progress_bar,
    status_text,
) -> pd.DataFrame:
    """Run each question through the query engine and collect results."""
    rows: list[dict] = []
    total = len(questions)

    for i, question in enumerate(questions):
        status_text.text(f"Processing question {i + 1}/{total}…")
        progress_bar.progress((i + 1) / total)

        try:
            response = query_engine.query(question)
            answer = str(response).strip()
            evidence = format_evidence(response.source_nodes)
            evidence_files = "; ".join(
                sorted({e["file_name"] for e in evidence})
            )
            evidence_excerpts = " ||| ".join(e["excerpt"] for e in evidence)
        except Exception as exc:
            answer = f"ERROR: {exc}"
            evidence_files = ""
            evidence_excerpts = ""

        rows.append(
            {
                "question": question,
                "answer": answer,
                "evidence_files": evidence_files,
                "evidence_excerpts": evidence_excerpts,
            }
        )

    return pd.DataFrame(rows)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------


def check_ollama() -> tuple[bool, str]:
    """Quick connectivity test – returns (ok, message)."""
    try:
        llm = get_llm()
        llm.complete("ping")
        return True, f"✅ Ollama is running — model **{LLM_MODEL}**"
    except Exception as exc:
        return False, (
            f"⚠️ Cannot reach Ollama: `{exc}`\n\n"
            "Make sure Ollama is running:\n"
            "```bash\nollama serve\n```\n"
            f"And that the model is pulled:\n"
            f"```bash\nollama pull {LLM_MODEL}\n```"
        )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="NIS2 RAG Auditor", page_icon="🛡️", layout="wide")

    init_settings()

    # ---- Session state defaults ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index" not in st.session_state:
        st.session_state.index = None

    # Try to load a previously persisted index on first run
    if st.session_state.index is None and _storage_exists():
        with st.spinner("Loading persisted index…"):
            st.session_state.index = load_index()

    # ===================================================================
    # SIDEBAR
    # ===================================================================
    with st.sidebar:
        st.header("🛡️ NIS2 RAG Auditor")
        st.caption("Local audit-compliance assistant")
        st.divider()

        # -- Ollama status --
        ollama_ok, ollama_msg = check_ollama()
        st.markdown(ollama_msg)
        st.divider()

        # -- Document upload --
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & drop PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="doc_uploader",
        )

        col1, col2 = st.columns(2)

        with col1:
            index_btn = st.button("💾 Save & Index", use_container_width=True, disabled=not uploaded_files)
        with col2:
            reindex_btn = st.button("🔄 Re-index All", use_container_width=True)

        if index_btn and uploaded_files:
            saved = save_uploaded_files(uploaded_files)
            st.success(f"Saved {len(saved)} file(s) to `data/`")
            with st.spinner("Indexing documents — this may take a while…"):
                docs = load_documents()
                if docs:
                    st.session_state.index = build_index(docs)
                    st.success(f"Index built — {len(docs)} document chunk(s)")
                else:
                    st.warning("No supported documents found in `data/`.")

        if reindex_btn:
            with st.spinner("Re-indexing all documents…"):
                docs = load_documents()
                if docs:
                    st.session_state.index = build_index(docs)
                    st.success(f"Index rebuilt — {len(docs)} document chunk(s)")
                else:
                    st.warning("No documents found in `data/`. Upload files first.")

        st.divider()

        # -- Index status --
        st.subheader("📊 Index Status")
        if st.session_state.index is not None:
            st.info("Index loaded and ready.")
        elif _storage_exists():
            st.info("Persisted index found on disk (not loaded yet).")
        else:
            st.warning("No index — upload and index documents first.")

        # -- Existing files in data/ --
        if DATA_DIR.exists():
            files_on_disk = sorted(
                p.name for p in DATA_DIR.iterdir() if p.suffix in {".pdf", ".docx", ".txt"}
            )
            if files_on_disk:
                with st.expander(f"Files on disk ({len(files_on_disk)})", expanded=False):
                    for fn in files_on_disk:
                        st.text(fn)

        st.divider()
        st.caption(f"LLM: `{LLM_MODEL}` | Embeddings: `{EMBED_MODEL_NAME}`")
        st.caption(f"Chunk: {CHUNK_SIZE} tokens | top_k: {TOP_K}")

    # ===================================================================
    # MAIN AREA
    # ===================================================================

    tab_chat, tab_batch = st.tabs(["💬 Chat", "📋 Batch Processing"])

    # ---------------------------------------------------------------
    # TAB 1 — Chat
    # ---------------------------------------------------------------
    with tab_chat:
        if not ollama_ok:
            st.error("Ollama is not reachable. Fix the connection (see sidebar) before querying.")
            st.stop()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("evidence"):
                    render_evidence(msg["evidence"])

        user_input = st.chat_input("Ask a question about your NIS2 documents…")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state.index is None:
                assistant_msg = "⚠️ No index available. Please upload and index documents first (see sidebar)."
                st.session_state.messages.append({"role": "assistant", "content": assistant_msg, "evidence": []})
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        qe = get_query_engine(st.session_state.index)
                        response = qe.query(user_input)
                        answer = str(response).strip()
                        evidence = format_evidence(response.source_nodes)

                    st.markdown(answer)
                    render_evidence(evidence)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "evidence": evidence}
                    )

    # ---------------------------------------------------------------
    # TAB 2 — Batch Processing
    # ---------------------------------------------------------------
    with tab_batch:
        st.subheader("Batch Question Processing")
        st.markdown(
            "Upload an **Excel (.xlsx) or CSV** file with questions. "
            "The system will answer each question and attach evidence."
        )

        if not ollama_ok:
            st.error("Ollama is not reachable. Fix the connection first.")
            st.stop()

        batch_file = st.file_uploader(
            "Upload question file",
            type=["xlsx", "csv"],
            key="batch_uploader",
        )

        if batch_file is not None:
            if batch_file.name.endswith(".csv"):
                df_input = pd.read_csv(batch_file)
            else:
                df_input = pd.read_excel(batch_file, engine="openpyxl")

            st.dataframe(df_input, use_container_width=True)

            columns = list(df_input.columns)
            question_col = st.selectbox(
                "Select the column containing questions",
                options=columns,
                index=0,
            )

            if st.button("▶️ Process All Questions", use_container_width=True):
                if st.session_state.index is None:
                    st.error("No index available. Upload and index documents first.")
                else:
                    questions = df_input[question_col].dropna().astype(str).tolist()
                    if not questions:
                        st.warning("No questions found in the selected column.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        qe = get_query_engine(st.session_state.index)
                        results_df = process_batch(questions, qe, progress_bar, status_text)

                        status_text.text("Done!")
                        progress_bar.progress(1.0)
                        time.sleep(0.3)
                        status_text.empty()
                        progress_bar.empty()

                        st.success(f"Processed {len(results_df)} question(s).")
                        st.dataframe(results_df, use_container_width=True)

                        col_csv, col_xlsx = st.columns(2)
                        with col_csv:
                            st.download_button(
                                "⬇️ Download CSV",
                                data=results_df.to_csv(index=False).encode("utf-8"),
                                file_name="nis2_answers.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with col_xlsx:
                            st.download_button(
                                "⬇️ Download Excel",
                                data=to_excel_bytes(results_df),
                                file_name="nis2_answers.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )


if __name__ == "__main__":
    main()
