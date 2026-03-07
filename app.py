"""
NIS2 RAG System -- Local Audit Compliance Tool
Runs entirely on a MacBook Air M4 (16GB) using Ollama + LlamaIndex + HuggingFace + Streamlit.
"""

from __future__ import annotations

import io
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

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

APP_VERSION = "1.1.0"
DATA_DIR = Path("./data")
STORAGE_DIR = Path("./storage")
INDEX_META_PATH = STORAGE_DIR / "index_meta.json"

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_TOP_K = 4
MAX_TOP_K = 6
DEFAULT_RESPONSE_MODE = "compact"

DEFAULT_EMBED_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 300
DEFAULT_NUM_PREDICT = 320
DEFAULT_TEMPERATURE = 0.1
DEFAULT_KEEP_ALIVE = "20m"

OLLAMA_API_BASE = "http://localhost:11434"

MODEL_PROFILES = {
    "Balanced (recommended)": {
        "model": "llama3.1:8b",
        "description": "Best quality/speed balance for NIS2 answers on 16GB.",
        "top_k": 4,
        "num_predict": 320,
    },
    "Fast": {
        "model": "llama3.2:3b",
        "description": "Lowest latency, lower reasoning depth.",
        "top_k": 3,
        "num_predict": 220,
    },
    "Alternative Fast": {
        "model": "qwen2.5:7b",
        "description": "Good multilingual speed option.",
        "top_k": 4,
        "num_predict": 280,
    },
    "Heavy (slower)": {
        "model": "qwen2.5:14b",
        "description": "Higher quality potential, can be slow on 16GB.",
        "top_k": 4,
        "num_predict": 300,
    },
}

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
# Ollama API helpers
# ---------------------------------------------------------------------------


def _ollama_get(path: str, timeout: int = 8) -> dict:
    req = url_request.Request(f"{OLLAMA_API_BASE}{path}", method="GET")
    with url_request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=15)
def get_ollama_tags() -> dict:
    return _ollama_get("/api/tags")


def get_installed_models() -> list[str]:
    tags = get_ollama_tags()
    models = [m.get("name", "") for m in tags.get("models", []) if m.get("name")]
    return sorted(models)


def check_ollama(selected_model: str) -> tuple[bool, str, list[str]]:
    try:
        installed_models = get_installed_models()
        msg = f"Ollama reachable at `{OLLAMA_API_BASE}`"
        if selected_model in installed_models:
            msg += f"\n\nSelected model available: **{selected_model}**"
        else:
            msg += f"\n\nSelected model is not installed yet: **{selected_model}**"
        return True, msg, installed_models
    except url_error.URLError as exc:
        return False, f"Cannot reach Ollama server: `{exc}`", []
    except Exception as exc:
        return False, f"Ollama check failed: `{exc}`", []


def download_model(model_name: str, progress_bar, status_text) -> tuple[bool, str]:
    payload = json.dumps({"model": model_name, "stream": True}).encode("utf-8")
    req = url_request.Request(
        f"{OLLAMA_API_BASE}/api/pull",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with url_request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                status = event.get("status", "Downloading model...")
                total = event.get("total", 0)
                completed = event.get("completed", 0)
                status_text.text(status)
                if total and completed:
                    progress_bar.progress(min(completed / total, 1.0))

        progress_bar.progress(1.0)
        status_text.text("Model download complete.")
        get_ollama_tags.clear()
        return True, f"Model `{model_name}` downloaded successfully."
    except Exception as exc:
        return False, f"Model download failed: `{exc}`"


# ---------------------------------------------------------------------------
# Cached model initialisation
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~2.3 GB)...")
def get_embed_model(model_name: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=model_name)


@st.cache_resource(show_spinner="Preparing Ollama client...")
def get_llm(
    model_name: str,
    request_timeout: int,
    num_predict: int,
    keep_alive: str,
    temperature: float,
) -> Ollama:
    """Create Ollama client with backward-compatible options."""
    kwargs = {
        "model": model_name,
        "request_timeout": request_timeout,
        "temperature": temperature,
    }

    # Newer wrappers accept additional_kwargs / keep_alive.
    try:
        return Ollama(
            **kwargs,
            keep_alive=keep_alive,
            additional_kwargs={"num_predict": num_predict},
        )
    except TypeError:
        return Ollama(**kwargs)


def init_settings(
    embed_model_name: str,
    llm_model_name: str,
    request_timeout: int,
    num_predict: int,
    keep_alive: str,
    temperature: float,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Configure LlamaIndex global settings once per session rerun."""
    Settings.embed_model = get_embed_model(embed_model_name)
    Settings.llm = get_llm(llm_model_name, request_timeout, num_predict, keep_alive, temperature)
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    Settings.transformations = [
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def save_uploaded_files(uploaded_files: list) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in uploaded_files:
        dest = DATA_DIR / f.name
        dest.write_bytes(f.getvalue())
        saved.append(dest)
    return saved


def load_documents():
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
# Index compatibility and management
# ---------------------------------------------------------------------------


def _storage_exists() -> bool:
    if not STORAGE_DIR.exists():
        return False
    required = {"docstore.json", "index_store.json", "default__vector_store.json"}
    existing = {p.name for p in STORAGE_DIR.iterdir()}
    return required.issubset(existing)


def write_index_metadata(embed_model_name: str, chunk_size: int, chunk_overlap: int) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embed_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    INDEX_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_index_metadata() -> dict | None:
    if not INDEX_META_PATH.exists():
        return None
    try:
        return json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_index_compatibility(current_embed_model_name: str) -> tuple[bool, str]:
    """
    Safe check before query.
    - Missing metadata (legacy index): allow with warning.
    - Embedding mismatch: block querying and require re-index.
    """
    if not _storage_exists():
        return False, "No persisted index found."

    meta = read_index_metadata()
    if meta is None:
        return True, (
            "Legacy index detected (no metadata). Querying is allowed, but if retrieval "
            "looks wrong, click Re-index All to regenerate with compatibility metadata."
        )

    indexed_embed = meta.get("embedding_model")
    if indexed_embed and indexed_embed != current_embed_model_name:
        return False, (
            "Index embedding mismatch: stored index uses "
            f"`{indexed_embed}` but current app uses `{current_embed_model_name}`. "
            "Re-index is required to avoid retrieval errors/timeouts."
        )

    return True, "Index compatibility check passed."


def load_index():
    if not _storage_exists():
        return None
    storage_ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_ctx)


def build_index(documents, embed_model_name: str, chunk_size: int, chunk_overlap: int) -> VectorStoreIndex:
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    write_index_metadata(embed_model_name, chunk_size, chunk_overlap)
    return index


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------


def make_query_engine(index: VectorStoreIndex, top_k: int, response_mode: str):
    return index.as_query_engine(
        similarity_top_k=top_k,
        response_mode=response_mode,
        text_qa_template=QA_PROMPT_TMPL,
    )


def get_cached_query_engine(index: VectorStoreIndex, signature: tuple):
    if (
        "query_engine" not in st.session_state
        or st.session_state.get("qe_signature") != signature
    ):
        st.session_state.query_engine = make_query_engine(
            index=index,
            top_k=signature[2],
            response_mode=signature[3],
        )
        st.session_state.qe_signature = signature
    return st.session_state.query_engine


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


def format_evidence(source_nodes) -> list[dict]:
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
    if not evidence:
        return
    with st.expander(f"Evidence ({len(evidence)} chunks)", expanded=False):
        for e in evidence:
            score_str = f" | score {e['score']}" if e["score"] is not None else ""
            st.markdown(f"**[{e['rank']}] {e['file_name']}** - page {e['page']}{score_str}")
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
    rows: list[dict] = []
    total = len(questions)

    for i, question in enumerate(questions):
        status_text.text(f"Processing question {i + 1}/{total}...")
        progress_bar.progress((i + 1) / total)

        try:
            t0 = time.perf_counter()
            response = query_engine.query(question)
            elapsed = time.perf_counter() - t0
            answer = str(response).strip()
            evidence = format_evidence(response.source_nodes)
            evidence_files = "; ".join(sorted({e["file_name"] for e in evidence}))
            evidence_excerpts = " ||| ".join(e["excerpt"] for e in evidence)
        except Exception as exc:
            answer = f"ERROR: {exc}"
            evidence_files = ""
            evidence_excerpts = ""
            elapsed = 0.0

        rows.append(
            {
                "question": question,
                "answer": answer,
                "response_seconds": round(elapsed, 2),
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
# Streamlit UI
# ---------------------------------------------------------------------------


def init_session_defaults() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "index_compat_ok" not in st.session_state:
        st.session_state.index_compat_ok = True
    if "index_compat_msg" not in st.session_state:
        st.session_state.index_compat_msg = ""
    if "selected_profile" not in st.session_state:
        st.session_state.selected_profile = "Balanced (recommended)"


def main() -> None:
    st.set_page_config(page_title="NIS2 RAG Auditor", page_icon="🛡️", layout="wide")
    init_session_defaults()

    profile = MODEL_PROFILES[st.session_state.selected_profile]

    with st.sidebar:
        st.header("NIS2 RAG Auditor")
        st.caption("Local audit-compliance assistant")
        st.divider()

        st.subheader("Model and Performance")
        selected_profile = st.selectbox(
            "Model profile",
            options=list(MODEL_PROFILES.keys()),
            key="selected_profile",
            help="Use Fast if responses are slow or timing out.",
        )
        profile = MODEL_PROFILES[selected_profile]

        selected_model = st.text_input(
            "Model tag",
            value=profile["model"],
            help="Any local Ollama tag, e.g. llama3.1:8b",
        ).strip()

        st.caption(profile["description"])

        perf_mode = st.selectbox("Performance mode", options=["Speed", "Balanced"], index=1)
        if perf_mode == "Speed":
            default_top_k = min(profile["top_k"], 3)
            default_num_predict = min(profile["num_predict"], 220)
            response_mode = "compact"
        else:
            default_top_k = profile["top_k"]
            default_num_predict = profile["num_predict"]
            response_mode = DEFAULT_RESPONSE_MODE

        top_k = st.slider("Retrieval top_k", min_value=2, max_value=MAX_TOP_K, value=default_top_k)
        num_predict = st.slider("Max output tokens", min_value=96, max_value=768, value=default_num_predict, step=32)
        request_timeout = st.slider("Request timeout (seconds)", min_value=120, max_value=900, value=DEFAULT_TIMEOUT, step=30)
        keep_alive = st.selectbox("Keep model loaded", options=["5m", "15m", "20m", "30m", "1h"], index=2)
        temperature = st.slider("Temperature", min_value=0.0, max_value=0.4, value=DEFAULT_TEMPERATURE, step=0.05)

        st.divider()
        ollama_ok, ollama_msg, installed_models = check_ollama(selected_model)
        if ollama_ok:
            st.success(ollama_msg)
        else:
            st.error(ollama_msg)
            st.code("ollama serve")

        if installed_models:
            with st.expander(f"Installed models ({len(installed_models)})", expanded=False):
                for name in installed_models:
                    st.text(name)

        if ollama_ok and selected_model and selected_model not in installed_models:
            st.warning(f"Model `{selected_model}` is not installed.")
            if st.button("Download selected model", use_container_width=True):
                dl_progress = st.progress(0)
                dl_status = st.empty()
                ok, msg = download_model(selected_model, dl_progress, dl_status)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.divider()
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag and drop PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="doc_uploader",
        )

        col1, col2 = st.columns(2)
        with col1:
            index_btn = st.button("Save and Index", use_container_width=True, disabled=not uploaded_files)
        with col2:
            reindex_btn = st.button("Re-index All", use_container_width=True)

    # Configure LlamaIndex after selecting runtime options.
    init_settings(
        embed_model_name=DEFAULT_EMBED_MODEL_NAME,
        llm_model_name=selected_model or DEFAULT_LLM_MODEL,
        request_timeout=request_timeout,
        num_predict=num_predict,
        keep_alive=keep_alive,
        temperature=temperature,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    # Try loading persisted index on first run.
    if st.session_state.index is None and _storage_exists():
        with st.spinner("Loading persisted index..."):
            st.session_state.index = load_index()

    # Compatibility checks (safe for legacy indexes).
    if st.session_state.index is not None:
        compat_ok, compat_msg = check_index_compatibility(DEFAULT_EMBED_MODEL_NAME)
        st.session_state.index_compat_ok = compat_ok
        st.session_state.index_compat_msg = compat_msg

    if index_btn and uploaded_files:
        saved = save_uploaded_files(uploaded_files)
        st.success(f"Saved {len(saved)} file(s) to data/")
        with st.spinner("Indexing documents... this may take a while."):
            docs = load_documents()
            if docs:
                st.session_state.index = build_index(
                    documents=docs,
                    embed_model_name=DEFAULT_EMBED_MODEL_NAME,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                )
                st.session_state.index_compat_ok = True
                st.session_state.index_compat_msg = "Index rebuilt with current compatibility metadata."
                st.success(f"Index built - {len(docs)} document(s)")
            else:
                st.warning("No supported documents found in data/.")

    if reindex_btn:
        with st.spinner("Re-indexing all documents..."):
            docs = load_documents()
            if docs:
                st.session_state.index = build_index(
                    documents=docs,
                    embed_model_name=DEFAULT_EMBED_MODEL_NAME,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                )
                st.session_state.index_compat_ok = True
                st.session_state.index_compat_msg = "Index rebuilt with current compatibility metadata."
                st.success(f"Index rebuilt - {len(docs)} document(s)")
            else:
                st.warning("No documents found in data/. Upload files first.")

    with st.sidebar:
        st.subheader("Index Status")
        if st.session_state.index is None:
            st.warning("No index loaded.")
        elif st.session_state.index_compat_ok:
            st.info("Index loaded and compatible.")
        else:
            st.error("Index loaded but incompatible with current embedding settings.")
        if st.session_state.index_compat_msg:
            st.caption(st.session_state.index_compat_msg)

        if DATA_DIR.exists():
            files_on_disk = sorted(
                p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt"}
            )
            if files_on_disk:
                with st.expander(f"Files on disk ({len(files_on_disk)})", expanded=False):
                    for fn in files_on_disk:
                        st.text(fn)

        st.divider()
        st.caption(f"LLM: `{selected_model}`")
        st.caption(f"Embeddings: `{DEFAULT_EMBED_MODEL_NAME}`")
        st.caption(f"response_mode: `{response_mode}` | top_k: {top_k} | max_tokens: {num_predict}")

    tab_chat, tab_batch = st.tabs(["Chat", "Batch Processing"])

    # Chat tab
    with tab_chat:
        if not ollama_ok:
            st.error("Ollama is not reachable. Start it first.")
            st.stop()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("evidence"):
                    render_evidence(msg["evidence"])

        user_input = st.chat_input("Ask a question about your NIS2 documents...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state.index is None:
                msg = "No index available. Please upload and index documents first."
                st.session_state.messages.append({"role": "assistant", "content": msg, "evidence": []})
                with st.chat_message("assistant"):
                    st.markdown(msg)
            elif not st.session_state.index_compat_ok:
                msg = (
                    "Index compatibility check failed. Please click Re-index All before querying."
                )
                st.session_state.messages.append({"role": "assistant", "content": msg, "evidence": []})
                with st.chat_message("assistant"):
                    st.markdown(msg)
            elif selected_model not in installed_models:
                msg = (
                    f"Selected model `{selected_model}` is not installed. "
                    "Use 'Download selected model' in the sidebar."
                )
                st.session_state.messages.append({"role": "assistant", "content": msg, "evidence": []})
                with st.chat_message("assistant"):
                    st.markdown(msg)
            else:
                signature = (
                    id(st.session_state.index),
                    selected_model,
                    top_k,
                    response_mode,
                    num_predict,
                    request_timeout,
                    keep_alive,
                    temperature,
                )
                qe = get_cached_query_engine(st.session_state.index, signature)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        t0 = time.perf_counter()
                        response = qe.query(user_input)
                        elapsed = time.perf_counter() - t0

                    answer = str(response).strip()
                    evidence = format_evidence(response.source_nodes)
                    st.markdown(answer)
                    st.caption(f"Response time: {elapsed:.2f}s")
                    render_evidence(evidence)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "evidence": evidence}
                    )

    # Batch tab
    with tab_batch:
        st.subheader("Batch Question Processing")
        st.markdown(
            "Upload an Excel (.xlsx) or CSV file with questions. "
            "The system answers each question and attaches evidence."
        )

        if not ollama_ok:
            st.error("Ollama is not reachable. Start it first.")
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

            question_col = st.selectbox(
                "Select question column",
                options=list(df_input.columns),
                index=0,
            )

            if st.button("Process All Questions", use_container_width=True):
                if st.session_state.index is None:
                    st.error("No index available. Upload and index documents first.")
                elif not st.session_state.index_compat_ok:
                    st.error("Index compatibility check failed. Re-index is required.")
                elif selected_model not in installed_models:
                    st.error("Selected model is not installed. Download it first.")
                else:
                    questions = df_input[question_col].dropna().astype(str).tolist()
                    if not questions:
                        st.warning("No questions found in selected column.")
                    else:
                        signature = (
                            id(st.session_state.index),
                            selected_model,
                            top_k,
                            response_mode,
                            num_predict,
                            request_timeout,
                            keep_alive,
                            temperature,
                        )
                        qe = get_cached_query_engine(st.session_state.index, signature)

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_df = process_batch(questions, qe, progress_bar, status_text)

                        status_text.text("Done.")
                        progress_bar.progress(1.0)
                        time.sleep(0.25)
                        status_text.empty()
                        progress_bar.empty()

                        st.success(f"Processed {len(results_df)} question(s).")
                        st.dataframe(results_df, use_container_width=True)

                        col_csv, col_xlsx = st.columns(2)
                        with col_csv:
                            st.download_button(
                                "Download CSV",
                                data=results_df.to_csv(index=False).encode("utf-8"),
                                file_name="nis2_answers.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with col_xlsx:
                            st.download_button(
                                "Download Excel",
                                data=to_excel_bytes(results_df),
                                file_name="nis2_answers.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )


if __name__ == "__main__":
    main()
