"""
NIS2 RAG System -- Local Audit Compliance Tool
Runs entirely on a MacBook Air M4 (16GB) using Ollama + LlamaIndex + HuggingFace + Streamlit.
"""

from __future__ import annotations

import io
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from tempfile import TemporaryDirectory
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
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

try:
    from PIL import Image
    import pytesseract

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


APP_VERSION = "1.2.0"
DATA_DIR = Path("./data")
STORAGE_DIR = Path("./storage")
INDEX_META_PATH = STORAGE_DIR / "index_meta.json"
BATCH_RUNS_DIR = Path("./batch_runs")

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_RESPONSE_MODE = "compact"
DEFAULT_EMBED_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 300
DEFAULT_TEMPERATURE = 0.1
MAX_TOP_K = 6
OLLAMA_API_BASE = "http://localhost:11434"
SUPPORTED_DOC_EXTS = {".pdf", ".docx", ".txt"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

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


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hash_row_id(source_row_index: int, question: str) -> str:
    payload = f"{source_row_index}::{question.strip()}".encode("utf-8", errors="ignore")
    return sha1(payload).hexdigest()[:16]


def safe_read_json(path: Path, default: dict | None = None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dataframe_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file, engine="openpyxl")


def _ollama_get(path: str, timeout: int = 8) -> dict:
    req = url_request.Request(f"{OLLAMA_API_BASE}{path}", method="GET")
    with url_request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=15)
def get_ollama_tags() -> dict:
    return _ollama_get("/api/tags")


def get_installed_models() -> list[str]:
    tags = get_ollama_tags()
    return sorted([m.get("name", "") for m in tags.get("models", []) if m.get("name")])


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
                status_text.text(event.get("status", "Downloading model..."))
                total = event.get("total", 0)
                completed = event.get("completed", 0)
                if total and completed:
                    progress_bar.progress(min(completed / total, 1.0))
        progress_bar.progress(1.0)
        status_text.text("Model download complete.")
        get_ollama_tags.clear()
        return True, f"Model `{model_name}` downloaded successfully."
    except Exception as exc:
        return False, f"Model download failed: `{exc}`"


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
    kwargs = {"model": model_name, "request_timeout": request_timeout, "temperature": temperature}
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
    Settings.embed_model = get_embed_model(embed_model_name)
    Settings.llm = get_llm(llm_model_name, request_timeout, num_predict, keep_alive, temperature)
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    Settings.transformations = [SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)]


def save_uploaded_files(uploaded_files: list) -> list[Path]:
    ensure_dir(DATA_DIR)
    saved: list[Path] = []
    for f in uploaded_files:
        dest = DATA_DIR / f.name
        dest.write_bytes(f.getvalue())
        saved.append(dest)
    return saved


def load_text_documents() -> list[Document]:
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        return []

    def _file_metadata(file_path: str) -> dict:
        return {"file_name": Path(file_path).name, "source_type": "document"}

    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=sorted(SUPPORTED_DOC_EXTS),
        file_metadata=_file_metadata,
        recursive=True,
    )
    return reader.load_data()


def load_image_documents_ocr(ocr_lang: str) -> tuple[list[Document], list[str]]:
    docs: list[Document] = []
    errors: list[str] = []
    if not OCR_AVAILABLE:
        return docs, ["OCR dependencies missing (install Pillow + pytesseract + tesseract binary)."]
    if not DATA_DIR.exists():
        return docs, errors
    for image_path in sorted(DATA_DIR.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img, lang=ocr_lang).strip()
            if text:
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "file_name": image_path.name,
                            "page_label": "image_1",
                            "source_type": "image_ocr",
                        },
                    )
                )
        except Exception as exc:
            errors.append(f"{image_path.name}: {exc}")
    return docs, errors


def load_documents(enable_image_ocr: bool, ocr_lang: str) -> tuple[list[Document], list[str]]:
    docs = load_text_documents()
    ocr_errors: list[str] = []
    if enable_image_ocr:
        image_docs, ocr_errors = load_image_documents_ocr(ocr_lang=ocr_lang)
        docs.extend(image_docs)
    return docs, ocr_errors


def _storage_exists() -> bool:
    if not STORAGE_DIR.exists():
        return False
    required = {"docstore.json", "index_store.json", "default__vector_store.json"}
    existing = {p.name for p in STORAGE_DIR.iterdir()}
    return required.issubset(existing)


def write_index_metadata(embed_model_name: str, chunk_size: int, chunk_overlap: int, image_ocr_enabled: bool, ocr_lang: str) -> None:
    meta = {
        "version": APP_VERSION,
        "created_at": now_utc_iso(),
        "embedding_model": embed_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "image_ocr_enabled": image_ocr_enabled,
        "ocr_lang": ocr_lang,
    }
    write_json(INDEX_META_PATH, meta)


def read_index_metadata() -> dict | None:
    return safe_read_json(INDEX_META_PATH, default=None)


def check_index_compatibility(current_embed_model_name: str) -> tuple[bool, str]:
    if not _storage_exists():
        return False, "No persisted index found."
    meta = read_index_metadata()
    if meta is None:
        return True, "Legacy index detected (no metadata). Querying is allowed with warning."
    indexed_embed = meta.get("embedding_model")
    if indexed_embed and indexed_embed != current_embed_model_name:
        return False, (
            "Index embedding mismatch: stored index uses "
            f"`{indexed_embed}` but current app uses `{current_embed_model_name}`. Re-index required."
        )
    return True, "Index compatibility check passed."


def load_index():
    if not _storage_exists():
        return None
    storage_ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_ctx)


def build_index(documents, embed_model_name: str, chunk_size: int, chunk_overlap: int, image_ocr_enabled: bool, ocr_lang: str) -> VectorStoreIndex:
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    ensure_dir(STORAGE_DIR)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    write_index_metadata(embed_model_name, chunk_size, chunk_overlap, image_ocr_enabled, ocr_lang)
    return index


def create_index_zip_bytes() -> bytes | None:
    if not _storage_exists():
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in STORAGE_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(STORAGE_DIR).as_posix())
    return buf.getvalue()


def import_index_from_zip(uploaded_zip, expected_embed_model: str) -> tuple[bool, str]:
    try:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            zip_path = tmp / "index.zip"
            zip_path.write_bytes(uploaded_zip.getvalue())
            extract_dir = tmp / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            required = {"docstore.json", "index_store.json", "default__vector_store.json", "index_meta.json"}
            existing = {p.name for p in extract_dir.iterdir() if p.is_file()}
            if not required.issubset(existing):
                return False, "Invalid index package: required files missing."
            meta = safe_read_json(extract_dir / "index_meta.json", default={}) or {}
            if not meta.get("version"):
                return False, "Invalid index package: metadata version missing."
            if meta.get("embedding_model") != expected_embed_model:
                return False, "Index import blocked due to embedding model mismatch."
            if STORAGE_DIR.exists():
                backup = STORAGE_DIR.with_name(f"storage_backup_{int(time.time())}")
                shutil.move(str(STORAGE_DIR), str(backup))
            shutil.copytree(extract_dir, STORAGE_DIR)
        return True, "Index imported successfully."
    except Exception as exc:
        return False, f"Index import failed: {exc}"


def make_query_engine(index: VectorStoreIndex, top_k: int, response_mode: str):
    return index.as_query_engine(similarity_top_k=top_k, response_mode=response_mode, text_qa_template=QA_PROMPT_TMPL)


def get_cached_query_engine(index: VectorStoreIndex, signature: tuple):
    if "query_engine" not in st.session_state or st.session_state.get("qe_signature") != signature:
        st.session_state.query_engine = make_query_engine(index=index, top_k=signature[2], response_mode=signature[3])
        st.session_state.qe_signature = signature
    return st.session_state.query_engine


def format_evidence(source_nodes) -> list[dict]:
    evidence: list[dict] = []
    for i, node in enumerate(source_nodes, 1):
        meta = node.metadata or {}
        evidence.append(
            {
                "rank": i,
                "file_name": meta.get("file_name", "unknown"),
                "page": meta.get("page_label", meta.get("page_number", "n/a")),
                "source_type": meta.get("source_type", "document"),
                "score": round(node.score, 4) if node.score is not None else None,
                "excerpt": node.get_content()[:300].replace("\n", " ").strip(),
            }
        )
    return evidence


def evidence_to_audit_fields(evidence: list[dict], max_excerpts: int = 3) -> dict:
    files = sorted({e["file_name"] for e in evidence})
    pages = sorted({str(e["page"]) for e in evidence})
    backlinks = [f"{e['file_name']}#page={e['page']}" for e in evidence]
    payload = {
        "source_files": "; ".join(files),
        "source_pages": "; ".join(pages),
        "source_backlinks": " | ".join(backlinks),
        "evidence_excerpts": " ||| ".join(e["excerpt"] for e in evidence),
    }
    for idx in range(max_excerpts):
        payload[f"proof_excerpt_{idx + 1}"] = evidence[idx]["excerpt"] if idx < len(evidence) else ""
    return payload


def render_evidence(evidence: list[dict]) -> None:
    if not evidence:
        return
    with st.expander(f"Evidence ({len(evidence)} chunks)", expanded=False):
        for e in evidence:
            score_str = f" | score {e['score']}" if e["score"] is not None else ""
            st.markdown(f"**[{e['rank']}] {e['file_name']}** - page {e['page']} ({e['source_type']}){score_str}")
            st.caption(e["excerpt"])
            st.divider()


def batch_job_dir(job_id: str) -> Path:
    return BATCH_RUNS_DIR / job_id


def job_paths(job_id: str) -> dict[str, Path]:
    root = batch_job_dir(job_id)
    return {
        "root": root,
        "meta": root / "job_meta.json",
        "checkpoint": root / "job_checkpoint.json",
        "progress": root / "job_progress.jsonl",
        "input": root / "input_questions.csv",
        "answers_csv": root / "answers_only.csv",
        "answers_xlsx": root / "answers_only.xlsx",
        "merged_xlsx": root / "merge_with_original.xlsx",
        "partial_xlsx": root / "job_output_partial.xlsx",
    }


def list_existing_jobs() -> list[dict]:
    ensure_dir(BATCH_RUNS_DIR)
    jobs: list[dict] = []
    for path in sorted(BATCH_RUNS_DIR.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        meta = safe_read_json(path / "job_meta.json", default={}) or {}
        cp = safe_read_json(path / "job_checkpoint.json", default={}) or {}
        jobs.append(
            {
                "job_id": path.name,
                "created_at": meta.get("created_at", "unknown"),
                "model": meta.get("model", "unknown"),
                "next_index": cp.get("next_index", 0),
                "done": cp.get("is_done", False),
            }
        )
    return jobs


def create_job(
    df_input: pd.DataFrame,
    question_col: str,
    model_tag: str,
    top_k: int,
    resume_mode: str,
    prior_answers_df: pd.DataFrame | None,
    reaudit_mode: bool,
    rerun_statuses: list[str],
) -> str:
    ensure_dir(BATCH_RUNS_DIR)
    job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
    paths = job_paths(job_id)
    ensure_dir(paths["root"])

    df = df_input.copy().reset_index(drop=False).rename(columns={"index": "source_row_index"})
    df["question"] = df[question_col].astype(str)
    df["row_id"] = df.apply(lambda r: hash_row_id(int(r["source_row_index"]), r["question"]), axis=1)
    df.to_csv(paths["input"], index=False)

    preprocessed_row_ids: set[str] = set()
    if prior_answers_df is not None and not prior_answers_df.empty:
        tmp = prior_answers_df.copy()
        if "row_id" in tmp.columns:
            if reaudit_mode and "answer_status" in tmp.columns and rerun_statuses:
                keep_done = tmp[~tmp["answer_status"].isin(rerun_statuses)]
                preprocessed_row_ids.update(keep_done["row_id"].dropna().astype(str).tolist())
            else:
                preprocessed_row_ids.update(tmp["row_id"].dropna().astype(str).tolist())
        elif "question" in tmp.columns:
            done_q = set(tmp["question"].dropna().astype(str).tolist())
            preprocessed_row_ids.update(df[df["question"].isin(done_q)]["row_id"].tolist())

    write_json(
        paths["meta"],
        {
            "job_id": job_id,
            "created_at": now_utc_iso(),
            "model": model_tag,
            "top_k": top_k,
            "question_col": question_col,
            "total_rows": int(len(df)),
            "resume_mode": resume_mode,
            "reaudit_mode": reaudit_mode,
            "rerun_statuses": rerun_statuses,
            "app_version": APP_VERSION,
        },
    )
    write_json(
        paths["checkpoint"],
        {
            "job_id": job_id,
            "next_index": 0,
            "processed_row_ids": sorted(preprocessed_row_ids),
            "processed_count": len(preprocessed_row_ids),
            "is_done": False,
            "last_updated": now_utc_iso(),
        },
    )
    paths["progress"].touch(exist_ok=True)
    return job_id


def load_job_input(job_id: str) -> pd.DataFrame:
    return pd.read_csv(job_paths(job_id)["input"])


def load_checkpoint(job_id: str) -> dict:
    cp = safe_read_json(job_paths(job_id)["checkpoint"], default={}) or {}
    cp.setdefault("next_index", 0)
    cp.setdefault("processed_row_ids", [])
    cp.setdefault("processed_count", 0)
    cp.setdefault("is_done", False)
    return cp


def save_checkpoint(job_id: str, checkpoint: dict) -> None:
    checkpoint["last_updated"] = now_utc_iso()
    write_json(job_paths(job_id)["checkpoint"], checkpoint)


def append_progress_row(job_id: str, row: dict) -> None:
    p = job_paths(job_id)["progress"]
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress_df(job_id: str) -> pd.DataFrame:
    p = job_paths(job_id)["progress"]
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def export_job_outputs(job_id: str) -> dict[str, Path]:
    paths = job_paths(job_id)
    df_input = load_job_input(job_id)
    df_progress = load_progress_df(job_id)
    if df_progress.empty:
        empty = pd.DataFrame(columns=["row_id", "question", "answer", "answer_status"])
        empty.to_csv(paths["answers_csv"], index=False)
        with pd.ExcelWriter(paths["answers_xlsx"], engine="openpyxl") as writer:
            empty.to_excel(writer, index=False, sheet_name="answers")
        with pd.ExcelWriter(paths["merged_xlsx"], engine="openpyxl") as writer:
            df_input.to_excel(writer, index=False, sheet_name="merged")
        return paths

    df_answers = df_progress.sort_values("source_row_index").drop_duplicates("row_id", keep="last")
    df_answers.to_csv(paths["answers_csv"], index=False)
    with pd.ExcelWriter(paths["answers_xlsx"], engine="openpyxl") as writer:
        df_answers.to_excel(writer, index=False, sheet_name="answers")
    merged = df_input.merge(df_answers, on=["row_id", "source_row_index", "question"], how="left")
    with pd.ExcelWriter(paths["merged_xlsx"], engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="merged")
    with pd.ExcelWriter(paths["partial_xlsx"], engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="partial")
    return paths


def process_batch_step(
    job_id: str,
    query_engine,
    model_tag: str,
    top_k: int,
    step_size: int,
    sleep_ms: int,
    autosave_every: int,
) -> dict:
    df_input = load_job_input(job_id)
    checkpoint = load_checkpoint(job_id)
    processed_set = set(checkpoint.get("processed_row_ids", []))
    processed_now = 0
    t_start = time.perf_counter()
    total_rows = len(df_input)
    idx = int(checkpoint.get("next_index", 0))

    while idx < total_rows and processed_now < step_size:
        row = df_input.iloc[idx]
        row_id = str(row["row_id"])
        question = str(row["question"])

        if row_id in processed_set:
            idx += 1
            checkpoint["next_index"] = idx
            continue

        started = time.perf_counter()
        status = "ok"
        evidence = []
        answer = ""
        try:
            response = query_engine.query(question)
            answer = str(response).strip()
            evidence = format_evidence(response.source_nodes)
        except Exception as exc:
            status = "error"
            answer = f"ERROR: {exc}"

        audit = evidence_to_audit_fields(evidence)
        append_progress_row(
            job_id,
            {
                "job_id": job_id,
                "row_id": row_id,
                "source_row_index": int(row["source_row_index"]),
                "question": question,
                "answer": answer,
                "answer_status": status,
                "response_seconds": round(time.perf_counter() - started, 3),
                "model_used": model_tag,
                "top_k_used": top_k,
                "processed_at_utc": now_utc_iso(),
                **audit,
            },
        )

        processed_set.add(row_id)
        processed_now += 1
        idx += 1
        checkpoint["next_index"] = idx
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    checkpoint["processed_row_ids"] = sorted(processed_set)
    checkpoint["processed_count"] = len(processed_set)
    checkpoint["is_done"] = checkpoint["next_index"] >= total_rows
    save_checkpoint(job_id, checkpoint)

    if autosave_every > 0 and (checkpoint["processed_count"] % autosave_every == 0 or checkpoint["is_done"]):
        export_job_outputs(job_id)

    elapsed = time.perf_counter() - t_start
    avg = (elapsed / max(processed_now, 1)) if processed_now else 0
    remaining = max(total_rows - checkpoint["next_index"], 0)
    return {
        "processed_now": processed_now,
        "processed_total": checkpoint["processed_count"],
        "next_index": checkpoint["next_index"],
        "total_rows": total_rows,
        "is_done": checkpoint["is_done"],
        "eta_seconds": int(remaining * avg),
    }


def init_session_defaults() -> None:
    defaults = {
        "messages": [],
        "index": None,
        "index_compat_ok": True,
        "index_compat_msg": "",
        "selected_profile": "Balanced (recommended)",
        "current_job_id": "",
        "batch_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def maybe_load_index() -> None:
    if st.session_state.index is None and _storage_exists():
        with st.spinner("Loading persisted index..."):
            st.session_state.index = load_index()


def query_engine_signature(index, selected_model, top_k, response_mode, num_predict, request_timeout, keep_alive, temperature):
    return (id(index), selected_model, top_k, response_mode, num_predict, request_timeout, keep_alive, temperature)


def render_model_section():
    st.subheader("Model and Performance")
    profile_key = st.selectbox("Model profile", options=list(MODEL_PROFILES.keys()), key="selected_profile")
    profile = MODEL_PROFILES[profile_key]
    selected_model = st.text_input("Model tag", value=profile["model"]).strip() or DEFAULT_LLM_MODEL
    overnight = st.checkbox("Overnight thermal-safe mode", value=True)
    speed = st.selectbox("Performance mode", ["Speed", "Balanced"], index=0 if overnight else 1)
    if speed == "Speed" or overnight:
        default_top_k = min(profile["top_k"], 3)
        default_predict = min(profile["num_predict"], 220)
    else:
        default_top_k = profile["top_k"]
        default_predict = profile["num_predict"]

    top_k = st.slider("Retrieval top_k", 2, MAX_TOP_K, default_top_k)
    num_predict = st.slider("Max output tokens", 96, 768, default_predict, step=32)
    request_timeout = st.slider("Request timeout", 120, 1200, DEFAULT_TIMEOUT, step=30)
    keep_alive = st.selectbox("Keep model loaded", ["5m", "15m", "20m", "30m", "1h"], index=2)
    temperature = st.slider("Temperature", 0.0, 0.4, DEFAULT_TEMPERATURE, step=0.05)

    st.subheader("Batch Runtime")
    step_size = st.slider("Rows per processing step", 1, 5, 1)
    sleep_ms = st.slider("Sleep between questions (ms)", 0, 3000, 500, step=100)
    autosave_every = st.slider("Autosave every N answered rows", 1, 50, 10)

    return {
        "selected_model": selected_model,
        "top_k": top_k,
        "num_predict": num_predict,
        "request_timeout": request_timeout,
        "keep_alive": keep_alive,
        "temperature": temperature,
        "response_mode": DEFAULT_RESPONSE_MODE,
        "step_size": step_size,
        "sleep_ms": sleep_ms,
        "autosave_every": autosave_every,
    }


def render_ollama_section(selected_model: str):
    st.divider()
    ok, msg, installed = check_ollama(selected_model)
    if ok:
        st.success(msg)
    else:
        st.error(msg)
        st.code("ollama serve")
    if installed:
        with st.expander(f"Installed models ({len(installed)})"):
            for m in installed:
                st.text(m)
    if ok and selected_model not in installed:
        if st.button("Download selected model", use_container_width=True):
            pb = st.progress(0)
            status = st.empty()
            dl_ok, dl_msg = download_model(selected_model, pb, status)
            if dl_ok:
                st.success(dl_msg)
            else:
                st.error(dl_msg)
    return ok, installed


def render_index_tools_section():
    st.divider()
    st.subheader("Index Tools")
    zip_bytes = create_index_zip_bytes()
    st.download_button(
        "Export index (.zip)",
        data=zip_bytes or b"",
        file_name="nis2_index_export.zip",
        mime="application/zip",
        disabled=zip_bytes is None,
        use_container_width=True,
    )
    import_zip = st.file_uploader("Import index zip", type=["zip"], key="index_import_zip")
    if st.button("Import index package", disabled=import_zip is None, use_container_width=True):
        ok, msg = import_index_from_zip(import_zip, expected_embed_model=DEFAULT_EMBED_MODEL_NAME)
        if ok:
            st.success(msg)
            st.session_state.index = load_index()
        else:
            st.error(msg)


def render_batch_tab(ollama_ok: bool, installed_models: list[str], selected_model: str, qe_signature: tuple, top_k: int, step_size: int, sleep_ms: int, autosave_every: int):
    st.subheader("Batch Question Processing")
    st.markdown("Overnight-safe processing with pause/resume, checkpointing, partial exports, and continuation.")
    if not ollama_ok:
        st.error("Ollama is not reachable. Start it first.")
        return
    if st.session_state.index is None:
        st.warning("No index loaded. Upload/index docs first.")
        return
    if not st.session_state.index_compat_ok:
        st.error("Index compatibility check failed. Re-index required.")
        return
    if selected_model not in installed_models:
        st.error("Selected model is not installed.")
        return

    jobs = list_existing_jobs()
    if jobs:
        labels = [f"{j['job_id']} | {j['created_at']} | next={j['next_index']}" for j in jobs]
        selected = st.selectbox("Load existing job", [""] + labels)
        if st.button("Load Job", disabled=selected == "", use_container_width=True):
            st.session_state.current_job_id = selected.split(" | ")[0]
            st.session_state.batch_running = False

    st.markdown("---")
    st.markdown("### Create New Job")
    batch_file = st.file_uploader("Upload question file", type=["xlsx", "csv"], key="batch_new_file")
    df_input = None
    question_col = None
    if batch_file is not None:
        df_input = load_dataframe_from_upload(batch_file)
        st.dataframe(df_input, use_container_width=True)
        question_col = st.selectbox("Question column", list(df_input.columns), index=0)
    resume_mode = st.selectbox("Resume mode", ["checkpoint", "append", "both"], index=2)
    prior_answers_file = st.file_uploader("Optional prior answers (for append/re-audit)", type=["xlsx", "csv"], key="prior_answers_upload")
    prior_df = load_dataframe_from_upload(prior_answers_file) if prior_answers_file is not None else None
    reaudit_mode = st.checkbox("Re-audit mode", value=False)
    rerun_statuses = st.multiselect("Rerun statuses", ["error", "", "skipped", "ok"], default=["error", ""])
    if st.button("Create Job", disabled=df_input is None or question_col is None, use_container_width=True):
        st.session_state.current_job_id = create_job(
            df_input=df_input,
            question_col=question_col,
            model_tag=selected_model,
            top_k=top_k,
            resume_mode=resume_mode,
            prior_answers_df=prior_df,
            reaudit_mode=reaudit_mode,
            rerun_statuses=rerun_statuses,
        )
        st.session_state.batch_running = False
        st.success(f"Created job `{st.session_state.current_job_id}`")

    st.markdown("---")
    st.markdown("### Current Job")
    job_id = st.session_state.current_job_id
    if not job_id:
        st.info("No job selected.")
        return
    paths = job_paths(job_id)
    if not paths["meta"].exists():
        st.error("Selected job metadata missing.")
        return

    meta = safe_read_json(paths["meta"], default={}) or {}
    cp = load_checkpoint(job_id)
    total_rows = int(meta.get("total_rows", 0))
    done_rows = int(cp.get("processed_count", 0))
    st.caption(f"Job `{job_id}` | model `{meta.get('model', 'unknown')}` | {done_rows}/{total_rows}")
    st.progress((done_rows / total_rows) if total_rows else 0.0)

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Start / Resume", use_container_width=True):
        st.session_state.batch_running = True
    if c2.button("Pause", use_container_width=True):
        st.session_state.batch_running = False
        save_checkpoint(job_id, cp)
    if c3.button("Stop and Export Now", use_container_width=True):
        st.session_state.batch_running = False
        export_job_outputs(job_id)
        st.success("Exported current state.")
    if c4.button("Export Outputs", use_container_width=True):
        export_job_outputs(job_id)
        st.success("Outputs exported.")

    out = export_job_outputs(job_id)
    d1, d2, d3 = st.columns(3)
    with d1:
        if out["answers_csv"].exists():
            st.download_button("answers_only.csv", out["answers_csv"].read_bytes(), f"{job_id}_answers_only.csv", "text/csv", use_container_width=True)
    with d2:
        if out["answers_xlsx"].exists():
            st.download_button("answers_only.xlsx", out["answers_xlsx"].read_bytes(), f"{job_id}_answers_only.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with d3:
        if out["merged_xlsx"].exists():
            st.download_button("merge_with_original.xlsx", out["merged_xlsx"].read_bytes(), f"{job_id}_merge_with_original.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    if st.session_state.batch_running:
        qe = get_cached_query_engine(st.session_state.index, qe_signature)
        result = process_batch_step(job_id, qe, selected_model, top_k, step_size, sleep_ms, autosave_every)
        st.info(f"Processed now: {result['processed_now']} | total: {result['processed_total']}/{result['total_rows']} | ETA: {result['eta_seconds']}s")
        if result["is_done"]:
            st.session_state.batch_running = False
            export_job_outputs(job_id)
            st.success("Batch completed and exported.")
        else:
            time.sleep(0.1)
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="NIS2 RAG Auditor", page_icon="🛡️", layout="wide")
    init_session_defaults()

    with st.sidebar:
        st.header("NIS2 RAG Auditor")
        st.caption("Local audit-compliance assistant")
        runtime = render_model_section()
        selected_model = runtime["selected_model"]
        ollama_ok, installed_models = render_ollama_section(selected_model)

        st.divider()
        st.subheader("Upload Documents")
        enable_image_ocr = st.checkbox("Enable image OCR indexing", value=False)
        ocr_lang = st.text_input("OCR language", value="eng")
        uploaded_files = st.file_uploader(
            "Drag and drop PDF, DOCX, TXT, PNG, JPG, TIFF files",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "tiff", "tif"],
            accept_multiple_files=True,
            key="doc_uploader",
        )
        a, b = st.columns(2)
        with a:
            index_btn = st.button("Save and Index", use_container_width=True, disabled=not uploaded_files)
        with b:
            reindex_btn = st.button("Re-index All", use_container_width=True)

    init_settings(
        embed_model_name=DEFAULT_EMBED_MODEL_NAME,
        llm_model_name=selected_model,
        request_timeout=runtime["request_timeout"],
        num_predict=runtime["num_predict"],
        keep_alive=runtime["keep_alive"],
        temperature=runtime["temperature"],
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    maybe_load_index()
    if st.session_state.index is not None:
        compat_ok, compat_msg = check_index_compatibility(DEFAULT_EMBED_MODEL_NAME)
        st.session_state.index_compat_ok = compat_ok
        st.session_state.index_compat_msg = compat_msg

    if index_btn and uploaded_files:
        saved = save_uploaded_files(uploaded_files)
        st.success(f"Saved {len(saved)} file(s) to data/")
        with st.spinner("Indexing documents..."):
            docs, ocr_errors = load_documents(enable_image_ocr, ocr_lang)
            if docs:
                st.session_state.index = build_index(
                    documents=docs,
                    embed_model_name=DEFAULT_EMBED_MODEL_NAME,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    image_ocr_enabled=enable_image_ocr,
                    ocr_lang=ocr_lang,
                )
                st.success(f"Index built - {len(docs)} document(s)")
            else:
                st.warning("No supported documents found.")
            for err in ocr_errors[:10]:
                st.caption(f"OCR: {err}")

    if reindex_btn:
        with st.spinner("Re-indexing all documents..."):
            docs, ocr_errors = load_documents(enable_image_ocr, ocr_lang)
            if docs:
                st.session_state.index = build_index(
                    documents=docs,
                    embed_model_name=DEFAULT_EMBED_MODEL_NAME,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    image_ocr_enabled=enable_image_ocr,
                    ocr_lang=ocr_lang,
                )
                st.success(f"Index rebuilt - {len(docs)} document(s)")
            else:
                st.warning("No documents found in data/.")
            for err in ocr_errors[:10]:
                st.caption(f"OCR: {err}")

    with st.sidebar:
        render_index_tools_section()
        st.divider()
        st.subheader("Index Status")
        if st.session_state.index is None:
            st.warning("No index loaded.")
        elif st.session_state.index_compat_ok:
            st.info("Index loaded and compatible.")
        else:
            st.error("Index loaded but incompatible.")
        if st.session_state.index_compat_msg:
            st.caption(st.session_state.index_compat_msg)
        if DATA_DIR.exists():
            files_on_disk = sorted([p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED_DOC_EXTS.union(SUPPORTED_IMAGE_EXTS)])
            if files_on_disk:
                with st.expander(f"Files on disk ({len(files_on_disk)})"):
                    for fn in files_on_disk:
                        st.text(fn)
        st.divider()
        st.caption(f"LLM: `{selected_model}`")
        st.caption(f"Embeddings: `{DEFAULT_EMBED_MODEL_NAME}`")
        st.caption(f"response_mode: `{runtime['response_mode']}` | top_k: {runtime['top_k']} | max_tokens: {runtime['num_predict']}")

    tab_chat, tab_batch = st.tabs(["Chat", "Batch Processing"])
    qe_sig = query_engine_signature(
        st.session_state.index,
        selected_model,
        runtime["top_k"],
        runtime["response_mode"],
        runtime["num_predict"],
        runtime["request_timeout"],
        runtime["keep_alive"],
        runtime["temperature"],
    )

    with tab_chat:
        if not ollama_ok:
            st.error("Ollama is not reachable.")
        else:
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
                    msg = "Index compatibility check failed. Re-index required."
                    st.session_state.messages.append({"role": "assistant", "content": msg, "evidence": []})
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                elif selected_model not in installed_models:
                    msg = f"Selected model `{selected_model}` is not installed."
                    st.session_state.messages.append({"role": "assistant", "content": msg, "evidence": []})
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                else:
                    qe = get_cached_query_engine(st.session_state.index, qe_sig)
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
                        st.session_state.messages.append({"role": "assistant", "content": answer, "evidence": evidence})

    with tab_batch:
        render_batch_tab(
            ollama_ok=ollama_ok,
            installed_models=installed_models,
            selected_model=selected_model,
            qe_signature=qe_sig,
            top_k=runtime["top_k"],
            step_size=runtime["step_size"],
            sleep_ms=runtime["sleep_ms"],
            autosave_every=runtime["autosave_every"],
        )


if __name__ == "__main__":
    main()
