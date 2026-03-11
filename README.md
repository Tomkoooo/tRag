# NIS2 RAG Auditor

A fully local Retrieval-Augmented Generation system for NIS2 audit compliance.
Runs entirely on a MacBook Air M4 (16 GB unified memory) with no cloud dependencies.

| Component   | Technology                     |
| ----------- | ------------------------------ |
| LLM         | Ollama (model-selectable)      |
| Embeddings  | HuggingFace — `BAAI/bge-m3`    |
| RAG         | LlamaIndex                     |
| UI          | Streamlit                      |
| Batch       | Checkpointed resumable jobs    |

---

## Setup Guide

The steps below assume a clean macOS machine with no Homebrew and no Python installed.

### 1. Install Xcode Command Line Tools (required)

```bash
xcode-select --install
```

Verify:

```bash
xcode-select -p
```

### 2. Install Homebrew

Install script:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add Homebrew to shell config (Apple Silicon path):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify:

```bash
brew --version
which brew
```

Expected `which brew`: `/opt/homebrew/bin/brew`

### 3. Install Python 3.11 and pip

```bash
brew install python@3.11
```

Verify:

```bash
python3 --version
pip3 --version
```

Expected: Python 3.11.x (or newer, 3.10+ is supported by this project).

### 4. Install Ollama

```bash
brew install ollama
```

Start Ollama server in a dedicated terminal:

```bash
ollama serve
```

In another terminal, pull the model (~4.7 GB):

```bash
ollama pull llama3.1:8b
```

Verify:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### 5. Create and activate Python virtual environment

```bash
cd /path/to/rag_system
python3 -m venv .venv
source .venv/bin/activate
python --version
```

### 6. Install project dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For image OCR support:

```bash
brew install tesseract
```

Optional Hungarian OCR language data:

```bash
brew install tesseract-lang
```

### 7. First run (downloads embeddings cache)

```bash
streamlit run app.py
```

The first startup downloads `BAAI/bge-m3` (~2.3 GB) to `~/.cache/huggingface/`.

### M4-specific notes

- **Metal acceleration**: Ollama uses Metal on Apple Silicon by default (no extra config needed).
- **Memory headroom**: `llama3.1:8b` + `BAAI/bge-m3` typically stays in a safe range for 16 GB when using current defaults (`CHUNK_SIZE=1024`, `TOP_K=4`).
- **Monitor usage**: Use *Activity Monitor → Memory* and keep pressure green/yellow.

### Install Debug Info (if something fails)

- **Homebrew not found after install**: run `eval "$(/opt/homebrew/bin/brew shellenv)"` and reopen terminal.
- **`python3` command missing**: check `which python3`; if empty, reinstall `python@3.11` and restart shell.
- **`pip` installs fail with SSL/cert errors**: run `python3 -m ensurepip --upgrade` then retry.
- **Ollama not responding**: check process with `ps aux | rg ollama`; restart via `ollama serve`.
- **Port conflict on 11434**: run `lsof -i :11434`; stop conflicting process or restart machine.
- **Model pull interrupted**: rerun `ollama pull llama3.1:8b` (it resumes).
- **HuggingFace download slow/fails**: verify internet and free disk space (`df -h`); retry app start.
- **Disk space check (recommended before first run)**:
  - `llama3.1:8b` model: ~5 GB
  - `BAAI/bge-m3` cache: ~2.3 GB
  - temporary indexing and docs: depends on dataset size

---

## Usage Guide

### Start the app

```bash
source .venv/bin/activate
streamlit run app.py
```

The browser opens at `http://localhost:8501`.

### Upload documents

1. In the **sidebar**, drag & drop PDF, DOCX, or TXT files into the upload area.
2. Click **Save & Index**. A spinner appears while documents are chunked, embedded, and stored.
3. The index is persisted to the `storage/` directory — restarting the app reloads it instantly.
4. A compatibility check runs for stored indexes:
   - If metadata is missing (legacy index), querying is allowed with a warning.
   - If embedding model mismatch is detected, querying is blocked and **Re-index All** is required.

### Ask questions (Chat tab)

Type a question in the chat input. The system retrieves relevant chunks (default top-k = 4), uses `compact` response synthesis for speed, and returns an answer with evidence.

Each evidence block shows:
- Source **file name**
- **Page number** (for PDFs)
- **Relevance score**
- A short **excerpt** from the chunk
- **Response time (seconds)** for quick latency feedback

### Batch processing (Batch tab)

1. Upload an Excel (`.xlsx`) or CSV file with a column of questions.
2. Select which column contains the questions.
3. Click **Process All Questions**.
4. A progress bar tracks completion.
5. Download results as CSV or Excel. Output columns include `row_id`, `question`, `answer`, `answer_status`, `response_seconds`, `source_backlinks`, and proof excerpts.

### Overnight batch workflow (8-12h safe run)

1. In sidebar, enable **Overnight thermal-safe mode**.
2. Use **Fast** model profile if latency/heat is a concern.
3. In **Batch Processing** tab:
   - Upload question Excel/CSV
   - Select question column
   - Choose resume mode (`checkpoint`, `append`, or `both`)
   - (Optional) upload prior answers file for append/re-audit
4. Click **Create Job**, then **Start / Resume**.
5. Use **Pause** any time. State is checkpointed in `batch_runs/`.
6. Use **Stop and Export Now** to safely stop and export partial results.
7. Later, load the same job with **Load Job** and continue from last checkpoint.

### Resume modes

- `checkpoint`: continue from internal checkpoint (`next_index` + processed row IDs)
- `append`: skip rows that already exist in prior answers file
- `both`: combines checkpoint and append skip logic

### Export formats for audit/re-audit

- `answers_only.csv`
- `answers_only.xlsx`
- `merge_with_original.xlsx`

All are generated under `batch_runs/<job_id>/` and can be exported mid-run.

### Index export/import

In sidebar **Index Tools**:

- **Export index (.zip)**: packages current `storage/` for transfer/backup
- **Import index package**: validates package metadata and required files before replacing local storage

Import is blocked when embedding model is incompatible.

### Image OCR indexing (optional)

- Enable **Enable image OCR indexing** before indexing.
- Supported image types: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`
- OCR text is indexed with provenance metadata (`source_type=image_ocr`, file name, image page label).
- This is optional and usually lower power than full audio transcription.

### Model selection and download

The sidebar now includes:

- **Model profile** (Balanced/Fast/Alternative Fast/Heavy)
- **Model tag** input (custom Ollama model tag)
- **Installed models** list (from `http://localhost:11434/api/tags`)
- **Download selected model** button (uses Ollama pull API with progress)

If a selected model is not installed, chat and batch processing are paused until the model is downloaded.

### Performance controls (M4 optimization)

The sidebar exposes runtime controls to avoid slow responses/timeouts:

- `top_k` (default 4, max 6)
- `max output tokens` (`num_predict`)
- `request timeout`
- `keep_alive` (keeps model warm between questions)
- `temperature`
- `Performance mode` (Speed/Balanced)

Recommended order if responses are slow:

1. Switch profile to **Fast** (`llama3.2:3b`)
2. Reduce `top_k` to 3
3. Reduce max output tokens to 192-256
4. Keep model warm with `keep_alive` >= 15m
5. Retry the same query and compare response time

---

## Example NIS2 Prompts

**Hungarian:**

- *"Milyen intézkedéseket tartalmaz a kockázatkezelési policy?"*
- *"Hogyan biztosítja a szervezet az incidenskezelési folyamatok megfelelőségét a NIS2 szerint?"*
- *"Kik a felelős személyek a kiberbiztonsági irányításért?"*
- *"Milyen képzési programokat ír elő a belső szabályzat?"*

**English:**

- *"What risk management measures are described in the policy documents?"*
- *"How does the organisation ensure compliance with NIS2 incident reporting requirements?"*
- *"Which roles are responsible for cybersecurity governance?"*
- *"What supply-chain security measures are documented?"*

---

## Test Scenarios

### Test 1 — Single PDF query

**Setup:** Create a short text file `test_policy.txt` with content:

```
NIS2 Risk Management Policy
Page 1

The organisation shall perform annual risk assessments covering all critical
information systems. Risk treatment plans must be approved by the CISO.

Page 2

Incident response teams must be notified within 24 hours of a detected breach.
All incidents must be reported to the national CSIRT within 72 hours.
```

**Steps:**
1. Upload `test_policy.txt` and click *Save & Index*.
2. Ask: *"What are the incident reporting timelines?"*

**Expected output:** The answer should mention "24 hours" and "72 hours", with evidence pointing to `test_policy.txt`.

**Verify:** Expand the evidence block and confirm the file name and excerpt match the source.

### Test 2 — Multi-document cross-reference

**Setup:** Create two text files:

- `access_control.txt` — describes role-based access policies
- `incident_response.txt` — describes incident handling procedures

**Steps:**
1. Upload both files and index.
2. Ask: *"Who is responsible for access management during a security incident?"*

**Expected output:** The answer should synthesise information from both files. Evidence should list both file names.

### Test 3 — Batch Excel processing

**Setup:** Create `questions.xlsx` with a single column named `question`:

| question |
| --- |
| What risk assessments are required? |
| How are incidents reported? |
| What training is mandatory? |

**Steps:**
1. Upload documents and index them.
2. Go to the *Batch Processing* tab, upload `questions.xlsx`.
3. Select the `question` column and click *Process All Questions*.

**Expected output:** A results table with 3 rows, each containing an answer and evidence columns. Download as CSV and verify all rows are populated.

### Test 4 — Persistence across restarts

**Steps:**
1. Upload and index documents.
2. Stop the Streamlit app (`Ctrl+C`).
3. Restart with `streamlit run app.py`.

**Expected output:** The sidebar should show index loaded status without re-indexing. Queries should work immediately.

**Verify:** Check that `storage/` contains `docstore.json`, `index_store.json`, `default__vector_store.json`, and `index_meta.json`.

### Test 5 — Memory stress test

**Steps:**
1. Upload 10+ PDF documents (ideally 50–100 pages total).
2. Index all documents.
3. Open *Activity Monitor → Memory* and note the memory pressure indicator.
4. Run a batch of 10 questions.

**Expected output:** Memory pressure stays in the green/yellow zone. Total memory used by Python + Ollama stays below ~13 GB.

**Troubleshooting if memory is too high:**
- Use Fast profile (`llama3.2:3b`).
- Reduce `top_k` from 4 to 3.
- Reduce max output tokens to 160-220.
- Keep chunk size at 1024 unless you are reindexing intentionally.
- Close other applications consuming memory.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Cannot reach Ollama" in sidebar | Run `ollama serve` in a terminal |
| Model not found | Run `ollama pull llama3.1:8b` |
| Selected model missing | Use **Download selected model** in sidebar or `ollama pull <model>` |
| Slow first query | Normal on cold start; keep model warm with `keep_alive` |
| Second query times out | Use Speed mode, reduce top_k/output tokens, and increase timeout |
| Out-of-memory (app killed) | Switch to Fast profile and reduce output tokens |
| Index seems stale after adding new files | Click **Re-index All** in the sidebar |
| Index compatibility failed | Re-index All (embedding mismatch detected) |
| Excel download is empty | Ensure you selected the correct question column |
| Overnight batch interrupted | Load same job from Batch tab and click **Start / Resume** |
| OCR returns empty text | Ensure image quality is high and Tesseract is installed |
| `pytesseract` error about binary | Install `tesseract` via Homebrew and restart terminal |

---

## Project Structure

```
rag_system/
├── app.py              ← Single-file Streamlit application
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── data/               ← Uploaded documents (created at runtime)
└── storage/            ← Persisted vector index (created at runtime)
└── batch_runs/         ← Checkpointed overnight batch jobs
```

## License

Internal use — NIS2 audit compliance tool.
