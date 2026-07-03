# 🎓 ChatIITK — Local RAG Chatbot

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a custom knowledge base — no external API calls, no cloud LLM costs, runs entirely on CPU.

Built as a full-stack application with a FastAPI backend and a lightweight vanilla JS + Tailwind frontend, on top of a FAISS vector store and a HuggingFace Transformers language model.

---

## ✨ Features

- **100% local inference** — no OpenAI/Anthropic API keys required, all embeddings and generation run on-device
- **Pure-Python dependency stack** — no C++ build tools, no CUDA toolchain, installs cleanly via `pip`
- **Multi-format ingestion** — `.txt`, `.pdf`, `.csv`, `.xlsx`, `.docx`, `.html`, `.md` all supported
- **FAISS vector search** for fast, accurate document retrieval
- **FastAPI REST backend** with CORS-enabled `/api/chat` endpoint
- **Responsive chat UI** — single-file HTML/Tailwind/vanilla JS frontend, no build step
- **Built-in evaluation harness** — measures retrieval latency, generation latency, and retrieval hit-rate

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| LLM Inference | HuggingFace `transformers` (CPU) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (`faiss-cpu`) |
| Orchestration | LangChain |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML + Tailwind CSS (CDN) + Vanilla JS |
| Document Loaders | `pdfminer.six`, `docx2txt`, `unstructured` |

---

## 📁 Repository Structure

```
ChatIITK-RAG/
├── SOURCE_DOCUMENTS/          # Your knowledge base files (.txt, .pdf, .csv, .xlsx, .docx)
├── constants.py                # Paths, model config, embedding function, document loader map
├── ingest.py                   # Loads, chunks, embeds, and indexes documents into FAISS
├── prompt_template_utils.py    # Context-grounded prompt template
├── run_ChatIITK.py             # CLI script to test a single hardcoded query
├── main.py                     # FastAPI backend (POST /api/chat)
├── ChatIITK_UI.py              # Streamlit UI (alternative to index.html)
├── index.html                  # Standalone chat frontend (Tailwind + vanilla JS)
├── evaluate.py                 # Benchmarks retrieval/generation latency + hit-rate
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ChatIITK-RAG.git
cd ChatIITK-RAG
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your knowledge base
Place your `.txt`, `.pdf`, `.csv`, `.xlsx`, or `.docx` files inside `SOURCE_DOCUMENTS/`.

### 5. Build the vector index
```bash
python ingest.py --device_type cpu
```
This creates a `DB/` folder containing the FAISS index (excluded from git, regenerate anytime).

### 6. Run a quick test query
```bash
python run_ChatIITK.py
```

### 7. Launch the backend API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
API docs available at `http://localhost:8000/docs`.

### 8. Launch the frontend
Open `index.html` directly in your browser, or serve it:
```bash
python -m http.server 5500
```
Then visit `http://localhost:5500`.

*(Alternative: run the Streamlit UI instead with `streamlit run ChatIITK_UI.py`)*

---

## 📊 Evaluation

Run the built-in benchmark suite:
```bash
python evaluate.py
```

Outputs:
- **Average Retrieval Latency (ms)**
- **Average Generation Latency (ms)**
- **Retrieval Hit-Rate (%)** — whether expected keywords appear in retrieved context

### Sample Results (10-query eval set)
| Metric | Value |
|---|---|
| Avg Retrieval Latency | 48.12 ms |
| Avg Generation Latency | 40,145 ms |
| Retrieval Hit-Rate | 100% |

---

## ⚙️ Configuration

All key settings live in `constants.py`:
- `MODEL_ID` — swap the LLM (defaults to a small CPU-friendly instruct model)
- `EMBEDDING_MODEL_NAME` — swap the embedding model
- `MAX_NEW_TOKENS` — cap generation length
- `DOCUMENT_MAP` — add/remove supported file types

---
