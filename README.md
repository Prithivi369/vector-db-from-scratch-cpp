# VectorDB — Vector Database from Scratch in C++

A vector database built from scratch in C++ with a web UI. Implements **HNSW**, **KD-Tree**, and **Brute Force** search side-by-side, with a **RAG pipeline** powered by a local LLM via Ollama.

---

## Features

- **3 Search Algorithms** — HNSW (production-grade), KD-Tree, Brute Force
- **3 Distance Metrics** — Cosine, Euclidean, Manhattan
- **16D Demo Vectors** — 20 pre-loaded vectors across 4 semantic categories
- **2D PCA Scatter Plot** — Live visualization of semantic clusters
- **Real Embeddings** — Paste any text → Ollama embeds it at 768D
- **RAG Pipeline** — Ask questions → HNSW retrieves context → local LLM answers
- **REST API** — Full CRUD: insert, delete, search, benchmark, hnsw-info

---

## Prerequisites

- [MSYS2](https://www.msys2.org) — provides `g++` on Windows
- [Ollama](https://ollama.com) — runs the local AI models

**Pull the required models:**
```bash
ollama pull nomic-embed-text   # ~274 MB, embedding model
ollama pull llama3.2           # ~2 GB, language model
```

---

## Setup & Run

**1. Compile**
```powershell
g++ -std=c++17 -O2 main.cpp -o db -lws2_32
```

**2. Start Ollama** (if not already running in the system tray)
```powershell
ollama serve
```

**3. Start the server**
```powershell
./db
```

Open **http://localhost:8080** in your browser.

---

## Project Structure

```
VectorDB/
├── main.cpp     ← C++ backend (HNSW, KD-Tree, BruteForce, HTTP server, RAG)
├── httplib.h    ← Single-header HTTP library (cpp-httplib)
├── index.html   ← Frontend (scatter plot, chat UI, benchmark tool)
└── README.md
```

**Core classes in `main.cpp`:**

| Class | Role |
|---|---|
| `BruteForce` | O(N·d) exact search, baseline |
| `KDTree` | O(log N) exact, axis-aligned partitioning |
| `HNSW` | O(log N) approximate, multilayer graph |
| `VectorDB` | Unified interface over all 3 (16D demo vectors) |
| `DocumentDB` | HNSW-only index for Ollama embeddings (768D) |
| `OllamaClient` | HTTP client → `/api/embeddings` + `/api/generate` |

---

## REST API

**Demo vectors**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | Graph structure & layer stats |
| `GET` | `/items` | List all vectors |
| `GET` | `/stats` | DB statistics |

**Documents & RAG**

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed & store document |
| `GET` | `/doc/list` | — | List stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status |

---

## How RAG Works

```
Question → nomic-embed-text (768D) → HNSW search → top-k chunks → llama3.2 → Answer
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` |
| `g++: command not found` | Add `C:\msys64\ucrt64\bin` to Windows PATH |
| Port 8080 in use | `netstat -ano \| findstr 8080` → `taskkill /PID <pid> /F` |
| LLM too slow | Switch to `llama3.2:1b` — edit `genModel` in `main.cpp` and recompile |

---

## License

MIT
