# VectorDB — Vector Database from Scratch in C++

A fully functional vector database built from scratch in C++, with a web UI and a complete RAG pipeline. Implements three search algorithms side by side — HNSW, KD-Tree, and Brute Force — across three distance metrics, with live benchmarking to compare them directly.

Built to understand how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood — not by using an existing library, but by implementing every data structure, distance function, graph traversal, and HTTP endpoint from the ground up.

---

## What It Does

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)     ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (C++)              ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search               ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)             ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

There are two separate indices running in parallel. A **16-dimensional demo index** (`VectorDB`) holds 20 hand-crafted vectors across 4 semantic categories (CS, Math, Food, Sports) and runs all three algorithms so you can compare them. A **768-dimensional document index** (`DocumentDB`) uses real Ollama embeddings from `nomic-embed-text` for actual semantic search and RAG.

---

## The Three Algorithms

### Brute Force — O(N·d)

The baseline. Computes the distance from the query to every single stored vector and sorts. Exact by definition. Gets slow as N grows but is the ground truth everything else is measured against.

### KD-Tree — O(log N) average, degrades at high dimensions

Binary space partitioning: each node splits the space along one axis (cycling through all dimensions). Search uses a max-heap of size k and prunes entire subtrees when the closest possible point in that subtree — bounded by the axis-aligned split distance `|q[ax] - node.emb[ax]|` — can't beat the current k-th best. This pruning is what makes it fast.

The problem is the curse of dimensionality. In high-dimensional spaces, almost all points end up near the boundary of any hyperplane split — the pruning condition almost never fires, and the tree degrades toward O(N). Works well for the 16D demo; essentially becomes brute force at 768D, which is exactly what the benchmark shows.

### HNSW — O(log N), works at any dimension

The algorithm used by Pinecone, Weaviate, Milvus, and Chroma. Builds a multilayer graph where each node is randomly assigned a maximum layer level using the probability `floor(-log(uniform(0,1)) * mL)` where `mL = 1/log(M)`. Layer 0 has all nodes with up to `M0 = 2*M` connections; higher layers have exponentially fewer nodes and longer-range connections.

**Insert:** Start from the top layer, greedily descend to find the closest entry point at each layer above the new node's assigned level (ef=1, just greedy). From the new node's level down to 0, run a beam search with `ef_construction=200` to find candidates, select the best M neighbors, connect bidirectionally. If any neighbor's connection list exceeds M after adding the new node, prune it back by re-sorting and keeping the M closest.

**Search:** Same greedy descent from the top layer. At layer 0, expand to `ef` nearest candidates using two priority queues — one min-heap for candidates to explore (ordered by distance ascending), one max-heap for the best found so far (ordered descending so you can check and evict the worst). Stop expanding a candidate if its distance exceeds the worst in the found set and the found set is already full.

**Why it doesn't degrade at high dimensions:** KD-Tree pruning relies on axis-aligned distance bounds that become meaningless in high dimensions. HNSW's graph connectivity doesn't depend on dimensional structure — it builds shortcuts through the data regardless of how many dimensions there are.

---

## Key Engineering Decisions

**Hand-rolled JSON parser.** Rather than pulling in a library, the server parses request bodies with `extractStr()` and `extractArr()` using `std::string::find` and explicit offset tracking, and serializes responses by building strings directly. This means handling escape sequences manually (`\"`, `\\`, `\n`, `\r`, `\t`) in both directions. It's more fragile than a real parser but zero dependencies.

**DocumentDB switches between brute force and HNSW at a threshold of 10 documents.** HNSW's graph traversal has overhead that doesn't pay off for tiny sets — with fewer than 10 chunks, brute force is both faster and exact. The switch is a single line: `(store.size() < 10) ? bf.knn(...) : hnsw.knn(...)`.

**Overlapping text chunks with a 30-word overlap on 250-word windows.** When a document is split for RAG, consecutive chunks share 30 words at their boundary. This prevents context from being cut off mid-sentence at a chunk boundary — a question about something that straddles a split would still retrieve a chunk containing the full context.

**KD-Tree is rebuilt on every delete.** Deletion from a KD-Tree while maintaining balance is genuinely complex. The practical solution: collect all remaining items, destroy the tree, and reinsert. Expensive for large datasets but correct, and deletion isn't a hot path for the demo use case.

**VectorDB holds three separate indices simultaneously.** Every insert goes into `BruteForce`, `KDTree`, and `HNSW` at the same time, under a single mutex. This is what makes the benchmark endpoint possible — one query, three timings, directly comparable because they're searching identical data.

---

## Architecture

```
VectorDB (main.cpp — 1089 lines)
│
├── Distance Metrics
│   ├── euclidean()       — sqrt of sum of squared differences
│   ├── cosine()          — 1 - dot(a,b)/(|a||b|), clamped at 1e-9
│   └── manhattan()       — sum of absolute differences
│
├── Search Algorithms
│   ├── BruteForce        — linear scan, exact
│   ├── KDTree            — recursive binary space partition, axis-cycling split
│   └── HNSW              — multilayer graph, beam search, probabilistic level assignment
│
├── VectorDB              — unified interface over all 3, mutex-protected, 16D
├── DocumentDB            — HNSW + BruteForce fallback, 768D, mutex-protected
│
├── OllamaClient          — HTTP client to Ollama's /api/embeddings + /api/generate
│   └── hand-rolled JSON  — escaping, embedding array parsing (depth-tracked bracket matching)
│
├── chunkText()           — sliding window word tokenizer, 250 words, 30-word overlap
│
└── HTTP Server (cpp-httplib)
    ├── /search           — k-NN search, selectable algo + metric
    ├── /benchmark        — all 3 algos timed on same query
    ├── /hnsw-info        — full graph structure: layers, node count, edge list
    ├── /insert /delete   — demo vector CRUD
    ├── /doc/insert       — chunk + embed + store
    ├── /doc/ask          — full RAG: embed question → HNSW retrieve → LLM generate
    └── /status           — Ollama availability + model info
```

---

## File Structure

```
VectorDB/
├── main.cpp      ← everything: algorithms, HTTP server, RAG pipeline (1089 lines)
├── httplib.h     ← single-header HTTP library (cpp-httplib)
├── index.html    ← frontend: PCA scatter plot, benchmark UI, RAG chat
└── README.md
```

---

## Prerequisites

- [MSYS2](https://www.msys2.org) for `g++` on Windows (or any C++17 compiler on macOS/Linux)
- [Ollama](https://ollama.com) for embeddings and generation

```powershell
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB  — generation model
```

---

## Build & Run

```powershell
# Compile
g++ -std=c++17 -O2 main.cpp -o db -lws2_32

# Start Ollama (if not already in system tray)
ollama serve

# Start the server
./db
```

Open **http://localhost:8080**

---

## REST API

**Demo vectors (16D)**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | All 3 algorithms timed on same query |
| `GET` | `/hnsw-info` | Full graph: layer counts, node list, edge list |
| `POST` | `/insert` | Insert a vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all vectors |
| `GET` | `/stats` | Counts, dims, available algorithms and metrics |

**Documents & RAG (768D)**

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Chunk, embed, and store |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | Full RAG pipeline |
| `GET` | `/doc/list` | — | All stored chunks with previews |
| `DELETE` | `/doc/delete/:id` | — | Delete a chunk |
| `GET` | `/status` | — | Ollama status, model names, doc count |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` |
| `g++: command not found` | Add `C:\msys64\ucrt64\bin` to Windows PATH |
| Port 8080 in use | `netstat -ano \| findstr 8080` → `taskkill /PID <pid> /F` |
| LLM too slow | `ollama pull llama3.2:1b` then change `genModel` in `main.cpp` and recompile |

---

## What's Next

**Persistent storage** — the index lives entirely in memory and is lost on restart. Adding serialization for the HNSW graph (node list, adjacency layers, entry point) would make it production-usable.

**Product Quantization** — compressing 768-dimensional float vectors to reduce memory and speed up distance calculations, the way Faiss does it. Interesting tradeoff between compression ratio and recall degradation.

**Filtered search** — restrict results to a specific category without scanning everything, which requires either a separate index per category or metadata-aware graph construction.

**True streaming RAG** — the current `/doc/ask` waits for the full LLM response before sending. Streaming the tokens as they're generated (Ollama supports it with `"stream":true`) would make the UI feel significantly more responsive.

---

## License

MIT
