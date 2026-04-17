<div align="center">
    <img src="https://raw.githubusercontent.com/milvus-io/milvus-lite/refs/heads/main/milvus_lite_logo.png" width="60%"/>
</div>

<h3 align="center">
    <p>Milvus Lite v2 &mdash; The next-generation lightweight Milvus</p>
</h3>

<p align="center">
    <a href="https://github.com/junjiejiangjjj/milvus-lite-v2/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/built%20with-vibe%20coding-ff69b4" alt="Vibe Coding">
</p>

# Introduction

Milvus Lite v2 is the next-generation lightweight version of [Milvus](https://github.com/milvus-io/milvus), rebuilt from scratch in **pure Python** to replace the original [milvus-lite](https://github.com/milvus-io/milvus-lite).

The original milvus-lite wraps the full C++ Milvus core via CGo bindings, inheriting its heavy build chain, platform restrictions (no Windows, no Alpine), and opaque debugging experience. Milvus Lite v2 takes a different approach: a clean-room Python implementation with an LSM-tree storage engine, delivering the same pymilvus-compatible API in a package that is easy to install, inspect, and extend.

This project is entirely **vibe coded** — designed, implemented, and tested through conversational AI pair programming with [Claude Code](https://claude.ai/code). From architecture decisions to 2100+ test cases, every line of code was produced through human-AI collaboration, demonstrating that complex database systems can be built effectively with the vibe coding workflow.

### Why replace milvus-lite?

| | milvus-lite (v1) | Milvus Lite v2 |
|---|---|---|
| Language | C++ core + CGo + Python wrapper | Pure Python |
| Install | `pip install` downloads ~200MB binary | `pip install` pulls lightweight Python packages |
| Platform | Linux/macOS only, no Windows/Alpine | Windows, macOS, Linux — anywhere Python runs |
| Debugging | Opaque C++ core, segfaults | Pure Python stack traces |
| Extensibility | Requires rebuilding C++ | Standard Python, easy to fork and modify |
| Index | FLAT, IVF_FLAT | HNSW, HNSW_SQ, IVF_FLAT, IVF_SQ8 (FAISS), FLAT, BM25 sparse inverted |
| Full text search | Tantivy (C++ binding) | Pure Python BM25 with pluggable analyzers |

# Requirements

- Python >= 3.10
- Any platform: macOS, Linux, Windows

# Installation

```bash
pip install litevecdb
```

Default install includes FAISS (HNSW/IVF_FLAT indexes), pymilvus, and gRPC — everything needed for `MilvusClient("./demo.db")` to work out of the box.

> **Note:** If you have the original `milvus-lite` installed, uninstall it first to avoid conflicts — both packages provide the `milvus_lite` Python module:
> ```bash
> pip uninstall milvus-lite -y
> pip install litevecdb
> ```

> **Important:** `.db` files created by milvus-lite v1 are **not compatible** with Milvus Lite v2. The v1 storage format uses SQLite + C++ internal structures, while v2 uses a completely different LSM-tree engine (WAL + Parquet). You must re-import your data into a new v2 database — there is no automatic migration.

For development:

```bash
git clone https://github.com/junjiejiangjjj/milvus-lite-v2.git
cd milvus-lite-v2
make dev                    # create .venv + install with dev deps
source .venv/bin/activate
```

| Command | Description |
|---|---|
| `make dev` | Create virtual environment and install with dev dependencies |
| `make install` | Create virtual environment and install (without dev deps) |
| `make test` | Run tests (excludes benchmark) |
| `make test-all` | Run all tests including benchmark |
| `make benchmark` | Run performance benchmark only |
| `make coverage` | Run tests with coverage report |
| `make serve` | Start gRPC server on port 19530 |
| `make clean` | Remove .venv, caches, build artifacts |

# Quick Start

### Option 1: Local .db file (simplest, just like milvus-lite)

Pass a `.db` path to `MilvusClient` — the server starts automatically in-process, no manual setup needed:

```python
from pymilvus import MilvusClient

client = MilvusClient("./demo.db")

# Create collection, insert, search — exactly like milvus-lite
client.create_collection("demo", dimension=384)

data = [
    {"id": i, "vector": [float(j) for j in range(384)], "text": f"doc {i}"}
    for i in range(100)
]
client.insert("demo", data)

results = client.search(
    "demo",
    data=[[float(i) for i in range(384)]],
    limit=5,
    output_fields=["text"],
)
for hits in results:
    for hit in hits:
        print(f"id={hit['id']}, distance={hit['distance']:.4f}, text={hit['entity']['text']}")

# Filter, query, delete — all standard pymilvus API
results = client.query("demo", filter="id < 10", output_fields=["text"], limit=5)
client.delete("demo", ids=[0, 1, 2])
client.drop_collection("demo")
```

This is a **drop-in replacement** for milvus-lite — just `pip install litevecdb` and your existing `MilvusClient("./xxx.db")` code works without changes.

### Option 2: Standalone gRPC server

For multi-client or long-running service scenarios, start the server explicitly:

```bash
litevecdb-grpc --data-dir ./data --port 19530
```

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
# Same API as above
```

### Option 3: Embedded engine (no server, no gRPC)

```python
from litevecdb import LiteVecDB, CollectionSchema, FieldSchema, DataType

schema = CollectionSchema(fields=[
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
])

with LiteVecDB("./data") as db:
    col = db.create_collection("docs", schema)
    col.insert([
        {"id": 1, "vec": [0.1] * 128, "category": "tech"},
        {"id": 2, "vec": [0.2] * 128, "category": "news"},
    ])
    col.create_index("vec", {
        "index_type": "HNSW", "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    })
    col.load()

    results = col.search(
        [[0.1] * 128], top_k=3, metric_type="COSINE",
        expr="category == 'tech'", output_fields=["category"],
    )
```

# Examples

## Full Text Search (BM25)

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri="http://localhost:19530")

schema = MilvusClient.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535,
                 enable_analyzer=True, enable_match=True)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(
    name="bm25", function_type=FunctionType.BM25,
    input_field_names=["text"], output_field_names=["sparse"],
))
client.create_collection("articles", schema=schema)

client.insert("articles", [
    {"text": "machine learning algorithms for classification"},
    {"text": "deep learning neural networks"},
    {"text": "natural language processing with transformers"},
])

idx = client.prepare_index_params()
idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
              metric_type="BM25", params={})
client.create_index("articles", idx)
client.load_collection("articles")

results = client.search(
    "articles", data=["machine learning"], anns_field="sparse",
    limit=3, output_fields=["text"],
)
```

## Hybrid Search (Dense + BM25)

```python
from pymilvus import AnnSearchRequest, WeightedRanker

dense_req = AnnSearchRequest(
    data=[[0.1] * 128], anns_field="vec",
    param={"metric_type": "COSINE"}, limit=10,
)
sparse_req = AnnSearchRequest(
    data=["search query"], anns_field="sparse",
    param={"metric_type": "BM25"}, limit=10,
)
results = client.hybrid_search(
    "articles", reqs=[dense_req, sparse_req],
    ranker=WeightedRanker(0.7, 0.3), limit=5,
)
```

## Text Embedding (Auto text-to-vector)

Define a `TEXT_EMBEDDING` function to auto-generate dense vectors from text fields during insert, and auto-embed text queries during search:

```python
from litevecdb import LiteVecDB, CollectionSchema, FieldSchema, DataType, Function, FunctionType

schema = CollectionSchema(fields=[
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=1536, is_function_output=True),
], functions=[
    Function(
        name="text_emb", function_type=FunctionType.TEXT_EMBEDDING,
        input_field_names=["text"], output_field_names=["vec"],
        params={"provider": "openai", "model_name": "text-embedding-3-small"},
    ),
])

with LiteVecDB("./data") as db:
    col = db.create_collection("docs", schema)

    # Insert text only — vectors are auto-generated via OpenAI API
    col.insert([
        {"text": "machine learning algorithms"},
        {"text": "deep learning neural networks"},
        {"text": "web development frameworks"},
    ])
    col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE",
                              "params": {"M": 16, "efConstruction": 200}})
    col.load()

    # Search with text — query is auto-embedded
    results = col.search(
        query_vectors=["machine learning"],
        top_k=3, metric_type="COSINE", anns_field="vec",
        output_fields=["text"],
    )
```

Requires `OPENAI_API_KEY` environment variable or `api_key` param. Uses `urllib` internally — no OpenAI SDK dependency.

## Rerank

### Semantic Rerank (Cohere)

Add a `RERANK` function with an external provider to re-score search results using a cross-encoder model:

```python
from litevecdb import Function, FunctionType

# Add to schema alongside TEXT_EMBEDDING
Function(
    name="my_reranker", function_type=FunctionType.RERANK,
    input_field_names=["text"], output_field_names=[],
    params={"provider": "cohere", "model_name": "rerank-v3.5"},
)
```

When searching with text queries, the engine first retrieves candidates via vector search, then calls the Cohere Rerank API to re-score by semantic relevance. Requires `COHERE_API_KEY` environment variable or `api_key` param.

### Decay Rerank (Field-value based)

Adjust search scores based on how close a numeric field is to a reference value. Three decay curves: `gauss`, `exp`, `linear`.

```python
Function(
    name="recency_decay", function_type=FunctionType.RERANK,
    input_field_names=["timestamp"],  # numeric field
    output_field_names=[],
    params={
        "reranker": "decay",
        "function": "gauss",   # or "exp", "linear"
        "origin": 1700000000,  # reference value (e.g. current timestamp)
        "scale": 86400,        # at this distance, score *= decay
        "offset": 3600,        # safe zone (no penalty within +-offset)
        "decay": 0.5,          # target factor at distance=scale
    },
)
```

Final score = `vector_relevance * decay_factor`. Items closer to `origin` rank higher. No external API call — pure local computation.

## Metadata Filtering

```python
# Comparison & logical
client.query("col", filter="age > 25 and status == 'active'", limit=10)

# IN operator
client.query("col", filter="category in ['tech', 'science']", limit=10)

# String matching
client.query("col", filter="name like 'John%'", limit=10)

# Dynamic field (bare name or $meta["key"] syntax)
client.query("col", filter='color == "red"', limit=10)
client.query("col", filter='$meta["color"] == "red"', limit=10)

# Text match (tokenized keyword search)
client.query("col", filter="text_match(title, 'machine learning')", limit=10)

# Array operations
client.query("col", filter='array_contains(tags, "python")', limit=10)
client.query("col", filter="array_length(scores) >= 3", limit=10)
client.query("col", filter="scores[0] > 90", limit=10)
```

# Supported Features

| Feature | Details |
|---|---|
| Vector types | `FLOAT_VECTOR`, `SPARSE_FLOAT_VECTOR` |
| Index types | `HNSW`, `HNSW_SQ`, `IVF_FLAT`, `IVF_SQ8` (FAISS), `FLAT` / `BRUTE_FORCE` / `AUTOINDEX`, `SPARSE_INVERTED_INDEX` |
| Metrics | `COSINE`, `L2`, `IP`, `BM25` |
| Scalar types | `INT8/16/32/64`, `FLOAT`, `DOUBLE`, `VARCHAR`, `BOOL`, `JSON`, `ARRAY` |
| Search | Dense ANN, sparse BM25, hybrid (multi-vector + reranker), range search, group-by |
| Filter | Comparison, logical, IN, LIKE, arithmetic, IS NULL, $meta, text_match, array ops, chained JSON path |
| CRUD | Insert, upsert (partial update), delete (by ID or filter), get, query, search |
| Partition key | `is_partition_key=True` — auto-bucket routing by field value hash |
| Partitions | Create, drop, list, per-partition insert/search |
| Pagination | `offset` parameter, query/search iterators |
| Auto ID | `auto_id=True` on INT64 primary key |
| Default values | `default_value` on schema fields, auto-filled on insert |
| Dynamic fields | `enable_dynamic_field=True`, filter with bare field names or `$meta["key"]` |
| BM25 | `Function(type=BM25)` auto-generates sparse vectors, per-segment inverted index |
| Text Embedding | `Function(type=TEXT_EMBEDDING)` auto text-to-vector via OpenAI API (insert + search) |
| Rerank | `Function(type=RERANK)` — semantic reranking (Cohere API) and decay reranking (gauss/exp/linear) |
| Nullable fields | Nullable scalars and vectors |
| Rename collection | `client.rename_collection("old", "new")` |
| gRPC | 25+ Milvus RPCs, pymilvus fully compatible |

# Performance Benchmark

Benchmarked using the Cohere 100K dataset from [VectorDBBench](https://github.com/zilliztech/VectorDBBench) (768-dim, COSINE, with ground truth).

### Environment

- Hardware: Apple Silicon (MacBook)
- Python: 3.14
- Index: HNSW (M=16, efConstruction=200, ef=128)

### Results

| Metric | Result |
|--------|--------|
| **Dataset** | 100,000 vectors (768-dim) |
| **Insert throughput** | 4,461 records/s |
| **Index build** | 9.32s (HNSW) |
| **Search QPS (nq=1)** | 48.6 |
| **Search VPS (nq=10)** | 340.7 |
| **Search latency P50** | 20.36 ms |
| **Search latency P95** | 21.22 ms |
| **Search latency P99** | 27.62 ms |
| **Recall@10** | **91.18%** |

> As a pure-Python embedded vector database, LiteVecDB delivers solid search quality (91% recall) with stable latency (P50 to P95 gap is only ~1ms). Batched queries (nq=10) reach 340+ VPS (vectors per second) thanks to amortized RPC overhead and FAISS HNSW batch processing.
>
> **Note on QPS vs VPS:** `QPS (nq=1)` counts one query per RPC. `VPS (nq=10)` counts total vector queries processed per second — these are not directly comparable because batching amortizes RPC and serialization overhead.

# Known Limitations

- **Single-process only** — one process per `data_dir` (file-level lock)
- **Synchronous flush** — no background compaction or async writes
- **No authentication / RBAC**
- **No binary / float16 / bfloat16 vectors**
- **No PQ indexes** — no Product Quantization; SQ is supported via IVF_SQ8 and HNSW_SQ
- **Per-segment BM25 IDF** — IDF statistics are segment-local, not global

# Architecture

```
pymilvus client
      |  gRPC
      v
+---------------------------------------------------+
| adapter/grpc/  MilvusServicer                      |
|   25+ RPCs, schema/search/records translators      |
+---------------------------------------------------+
      |
+---------------------------------------------------+
| engine/  Collection                                |
|   insert, delete, search, query, flush,            |
|   compaction, recovery, load/release               |
+---------------------------------------------------+
      |                           |
+------------------+    +---------------------+
| storage/         |    | search/             |
|   WAL, MemTable, |    |   assembler,        |
|   Segment,       |    |   executor,         |
|   Manifest       |    |   filter/ (3 BE)    |
+------------------+    +---------------------+
      |                           |
+------------------+    +---------------------+
| index/           |    | analyzer/           |
|   HNSW, HNSW_SQ, |    |   Standard/Jieba,   |
|   IVF_FLAT,      |    |   BM25 sparse,      |
|   IVF_SQ8,       |    |   term hash         |
|   BruteForce,    |    +---------------------+
|   SparseInverted |              |
+------------------+    +---------------------+
      |               | embedding/          |
+------------------+  |   OpenAI provider,  |
| rerank/          |  |   auto text→vector  |
|   Cohere API,    |  +---------------------+
|   Decay (local)  |
+------------------+
      |
+---------------------------------------------------+
| schema/  DataType, FieldSchema, Function,          |
|   validation, Arrow builders, persistence          |
+---------------------------------------------------+
```

Storage is LSM-tree style: WAL (Arrow IPC) -> MemTable -> immutable Parquet segments. Vector indexes are segment-level (one `.idx` per Parquet file). Manifest is the single source of truth, updated atomically via tmp+rename.

# Built with Vibe Coding

This entire project — architecture design, implementation, test suite, documentation — was built through conversational AI pair programming using [Claude Code](https://claude.ai/code).

The development process:
1. **Design** — discuss architecture in natural language, produce design docs
2. **Implement** — describe what to build, review and iterate on generated code
3. **Test** — 2100+ tests including recall validation and Milvus compatibility suites
4. **Iterate** — fix bugs, optimize performance, add features — all through conversation

No boilerplate was hand-typed. No Stack Overflow was consulted. Just a human with a vision and an AI that codes.

# Testing

```bash
pytest                                  # 2100+ tests
pytest tests/adapter/ -k "grpc"         # gRPC integration tests
pytest tests/index/test_index_differential.py  # recall validation
pytest --cov=litevecdb                  # with coverage
```

# Contributing

Issues and pull requests are welcome at [GitHub](https://github.com/junjiejiangjjj/milvus-lite-v2).

# License

Apache License 2.0. See [LICENSE](LICENSE).
