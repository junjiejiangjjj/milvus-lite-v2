# LiteVecDB

A local embedded vector database — designed as a **local version of Milvus**, written in pure Python.

LiteVecDB is an LSM-tree-style storage engine with PyArrow in-memory and Parquet on disk, per-segment FAISS HNSW indexing, BM25 full text search, and a Milvus-compatible gRPC adapter. pymilvus clients connect without code changes.

> **Status**: pre-1.0. Phases 0–14 landed — storage, engine, scalar filter, vector index, gRPC adapter, full text search, hybrid search, group by, range search. **1449 tests passing.**

---

## Highlights

- **LSM-Tree storage** — WAL → MemTable → Parquet, with size-tiered compaction and tombstone GC
- **Crash-safe** — atomic Manifest with `.prev` backup; WAL replay on every restart
- **Milvus-style API** — `insert / delete / get / search / query`, partition CRUD, `_seq` global ordering, upsert semantics
- **Scalar filter expressions** — `age > 18 and category in ['tech', 'news']` style; three backends (`arrow` / `hybrid` / `python`) with automatic dispatch
- **FAISS HNSW vector index** — per-segment binding, `IDSelectorBitmap` pre-filter, optional via `[faiss]` extra
- **BM25 full text search** — `Function(type=BM25)` auto-generates sparse vectors from text; inverted index with BM25 scoring
- **text_match filter** — `text_match(field, 'tokens')` tokenized keyword matching with OR logic
- **Hybrid search** — multi-vector fusion with `WeightedRanker` / `RRFRanker`
- **Group By search** — `group_by_field` deduplicates results by scalar field, with `group_size` control
- **Range search** — `radius` / `range_filter` distance bounds
- **gRPC adapter** — `pymilvus.MilvusClient` fully compatible; embeddable + standalone server
- **load / release state machine** — mirrors Milvus client behavior
- **1449 tests**, including recall@10 differential tests and pymilvus end-to-end compatibility suites

---

## Quick start

### Install

```bash
git clone <this repo>
cd milvus-lite-v2
pip install -e ".[dev]"              # base + test deps
pip install -e ".[dev,faiss]"        # add FAISS HNSW (recommended)
pip install -e ".[dev,faiss,grpc]"   # add pymilvus-compatible gRPC server
```

Requires Python >= 3.10. Core deps: `pyarrow >= 15.0`, `numpy >= 1.24`. Optional: `faiss-cpu >= 1.7.4` for HNSW, `pymilvus >= 2.4 + grpcio >= 1.50` for gRPC, `jieba >= 0.42` for Chinese tokenization.

### Embedded usage

```python
from litevecdb import LiteVecDB, CollectionSchema, FieldSchema, DataType

schema = CollectionSchema(fields=[
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR),
])

with LiteVecDB("/path/to/data") as db:
    col = db.create_collection("docs", schema)
    col.insert([
        {"id": 1, "vec": [...], "title": "intro", "category": "tech"},
        {"id": 2, "vec": [...], "title": "blog",  "category": "news"},
    ])

    col.create_index("vec", {
        "index_type": "HNSW", "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    })
    col.load()

    # Vector search with scalar filter
    results = col.search(
        [[0.1, 0.2, ...]], top_k=10, metric_type="COSINE",
        expr="category == 'tech'", output_fields=["title"],
    )

    # Group by search
    results = col.search(
        [[0.1, 0.2, ...]], top_k=5,
        group_by_field="category", group_size=2,
    )

    # Range search
    results = col.search(
        [[0.1, 0.2, ...]], top_k=10,
        radius=0.1, range_filter=0.8,
    )

    # Scalar query
    rows = col.query("category == 'tech'", limit=20)
```

### Full text search (BM25)

```python
from litevecdb.schema.types import Function, FunctionType

schema = CollectionSchema(fields=[
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR,
                enable_analyzer=True,
                analyzer_params={"tokenizer": "standard"}),
    FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="sparse_emb", dtype=DataType.SPARSE_FLOAT_VECTOR,
                is_function_output=True),
], functions=[
    Function(name="bm25_fn", function_type=FunctionType.BM25,
             input_field_names=["text"], output_field_names=["sparse_emb"]),
])

with LiteVecDB("/path/to/data") as db:
    col = db.create_collection("articles", schema)
    col.insert([
        {"id": 1, "text": "machine learning algorithms", "dense": [...]},
        {"id": 2, "text": "deep learning neural networks", "dense": [...]},
    ])
    col.load()

    # BM25 text search
    results = col.search(
        ["machine learning"], top_k=10,
        metric_type="BM25", anns_field="sparse_emb",
    )

    # text_match filter
    rows = col.query("text_match(text, 'machine learning')", limit=20)
```

### gRPC server (pymilvus client)

```bash
litevecdb-grpc --data-dir ./data --port 19530
```

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, WeightedRanker

client = MilvusClient(uri="http://localhost:19530")

# Create collection, insert, index, load, search — all standard pymilvus API
schema = MilvusClient.create_schema(auto_id=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535,
                 enable_analyzer=True, enable_match=True)
schema.add_field("vec", DataType.FLOAT_VECTOR, dim=128)
schema.add_field("bm25_emb", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(
    name="bm25_fn", function_type=FunctionType.BM25,
    input_field_names=["text"], output_field_names=["bm25_emb"],
))
client.create_collection("articles", schema=schema)
client.insert("articles", data=[...])

# Hybrid search: dense + BM25
dense_req = AnnSearchRequest(data=[[...]], anns_field="vec", param={}, limit=10)
bm25_req = AnnSearchRequest(data=[{term: 1.0}], anns_field="bm25_emb",
                            param={"metric_type": "BM25"}, limit=10)
results = client.hybrid_search("articles",
    reqs=[dense_req, bm25_req],
    ranker=WeightedRanker(0.6, 0.4), limit=10)
```

---

## Architecture

```
+-------------------------------------------------+
| adapter/grpc           Phase 10-14               |
|   pymilvus protocol -> engine API translation    |
|   + BM25/sparse/hybrid/group_by/range support    |
+-------------------------------------------------+
                         |
+-------------------------------------------------+
| db.LiteVecDB      Multi-collection lifecycle     |
+-------------------------------------------------+
                         |
+-------------------------------------------------+
| engine/Collection   Insert/Delete/Get/Search/    |
|   Query, BM25 auto-gen, group_by, range filter,  |
|   create_index/load/release, partition CRUD,      |
|   flush, compaction, recovery                     |
+-------------------------------------------------+
          |                              |
+---------------------+    +--------------------------+
| storage/            |    | search/                   |
|   WAL, MemTable,    |    |   bitmap pipeline,        |
|   Manifest (v2),    |    |   distance, assembler,    |
|   Segment + index   |    |   executor (with index),  |
+---------------------+    |   filter/ (Phase 8),      |
          |                |   text_match (Phase 11)   |
          |                +--------------------------+
          |                              |
+---------------------+    +--------------------------+
| index/              |    | analyzer/   Phase 11      |
|   VectorIndex,      |    |   StandardAnalyzer,       |
|   BruteForceIndex,  |    |   JiebaAnalyzer,          |
|   FaissHnswIndex,   |    |   BM25 sparse codec,      |
|   SparseInverted    |    |   term hash (FNV-1a)      |
+---------------------+    +--------------------------+
                         |
+-------------------------------------------------+
| schema/     DataType (incl. SPARSE_FLOAT_VECTOR), |
|   FieldSchema, Function/FunctionType,             |
|   validation, Arrow builders, persistence         |
+-------------------------------------------------+
```

### Key design decisions

- **`_seq` is the global ordering** — every override / discard decision compares `_seq`, never call order
- **Files are immutable** — data Parquet, delta Parquet, WAL, .idx are write-once. Manifest is the only mutable state (atomic tmp + rename)
- **Vector index is segment-level** — one `.idx` per data Parquet, 1:1 lifetime. Avoids FAISS HNSW "no real delete" trap
- **BM25 uses per-segment inverted index** — TF stored at insert time, IDF computed at search time from segment statistics
- **Hybrid search = multi-route + rerank** — each route executes independently, results merged by WeightedRanker or RRFRanker
- **Filter parser / evaluator decoupled via AST** — three backends chosen at compile time; text_match forces python backend

---

## Filter expression syntax

| Category | Operators / Forms |
|---|---|
| Comparison | `==` `!=` `<` `<=` `>` `>=` |
| Logical | `and` `or` `not` + `&&` `\|\|` `!` |
| Set membership | `field in [...]` `field not in [...]` |
| String | `field like "pattern%"` (SQL LIKE: `%` and `_`) |
| Arithmetic | `+` `-` `*` `/` |
| Null check | `field is null` `field is not null` |
| Dynamic field | `$meta["key"]` (requires `enable_dynamic_field=True`) |
| Text match | `text_match(field, 'token1 token2')` (OR logic, requires `enable_match=True`) |

---

## Engine API reference

```python
Collection
  # Writes
  .insert(records: List[dict], partition_name="_default") -> List[pk]
  .delete(pks: List, partition_name=None) -> int

  # Reads (require loaded state if index configured)
  .get(pks, partition_names=None, expr=None) -> List[dict]
  .search(query_vectors, top_k=10, metric_type="COSINE",
          partition_names=None, expr=None, output_fields=None,
          anns_field=None,
          group_by_field=None, group_size=1, strict_group_size=False,
          radius=None, range_filter=None)
          -> List[List[dict]]
  .query(expr, output_fields=None, partition_names=None, limit=None)
         -> List[dict]

  # Index lifecycle
  .create_index(field_name, index_params)
  .drop_index(field_name=None)
  .load() / .release()
  .load_state -> "released" | "loading" | "loaded"

  # Partitions
  .create_partition(name) / .drop_partition(name)
  .list_partitions() / .has_partition(name)

  # Stats
  .num_entities -> int
  .describe() -> dict
```

---

## Testing

```bash
pytest                              # 1449 default tests (~60s)
pytest -m slow                      # long-running stress tests
pytest --cov=litevecdb              # with coverage
pytest tests/adapter/ -k fts        # specific tests
```

| Path | Coverage |
|---|---|
| `tests/schema/` | type system, validation, Arrow builders, FTS schema extensions |
| `tests/storage/` | WAL, MemTable, Parquet IO, Manifest, DeltaIndex, Segment |
| `tests/engine/` | Collection CRUD, flush, recovery, compaction, partitions, anns_field |
| `tests/search/` | bitmap, distance, executor, filter (parser/semantic/backends), text_match |
| `tests/index/` | BruteForceIndex, FaissHnswIndex, SparseInvertedIndex, recall differential |
| `tests/analyzer/` | StandardAnalyzer, JiebaAnalyzer, sparse codec, BM25 auto-gen |
| `tests/adapter/` | gRPC server, schema/records/search translators, pymilvus compat (42+24+14+7 cases), hybrid search, group by, range search, FTS compat |

---

## Project status & roadmap

| Phase | Scope | Status |
|---|---|---|
| 0-7 | Storage, engine, search, partitions, compaction | done |
| **8** | Scalar filter expressions (F1-F3+, three backends) | done |
| **9** | FAISS HNSW vector index (per-segment, load/release) | done |
| **10** | gRPC adapter (pymilvus compatible, 25+ RPCs) | done |
| **11** | Full text search (BM25, Analyzer, text_match) | done |
| **12** | Hybrid search (WeightedRanker, RRFRanker) | done |
| **13** | Group By search (scalar field grouping) | done |
| **14** | Range search (distance bounds filtering) | done |

For the full roadmap see `plan/roadmap.md`.

---

## Out of scope (post-MVP)

- IVF / IVF-PQ quantized vector indexes
- Binary / Float16 / BFloat16 vectors
- Auto ID / Partition Key
- phrase_match / Multi-Analyzer / Highlighter
- Search Iterator
- Authentication / RBAC
- Async flush + compaction
- Multi-process / distributed

---

## Repository layout

```
milvus-lite-v2/
+-- README.md
+-- pyproject.toml             # hatchling build, [faiss] / [grpc] / [chinese] extras
|
+-- litevecdb/
|   +-- schema/                # DataType, FieldSchema, Function, validation, Arrow builders
|   +-- storage/               # WAL, MemTable, DataFile, DeltaFile, DeltaIndex, Segment, Manifest
|   +-- engine/                # Collection, Operation, flush, recovery, compaction
|   +-- search/                # bitmap, distance, assembler, executor, filter/
|   +-- index/                 # VectorIndex, BruteForceIndex, FaissHnswIndex, SparseInvertedIndex
|   +-- analyzer/              # Analyzer, StandardAnalyzer, JiebaAnalyzer, sparse codec, hash
|   +-- adapter/grpc/          # MilvusServicer, server, reranker, errors, translators/
|
+-- tests/                     # 1449 tests
+-- examples/                  # m2_demo.py ... m10_demo.py
+-- plan/                      # design docs (Chinese)
```

---

## License

TBD.
