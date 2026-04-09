# LiteVecDB

A local embedded vector database — designed as a **local version of Milvus**, written in pure Python.

LiteVecDB is an LSM-tree-style storage engine with PyArrow in-memory and Parquet on disk, plus per-segment FAISS HNSW indexing and a Milvus-style scalar filter expression system. A gRPC adapter (work in progress) lets pymilvus clients connect without code changes.

> **Status**: pre-1.0. Phases 0–9 are landed (storage, engine, search, scalar filter, vector index). Phase 10 (gRPC adapter) is in progress — Collection lifecycle RPCs work today; CRUD / Search / Index RPCs are next.

---

## Highlights

- **LSM-Tree storage** — WAL → MemTable → Parquet, with size-tiered compaction and tombstone GC
- **Crash-safe** — atomic Manifest with `.prev` backup; WAL replay on every restart
- **Milvus-style API** — `insert / delete / get / search / query`, partition CRUD, `_seq` global ordering, upsert semantics
- **Scalar filter expressions** (Phase 8) — `age > 18 and category in ['tech', 'news']` style; three backends (`arrow` / `hybrid` / `python`) with automatic dispatch + LRU cache
- **FAISS HNSW vector index** (Phase 9) — per-segment binding, `IDSelectorBitmap` pre-filter integration, optional via `[faiss]` extra
- **Persistent indexes** — `.idx` sidecar files; cold start reads from disk, no rebuild
- **load / release state machine** — mirrors Milvus client behavior
- **gRPC adapter** (Phase 10, in progress) — `pymilvus.MilvusClient` compatible; embeddable + standalone server modes
- **1119 tests passing**, including a recall@10 differential test (FAISS HNSW vs brute force ≥ 0.95)

---

## Quick start

### Install

```bash
git clone <this repo>
cd milvus-lite-v2
pip install -e ".[dev]"           # base + test deps
pip install -e ".[dev,faiss]"     # add FAISS HNSW (recommended)
pip install -e ".[dev,faiss,grpc]"  # add the pymilvus-compatible gRPC server
```

Requires Python ≥ 3.10. Core dependencies: `pyarrow ≥ 15.0`, `numpy ≥ 1.24`. Optional: `faiss-cpu ≥ 1.7.4` for HNSW, `pymilvus ≥ 2.4 + grpcio ≥ 1.50` for the gRPC adapter.

### Embedded usage

```python
from litevecdb import LiteVecDB, CollectionSchema, FieldSchema, DataType

schema = CollectionSchema(fields=[
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="active", dtype=DataType.BOOL),
])

with LiteVecDB("/path/to/data") as db:
    col = db.create_collection("docs", schema)

    col.insert([
        {"id": 1, "vec": [...], "title": "intro", "active": True},
        {"id": 2, "vec": [...], "title": "blog",  "active": False},
        # ...
    ])

    # Vector index — required before search if you want HNSW.
    # Skip create_index entirely to keep brute-force search.
    col.create_index("vec", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    })
    col.load()

    # Vector search with scalar filter
    results = col.search(
        [[0.1, 0.2, ...]],         # query vectors
        top_k=10,
        metric_type="COSINE",
        expr="active == true and title like 'b%'",
        output_fields=["title"],
    )

    # Scalar query (no vector)
    rows = col.query("active == true", limit=20)

    # Get by primary key
    rows = col.get([1, 2, 3])

    # Delete
    col.delete(pks=[2])
```

See `examples/m9_demo.py` for the full Phase 9 lifecycle (10K records, FAISS HNSW, partitions, filters, reload after restart).

### gRPC server (pymilvus client)

> Phase 10 is in progress. Today the server starts and supports Collection lifecycle RPCs (create / drop / list / has / describe). Insert / Search / Index RPCs are coming in 10.3 / 10.4.

```bash
litevecdb-grpc --data-dir ./data --port 19530
# or
python -m litevecdb.adapter.grpc --data-dir ./data --port 19530
```

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vec", DataType.FLOAT_VECTOR, dim=128)

client.create_collection("docs", schema=schema)
print(client.list_collections())          # → ['docs']
print(client.has_collection("docs"))       # → True
print(client.describe_collection("docs"))  # → schema dict
client.drop_collection("docs")
```

---

## Architecture

LiteVecDB is layered bottom-up. Each layer only depends on the layers below it.

```
┌─────────────────────────────────────────────────┐
│ adapter/grpc        Phase 10  (in progress)      │
│   pymilvus protocol → engine API translation     │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│ db.LiteVecDB      Multi-collection lifecycle     │
│                   LOCK file, single-process safe │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│ engine/Collection   Insert/Delete/Get/Search/    │
│                     Query, _seq alloc,           │
│                     create_index/load/release,   │
│                     partition CRUD, flush,       │
│                     compaction, recovery         │
└─────────────────────────────────────────────────┘
            │                              │
┌───────────────────────┐    ┌───────────────────────┐
│ storage/              │    │ search/                │
│   WAL                 │    │   bitmap pipeline       │
│   MemTable            │    │   (dedup + tombstone +  │
│   Manifest (v2)       │    │    scalar filter)        │
│   DataFile / DeltaFile│    │   distance               │
│   DeltaIndex          │    │   assembler              │
│   Segment + index slot│    │   executor (with index) │
└───────────────────────┘    │   filter/  (Phase 8)    │
            │                │     parse → compile →   │
            │                │     evaluate            │
            │                └───────────────────────┘
            │                              │
            └──────────┬───────────────────┘
                       │
┌─────────────────────────────────────────────────┐
│ index/      Phase 9                              │
│   VectorIndex protocol                           │
│   BruteForceIndex  (NumPy, fallback + baseline)  │
│   FaissHnswIndex   (FAISS HNSW + IDSelectorBatch)│
│   factory          (try-import faiss degradation)│
└─────────────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────┐
│ schema/     DataType, FieldSchema, validation,    │
│             4 Arrow schema variants               │
└─────────────────────────────────────────────────┘
```

### Data hierarchy

```
DB ("my_app")                       ← namespace = root directory
  ├── Collection ("documents")      ← schema owner; WAL/MemTable/_seq are Collection-level
  │     ├── Partition ("2024_Q1")   ← per-Partition data files
  │     ├── Partition ("2024_Q2")
  │     └── Partition ("_default")  ← always exists, cannot drop
  └── Collection ("images")
        └── Partition ("_default")
```

### On-disk layout

```
data_dir/
├── LOCK                                          # fcntl flock — single-process gate
└── collections/
    └── docs/
        ├── schema.json                            # schema persistence
        ├── manifest.json                          # single source of truth (atomic + .prev)
        ├── manifest.json.prev
        ├── wal/
        │   ├── wal_data_000003.arrow              # Arrow IPC streaming
        │   └── wal_delta_000003.arrow
        └── partitions/
            └── _default/
                ├── data/
                │   └── data_000001_000500.parquet
                ├── delta/
                │   └── delta_000501_000503.parquet
                └── indexes/                       # Phase 9
                    └── data_000001_000500.hnsw.idx
```

### Key design decisions

- **`_seq` is the global ordering** — every override / discard / dedup decision compares `_seq`, never relies on call order or file physical order. Recovery can replay operations in any order.
- **Files are immutable** — data Parquet, delta Parquet, WAL, .idx files are write-once-then-delete. Manifest is the only mutable state, updated atomically via tmp + rename + `.prev` backup.
- **Insert and delete are separate streams** — different file types (data Parquet vs delta Parquet). Filters happen in memory via `delta_index`.
- **Vector index is segment-level** — one `.idx` per data Parquet, 1:1 lifetime. Compaction creates a new merged segment + new index, drops the old ones. Avoids the FAISS HNSW "no real delete" trap.
- **FAISS pre-filter via IDSelectorBitmap** — the Phase 8 bitmap pipeline produces the same `valid_mask` shape that FAISS's selector consumes, so scalar filter + vector search compose without post-filtering.
- **Filter parser ↔ evaluator decoupled via AST** — three backends (`arrow` / `hybrid` / `python`) chosen at compile time based on whether the expression touches `$meta` dynamic fields.
- **load / release state machine** mirrors Milvus — collections without an index auto-load; collections with an index require explicit `load()` before `search/get/query`.

For full design details see `plan/`:
- `MVP.md` — overall MVP design (Chinese)
- `wal-design.md` — WAL + segment + search pipeline deep dive
- `filter-design.md` — Phase 8 scalar filter subsystem
- `index-design.md` — Phase 9 vector index subsystem
- `grpc-adapter-design.md` — Phase 10 gRPC adapter
- `modules.md` — file-by-file module reference (per-class/method signatures)
- `roadmap.md` — phased implementation plan

---

## Engine API reference

```python
LiteVecDB(data_dir)                                          # context manager
  .create_collection(name, schema) → Collection
  .get_collection(name) → Collection
  .drop_collection(name)
  .has_collection(name) → bool
  .list_collections() → List[str]
  .get_collection_stats(name) → {"row_count": int}
  .close()

Collection
  # Writes (no loaded state required)
  .insert(records: List[dict], partition_name="_default") → List[pk]
  .delete(pks: List, partition_name=None) → int

  # Reads (require loaded state if an index is configured)
  .get(pks, partition_names=None, expr=None) → List[dict]
  .search(query_vectors, top_k=10, metric_type="COSINE",
          partition_names=None, expr=None, output_fields=None)
          → List[List[dict]]
  .query(expr, output_fields=None, partition_names=None, limit=None)
         → List[dict]

  # Index lifecycle (Phase 9)
  .create_index(field_name, index_params) → None
  .drop_index(field_name=None) → None
  .has_index() → bool
  .get_index_info() → Optional[dict]
  .load() → None
  .release() → None
  .load_state → "released" | "loading" | "loaded"

  # Partitions
  .create_partition(name)
  .drop_partition(name)
  .list_partitions() → List[str]
  .has_partition(name) → bool

  # Stats / introspection
  .num_entities → int
  .describe() → dict
  .name → str
  .schema → CollectionSchema

  # Maintenance
  .flush()
  .close()
```

---

## Filter expression syntax

LiteVecDB implements a Milvus-compatible subset (Phase 8 / F1-F3+):

| Category | Operators / Forms |
|---|---|
| Comparison | `==` `!=` `<` `<=` `>` `>=` |
| Logical | `and` `or` `not` (case-insensitive) + `&&` `\|\|` `!` |
| Set membership | `field in [...]` `field not in [...]` |
| String | `field like "pattern%"` (SQL LIKE: `%` and `_`) |
| Arithmetic | `+` `-` `*` `/` `%` |
| Null check | `field is null` `field is not null` |
| Dynamic field | `$meta["key"]` (requires `enable_dynamic_field=True`) |

Examples:

```python
col.search(q, expr="age > 18 and category in ['tech', 'news']")
col.query("title like 'AI%' and score > 0.5", limit=20)
col.search(q, expr='$meta["priority"] >= 5 or active == true')
col.query("age >= 30 and (price * quantity > 1000.0 or discount is not null)")
```

Errors carry a caret-style location pointer + did-you-mean suggestions:

```
FilterFieldError: unknown field 'agg' at column 1
  agg > 18
  ^^^
did you mean 'age'?
```

---

## Testing

```bash
pytest                              # default — skips slow tests (~3s)
pytest -m slow                      # long-running stress tests
pytest --cov=litevecdb              # with coverage
pytest tests/index/                 # specific package
pytest tests/adapter/ -k grpc       # specific tests
```

The suite is **1119 default tests + 8 slow tests**, organized into:

| Path | Coverage |
|---|---|
| `tests/schema/` | type system, validation, Arrow builders |
| `tests/storage/` | WAL, MemTable, Parquet IO, Manifest, DeltaIndex, Segment |
| `tests/engine/` | Collection insert/get/search/delete, flush, recovery, compaction, partition CRUD, load/release, index persistence |
| `tests/search/` | bitmap, distance, executor, filter (parser, semantic, arrow/hybrid/python backends, end-to-end differential) |
| `tests/index/` | BruteForceIndex, FaissHnswIndex (skipif faiss missing), IndexSpec, recall@10 differential vs FAISS |
| `tests/adapter/` | gRPC server startup, schema translator, Collection lifecycle RPCs (skipif pymilvus missing) |
| `tests/test_db.py` | LiteVecDB lifecycle, LOCK, multi-collection |
| `tests/test_smoke_e2e.py` | end-to-end via the public API |

Differential testing is the main correctness safety net:
- Phase 8 filter: `arrow_backend == hybrid_backend == python_backend` for every supported expression
- Phase 9 index: `FaissHnswIndex` recall@10 ≥ 0.95 vs `BruteForceIndex` baseline + distance value parity within 1e-3

---

## Examples

`examples/m{N}_demo.py` — one self-contained script per major milestone:

| File | What it shows |
|---|---|
| `m2_demo.py` | Phase 2: insert + get on memtable |
| `m3_demo.py` | Phase 3: flush + recovery (data survives restart) |
| `m4_demo.py` | Phase 4: vector search (brute force + bitmap pipeline) |
| `m5_demo.py` | Phase 5: delete (tombstones, restart correctness) |
| `m6_demo.py` | Phase 6: compaction + tombstone GC |
| `m7_demo.py` | Phase 7: multi-collection DB layer |
| `m8_demo.py` | Phase 8: scalar filter expressions through search/get/query |
| `m9_demo.py` | Phase 9: vector index lifecycle (HNSW, load/release, persistence) |

Run any demo from the project root: `python examples/m9_demo.py`.

---

## Project status & roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Design freeze | done |
| 1 | WAL hardening, schema scaffolding | done |
| 2 | Insert + get over WAL + MemTable | done |
| 3 | Flush + crash recovery (Parquet on disk) | done |
| 4 | Vector search (brute force) | done |
| 5 | Delete + delta files + delta_index | done |
| 6 | Size-tiered compaction + tombstone GC | done |
| 7 | DB layer + LOCK + multi-collection | done |
| **8 (filter)** | F1: basic grammar | done |
|   | F2a: arithmetic, LIKE, IS NULL | done |
|   | F2b: `$meta` dynamic field | done |
|   | F2c: LRU cache + `query()` | done |
|   | F3+: hybrid backend (per-batch JSON preprocessing) | done |
| **9 (index)** | 9.1: pymilvus-prereq API (partition CRUD, num_entities, describe, output_fields, get_collection_stats) | done |
|   | 9.2: VectorIndex protocol + BruteForceIndex + execute_search_with_index | done |
|   | 9.3: IndexSpec + Manifest v2 + load/release state machine | done |
|   | 9.4: index file persistence + flush/compaction/recovery hooks | done |
|   | 9.5: FAISS HNSW + factory + metric alignment + recall differential | done |
|   | 9.6: m9 demo + long-running stress test | done |
| **10 (gRPC adapter)** | 10.1: server skeleton + Connect/GetVersion/CheckHealth | done |
|   | 10.2: Collection lifecycle RPCs + schema translator | done |
|   | 10.3: insert / upsert / delete / query / get + records translator | next |
|   | 10.4: search + create_index + load + release RPCs | — |
|   | 10.5: partition + flush + stats + m10 demo + pymilvus quickstart smoke | — |
|   | 10.6: error code mapping + UNIMPLEMENTED friendly messages | — |

For the full roadmap with sub-task breakdowns and rationale see `plan/roadmap.md`.

---

## Out of scope (post-MVP)

These are deferred to future iterations and tracked in `plan/MVP.md` §10:

- IVF / IVF-PQ / OPQ quantized vector indexes (Phase 9 ships HNSW only)
- Sparse / Binary / Float16 / BFloat16 vectors
- Multiple vector fields per collection
- Hybrid search (multi-vector recall)
- Auto ID
- Partition Key (automatic hash partitioning)
- Backup / Restore RPC
- Authentication / RBAC
- Async / background flush + compaction
- Multi-process concurrency
- Distributed deployment
- Snapshot / time travel queries

---

## Repository layout

```
milvus-lite-v2/
├── README.md                  # this file
├── CLAUDE.md                  # project conventions for Claude Code
├── pyproject.toml             # hatchling build, deps, [faiss] / [grpc] extras
│
├── litevecdb/
│   ├── __init__.py            # public API: LiteVecDB, CollectionSchema, ...
│   ├── constants.py
│   ├── exceptions.py
│   ├── db.py                  # LiteVecDB top-level
│   ├── schema/                # DataType, FieldSchema, CollectionSchema, validation, Arrow builders, persistence
│   ├── storage/               # WAL, MemTable, DataFile, DeltaFile, DeltaIndex, Segment, Manifest
│   ├── engine/                # Collection, Operation, flush, recovery, compaction
│   ├── search/                # bitmap, distance, assembler, executor (+executor_indexed), filter/
│   ├── index/                 # VectorIndex protocol, BruteForceIndex, FaissHnswIndex, IndexSpec, factory
│   └── adapter/grpc/          # MilvusServicer, server, cli, errors, translators/
│
├── tests/                     # 1119 default + 8 slow tests
│   ├── schema/  storage/  engine/  search/  index/  adapter/
│   ├── test_db.py
│   └── test_smoke_e2e.py
│
├── examples/                  # m2_demo.py … m9_demo.py
└── plan/                      # design docs (Chinese): MVP, wal-design, filter-design,
                               # index-design, grpc-adapter-design, modules, roadmap
```

---

## License

TBD.
