# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteVecDB — a local embedded vector database, designed as a **local version of Milvus**. Implementation is in Python. The code lives in `milvus-lite-v2/litevecdb/`.

## Repository Layout

- **Design docs** (root-level, written in Chinese): `MVP.md`, `write-pipeline.md`, `research.md`, `modules.md`, `wal-design.md`, `filter-design.md`, `index-design.md`, `grpc-adapter-design.md`, `fts-design.md`, `roadmap.md`
- **Code repo**: `milvus-lite-v2/` (git root) — design docs are also copied into `milvus-lite-v2/plan/`
- **`modules.md`**: Authoritative module design — file-by-file breakdown with per-class/function signatures. Consult before implementing any module. §9.19-9.28 cover Phase 8 filter, §10 covers Phase 9 index, §11 covers Phase 10 gRPC adapter.
- **`wal-design.md`**: Deep-dive on WAL (Arrow IPC Streaming), segment architecture, and search pipeline.
- **`filter-design.md`**: Deep-dive on the Phase 8 scalar filter subsystem (grammar, AST, three-stage compilation, three backends arrow/hybrid/python).
- **`index-design.md`**: Deep-dive on the Phase 9 vector index subsystem (FAISS HNSW, segment-level binding, load/release state machine, recall validation).
- **`grpc-adapter-design.md`**: Deep-dive on the Phase 10 gRPC adapter (proto cropping, RPC mapping, FieldData transposition, error code translation).
- **`fts-design.md`**: Deep-dive on the Phase 11 full text search subsystem (BM25, Analyzer, SparseInvertedIndex, text_match filter).
- **`roadmap.md`**: Phased implementation plan. Phases 0-10 are landed; Phase 11 (full text search) is current.

## Development Commands

All commands run from `milvus-lite-v2/`:

```bash
# Install in editable mode with dev deps
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/storage/test_wal.py

# Run a specific test
pytest tests/storage/test_wal.py::test_write_recover -v

# Run with coverage
pytest --cov=litevecdb
```

Build system: Hatchling. Dependencies: `pyarrow>=15.0`, `numpy>=1.24`. Dev: `pytest>=8.0`, `pytest-cov>=5.0`. Requires Python >=3.10.

## Architecture (Two-Layer)

- **Internal Engine**: LSM-Tree style storage with PyArrow in-memory and Parquet on disk. Phase 9 adds per-segment vector indexes (FAISS HNSW).
- **gRPC Adapter Layer** (Phase 10): Sits on top of the engine to provide Milvus protocol compatibility, allowing pymilvus to connect directly without code changes.

## Code Structure (`litevecdb/`)

Seven packages, layered bottom-up:

1. **`schema/`** — Data model & type system. `DataType` enum (incl. `SPARSE_FLOAT_VECTOR`), `FieldSchema`, `CollectionSchema`, `Function`/`FunctionType`, record validation, Arrow schema builders (4 variants: data/delta/wal_data/wal_delta), schema.json persistence.
2. **`storage/`** — Persistence layer. WAL (Arrow IPC Streaming, dual-file), MemTable (RecordBatch list + pk_index + delete_index, seq-aware), DataFile / DeltaFile (Parquet IO), DeltaIndex (in-memory tombstone map + gc_below), Segment (immutable Parquet cache + optional VectorIndex), Manifest (JSON + .prev backup, atomic via tmp+rename, Phase 9.3 adds index_spec field).
3. **`engine/`** — Core logic orchestration. `Collection` class (entry point, `_seq` allocation, insert/delete/get/search/query, _apply orchestration, Phase 9.3 adds create_index/drop_index/load/release + load_state machine), Operation abstraction, Flush pipeline (7+1 steps, sync, Phase 9.4 adds index hook), Recovery (5+1 steps, Phase 9.4 adds orphan .idx cleanup), Compaction (Size-Tiered + tombstone GC + Phase 9.4 index lifecycle).
4. **`search/`** — Vector retrieval. Bitmap pipeline (dedup + delete + scalar filter), distance functions (cosine/L2/IP via NumPy), assembler (segments + memtable → numpy + filter mask), executor (top-k; Phase 9.2 adds `execute_search_with_index` path). **`search/filter/`** subpackage (Phase 8): tokenizer + Pratt parser + semantic check + three backends (arrow/hybrid/python) for Milvus-style scalar expressions.
5. **`index/`** (Phase 9) — Vector index abstraction. `VectorIndex` ABC, `IndexSpec`, `BruteForceIndex` (NumPy, fallback + differential baseline), `FaissHnswIndex` (FAISS HNSW + IDSelectorBitmap), factory with try-import faiss degradation. Bound to segments 1:1 via `.idx` files.
6. **`analyzer/`** (Phase 11) — Text analysis for full text search. `Analyzer` ABC, `StandardAnalyzer` (regex tokenizer), `JiebaAnalyzer` (optional Chinese), `create_analyzer` factory, `term_to_id` hash function.
7. **`adapter/`** (Phase 10) — Protocol translation. `adapter/grpc/` provides `MilvusServicer` mapping pymilvus RPCs to engine API. Translators handle Milvus FieldData ↔ records transposition, schema, search, results, expressions, index params. proto stubs generated from milvus-io/milvus-proto and committed to repo.

Top-level: `db.py` (`LiteVecDB` — multi-collection lifecycle), `constants.py`, `exceptions.py`.

## Data Hierarchy

DB → Collection → Partition. WAL/MemTable/`_seq` are Collection-level shared. Data files are Partition-level isolated (directory per partition).

## Internal Engine API

All inputs are Lists — no single-value normalization:

- `insert(records: List[dict], partition_name="_default") → List[pk]` — upsert semantics
- `delete(pks: List, partition_name=None) → int` — None means cross-all-partitions
- `get(pks: List, partition_names=None, expr=None) → List[dict]`
- `search(query_vectors: List[list], top_k=10, metric_type="COSINE", partition_names=None, expr=None, output_fields=None) → List[List[dict]]`
- `query(expr: str, output_fields=None, partition_names=None, limit=None) → List[dict]` — Phase 8, pure scalar query

Phase 9 adds index lifecycle:
- `create_index(field_name: str, index_params: dict) → None`
- `drop_index(field_name: str) → None`
- `has_index() → bool`
- `get_index_info() → Optional[dict]`
- `load() → None` — required before `search/get/query` (raises CollectionNotLoadedError otherwise)
- `release() → None`
- `load_state` property — `"released" | "loading" | "loaded"`

Phase 9.1 also adds:
- `Collection.create_partition / drop_partition / list_partitions / has_partition / num_entities / describe`
- `LiteVecDB.has_collection / get_collection_stats`

Write ops take `partition_name` (singular str). Read ops take `partition_names` (plural List[str]).
The optional `expr` parameter on read ops is a Milvus-style scalar filter expression (Phase 8).

## Key Design Decisions

- `_seq` is the global ordering for all override/discard decisions; never depend on call order or file physical order
- MemTable cross-clear is seq-aware (apply_insert / apply_delete are order-independent)
- Insert and Delete are separate data flows → different file types (data Parquet vs delta Parquet)
- Batch delete shares one `_seq`; batch insert assigns independent `_seq` per record
- Manifest is the single source of truth (atomic update via write-tmp + rename, with `.prev` backup)
- All disk files are immutable (create → never modify → delete whole file)
- Four distinct Arrow/Parquet schemas: `wal_data`, `wal_delta`, `data` (Parquet), `delta` (Parquet) — see `schema/arrow_builder.py`
- MVP synchronous flush; async deferred to future
- Single writer per Collection; single process per data_dir (fcntl.flock LOCK file)
- Schema is immutable (no alter table in MVP)
- WAL default `sync_mode="close"` for OOM-restart safety
- Phase 8 filter: parser and evaluator are decoupled via the AST — parser implementation can be swapped (hand-written → ANTLR) without touching type checker / backends
- **Phase 9: Vector index is segment-level** (1:1 with data parquet files), matching the LSM immutable architecture. Compaction creates new merged segments + new indexes; old indexes are dropped. Avoids the FAISS HNSW "no real delete" trap.
- **Phase 9: FAISS-cpu is the default index** because `IDSelectorBitmap` aligns natively with the Phase 8 bitmap pipeline, enabling pre-filter (not post-filter) for combined scalar+vector search
- **Phase 9: BruteForceIndex is a long-lived first-class implementation**, not a placeholder — it's the differential test baseline for FAISS, the fallback when faiss is not installed, and the implementation for small segments below `INDEX_BUILD_THRESHOLD`
- **Phase 9: Load/release state machine** (`released → loading → loaded`) mirrors Milvus behavior. `search/get/query` require `loaded` state (raise `CollectionNotLoadedError` otherwise). Restart defaults to `released` — caller must explicitly `load()`
- **Phase 9: Distance normalization happens inside VectorIndex.search**. FAISS L2/IP/cosine internal conventions don't match the engine's "smaller = more similar" convention; normalization in `FaissHnswIndex.search` keeps the upper-layer behavior identical to `BruteForceIndex` (enforced by differential tests)
- **Phase 10: gRPC adapter only translates, never adds capability**. Unsupported RPCs return `UNIMPLEMENTED` with friendly messages — never silent fail. proto stubs from milvus-io/milvus-proto are committed to repo, not generated at runtime
- Design documents are written in Chinese

## Optional Dependencies (Phase 9 / 10)

```bash
pip install -e ".[dev]"               # base + test deps
pip install -e ".[dev,faiss]"         # add FAISS HNSW (Phase 9)
pip install -e ".[dev,grpc]"          # add gRPC adapter (Phase 10)
pip install -e ".[dev,faiss,grpc]"    # everything

# Run gRPC server (Phase 10)
python -m litevecdb.adapter.grpc --data-dir ./data --port 19530
```
