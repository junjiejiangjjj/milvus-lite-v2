# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

milvus-lite — lightweight version of Milvus for local development and testing. Pure-Python implementation. The code lives in `milvus_lite/`.

## Repository Layout

- **Design docs** (`docs/`, written in English): `MVP.md`, `write-pipeline.md`, `modules.md`, `wal-design.md`, `filter-design.md`, `index-design.md`, `grpc-adapter-design.md`, `fts-design.md`, `roadmap.md`, `search-iterator-design.md`
- **`modules.md`**: Authoritative module design — file-by-file breakdown with per-class/function signatures.
- **`wal-design.md`**: Deep-dive on WAL (Arrow IPC Streaming), segment architecture, and search pipeline.
- **`filter-design.md`**: Deep-dive on the scalar filter subsystem (grammar, AST, three-stage compilation, three backends).
- **`index-design.md`**: Deep-dive on the vector index subsystem (FAISS HNSW, segment-level binding, load/release state machine).
- **`grpc-adapter-design.md`**: Deep-dive on the gRPC adapter (proto cropping, RPC mapping, FieldData transposition).
- **`fts-design.md`**: Deep-dive on full text search (BM25, Analyzer, SparseInvertedIndex, text_match filter).
- **`roadmap.md`**: Phased implementation plan.

## Development Commands

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
pytest --cov=milvus_lite
```

Build system: Hatchling. Dependencies: `pyarrow>=15.0`, `numpy>=1.24`, `faiss-cpu>=1.7.4`, `pymilvus>=2.4`, `grpcio>=1.50`. Requires Python >=3.10.

## Architecture (Two-Layer)

- **Internal Engine**: LSM-Tree style storage with PyArrow in-memory and Parquet on disk. Per-segment vector indexes (FAISS HNSW).
- **gRPC Adapter Layer**: Sits on top of the engine to provide Milvus protocol compatibility, allowing pymilvus to connect directly without code changes.

## Code Structure (`milvus_lite/`)

Seven packages, layered bottom-up:

1. **`schema/`** — Data model & type system. `DataType` enum (incl. `SPARSE_FLOAT_VECTOR`), `FieldSchema`, `CollectionSchema`, `Function`/`FunctionType`, record validation, Arrow schema builders (4 variants: data/delta/wal_data/wal_delta), schema.json persistence.
2. **`storage/`** — Persistence layer. WAL (Arrow IPC Streaming, dual-file), MemTable (RecordBatch list + pk_index + delete_index, seq-aware), DataFile / DeltaFile (Parquet IO), DeltaIndex (in-memory tombstone map + gc_below), Segment (immutable Parquet cache + optional VectorIndex), Manifest (JSON + .prev backup, atomic via tmp+rename).
3. **`engine/`** — Core logic orchestration. `Collection` class (entry point, `_seq` allocation, insert/delete/get/search/query, create_index/drop_index/load/release + load_state machine), Operation abstraction, Flush pipeline, Recovery, Compaction (Size-Tiered + tombstone GC + index lifecycle).
4. **`search/`** — Vector retrieval. Bitmap pipeline (dedup + delete + scalar filter), distance functions (cosine/L2/IP via NumPy), assembler (segments + memtable → numpy + filter mask), executor (top-k + index-aware path). **`search/filter/`** subpackage: tokenizer + Pratt parser + semantic check + three backends (arrow/hybrid/python) for Milvus-style scalar expressions.
5. **`index/`** — Vector index abstraction. `VectorIndex` ABC, `IndexSpec`, `BruteForceIndex` (NumPy, fallback + differential baseline), `FaissHnswIndex` (FAISS HNSW + IDSelectorBitmap), IVF_FLAT, IVF_SQ8, HNSW_SQ, SparseInvertedIndex. Bound to segments 1:1 via `.idx` files.
6. **`analyzer/`** — Text analysis for full text search. `Analyzer` ABC, `StandardAnalyzer` (regex tokenizer), `JiebaAnalyzer` (optional Chinese), `create_analyzer` factory, `term_to_id` hash function.
7. **`adapter/`** — Protocol translation. `adapter/grpc/` provides `MilvusServicer` mapping pymilvus RPCs to engine API. Translators handle Milvus FieldData ↔ records transposition, schema, search, results, expressions, index params.

Top-level: `db.py` (`MilvusLite` — multi-collection lifecycle), `server_manager.py` (pymilvus integration), `constants.py`, `exceptions.py`.

## Data Hierarchy

DB → Collection → Partition. WAL/MemTable/`_seq` are Collection-level shared. Data files are Partition-level isolated (directory per partition).

## Internal Engine API

All inputs are Lists — no single-value normalization:

- `insert(records: List[dict], partition_name="_default") → List[pk]` — upsert semantics
- `delete(pks: List, partition_name=None) → int` — None means cross-all-partitions
- `get(pks: List, partition_names=None, expr=None) → List[dict]`
- `search(query_vectors: List[list], top_k=10, metric_type="COSINE", partition_names=None, expr=None, output_fields=None) → List[List[dict]]`
- `query(expr: str, output_fields=None, partition_names=None, limit=None) → List[dict]`

Index lifecycle:
- `create_index(field_name: str, index_params: dict) → None`
- `drop_index(field_name: str) → None`
- `has_index() → bool`
- `get_index_info() → Optional[dict]`
- `load() → None` — required before `search/get/query` (raises CollectionNotLoadedError otherwise)
- `release() → None`
- `load_state` property — `"released" | "loading" | "loaded"`

Write ops take `partition_name` (singular str). Read ops take `partition_names` (plural List[str]).

## Key Design Decisions

- `_seq` is the global ordering for all override/discard decisions; never depend on call order or file physical order
- Insert and Delete are separate data flows → different file types (data Parquet vs delta Parquet)
- Manifest is the single source of truth (atomic update via write-tmp + rename, with `.prev` backup)
- All disk files are immutable (create → never modify → delete whole file)
- Single writer per Collection; single process per data_dir (fcntl.flock LOCK file)
- Vector index is segment-level (1:1 with data parquet files), matching the LSM immutable architecture
- FAISS-cpu is the default index because `IDSelectorBitmap` aligns natively with the bitmap pipeline
- BruteForceIndex is a long-lived first-class implementation (fallback + differential baseline + small segments)
- Load/release state machine (`released → loading → loaded`) mirrors Milvus behavior
- gRPC adapter only translates, never adds capability. Unsupported RPCs return `UNIMPLEMENTED`
- Design documents are written in English

## pymilvus Integration

```python
# pymilvus detects .db URI → imports server_manager_instance
from milvus_lite.server_manager import server_manager_instance
uri = server_manager_instance.start_and_get_uri("./demo.db")
# → starts gRPC server in background thread, returns http://127.0.0.1:{port}
```

```bash
# Start gRPC server directly
milvus-lite-server --data-dir ./data --port 19530
```
