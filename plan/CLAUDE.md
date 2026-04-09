# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteVecDB — a local embedded vector database, designed as a **local version of Milvus**. Implementation is in Python. The code lives in `milvus-lite-v2/litevecdb/`.

## Repository Layout

- **Design docs** (root-level, written in Chinese): `MVP.md`, `write-pipeline.md`, `research.md`, `modules.md`, `wal-design.md`, `filter-design.md`, `roadmap.md`
- **Code repo**: `milvus-lite-v2/` (git root) — design docs are also copied into `milvus-lite-v2/plan/`
- **`modules.md`**: Authoritative module design — file-by-file breakdown with per-class/function signatures. Consult before implementing any module. §9.19-9.28 cover the Phase 8 filter subsystem.
- **`wal-design.md`**: Deep-dive on WAL (Arrow IPC Streaming), segment architecture, and search pipeline.
- **`filter-design.md`**: Deep-dive on the Phase 8 scalar filter subsystem (grammar, AST, three-stage compilation, dual backend).
- **`roadmap.md`**: Phased implementation plan from current state through MVP. Phases 0-7 are landed; Phase 8 (filter) has subphases F1/F2/F3.

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

- **Internal Engine**: LSM-Tree style storage with PyArrow in-memory and Parquet on disk. This is the current design focus.
- **gRPC Adapter Layer** (future): Sits on top of the engine to provide Milvus protocol compatibility, allowing pymilvus to connect directly.

## Code Structure (`litevecdb/`)

Four packages, layered bottom-up:

1. **`schema/`** — Data model & type system. `DataType` enum, `FieldSchema`, `CollectionSchema`, record validation, Arrow schema builders (4 variants: data/delta/wal_data/wal_delta), schema.json persistence.
2. **`storage/`** — Persistence layer. WAL (Arrow IPC Streaming, dual-file), MemTable (RecordBatch list + pk_index + delete_index, seq-aware), DataFile / DeltaFile (Parquet IO), DeltaIndex (in-memory tombstone map + gc_below), Segment (immutable Parquet cache), Manifest (JSON + .prev backup, atomic via tmp+rename).
3. **`engine/`** — Core logic orchestration. `Collection` class (entry point, `_seq` allocation, insert/delete/get/search/query, _apply orchestration), Operation abstraction (InsertOp/DeleteOp), Flush pipeline (7 steps, sync), Recovery (5 steps), Compaction (Size-Tiered + tombstone GC).
4. **`search/`** — Vector retrieval. Bitmap pipeline (dedup + delete + scalar filter), distance functions (cosine/L2/IP via NumPy), assembler (segments + memtable → numpy + filter mask), executor (top-k). **`search/filter/`** subpackage (Phase 8): tokenizer + Pratt parser + semantic check + dual backend (pyarrow.compute primary, Python row-wise fallback) for Milvus-style scalar expressions.

Top-level: `db.py` (`LiteVecDB` — multi-collection lifecycle), `constants.py`, `exceptions.py`.

## Data Hierarchy

DB → Collection → Partition. WAL/MemTable/`_seq` are Collection-level shared. Data files are Partition-level isolated (directory per partition).

## Internal Engine API

All inputs are Lists — no single-value normalization:

- `insert(records: List[dict], partition_name="_default") → List[pk]` — upsert semantics
- `delete(pks: List, partition_name=None) → int` — None means cross-all-partitions
- `get(pks: List, partition_names=None, expr=None) → List[dict]`
- `search(query_vectors: List[list], top_k=10, metric_type="COSINE", partition_names=None, expr=None) → List[List[dict]]`
- `query(expr: str, output_fields=None, partition_names=None, limit=None) → List[dict]` — Phase 8, pure scalar query

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
- Design documents are written in Chinese
