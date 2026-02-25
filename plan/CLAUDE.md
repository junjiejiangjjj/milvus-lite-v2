# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteVecDB — a local embedded vector database, designed as a **local version of Milvus**. Implementation is in Python. The code lives in `milvus-lite-v2/litevecdb/`.

## Repository Layout

- **Design docs** (root-level, written in Chinese): `MVP.md`, `write-pipeline.md`, `research.md`, `modules.md`, `wal-design.md`
- **Code repo**: `milvus-lite-v2/` (git root) — design docs are also copied into `milvus-lite-v2/plan/`
- **`modules.md`**: Authoritative module design — contains the complete file-by-file breakdown with per-class/function signatures and responsibilities. Consult this before implementing any module.
- **`wal-design.md`**: Deep-dive on WAL (Arrow IPC Streaming), segment architecture, and search pipeline.

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
2. **`storage/`** — Persistence layer. WAL (Arrow IPC Streaming, dual-file), MemTable (insert_buf + delete_buf with partition routing), DataFile (Parquet read/write), DeltaLog (deleted_map), Manifest (JSON, atomic update via tmp+rename).
3. **`engine/`** — Core logic orchestration. `Collection` class (entry point, `_seq` allocation, CRUD + partition management), Flush pipeline (7 steps), crash Recovery (5 steps), Compaction (Size-Tiered, per-partition).
4. **`search/`** — Vector retrieval. Bitmap pipeline (dedup + delete filtering), distance functions (cosine/L2/IP via NumPy), search executor (collect data → bitmap → vector search → top-k).

Top-level: `db.py` (`LiteVecDB` — multi-collection lifecycle), `constants.py`, `exceptions.py`.

## Data Hierarchy

DB → Collection → Partition. WAL/MemTable/`_seq` are Collection-level shared. Data files are Partition-level isolated (directory per partition).

## Internal Engine API

All inputs are Lists — no single-value normalization:

- `insert(records: List[dict], partition_name="_default") → List[pk]` — upsert semantics
- `delete(pks: List, partition_name=None) → int` — None means cross-all-partitions
- `get(pks: List, partition_names=None) → List[dict]`
- `search(vectors: List[list], top_k=10, metric_type="COSINE", partition_names=None) → List[List[dict]]`

Write ops take `partition_name` (singular str). Read ops take `partition_names` (plural List[str]).

## Key Design Decisions

- Insert and Delete are separate data flows → different file types (data Parquet vs delta Parquet)
- Batch delete shares one `_seq`; batch insert assigns independent `_seq` per record
- Manifest is the single source of truth (atomic update via write-tmp + rename)
- All disk files are immutable (create → never modify → delete whole file)
- Four distinct Arrow/Parquet schemas: `wal_data`, `wal_delta`, `data` (Parquet), `delta` (Parquet) — see `schema/arrow_builder.py`
- Design documents are written in Chinese
