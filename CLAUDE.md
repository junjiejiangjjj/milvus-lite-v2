# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteVecDB — a local embedded vector database, designed as a **local version of Milvus**. Pure Python, two-layer plan: an LSM-style internal engine (current focus) and a future gRPC adapter that will speak the Milvus protocol so pymilvus can connect directly.

## Design Docs (read before implementing)

Authoritative design lives in `plan/`, written in **Chinese**:

- **`plan/modules.md`** — file-by-file module breakdown with per-class/function signatures and responsibilities. Consult this before implementing or modifying any module.
- **`plan/wal-design.md`** — deep-dive on WAL (Arrow IPC Streaming), segment architecture, and the search pipeline.
- `plan/MVP.md`, `plan/write-pipeline.md`, `plan/research.md` — scope, write flow, and background research.

These files are mirrored from the parent `lite-v2/` directory; treat `plan/` as the in-repo source of truth.

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

# Coverage
pytest --cov=litevecdb
```

Build system: Hatchling. Runtime deps: `pyarrow>=15.0`, `numpy>=1.24`. Dev: `pytest>=8.0`, `pytest-cov>=5.0`. Requires Python >=3.10.

## Code Layout (`litevecdb/`)

Four packages, layered bottom-up. **The codebase is in early implementation** — most modules are still stubs; consult `plan/modules.md` for the target shape before adding code.

1. **`schema/`** — Data model & type system. Implemented: `types.py` (`DataType`, `FieldSchema`, `CollectionSchema`, `TYPE_MAP`) and `arrow_builder.py` (four schema builders: `data`, `delta`, `wal_data`, `wal_delta`).
2. **`storage/`** — Persistence layer. Implemented: `wal.py` (Arrow IPC Streaming, dual-file with lazy init, recovery tolerant of truncated tail). Planned: MemTable, DataFile, DeltaLog, Manifest.
3. **`engine/`** — Core orchestration. Planned: `Collection` (entry point, `_seq` allocation, CRUD, partition management), Flush pipeline, crash Recovery, Size-Tiered Compaction.
4. **`search/`** — Vector retrieval. Planned: bitmap pipeline (dedup + delete filtering), distance functions (cosine/L2/IP via NumPy), top-k executor.

Top-level: `db.py` (`LiteVecDB` — currently a stub for the multi-collection lifecycle), `constants.py` (size limits, file-name templates, partition sentinels), `exceptions.py`.

## Architectural Invariants

These hold across the engine and should guide every change:

- **Data hierarchy**: DB → Collection → Partition. WAL, MemTable, and the `_seq` counter are **Collection-level** shared state. Data files are **Partition-level** isolated (one directory per partition).
- **Insert vs Delete are separate flows** with different file types: data Parquet vs delta Parquet, plus matching WAL variants.
- **`_seq` allocation**: a batch insert assigns an independent `_seq` per record; a batch delete shares one `_seq` for the whole batch.
- **Manifest is the single source of truth**, atomically updated via write-tmp + rename.
- **All disk files are immutable**: create → never modify → delete the whole file. No in-place edits.
- **Four distinct Arrow/Parquet schemas** — see `litevecdb/schema/arrow_builder.py`:
  - `data` (Parquet): `_seq` + user fields + optional `$meta`
  - `delta` (Parquet): `{pk}` + `_seq`
  - `wal_data` (Arrow IPC): `_seq` + `_partition` + user fields + optional `$meta`
  - `wal_delta` (Arrow IPC): `{pk}` + `_seq` + `_partition`
- **WAL files are paired**: each round produces `wal_data_{N}.arrow` + `wal_delta_{N}.arrow`; both are deleted together after a successful flush. Writers are lazily initialized so unused files are never created.
- **WAL recovery tolerates truncation**: `_read_wal_file` keeps batches read before an `ArrowInvalid` and returns `[]` for missing/severely corrupted files.

## Internal Engine API (target shape)

All inputs are Lists — no single-value normalization:

- `insert(records: List[dict], partition_name="_default") → List[pk]` — upsert semantics
- `delete(pks: List, partition_name=None) → int` — `None` means cross-all-partitions
- `get(pks: List, partition_names=None) → List[dict]`
- `search(vectors: List[list], top_k=10, metric_type="COSINE", partition_names=None) → List[List[dict]]`

**Convention**: write ops take `partition_name` (singular `str`); read ops take `partition_names` (plural `List[str]`).

## Constants & File Naming

Defined in `litevecdb/constants.py`:

- `MEMTABLE_SIZE_LIMIT = 10_000`
- Compaction: `MAX_DATA_FILES = 32`, Size-Tiered with `COMPACTION_BUCKET_BOUNDARIES = [1MB, 10MB, 100MB]`, `COMPACTION_MIN_FILES_PER_BUCKET = 4`
- File templates use `SEQ_FORMAT_WIDTH = 6` zero-padded numbers: `data_{min}_{max}.parquet`, `delta_{min}_{max}.parquet`, `wal_data_{N}.arrow`, `wal_delta_{N}.arrow`
- Partition sentinels: `DEFAULT_PARTITION = "_default"`, `ALL_PARTITIONS = "_all"`
