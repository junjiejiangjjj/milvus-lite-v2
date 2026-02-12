# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteVecDB — a local embedded vector database, designed as a **local version of Milvus**. This repository currently contains **design documents only** (no code yet). The implementation will be in Python.

## Architecture (Two-Layer)

- **Internal Engine**: LSM-Tree style storage with PyArrow in-memory and Parquet on disk. This is the current design focus.
- **gRPC Adapter Layer** (future): Sits on top of the engine to provide Milvus protocol compatibility, allowing pymilvus to connect directly. Handles parameter normalization, expression parsing, and response format wrapping.

## Key Design Documents

- **MVP.md**: Complete MVP design — data model, Collection Schema, four-schema system (WAL vs Parquet), core components (WAL, MemTable, Manifest, DeltaLog, Compaction, Search), read/write paths, internal engine API, disk layout, implementation priorities (P0-P11).
- **write-pipeline.md**: Detailed write path — Insert/Delete flows, flush pipeline, crash safety analysis, concurrency control, MemTable semantics, WAL details, invariants.
- **research.md**: Competitive analysis of LanceDB, Turbopuffer, and Milvus partition design, with comparison tables and design implications.

## Data Hierarchy

DB → Collection → Partition. WAL/MemTable/\_seq are Collection-level shared. Data files are Partition-level isolated (directory per partition).

## Internal Engine API (all inputs are Lists, no single-value normalization)

- `insert(records: List[dict], partition_name="_default") → List[pk]` — upsert semantics (PK exists → overwrite)
- `delete(pks: List, partition_name=None) → int` — None means cross-all-partitions
- `get(pks: List, partition_names=None) → List[dict]`
- `search(vectors: List[list], top_k=10, metric_type="COSINE", partition_names=None) → List[List[dict]]`

Write ops take `partition_name` (singular str). Read ops take `partition_names` (plural List[str]).

## Key Design Decisions

- Insert and Delete are separate data flows → different file types (data Parquet vs delta Parquet)
- Batch delete shares one `_seq`; batch insert assigns independent `_seq` per record
- Manifest is the single source of truth (atomic update via write-tmp + rename)
- All disk files are immutable (create → never modify → delete whole file)
- Documents are written in Chinese
