"""Collection — engine entry point.

Phase 3 scope:
    - insert(records, partition_name="_default")
    - get(pks, partition_names=None)
    - Synchronous flush triggered when MemTable.size() >= MEMTABLE_SIZE_LIMIT
    - Crash recovery on construction (replays WAL, rebuilds delta_index)
    - WAL + MemTable + Manifest + DeltaIndex

NOT yet:
    - search (Phase 4)
    - delete (Phase 5) — Collection.delete is not exposed, but the
      plumbing (DeleteOp dispatch in _apply, MemTable.apply_delete)
      is in place so Phase 5 just adds the public method.
    - compaction (Phase 6)
    - partition CRUD (Phase 7)

Layering: Collection sits at the top of the engine layer. It is the
only place that knows about Operation dispatch — storage/wal.py and
storage/memtable.py both still take raw RecordBatches. This keeps the
storage layer free of engine-layer types.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import pyarrow as pa

from litevecdb.constants import DEFAULT_PARTITION, MEMTABLE_SIZE_LIMIT
from litevecdb.engine.flush import execute_flush
from litevecdb.engine.operation import DeleteOp, InsertOp, Operation
from litevecdb.engine.recovery import execute_recovery
from litevecdb.exceptions import PartitionNotFoundError
from litevecdb.schema.arrow_builder import (
    build_wal_data_schema,
    build_wal_delta_schema,
    get_primary_field,
    get_vector_field,
)
from litevecdb.schema.types import CollectionSchema
from litevecdb.schema.validation import (
    separate_dynamic_fields,
    validate_record,
    validate_schema,
)
from litevecdb.storage.manifest import Manifest
from litevecdb.storage.memtable import MemTable
from litevecdb.storage.wal import WAL


class Collection:
    """A single Collection — schema + WAL + MemTable + Manifest + DeltaIndex.

    Construction is non-destructive and crash-tolerant:
        1. Load Manifest (with .prev fallback if current is corrupted).
        2. Run recovery — replay any WAL files, rebuild DeltaIndex,
           clean orphan Parquet files.
        3. Allocate a fresh WAL number = max(found WALs + 1, manifest's
           active_wal_number).

    insert() validates fail-fast (no partial state on failure), allocates
    one _seq per row, writes to WAL then MemTable, and triggers a
    synchronous flush if the MemTable hit the size limit.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        schema: CollectionSchema,
    ) -> None:
        validate_schema(schema)

        self._name = name
        self._data_dir = data_dir
        self._schema = schema
        self._pk_name = get_primary_field(schema).name
        self._vector_name = get_vector_field(schema).name

        self._wal_data_schema = build_wal_data_schema(schema)
        self._wal_delta_schema = build_wal_delta_schema(schema)

        os.makedirs(data_dir, exist_ok=True)

        # ── 1. load manifest ────────────────────────────────────
        self._manifest = Manifest.load(data_dir)

        # ── 2. recovery ─────────────────────────────────────────
        self._memtable, self._delta_index, next_wal_number = execute_recovery(
            data_dir=data_dir,
            schema=schema,
            manifest=self._manifest,
        )

        # ── 3. fresh WAL ────────────────────────────────────────
        # next_seq must clear both manifest's recorded seq AND any seq
        # we just learned from WAL replay.
        self._next_seq = max(
            self._manifest.current_seq, self._memtable.max_seq
        ) + 1

        wal_dir = os.path.join(data_dir, "wal")
        self._wal = WAL(
            wal_dir=wal_dir,
            wal_data_schema=self._wal_data_schema,
            wal_delta_schema=self._wal_delta_schema,
            wal_number=next_wal_number,
        )

    # ── public API ──────────────────────────────────────────────

    def insert(
        self,
        records: List[dict],
        partition_name: str = DEFAULT_PARTITION,
    ) -> List[Any]:
        """Insert records into the collection. Returns the list of pks.

        Each record is validated up-front (fail fast — no partial state
        on validation error). After WAL+MemTable apply, if the MemTable
        has hit MEMTABLE_SIZE_LIMIT, a synchronous flush runs before
        returning.
        """
        if not isinstance(records, list):
            raise TypeError(f"records must be a list, got {type(records).__name__}")
        if not records:
            return []

        if not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # 1. validate every record up-front
        for r in records:
            validate_record(r, self._schema)

        # 2. allocate seqs
        seq_start = self._next_seq
        self._next_seq += len(records)
        seqs = list(range(seq_start, seq_start + len(records)))

        # 3. build wal_data RecordBatch
        batch = self._build_wal_data_batch(records, partition_name, seqs)

        # 4. construct Operation and dispatch
        op = InsertOp(partition=partition_name, batch=batch)
        self._apply(op)

        # 5. trigger flush if we hit the size limit
        if self._memtable.size() >= MEMTABLE_SIZE_LIMIT:
            self._trigger_flush()

        return [r[self._pk_name] for r in records]

    def get(
        self,
        pks: List[Any],
        partition_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """Point read.

        Phase 3 reads only the MemTable. Phase 4 will also read disk
        segments and merge results.
        """
        if not isinstance(pks, list):
            raise TypeError(f"pks must be a list, got {type(pks).__name__}")
        out: List[dict] = []
        for pk in pks:
            rec = self._memtable.get(pk)
            if rec is not None:
                out.append(rec)
        return out

    def flush(self) -> None:
        """Force a synchronous flush of the current MemTable.

        No-op if the MemTable is empty.
        """
        if self._memtable.size() == 0:
            return
        self._trigger_flush()

    # ── orchestration ───────────────────────────────────────────

    def _apply(self, op: Operation) -> None:
        """Single write entry point.

        Dispatches Operation to WAL and MemTable. Storage layer methods
        take raw batches (no Operation knowledge), so the dispatch is
        explicit here.
        """
        if isinstance(op, InsertOp):
            self._wal.write_insert(op.batch)
            self._memtable.apply_insert(op.batch)
        else:  # DeleteOp
            self._wal.write_delete(op.batch)
            self._memtable.apply_delete(op.batch)

    def _trigger_flush(self) -> None:
        """Step 1 of the flush pipeline + execute_flush for Steps 2-7.

        Step 1 (here): freeze the current (MemTable, WAL), swap in fresh
        ones on the Collection. Then call execute_flush on the frozen
        pair, which handles disk writes, manifest commit, WAL cleanup.
        """
        # ── Step 1: freeze ──────────────────────────────────────
        frozen_memtable = self._memtable
        frozen_wal = self._wal
        new_wal_number = frozen_wal.number + 1

        # Swap in fresh ones BEFORE running execute_flush so that any
        # subsequent insert calls (in case of async future) hit the new
        # MemTable. In sync mode this is order-preserving anyway.
        self._memtable = MemTable(self._schema)
        wal_dir = os.path.join(self._data_dir, "wal")
        self._wal = WAL(
            wal_dir=wal_dir,
            wal_data_schema=self._wal_data_schema,
            wal_delta_schema=self._wal_delta_schema,
            wal_number=new_wal_number,
        )

        # ── Steps 2-7: execute_flush ────────────────────────────
        execute_flush(
            frozen_memtable=frozen_memtable,
            frozen_wal=frozen_wal,
            data_dir=self._data_dir,
            schema=self._schema,
            manifest=self._manifest,
            delta_index=self._delta_index,
            new_wal_number=new_wal_number,
        )

    # ── batch builders ──────────────────────────────────────────

    def _build_wal_data_batch(
        self,
        records: List[dict],
        partition_name: str,
        seqs: List[int],
    ) -> pa.RecordBatch:
        """Build a RecordBatch matching wal_data_schema.

        Splits dynamic fields into $meta if enable_dynamic_field is set.
        """
        n = len(records)
        cols: dict[str, list] = {
            "_seq": seqs,
            "_partition": [partition_name] * n,
        }

        for f in self._schema.fields:
            cols[f.name] = []

        meta_col: Optional[List[Optional[str]]] = None
        if self._schema.enable_dynamic_field:
            meta_col = []

        for r in records:
            schema_part, meta_json = separate_dynamic_fields(r, self._schema)
            for f in self._schema.fields:
                cols[f.name].append(schema_part.get(f.name))
            if meta_col is not None:
                meta_col.append(meta_json)

        if meta_col is not None:
            cols["$meta"] = meta_col

        return pa.RecordBatch.from_pydict(cols, schema=self._wal_data_schema)

    # ── lifecycle ───────────────────────────────────────────────

    def close(self) -> None:
        """Flush any pending state and shut down the WAL.

        Phase-3 close runs a final flush so the on-disk state is
        consistent with whatever insert calls returned successfully.
        """
        if self._memtable.size() > 0:
            self._trigger_flush()
        else:
            # Even an empty MemTable needs WAL cleanup so we don't leave
            # an empty wal file behind.
            self._wal.close_and_delete()

    # ── introspection ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def schema(self) -> CollectionSchema:
        return self._schema

    @property
    def pk_field(self) -> str:
        return self._pk_name

    def count(self) -> int:
        """Number of live records in the MemTable.

        NOTE: this is the in-memory count only. Phase 4 will add a
        full collection count that includes flushed Parquet files.
        """
        return self._memtable.size()
