"""Collection — engine entry point.

Phase 2 scope (per roadmap):
    - insert(records, partition_name="_default")
    - get(pks, partition_names=None)
    - _alloc_seq + _apply orchestration
    - WAL + MemTable + minimal Manifest

NOT yet:
    - flush, recovery, search, delete, compaction, partition CRUD
    - These land in Phase 3-7.

Layering: Collection sits at the top of the engine layer. It is the only
place that knows about Operation dispatch — storage/wal.py and
storage/memtable.py both still take raw RecordBatches. This keeps the
storage layer free of engine-layer types.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import pyarrow as pa

from litevecdb.constants import DEFAULT_PARTITION
from litevecdb.engine.operation import InsertOp, Operation
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
    """A single Collection — schema + WAL + MemTable + Manifest.

    Phase 2 supports insert/get only. Construction is non-destructive:
    if data_dir already has a manifest, the existing one is loaded; if
    not, a fresh state with the _default partition is created.

    Phase-2 limitations to know about:
        - No flush. MemTable grows unbounded; this is intentional for
          the M2 milestone (we're verifying the write→memory→read path).
        - No recovery. Restarting the process loses all in-memory state.
          Phase 3 will add recovery + flush.
        - Only the _default partition is supported as a write target.
        - No delete (Phase 5).
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

        # Manifest: load if present, else fresh.
        self._manifest = Manifest.load(data_dir)

        # Seq counter — Phase 2 starts from manifest.current_seq.
        # Phase 3 recovery will bump this past any seqs found in WAL.
        self._next_seq = self._manifest.current_seq + 1

        # MemTable.
        self._memtable = MemTable(schema)

        # WAL: a fresh round per Collection construction. Phase 3 will
        # tie this to the manifest's active_wal_number.
        wal_dir = os.path.join(data_dir, "wal")
        wal_number = (self._manifest.active_wal_number or 0) + 1
        self._wal = WAL(
            wal_dir=wal_dir,
            wal_data_schema=self._wal_data_schema,
            wal_delta_schema=self._wal_delta_schema,
            wal_number=wal_number,
        )

    # ── public API ──────────────────────────────────────────────

    def insert(
        self,
        records: List[dict],
        partition_name: str = DEFAULT_PARTITION,
    ) -> List[Any]:
        """Insert records into the collection. Returns the list of pks.

        Each record is validated, dynamic fields are separated to $meta
        (if enabled), and a unique _seq is allocated per record. The
        whole batch is then flushed through WAL → MemTable.
        """
        if not isinstance(records, list):
            raise TypeError(f"records must be a list, got {type(records).__name__}")
        if not records:
            return []

        if not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # 1. validate every record up-front (fail fast, no partial state)
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

        # 5. return pks (in the same order as input)
        return [r[self._pk_name] for r in records]

    def get(
        self,
        pks: List[Any],
        partition_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """Point read. Returns records in the same order as ``pks``;
        missing pks are skipped (NOT padded with None).

        Phase 2: only reads MemTable. Phase 4 will also read disk segments.
        """
        if not isinstance(pks, list):
            raise TypeError(f"pks must be a list, got {type(pks).__name__}")
        # partition_names ignored in Phase 2 — only _default exists for writes.
        out: List[dict] = []
        for pk in pks:
            rec = self._memtable.get(pk)
            if rec is not None:
                out.append(rec)
        return out

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
        # Per-row schema field extraction.
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
        """Close the WAL.

        Phase 2 has no flush, so close just shuts down the WAL writer.
        The MemTable is dropped with the Collection instance.
        """
        # close_and_delete also deletes the WAL files. For Phase 2 with no
        # recovery, that's fine — restarting will lose memory state anyway,
        # and there's nothing on disk worth keeping. Phase 3 will replace
        # this with proper flush + recovery.
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
        """Phase-2 helper: number of live records in the MemTable."""
        return self._memtable.size()
