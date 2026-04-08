"""Collection-level in-memory write buffer.

Internal representation (per modules.md §9.7):
    _insert_batches: list[pa.RecordBatch]              append-only
    _pk_index:       dict[pk → (batch_idx, row_idx, seq)]
    _delete_index:   dict[pk → max_delete_seq]

The list is append-only so we never copy / mutate Arrow buffers on the
write path. Stale rows in older batches are reachable only via _pk_index;
flush() (Phase 3) will dedup by walking _pk_index.values().

**Architectural invariant §1-2**: every cross-buffer decision is keyed on
``_seq``. ``apply_insert`` and ``apply_delete`` are order-independent —
calling them in any order with the same set of (pk, _seq) operations
yields the same final state. This is what makes recovery safe even if it
replays operations out of physical order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa

from litevecdb.schema.arrow_builder import get_primary_field
from litevecdb.schema.types import CollectionSchema


# (batch_idx, row_idx, seq)
_PkPos = Tuple[int, int, int]


class MemTable:
    """Collection-level in-memory write buffer.

    Phase 2 supports:
        - apply_insert / apply_delete (seq-aware, order-independent)
        - get(pk) for point reads
        - size() for flush triggering
        - get_active_records() for search/get during the read path

    Phase 3 will add flush() which dedups via _pk_index and emits Arrow
    Tables ready to write to Parquet.
    """

    def __init__(self, schema: CollectionSchema) -> None:
        self._schema = schema
        self._pk_name = get_primary_field(schema).name

        self._insert_batches: List[pa.RecordBatch] = []
        self._pk_index: Dict[Any, _PkPos] = {}
        self._delete_index: Dict[Any, int] = {}

    # ── write path ──────────────────────────────────────────────

    def apply_insert(self, batch: pa.RecordBatch) -> None:
        """Apply a wal_data RecordBatch.

        Each row is checked against _delete_index and the existing
        _pk_index entry — only rows whose seq is the new maximum take
        effect. Filtered-out rows still live physically in the appended
        batch but are not reachable via the index.
        """
        if batch.num_rows == 0:
            return
        self._validate_insert_schema(batch)

        batch_idx = len(self._insert_batches)
        pk_col = batch.column(self._pk_name)
        seq_col = batch.column("_seq")

        any_kept = False
        for row_idx in range(batch.num_rows):
            pk = pk_col[row_idx].as_py()
            seq = seq_col[row_idx].as_py()

            # seq-aware: a newer delete blocks this insert
            existing_delete = self._delete_index.get(pk)
            if existing_delete is not None and existing_delete >= seq:
                continue

            # seq-aware: a newer insert blocks this insert
            existing_pos = self._pk_index.get(pk)
            if existing_pos is not None and existing_pos[2] >= seq:
                continue

            # take effect
            self._pk_index[pk] = (batch_idx, row_idx, seq)
            # our seq > existing_delete (we passed the check above)
            self._delete_index.pop(pk, None)
            any_kept = True

        if any_kept:
            self._insert_batches.append(batch)

    def apply_delete(self, batch: pa.RecordBatch) -> None:
        """Apply a wal_delta RecordBatch.

        All rows in a delta batch share the same ``_seq``. The batch
        itself is not retained — only the (pk, seq) pairs are folded
        into _delete_index.
        """
        if batch.num_rows == 0:
            return
        self._validate_delete_schema(batch)

        pk_col = batch.column(self._pk_name)
        seq_col = batch.column("_seq")
        # delete batches share one seq; verify by reading first row.
        seq = seq_col[0].as_py()

        for row_idx in range(batch.num_rows):
            pk = pk_col[row_idx].as_py()

            # seq-aware: a newer insert blocks this delete
            existing_pos = self._pk_index.get(pk)
            if existing_pos is not None and existing_pos[2] >= seq:
                continue

            # update delete watermark to the larger seq
            existing_delete = self._delete_index.get(pk, -1)
            if seq > existing_delete:
                self._delete_index[pk] = seq

            # if pk_index entry exists with smaller seq, evict it
            if existing_pos is not None and existing_pos[2] < seq:
                self._pk_index.pop(pk, None)

    # ── read path ───────────────────────────────────────────────

    def get(self, pk_value: Any) -> Optional[dict]:
        """Point read for a single pk.

        Returns the live record dict (without _seq / _partition) or None
        if the pk is unknown or has been deleted.
        """
        pos = self._pk_index.get(pk_value)
        if pos is None:
            return None
        # _pk_index entries are guaranteed to NOT be shadowed by a delete:
        # apply_delete pops the entry, apply_insert pops the delete entry.
        # So we can return directly without re-checking _delete_index.
        batch_idx, row_idx, _ = pos
        batch = self._insert_batches[batch_idx]
        return self._row_to_dict(batch, row_idx)

    def get_active_records(
        self, partition_names: Optional[List[str]] = None
    ) -> List[dict]:
        """Return all live records, optionally filtered by partition.

        Used by Collection.search / Collection.get during the read path.
        Returned dicts are clean (no _seq, no _partition).
        """
        out: List[dict] = []
        partition_filter: Optional[set] = None
        if partition_names is not None:
            partition_filter = set(partition_names)

        for pk, (batch_idx, row_idx, _seq) in self._pk_index.items():
            batch = self._insert_batches[batch_idx]
            if partition_filter is not None:
                partition = batch.column("_partition")[row_idx].as_py()
                if partition not in partition_filter:
                    continue
            out.append(self._row_to_dict(batch, row_idx))
        return out

    def size(self) -> int:
        """Active pk count + tombstone count.

        Used by flush triggering. NOT the physical row count of
        _insert_batches — that includes shadowed rows that no longer
        contribute to the visible state.
        """
        return len(self._pk_index) + len(self._delete_index)

    # ── introspection (test/debug) ──────────────────────────────

    @property
    def pk_name(self) -> str:
        return self._pk_name

    def num_physical_rows(self) -> int:
        """Total rows physically retained in _insert_batches.
        Includes shadowed rows. For test/debug only."""
        return sum(b.num_rows for b in self._insert_batches)

    # ── internal helpers ────────────────────────────────────────

    def _row_to_dict(self, batch: pa.RecordBatch, row_idx: int) -> dict:
        """Extract one row as a dict, stripping _seq and _partition."""
        result = {}
        for name in batch.schema.names:
            if name in ("_seq", "_partition"):
                continue
            result[name] = batch.column(name)[row_idx].as_py()
        return result

    def _validate_insert_schema(self, batch: pa.RecordBatch) -> None:
        names = set(batch.schema.names)
        required = {"_seq", "_partition", self._pk_name}
        missing = required - names
        if missing:
            raise ValueError(
                f"insert batch missing required columns: {sorted(missing)}"
            )

    def _validate_delete_schema(self, batch: pa.RecordBatch) -> None:
        names = set(batch.schema.names)
        required = {"_seq", "_partition", self._pk_name}
        missing = required - names
        if missing:
            raise ValueError(
                f"delete batch missing required columns: {sorted(missing)}"
            )
