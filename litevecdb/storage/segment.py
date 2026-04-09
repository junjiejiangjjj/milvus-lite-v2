"""In-memory cache for one immutable data Parquet file.

Loaded once and never invalidated (the underlying Parquet is immutable —
modules.md architectural invariant §4). Compaction (Phase 6) drops the
old segment from the cache and adds the new merged one.

Why pre-extract numpy arrays at load time:
    - Search hot path needs (N, dim) float32 vectors for distance
      computation. PyArrow's FixedSizeListArray needs reshaping every
      access — doing it once at load amortizes the cost across all
      future searches.
    - pks and seqs are used by the bitmap pipeline (dedup + tombstone
      check), also per-search.

The original pa.Table is retained so that returning entity fields
(non-vector columns) for top-k results doesn't require a second read.

Phase 9.2: Each Segment may carry an attached VectorIndex. The index
is bound 1:1 to the segment and shares its lifetime — when the segment
is evicted (compaction, drop_partition), the index is dropped with it.
The index is None until Collection.load() (or a flush/compaction hook)
attaches one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pyarrow as pa

from litevecdb.storage.data_file import read_data_file

if TYPE_CHECKING:
    from litevecdb.index.protocol import VectorIndex


class Segment:
    """A single data Parquet file, loaded into memory.

    Public state (read-only):
        file_path: absolute path of the source Parquet
        partition: name of the partition this segment belongs to
        pks:       list of pk values (Python list, not numpy — pk dtype
                   may be string or int and a Python list is the simplest
                   uniform handling)
        seqs:      np.ndarray[uint64], shape (N,)
        vectors:   np.ndarray[float32], shape (N, dim)
        table:     original pa.Table for entity-field extraction
        pk_to_row: {pk_value: row_index} for O(1) point reads
    """

    __slots__ = (
        "file_path",
        "partition",
        "pks",
        "seqs",
        "vectors",
        "table",
        "pk_to_row",
        "index",
        "_pk_field",
        "_vector_field",
    )

    def __init__(
        self,
        file_path: str,
        partition: str,
        pk_field: str,
        vector_field: str,
        pks: List[Any],
        seqs: np.ndarray,
        vectors: np.ndarray,
        table: pa.Table,
    ) -> None:
        self.file_path = file_path
        self.partition = partition
        self._pk_field = pk_field
        self._vector_field = vector_field
        self.pks = pks
        self.seqs = seqs
        self.vectors = vectors
        self.table = table
        self.pk_to_row: Dict[Any, int] = {pk: i for i, pk in enumerate(pks)}
        # Phase 9.2: optional attached VectorIndex. None until Collection
        # load() (or flush/compaction hooks) attaches one.
        self.index: Optional["VectorIndex"] = None

    # ── factory ─────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        file_path: str,
        partition: str,
        pk_field: str,
        vector_field: str,
    ) -> "Segment":
        """Load a Parquet file from *file_path* into a Segment."""
        table = read_data_file(file_path)
        pks = table.column(pk_field).to_pylist()
        seqs = np.asarray(table.column("_seq").to_pylist(), dtype=np.uint64)
        vectors = _extract_vector_array(table.column(vector_field))
        return cls(
            file_path=file_path,
            partition=partition,
            pk_field=pk_field,
            vector_field=vector_field,
            pks=pks,
            seqs=seqs,
            vectors=vectors,
            table=table,
        )

    # ── point read ──────────────────────────────────────────────

    def find_row(self, pk_value: Any) -> Optional[int]:
        """Return the row index for *pk_value*, or None."""
        return self.pk_to_row.get(pk_value)

    def row_to_dict(self, row_idx: int) -> dict:
        """Materialize a row as a dict (excluding _seq).

        Used by Collection.get() and search executor's result builder.
        """
        result: dict = {}
        for name in self.table.schema.names:
            if name == "_seq":
                continue
            result[name] = self.table.column(name)[row_idx].as_py()
        return result

    # ── index lifecycle (Phase 9.2) ─────────────────────────────

    def attach_index(self, index: "VectorIndex") -> None:
        """Attach a built or loaded VectorIndex to this segment.

        Idempotent — replaces any existing index. Used by:
            - Collection.load() (Phase 9.3)
            - flush.execute_flush() index hook (Phase 9.4)
            - compaction.run_compaction() index hook (Phase 9.4)
        """
        self.index = index

    def release_index(self) -> None:
        """Drop the index reference. Memory is freed when GC collects.

        Called by Collection.release() and during drop_partition /
        drop_index. Calling on a segment with no index is a no-op.
        """
        self.index = None

    # ── introspection ───────────────────────────────────────────

    @property
    def num_rows(self) -> int:
        return len(self.pks)

    @property
    def vector_dim(self) -> int:
        return self.vectors.shape[1] if self.vectors.size > 0 else 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_vector_array(arr: pa.ChunkedArray) -> np.ndarray:
    """Convert a FixedSizeList<float32, dim> column to a (N, dim) numpy array.

    Handles ChunkedArray (multi-chunk) by concatenating, since pyarrow
    parquet files may load with multiple chunks.
    """
    # Combine chunks if necessary.
    if isinstance(arr, pa.ChunkedArray):
        if arr.num_chunks == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if arr.num_chunks == 1:
            arr = arr.chunk(0)
        else:
            arr = arr.combine_chunks()

    if not isinstance(arr.type, pa.FixedSizeListType):
        raise ValueError(
            f"vector column must be FixedSizeList, got {arr.type}"
        )

    n = len(arr)
    dim = arr.type.list_size
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)

    # arr.values is the underlying flat float32 array (length n * dim)
    flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return flat.reshape(n, dim)
