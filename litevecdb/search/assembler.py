"""Search candidate assembler.

Merges segments (on-disk Parquet caches) and the live MemTable into a
single candidate set ready for the bitmap pipeline + distance computation.

This is the only module in the search/ package that knows about both
storage layer types (Segment, MemTable) AND the filter subsystem —
keeping the rest of the search package storage-agnostic AND filter-
agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from litevecdb.search.filter.semantic import CompiledExpr
    from litevecdb.storage.memtable import MemTable
    from litevecdb.storage.segment import Segment


def assemble_candidates(
    segments: Iterable["Segment"],
    memtable: "MemTable",
    vector_field: str,
    partition_names: Optional[List[str]] = None,
    filter_compiled: Optional["CompiledExpr"] = None,
) -> Tuple[List[Any], np.ndarray, np.ndarray, List[dict], Optional[np.ndarray]]:
    """Build the unified candidate arrays.

    Returns:
        all_pks:     list of pk values (length N total)
        all_seqs:    np.ndarray[uint64], shape (N,)
        all_vectors: np.ndarray[float32], shape (N, dim)
        all_records: list of dicts (length N), in the same order
        filter_mask: np.ndarray[bool] (length N) or None — only set if
                     filter_compiled was provided

    Order is "segments first, then MemTable". This is irrelevant for
    correctness — the bitmap pipeline does dedup by max-seq — but it
    keeps the candidate layout deterministic for testing AND ensures
    the filter mask aligns with the candidate arrays.
    """
    # Local import to avoid creating an import cycle at module load time
    # — search/filter doesn't import storage, but storage tests can be
    # imported before search/filter sometimes; lazy import is safest.
    from litevecdb.search.filter.eval import evaluate as filter_evaluate

    partition_filter = set(partition_names) if partition_names else None

    pk_chunks: List[List[Any]] = []
    seq_chunks: List[np.ndarray] = []
    vec_chunks: List[np.ndarray] = []
    record_chunks: List[List[dict]] = []
    filter_chunks: List[np.ndarray] = []

    # ── segments ────────────────────────────────────────────────
    for seg in segments:
        if partition_filter is not None and seg.partition not in partition_filter:
            continue
        if seg.num_rows == 0:
            continue
        pk_chunks.append(list(seg.pks))
        seq_chunks.append(seg.seqs)
        vec_chunks.append(seg.vectors)
        record_chunks.append(
            [seg.row_to_dict(i) for i in range(seg.num_rows)]
        )
        if filter_compiled is not None:
            mask = filter_evaluate(filter_compiled, seg.table)
            filter_chunks.append(mask.to_numpy(zero_copy_only=False))

    # ── memtable ────────────────────────────────────────────────
    mt_pks, mt_seqs, mt_vecs, mt_records = memtable.to_search_arrays(
        vector_field=vector_field,
        partition_names=partition_names,
    )
    if mt_pks:
        pk_chunks.append(mt_pks)
        seq_chunks.append(mt_seqs)
        vec_chunks.append(mt_vecs)
        record_chunks.append(mt_records)
        if filter_compiled is not None:
            mt_table = memtable.to_arrow_table(partition_names=partition_names)
            mask = filter_evaluate(filter_compiled, mt_table)
            filter_chunks.append(mask.to_numpy(zero_copy_only=False))

    # ── concatenate ─────────────────────────────────────────────
    if not pk_chunks:
        empty_filter = np.zeros(0, dtype=bool) if filter_compiled is not None else None
        return (
            [],
            np.zeros(0, dtype=np.uint64),
            np.zeros((0, 0), dtype=np.float32),
            [],
            empty_filter,
        )

    all_pks: List[Any] = []
    for chunk in pk_chunks:
        all_pks.extend(chunk)

    all_seqs = np.concatenate(seq_chunks)

    # All vector chunks should have the same dim. Use the first non-empty.
    dim = next((v.shape[1] for v in vec_chunks if v.shape[1] > 0), 0)
    if dim == 0:
        all_vectors = np.zeros((len(all_pks), 0), dtype=np.float32)
    else:
        normalized_chunks = []
        for v in vec_chunks:
            if v.size == 0:
                continue
            if v.shape[1] != dim:
                raise ValueError(
                    f"vector dim mismatch across candidate sources: {v.shape[1]} vs {dim}"
                )
            normalized_chunks.append(v.astype(np.float32, copy=False))
        all_vectors = np.concatenate(normalized_chunks, axis=0) if normalized_chunks else np.zeros((0, dim), dtype=np.float32)

    all_records: List[dict] = []
    for chunk in record_chunks:
        all_records.extend(chunk)

    if filter_compiled is not None:
        filter_mask = np.concatenate(filter_chunks) if filter_chunks else np.zeros(0, dtype=bool)
    else:
        filter_mask = None

    return all_pks, all_seqs, all_vectors, all_records, filter_mask
