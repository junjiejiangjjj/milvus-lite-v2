"""Bitmap pipeline — produce a boolean mask of "valid" search candidates.

Two filters (in order):
    1. Dedup by max-seq: when the same pk appears multiple times in the
       candidate set (e.g. across multiple data files due to upsert),
       only the row with the largest _seq is kept.
    2. Tombstone filter: rows whose pk has a delete entry in delta_index
       with a strictly larger seq are dropped.

Future Phase 6+: scalar predicate filtering will hook in here.

Performance note: this is O(N) Python with dict lookups. For 1M
candidates it's ~1s, which is fine for an embedded MVP. Vectorizing
with numpy would help only if pks are uniformly typed (all int or all
str); for the current API the dict approach is the simplest correct path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import numpy as np

if TYPE_CHECKING:
    from litevecdb.storage.delta_index import DeltaIndex


def build_valid_mask(
    all_pks: List[Any],
    all_seqs: np.ndarray,
    delta_index: "DeltaIndex",
) -> np.ndarray:
    """Return a boolean mask over the candidate rows.

    Args:
        all_pks: list of pk values, length N. Python list (not numpy)
            because pk dtype may be string or int.
        all_seqs: np.ndarray[uint64], shape (N,)
        delta_index: tombstone source.

    Returns:
        np.ndarray[bool] shape (N,). True means "this row is the latest
        live version of its pk and is not deleted".
    """
    n = len(all_pks)
    if n == 0:
        return np.zeros(0, dtype=bool)

    if all_seqs.shape[0] != n:
        raise ValueError(
            f"all_pks ({n}) and all_seqs ({all_seqs.shape[0]}) must have the same length"
        )

    # Step 1: per-pk max seq
    max_seqs: dict[Any, int] = {}
    for i in range(n):
        pk = all_pks[i]
        seq = int(all_seqs[i])
        existing = max_seqs.get(pk)
        if existing is None or seq > existing:
            max_seqs[pk] = seq

    # Step 2: build mask — keep rows whose seq is the max for their pk AND
    # whose pk is not tombstoned with a larger seq.
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        pk = all_pks[i]
        seq = int(all_seqs[i])
        if seq != max_seqs[pk]:
            continue  # an older version of this pk exists later in the array
        if delta_index.is_deleted(pk, seq):
            continue
        mask[i] = True
    return mask
