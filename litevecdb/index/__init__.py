"""Vector index subsystem (Phase 9).

Each VectorIndex is bound 1:1 to a Segment (one .idx sidecar per data
Parquet file). Indexes are immutable — compaction creates a new merged
segment + a new index, dropping the old ones.

Public exports:
    VectorIndex     — abstract protocol (index/protocol.py)
    BruteForceIndex — NumPy implementation (Phase 9.2)
                      Long-lived: differential test baseline + faiss
                      fallback + small-segment chosen impl.

Phase 9.5 will add ``FaissHnswIndex`` here behind a ``try: import faiss``
guard so installations without faiss-cpu still get the BruteForce path.
"""

from litevecdb.index.brute_force import BruteForceIndex
from litevecdb.index.protocol import VectorIndex

__all__ = ["VectorIndex", "BruteForceIndex"]
