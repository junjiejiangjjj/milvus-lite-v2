"""Rerank subsystem.

Supports RERANK Function type with two modes:

1. **Semantic reranking** (external API): re-scores results using a
   cross-encoder model (e.g. Cohere).
2. **Decay reranking** (local): adjusts scores based on a numeric
   field's proximity to an origin value (gauss/exp/linear curves).

Public exports:
    RerankProvider     — abstract protocol for semantic rerankers
    RerankResult       — single rerank result
    DecayReranker      — local decay reranker
    create_rerank_provider — factory dispatch for semantic providers
"""

from litevecdb.rerank.protocol import RerankProvider, RerankResult
from litevecdb.rerank.decay import DecayReranker
from litevecdb.rerank.factory import create_rerank_provider

__all__ = [
    "RerankProvider",
    "RerankResult",
    "DecayReranker",
    "create_rerank_provider",
]
