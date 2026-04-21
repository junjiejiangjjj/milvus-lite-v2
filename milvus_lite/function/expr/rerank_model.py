"""RerankModelExpr — semantic reranking via external model (rerank stage).

Calls an external rerank provider (e.g., Cohere) to rescore documents
by relevance to a query.  Query texts are bound at chain construction
time; ``ctx.chunk_idx`` selects which query applies to the current chunk.
"""

from __future__ import annotations

from typing import FrozenSet, List

from milvus_lite.function.types import STAGE_RERANK, FuncContext, FunctionExpr


class RerankModelExpr(FunctionExpr):
    """document_text column -> relevance_score column."""

    name = "rerank_model"
    supported_stages: FrozenSet[str] = frozenset({STAGE_RERANK})

    def __init__(self, provider, query_texts: List[str]) -> None:
        self._provider = provider
        self._query_texts = query_texts

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        doc_texts = inputs[0]
        # Select the query for this chunk
        query_text = self._query_texts[ctx.chunk_idx]
        rerank_results = self._provider.rerank(
            query_text, doc_texts, top_n=len(doc_texts)
        )
        scores = [0.0] * len(doc_texts)
        for r in rerank_results:
            scores[r.index] = r.relevance_score
        return [scores]
