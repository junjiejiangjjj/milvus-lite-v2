"""Chain builders — construct FuncChains from schema functions.

``build_ingestion_chain`` creates a chain for insert-time auto-generation
(BM25, TEXT_EMBEDDING).

``build_rerank_chain`` creates a chain for search-time reranking from
schema-level RERANK/DECAY functions (4 patterns: RRF/Weighted/Decay/Model).

``build_hybrid_rerank_chain`` creates a chain for HybridSearch from
request-level rank_params (RRF/Weighted, no schema functions involved).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from milvus_lite.function.chain import FuncChain
from milvus_lite.function.types import (
    ID_FIELD,
    SCORE_FIELD,
    GROUP_SCORE_FIELD,
    STAGE_INGESTION,
    STAGE_RERANK,
)


# ── Ingestion ────────────────────────────────────────────────


def build_ingestion_chain(
    schema,
    field_by_name: Dict[str, Any],
) -> Optional[FuncChain]:
    """Build an ingestion chain from ``schema.functions``.

    Iterates all functions in the schema and adds those that support
    the ingestion stage to the chain in declaration order.

    Args:
        schema: A :class:`CollectionSchema` with ``functions`` attribute.
        field_by_name: Mapping from field name to :class:`FieldSchema`.

    Returns:
        A :class:`FuncChain` or ``None`` when no ingestion functions exist.
    """
    if not schema.functions:
        return None

    from milvus_lite.schema.types import FunctionType

    chain = FuncChain("ingestion", STAGE_INGESTION)
    has_steps = False

    for func in schema.functions:
        if func.function_type == FunctionType.BM25:
            from milvus_lite.analyzer.factory import create_analyzer
            from milvus_lite.function.expr.bm25_expr import BM25Expr

            in_name = func.input_field_names[0]
            out_name = func.output_field_names[0]
            in_field = field_by_name[in_name]
            analyzer = create_analyzer(in_field.analyzer_params)
            chain.map(BM25Expr(analyzer), [in_name], [out_name])
            has_steps = True

        elif func.function_type == FunctionType.TEXT_EMBEDDING:
            from milvus_lite.embedding.factory import create_embedding_provider
            from milvus_lite.function.expr.embedding_expr import EmbeddingExpr

            in_name = func.input_field_names[0]
            out_name = func.output_field_names[0]
            provider = create_embedding_provider(func.params)
            chain.map(EmbeddingExpr(provider), [in_name], [out_name])
            has_steps = True

    return chain if has_steps else None


# ── Rerank (schema-level functions) ──────────────────────────


def build_rerank_chain(
    schema,
    search_params: Dict[str, Any],
    search_metrics: Optional[List[str]] = None,
) -> Optional[FuncChain]:
    """Build a rerank chain from RERANK/DECAY functions in ``schema.functions``.

    4 chain patterns (aligned with Milvus ``rerank_builder.go``):

    - **RRF**:      ``Merge(rrf) -> Sort -> Limit -> [RoundDecimal] -> Select``
    - **Weighted**: ``Merge(weighted) -> Sort -> Limit -> [RoundDecimal] -> Select``
    - **Decay**:    ``Merge(score_mode) -> Map(Decay) -> Map(ScoreCombine) -> Sort -> Limit -> [RoundDecimal] -> Select``
    - **Model**:    ``Merge(max) -> Map(RerankModel) -> Sort -> Limit -> [RoundDecimal] -> Select``

    Args:
        schema: CollectionSchema with ``functions`` attribute.
        search_params: dict with keys ``limit``, ``offset``,
            ``round_decimal``, ``group_by_field``, ``group_size``.
        search_metrics: metric type per search path (for weighted normalization).

    Returns:
        A :class:`FuncChain` or ``None`` when no RERANK functions exist.
    """
    rerank_func = _find_rerank_function(schema)
    if rerank_func is None:
        return None

    chain = FuncChain("rerank", STAGE_RERANK)
    reranker_type = _get_reranker_type(rerank_func)

    # ── Head: Merge ──
    _build_rerank_head(chain, reranker_type, rerank_func, search_metrics or [])

    # ── Tail: Sort/GroupBy -> [RoundDecimal] -> Select ──
    _build_rerank_tail(chain, search_params)

    return chain


# ── Hybrid rerank (request-level params) ─────────────────────


def build_hybrid_rerank_chain(
    strategy: str,
    params: Dict[str, Any],
    search_params: Dict[str, Any],
) -> FuncChain:
    """Build a rerank chain for HybridSearch from rank_params.

    Used by ``servicer.HybridSearch`` where the rerank strategy comes
    from the request (not schema functions).

    Args:
        strategy: ``"rrf"`` or ``"weighted"``.
        params: strategy-specific params (``k`` for RRF, ``weights`` for weighted).
        search_params: dict with ``limit``, ``offset``, ``round_decimal``,
            ``group_by_field``, ``group_size``.

    Returns:
        A :class:`FuncChain`.
    """
    chain = FuncChain("hybrid_rerank", STAGE_RERANK)

    if strategy == "rrf":
        rrf_k = params.get("k", 60.0)
        chain.merge("rrf", rrf_k=rrf_k)
    elif strategy == "weighted":
        weights = params.get("weights", [])
        chain.merge("weighted", weights=weights)
    else:
        raise ValueError(f"Unsupported hybrid rerank strategy: {strategy!r}")

    _build_rerank_tail(chain, search_params)
    return chain


# ── Helpers ──────────────────────────────────────────────────


def _find_rerank_function(schema):
    """Find the first RERANK function in schema, or None."""
    from milvus_lite.schema.types import FunctionType

    if not schema.functions:
        return None
    for func in schema.functions:
        if func.function_type == FunctionType.RERANK:
            return func
    return None


def _get_reranker_type(func) -> str:
    """Determine reranker type from function params."""
    reranker = func.params.get("reranker", "").lower()
    if reranker == "decay":
        return "decay"
    provider = func.params.get("provider", "").lower()
    if provider:
        return "model"
    # Default to RRF if no specific type
    return func.params.get("strategy", "rrf").lower()


def _build_rerank_head(
    chain: FuncChain,
    reranker_type: str,
    rerank_func,
    search_metrics: List[str],
) -> None:
    """Build the head of a rerank chain (Merge + optional Map steps)."""
    from milvus_lite.function.expr.decay_expr import DecayExpr
    from milvus_lite.function.expr.rerank_model import RerankModelExpr
    from milvus_lite.function.expr.score_combine import ScoreCombineExpr

    if reranker_type == "rrf":
        rrf_k = rerank_func.params.get("k", 60.0)
        chain.merge("rrf", rrf_k=rrf_k)

    elif reranker_type == "weighted":
        weights = rerank_func.params.get("weights", [])
        normalize = rerank_func.params.get("norm_score", False)
        chain.merge(
            "weighted",
            weights=weights,
            metric_types=search_metrics,
            normalize=normalize,
        )

    elif reranker_type == "decay":
        score_mode = rerank_func.params.get("score_mode", "max")
        chain.merge(score_mode, metric_types=search_metrics)
        # Map(DecayExpr)
        in_name = rerank_func.input_field_names[0]
        decay_expr = DecayExpr(
            function=rerank_func.params["function"],
            origin=rerank_func.params["origin"],
            scale=rerank_func.params["scale"],
            offset=rerank_func.params.get("offset", 0.0),
            decay=rerank_func.params.get("decay", 0.5),
        )
        chain.map(decay_expr, [in_name], ["_decay_score"])
        # Map(ScoreCombineExpr)
        chain.map(
            ScoreCombineExpr("multiply"),
            [SCORE_FIELD, "_decay_score"],
            [SCORE_FIELD],
        )

    elif reranker_type == "model":
        chain.merge("max")
        in_name = rerank_func.input_field_names[0]
        from milvus_lite.rerank.factory import create_rerank_provider

        provider = create_rerank_provider(rerank_func.params)
        # query_texts injected at execute time via FuncContext
        model_expr = RerankModelExpr(provider, query_texts=[])
        chain.map(model_expr, [in_name], [SCORE_FIELD])

    else:
        raise ValueError(f"Unknown reranker type: {reranker_type!r}")


def _build_rerank_tail(chain: FuncChain, search_params: Dict[str, Any]) -> None:
    """Build the common tail: Sort/GroupBy -> [RoundDecimal] -> Select."""
    from milvus_lite.function.expr.round_decimal import RoundDecimalExpr

    group_by_field = search_params.get("group_by_field")
    limit = search_params.get("limit", 10)
    offset = search_params.get("offset", 0)
    round_decimal = search_params.get("round_decimal", -1)

    if group_by_field:
        group_size = search_params.get("group_size", 1)
        chain.group_by(group_by_field, group_size, limit, offset)
    else:
        chain.sort(SCORE_FIELD, desc=True)
        if limit > 0:
            chain.limit(limit, offset)

    if round_decimal >= 0:
        chain.map(
            RoundDecimalExpr(round_decimal), [SCORE_FIELD], [SCORE_FIELD]
        )

    select_cols = [ID_FIELD, SCORE_FIELD]
    if group_by_field:
        select_cols.extend([group_by_field, GROUP_SCORE_FIELD])
    chain.select(*select_cols)
