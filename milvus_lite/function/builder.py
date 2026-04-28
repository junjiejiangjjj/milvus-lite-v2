"""Chain builders — construct FuncChains from schema functions.

``build_ingestion_chain`` creates a chain for insert-time auto-generation
(BM25, TEXT_EMBEDDING).

``build_rerank_chain`` creates a chain for search-time reranking from
schema-level RERANK/DECAY functions (4 patterns: RRF/Weighted/Decay/Model).

``build_hybrid_rerank_chain`` creates a chain for HybridSearch from
request-level rank_params (RRF/Weighted, no schema functions involved).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from milvus_lite.function.chain import FuncChain
from milvus_lite.function.types import (
    DECAY_SCORE_FIELD,
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
        normalize = params.get("norm_score", True)
        chain.merge("weighted", weights=weights, normalize=normalize)
    else:
        raise ValueError(f"Unsupported hybrid rerank strategy: {strategy!r}")

    # skip_select=True: HybridSearch caller handles field filtering
    # during format conversion (entity fields must pass through).
    _build_rerank_tail(chain, search_params, skip_select=True)
    return chain


def build_hybrid_function_score_chain(
    rerank_func,
    search_params: Dict[str, Any],
    search_metrics: Optional[List[str]] = None,
) -> FuncChain:
    """Build the L2 HybridSearch chain from FunctionScore.reranker."""
    chain = FuncChain("hybrid_rerank", STAGE_RERANK)
    reranker_type = _get_reranker_type(rerank_func)
    query_texts = None
    if reranker_type == "model":
        query_texts = _extract_model_queries(rerank_func)
    _build_rerank_head(chain, reranker_type, rerank_func, search_metrics or [])
    if reranker_type == "model":
        from milvus_lite.function.expr.rerank_model import RerankModelExpr
        from milvus_lite.function.ops.map_op import MapOp

        for op in chain.operators:
            if isinstance(op, MapOp) and isinstance(op.expr, RerankModelExpr):
                op.expr.query_texts = query_texts
    _build_rerank_tail(chain, search_params, skip_select=True)
    return chain


# ── Single-search rerank (no MergeOp) ────────────────────────


def build_single_rerank_chain(
    rerank_func=None,
    decay_func=None,
) -> FuncChain:
    """Build a rerank chain for single-path search (no MergeOp).

    Used by ``Collection.search()`` when the schema has RERANK or DECAY
    functions.  The chain is pre-built at init time and reused across
    searches; only ``RerankModelExpr.query_texts`` needs to be set before
    each execution.

    Unlike ``build_rerank_chain`` (which always starts with MergeOp for
    multi-path fusion), this chain has no Merge and no tail (Sort/Limit
    are added by the caller per-search since top_k/offset vary).

    Args:
        rerank_func: schema ``Function`` with a model reranker, or None.
        decay_func: schema ``Function`` with decay reranker, or None.

    Returns:
        A :class:`FuncChain` with Map steps only (no Sort/Limit/Select).
    """
    chain = FuncChain("single_rerank", STAGE_RERANK)

    if rerank_func is not None:
        from milvus_lite.function.expr.rerank_model import RerankModelExpr
        from milvus_lite.rerank.factory import create_rerank_provider

        in_name = rerank_func.input_field_names[0]
        provider = create_rerank_provider(rerank_func.params)
        chain.map(RerankModelExpr(provider), [in_name], [SCORE_FIELD])

    if decay_func is not None:
        from milvus_lite.function.expr.decay_expr import DecayExpr
        from milvus_lite.function.expr.score_combine import ScoreCombineExpr

        p = decay_func.params
        in_name = decay_func.input_field_names[0]
        chain.map(
            DecayExpr(
                function=p["function"],
                origin=p["origin"],
                scale=p["scale"],
                offset=p.get("offset", 0.0),
                decay=p.get("decay", 0.5),
            ),
            [in_name],
            [DECAY_SCORE_FIELD],
        )
        chain.map(
            ScoreCombineExpr("multiply"),
            [SCORE_FIELD, DECAY_SCORE_FIELD],
            [SCORE_FIELD],
        )

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


def _extract_model_queries(func) -> List[str]:
    params = getattr(func, "params", {}) or {}
    if "queries" not in params:
        raise ValueError("model reranker requires params.queries")
    queries = params["queries"]
    if isinstance(queries, str):
        try:
            queries = json.loads(queries)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise ValueError("model reranker params.queries must be a JSON array")
    if not isinstance(queries, list) or not queries:
        raise ValueError("model reranker params.queries must be a non-empty array")
    if not all(isinstance(q, str) for q in queries):
        raise ValueError("model reranker params.queries must contain only strings")
    return list(queries)


def _get_reranker_type(func) -> str:
    """Determine reranker type from function params."""
    reranker = func.params.get("reranker", "").lower()
    if reranker in ("rrf", "weighted", "decay", "model"):
        return reranker
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
        chain.map(decay_expr, [in_name], [DECAY_SCORE_FIELD])
        # Map(ScoreCombineExpr)
        chain.map(
            ScoreCombineExpr("multiply"),
            [SCORE_FIELD, DECAY_SCORE_FIELD],
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


def _build_rerank_tail(
    chain: FuncChain,
    search_params: Dict[str, Any],
    *,
    skip_select: bool = False,
) -> None:
    """Build the common tail: Sort/GroupBy -> [RoundDecimal] -> [Select].

    Args:
        skip_select: if True, omit the final SelectOp.  Used by
            ``build_hybrid_rerank_chain`` where the caller handles
            field filtering during format conversion.
    """
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

    if not skip_select:
        select_cols = [ID_FIELD, SCORE_FIELD]
        if group_by_field:
            select_cols.extend([group_by_field, GROUP_SCORE_FIELD])
        chain.select(*select_cols)
