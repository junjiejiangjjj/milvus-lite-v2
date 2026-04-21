"""Chain builders — construct FuncChains from schema functions.

``build_ingestion_chain`` creates a chain for insert-time auto-generation
(BM25, TEXT_EMBEDDING).  ``build_rerank_chain`` (FC-6) will handle
search-time reranking.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from milvus_lite.function.chain import FuncChain
from milvus_lite.function.types import STAGE_INGESTION


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
