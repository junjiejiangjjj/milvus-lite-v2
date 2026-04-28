"""Tests for build_rerank_chain, build_hybrid_rerank_chain, build_single_rerank_chain."""

import pytest

from milvus_lite.function.builder import (
    build_hybrid_function_score_chain,
    build_hybrid_rerank_chain,
    build_rerank_chain,
    build_single_rerank_chain,
)
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.group_by_op import GroupByOp
from milvus_lite.function.ops.limit_op import LimitOp
from milvus_lite.function.ops.merge_op import MergeOp
from milvus_lite.function.ops.select_op import SelectOp
from milvus_lite.function.ops.sort_op import SortOp
from milvus_lite.function.ops.map_op import MapOp
from milvus_lite.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD


def _schema_with_rerank(**params):
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="ts", dtype=DataType.INT64),
        ],
        functions=[
            Function(
                name="rerank_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["text"] if "provider" in params else ["ts"],
                output_field_names=[],
                params=params,
            ),
        ],
    )


def _op_names(chain):
    return [op.name for op in chain.operators]


# ── build_rerank_chain ───────────────────────────────────────


def test_build_rrf_chain():
    schema = _schema_with_rerank(strategy="rrf", k=60)
    chain = build_rerank_chain(schema, {"limit": 10})
    names = _op_names(chain)
    assert names[0] == "Merge"
    assert "Sort" in names
    assert "Limit" in names
    assert "Select" in names


def test_build_decay_chain():
    schema = _schema_with_rerank(
        reranker="decay", function="gauss", origin=0, scale=100, decay=0.5
    )
    chain = build_rerank_chain(schema, {"limit": 10})
    names = _op_names(chain)
    # Merge -> Map(Decay) -> Map(ScoreCombine) -> Sort -> Limit -> Select
    assert names[0] == "Merge"
    assert names.count("Map") == 2
    assert "Sort" in names
    assert "Select" in names


def test_build_decay_chain_exprs():
    schema = _schema_with_rerank(
        reranker="decay", function="exp", origin=50, scale=10, decay=0.3
    )
    chain = build_rerank_chain(schema, {"limit": 5})
    map_ops = [op for op in chain.operators if isinstance(op, MapOp)]
    assert map_ops[0].expr.name == "decay"
    assert map_ops[1].expr.name == "score_combine"


def test_build_rerank_no_functions():
    schema = CollectionSchema(
        fields=[FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)]
    )
    assert build_rerank_chain(schema, {"limit": 10}) is None


def test_build_rerank_with_group_by():
    schema = _schema_with_rerank(strategy="rrf")
    chain = build_rerank_chain(
        schema, {"limit": 10, "group_by_field": "text", "group_size": 3}
    )
    names = _op_names(chain)
    assert "GroupBy" in names
    # No Sort/Limit when GroupBy is used
    assert "Sort" not in names
    assert "Limit" not in names


def test_build_rerank_with_round_decimal():
    schema = _schema_with_rerank(strategy="rrf")
    chain = build_rerank_chain(schema, {"limit": 10, "round_decimal": 4})
    map_ops = [op for op in chain.operators if isinstance(op, MapOp)]
    assert any(op.expr.name == "round_decimal" for op in map_ops)


def test_build_rerank_no_round_decimal_when_negative():
    schema = _schema_with_rerank(strategy="rrf")
    chain = build_rerank_chain(schema, {"limit": 10, "round_decimal": -1})
    map_ops = [op for op in chain.operators if isinstance(op, MapOp)]
    assert not any(op.expr.name == "round_decimal" for op in map_ops)


def test_build_weighted_chain_honors_norm_score_false():
    schema = _schema_with_rerank(
        strategy="weighted", weights=[0.25, 0.75], norm_score=False
    )
    chain = build_rerank_chain(schema, {"limit": 10})
    path0 = DataFrame([[
        {ID_FIELD: 1, SCORE_FIELD: 10.0},
        {ID_FIELD: 2, SCORE_FIELD: 1.0},
    ]])
    path1 = DataFrame([[
        {ID_FIELD: 1, SCORE_FIELD: 3.0},
        {ID_FIELD: 2, SCORE_FIELD: 5.0},
    ]])

    result = chain.execute(path0, path1).chunk(0)

    assert result[0] == {ID_FIELD: 1, SCORE_FIELD: 4.75}
    assert result[1] == {ID_FIELD: 2, SCORE_FIELD: 4.0}


def test_build_weighted_l2_chain_sorts_ascending_without_norm():
    schema = _schema_with_rerank(
        strategy="weighted", weights=[1.0], norm_score=False
    )
    chain = build_rerank_chain(
        schema,
        {"limit": 2},
        search_metrics=["L2"],
    )
    result = chain.execute(DataFrame([[
        {ID_FIELD: 1, SCORE_FIELD: 0.2},
        {ID_FIELD: 2, SCORE_FIELD: 1.0},
        {ID_FIELD: 3, SCORE_FIELD: 3.0},
    ]])).chunk(0)

    assert [row[ID_FIELD] for row in result] == [1, 2]


# ── build_hybrid_rerank_chain ────────────────────────────────


def test_build_hybrid_rrf():
    chain = build_hybrid_rerank_chain("rrf", {"k": 60}, {"limit": 10})
    names = _op_names(chain)
    assert names[0] == "Merge"
    assert "Sort" in names
    # Hybrid chains skip SelectOp (caller handles field filtering)
    assert "Select" not in names


def test_build_hybrid_weighted():
    chain = build_hybrid_rerank_chain(
        "weighted", {"weights": [0.7, 0.3]}, {"limit": 5}
    )
    names = _op_names(chain)
    assert names[0] == "Merge"


def test_build_hybrid_with_group_by():
    chain = build_hybrid_rerank_chain(
        "rrf", {}, {"limit": 10, "group_by_field": "cat", "group_size": 2}
    )
    names = _op_names(chain)
    assert "GroupBy" in names
    assert "Sort" not in names


def test_build_hybrid_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported hybrid rerank strategy"):
        build_hybrid_rerank_chain("unknown", {}, {"limit": 10})


def test_build_hybrid_model_requires_queries():
    func = Function(
        name="model_rank",
        function_type=FunctionType.RERANK,
        input_field_names=["text"],
        output_field_names=[],
        params={"reranker": "model", "provider": "mock"},
    )

    with pytest.raises(ValueError, match="queries"):
        build_hybrid_function_score_chain(func, {"limit": 10})


def test_build_hybrid_model_uses_queries_from_params(monkeypatch):
    class _Provider:
        def rerank(self, query, docs, top_n=None):
            return []

    monkeypatch.setattr(
        "milvus_lite.rerank.factory.create_rerank_provider",
        lambda params: _Provider(),
    )
    func = Function(
        name="model_rank",
        function_type=FunctionType.RERANK,
        input_field_names=["text"],
        output_field_names=[],
        params={
            "reranker": "model",
            "provider": "mock",
            "queries": ["query 1", "query 2"],
        },
    )

    chain = build_hybrid_function_score_chain(func, {"limit": 10})
    model_maps = [
        op for op in chain.operators
        if isinstance(op, MapOp) and op.expr.name == "rerank_model"
    ]
    assert model_maps[0].expr.query_texts == ["query 1", "query 2"]


# ── build_single_rerank_chain ────────────────────────────────


def test_build_single_rerank_chain_both_none():
    """No rerank_func and no decay_func → empty chain (no ops)."""
    chain = build_single_rerank_chain(rerank_func=None, decay_func=None)
    assert chain.operators == []


def test_build_single_rerank_chain_decay_only():
    func = Function(
        name="decay_fn",
        function_type=FunctionType.RERANK,
        input_field_names=["ts"],
        output_field_names=[],
        params={
            "reranker": "decay", "function": "gauss",
            "origin": 0, "scale": 100, "decay": 0.5,
        },
    )
    chain = build_single_rerank_chain(decay_func=func)
    names = _op_names(chain)
    # Map(Decay) → Map(ScoreCombine), no Sort/Limit
    assert names == ["Map", "Map"]


# ── _build_rerank_tail: limit=0 ──────────────────────────────


def test_build_rerank_tail_limit_zero_skips_limit_op():
    """When limit=0, LimitOp should not be added."""
    schema = _schema_with_rerank(strategy="rrf")
    chain = build_rerank_chain(schema, {"limit": 0})
    names = _op_names(chain)
    assert "Sort" in names
    assert "Limit" not in names
