"""Tests for build_rerank_chain and build_hybrid_rerank_chain."""

from milvus_lite.function.builder import (
    build_hybrid_rerank_chain,
    build_rerank_chain,
)
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
