"""Tests for GroupByOp."""

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.group_by_op import GroupByOp
from milvus_lite.function.types import (
    GROUP_SCORE_FIELD,
    ID_FIELD,
    SCORE_FIELD,
    STAGE_RERANK,
    FuncContext,
)


def _ctx():
    return FuncContext(STAGE_RERANK)


def _hit(pk, score, cat):
    return {ID_FIELD: pk, SCORE_FIELD: score, "cat": cat}


def test_group_by_basic():
    df = DataFrame([[
        _hit(1, 0.9, "A"), _hit(2, 0.8, "A"),
        _hit(3, 0.7, "B"), _hit(4, 0.6, "B"), _hit(5, 0.5, "B"),
    ]])
    op = GroupByOp("cat", group_size=2, limit=2)
    result = op.execute(_ctx(), df)
    chunk = result.chunk(0)
    # 2 groups × 2 hits each = 4 total
    assert len(chunk) == 4
    # group A (max score 0.9) should come first
    assert chunk[0]["cat"] == "A"


def test_group_by_scorer_max():
    df = DataFrame([[_hit(1, 0.9, "A"), _hit(2, 0.3, "A")]])
    op = GroupByOp("cat", group_size=2, limit=1, scorer="max")
    result = op.execute(_ctx(), df)
    assert result.chunk(0)[0][GROUP_SCORE_FIELD] == 0.9


def test_group_by_scorer_sum():
    df = DataFrame([[_hit(1, 0.9, "A"), _hit(2, 0.3, "A")]])
    op = GroupByOp("cat", group_size=2, limit=1, scorer="sum")
    result = op.execute(_ctx(), df)
    assert abs(result.chunk(0)[0][GROUP_SCORE_FIELD] - 1.2) < 1e-9


def test_group_by_scorer_avg():
    df = DataFrame([[_hit(1, 0.9, "A"), _hit(2, 0.3, "A")]])
    op = GroupByOp("cat", group_size=2, limit=1, scorer="avg")
    result = op.execute(_ctx(), df)
    assert abs(result.chunk(0)[0][GROUP_SCORE_FIELD] - 0.6) < 1e-9


def test_group_by_offset():
    df = DataFrame([[
        _hit(1, 0.9, "A"), _hit(2, 0.7, "B"), _hit(3, 0.5, "C"),
    ]])
    op = GroupByOp("cat", group_size=1, limit=1, offset=1)
    result = op.execute(_ctx(), df)
    # skip first group (A), take second (B)
    assert len(result.chunk(0)) == 1
    assert result.chunk(0)[0]["cat"] == "B"


def test_group_by_adds_group_score():
    df = DataFrame([[_hit(1, 0.9, "A")]])
    op = GroupByOp("cat", group_size=1, limit=1)
    result = op.execute(_ctx(), df)
    assert GROUP_SCORE_FIELD in result.chunk(0)[0]


def test_group_by_group_size_truncates():
    df = DataFrame([[
        _hit(1, 0.9, "A"), _hit(2, 0.8, "A"), _hit(3, 0.7, "A"),
    ]])
    op = GroupByOp("cat", group_size=2, limit=1)
    result = op.execute(_ctx(), df)
    # only top 2 from group A
    assert len(result.chunk(0)) == 2
    pks = {h[ID_FIELD] for h in result.chunk(0)}
    assert pks == {1, 2}
