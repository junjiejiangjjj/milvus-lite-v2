"""Tests for MergeOp."""

import pytest

from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.ops.merge_op import MergeOp
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD, STAGE_RERANK, FuncContext


def _ctx():
    return FuncContext(STAGE_RERANK)


def _df(hits_per_query):
    """Build a DataFrame from per-query hit lists."""
    return DataFrame(hits_per_query)


def _hit(pk, score, **extra):
    h = {ID_FIELD: pk, SCORE_FIELD: score}
    h.update(extra)
    return h


# ── RRF ──────────────────────────────────────────────────────


def test_merge_rrf_basic():
    path0 = _df([[_hit(1, 0.9), _hit(2, 0.8)]])
    path1 = _df([[_hit(3, 0.7), _hit(1, 0.6)]])
    op = MergeOp("rrf", rrf_k=60.0)
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    pks = {h[ID_FIELD] for h in chunk}
    assert pks == {1, 2, 3}
    # pk=1 appears in both paths: rank 0 in path0 + rank 1 in path1
    pk1 = next(h for h in chunk if h[ID_FIELD] == 1)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert abs(pk1[SCORE_FIELD] - expected) < 1e-9


def test_merge_rrf_dedup():
    """Same pk in multiple paths should appear once in output."""
    path0 = _df([[_hit(1, 0.9)]])
    path1 = _df([[_hit(1, 0.5)]])
    op = MergeOp("rrf")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert len(result.chunk(0)) == 1


# ── Weighted ─────────────────────────────────────────────────


def test_merge_weighted_basic():
    path0 = _df([[_hit(1, 0.8), _hit(2, 0.4)]])
    path1 = _df([[_hit(1, 0.6), _hit(3, 0.9)]])
    op = MergeOp("weighted", weights=[0.7, 0.3])
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    assert len(chunk) == 3  # pk 1, 2, 3


def test_merge_weighted_normalization():
    """With two hits per route, min-max norm should produce 0.0 and 1.0."""
    path0 = _df([[_hit(1, 10.0), _hit(2, 20.0)]])
    path1 = _df([[_hit(1, 5.0)]])
    op = MergeOp("weighted", weights=[0.5, 0.5])
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    pk1 = next(h for h in chunk if h[ID_FIELD] == 1)
    pk2 = next(h for h in chunk if h[ID_FIELD] == 2)
    # path0: pk1 norm=0.0, pk2 norm=1.0; path1: pk1 norm=1.0 (only one)
    # pk1 final = 0.5*0.0 + 0.5*1.0 = 0.5
    # pk2 final = 0.5*1.0 + 0 = 0.5
    assert abs(pk1[SCORE_FIELD] - 0.5) < 1e-9
    assert abs(pk2[SCORE_FIELD] - 0.5) < 1e-9


# ── Simple strategies ────────────────────────────────────────


def test_merge_max():
    path0 = _df([[_hit(1, 0.3)]])
    path1 = _df([[_hit(1, 0.9)]])
    op = MergeOp("max")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert result.chunk(0)[0][SCORE_FIELD] == 0.9


def test_merge_sum():
    path0 = _df([[_hit(1, 0.3)]])
    path1 = _df([[_hit(1, 0.7)]])
    op = MergeOp("sum")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert abs(result.chunk(0)[0][SCORE_FIELD] - 1.0) < 1e-9


def test_merge_avg():
    path0 = _df([[_hit(1, 0.2)]])
    path1 = _df([[_hit(1, 0.8)]])
    op = MergeOp("avg")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert abs(result.chunk(0)[0][SCORE_FIELD] - 0.5) < 1e-9


# ── Edge cases ───────────────────────────────────────────────


def test_merge_single_input_passthrough():
    df = _df([[_hit(1, 0.5)]])
    op = MergeOp("rrf")
    result = op.execute_multi(_ctx(), [df])
    assert result is df


def test_merge_multi_query():
    path0 = _df([[_hit(1, 0.9)], [_hit(2, 0.8)], [_hit(3, 0.7)]])
    path1 = _df([[_hit(4, 0.6)], [_hit(5, 0.5)], [_hit(6, 0.4)]])
    op = MergeOp("rrf")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert result.num_chunks == 3
    assert len(result.chunk(0)) == 2
    assert len(result.chunk(1)) == 2
    assert len(result.chunk(2)) == 2


def test_merge_execute_raises():
    op = MergeOp("rrf")
    with pytest.raises(RuntimeError):
        op.execute(_ctx(), _df([[_hit(1, 0.5)]]))


# ── Weighted: all scores identical (range=0) ─────────────────


def test_merge_weighted_identical_scores():
    """When all scores in a route are the same, range=0 → norm defaults to 1.0."""
    path0 = _df([[_hit(1, 5.0), _hit(2, 5.0)]])
    path1 = _df([[_hit(1, 3.0)]])
    op = MergeOp("weighted", weights=[0.6, 0.4])
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    pk1 = next(h for h in chunk if h[ID_FIELD] == 1)
    pk2 = next(h for h in chunk if h[ID_FIELD] == 2)
    # path0 range=0 → norm=1.0 for both; path1 single → norm=1.0
    # pk1 = 0.6*1.0 + 0.4*1.0 = 1.0
    # pk2 = 0.6*1.0 = 0.6
    assert abs(pk1[SCORE_FIELD] - 1.0) < 1e-9
    assert abs(pk2[SCORE_FIELD] - 0.6) < 1e-9


# ── Inputs with mismatched num_chunks ────────────────────────


def test_merge_mismatched_num_chunks():
    """path0 has 2 queries, path1 has 1 → query 1 uses empty chunk for path1."""
    path0 = _df([[_hit(1, 0.9)], [_hit(2, 0.8)]])
    path1 = _df([[_hit(3, 0.7)]])  # only 1 query
    op = MergeOp("rrf")
    result = op.execute_multi(_ctx(), [path0, path1])
    assert result.num_chunks == 2
    # query 0: pk1 + pk3
    assert len(result.chunk(0)) == 2
    # query 1: only pk2 (path1 has no chunk 1 → treated as empty)
    assert len(result.chunk(1)) == 1
    assert result.chunk(1)[0][ID_FIELD] == 2


# ── Weights longer than routes ───────────────────────────────


def test_merge_weighted_extra_weights_ignored():
    """Extra weights beyond num_routes are silently ignored."""
    path0 = _df([[_hit(1, 0.8)]])
    path1 = _df([[_hit(1, 0.4)]])
    op = MergeOp("weighted", weights=[0.7, 0.3, 0.5])  # 3 weights, 2 routes
    result = op.execute_multi(_ctx(), [path0, path1])
    chunk = result.chunk(0)
    assert len(chunk) == 1
