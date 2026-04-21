"""End-to-end tests for rerank chains."""

from milvus_lite.function.builder import build_hybrid_rerank_chain
from milvus_lite.function.dataframe import DataFrame
from milvus_lite.function.types import ID_FIELD, SCORE_FIELD


def _hit(pk, score, **extra):
    h = {ID_FIELD: pk, SCORE_FIELD: score}
    h.update(extra)
    return h


def test_rrf_chain_e2e():
    chain = build_hybrid_rerank_chain("rrf", {"k": 60}, {"limit": 2})
    path0 = DataFrame([[_hit(1, 0.9), _hit(2, 0.8), _hit(3, 0.7)]])
    path1 = DataFrame([[_hit(2, 0.6), _hit(4, 0.5)]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    assert len(chunk) == 2  # limit=2
    # pk=2 appears in both routes → highest RRF score
    pks = [h[ID_FIELD] for h in chunk]
    assert 2 in pks
    # Only $id and $score columns remain (Select)
    assert set(chunk[0].keys()) == {ID_FIELD, SCORE_FIELD}


def test_weighted_chain_e2e():
    chain = build_hybrid_rerank_chain(
        "weighted", {"weights": [0.7, 0.3]}, {"limit": 3}
    )
    path0 = DataFrame([[_hit(1, 0.9), _hit(2, 0.1)]])
    path1 = DataFrame([[_hit(1, 0.1), _hit(3, 0.9)]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    assert len(chunk) == 3
    # Results sorted by score desc
    scores = [h[SCORE_FIELD] for h in chunk]
    assert scores == sorted(scores, reverse=True)


def test_rrf_multi_query():
    chain = build_hybrid_rerank_chain("rrf", {}, {"limit": 10})
    path0 = DataFrame([
        [_hit(1, 0.9)],
        [_hit(2, 0.8)],
    ])
    path1 = DataFrame([
        [_hit(3, 0.7)],
        [_hit(4, 0.6)],
    ])
    result = chain.execute(path0, path1)
    assert result.num_chunks == 2
    assert len(result.chunk(0)) == 2
    assert len(result.chunk(1)) == 2


def test_rrf_with_offset():
    chain = build_hybrid_rerank_chain("rrf", {}, {"limit": 1, "offset": 1})
    path0 = DataFrame([[_hit(1, 0.9), _hit(2, 0.8), _hit(3, 0.7)]])
    path1 = DataFrame([[]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    assert len(chunk) == 1
    # Should be the 2nd result (offset=1)
    assert chunk[0][ID_FIELD] != 1  # not the top result


def test_chain_with_group_by_e2e():
    chain = build_hybrid_rerank_chain(
        "rrf", {},
        {"limit": 2, "group_by_field": "cat", "group_size": 1},
    )
    path0 = DataFrame([[
        _hit(1, 0.9, cat="A"),
        _hit(2, 0.8, cat="A"),
        _hit(3, 0.7, cat="B"),
    ]])
    path1 = DataFrame([[]])
    result = chain.execute(path0, path1)
    chunk = result.chunk(0)
    # 2 groups × 1 per group = 2 results
    assert len(chunk) == 2
    cats = {h["cat"] for h in chunk}
    assert cats == {"A", "B"}
