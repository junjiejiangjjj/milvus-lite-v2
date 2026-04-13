"""Phase 9.3 — Collection.create_index / drop_index / load / release tests.

Validates the state machine and the search/get/query loaded-guard.
The semantics chosen for backward compatibility:

    - Collection without IndexSpec → auto-loaded on construction
      (preserves all pre-Phase-9 tests; matches "no index = no
      need to load" intuition)
    - After create_index → released; user must call load() before
      search/get/query (mirrors Milvus)
    - After drop_index → loaded again
    - After release → released
"""

import pytest

from litevecdb.engine.collection import Collection
from litevecdb.exceptions import (
    CollectionNotLoadedError,
    IndexAlreadyExistsError,
    IndexNotFoundError,
    SchemaValidationError,
)
from litevecdb.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=64, nullable=True),
    ])


@pytest.fixture
def col(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    yield c
    c.close()


def _vec(i):
    return [float(i), float(i + 1), float(i + 2), float(i + 3)]


HNSW_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}

BRUTE_PARAMS = {
    "index_type": "BRUTE_FORCE",
    "metric_type": "L2",
    "params": {},
}


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_fresh_collection_has_no_index(col):
    assert col.has_index() is False
    assert col.get_index_info() is None
    # No index → auto-loaded (preserves backward compat).
    assert col.load_state == "loaded"


def test_search_works_when_no_index_attached(col):
    """Backward compat: existing tests don't call load(), and shouldn't have to."""
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    res = col.search([_vec(0)], top_k=3)
    assert len(res[0]) == 3


# ---------------------------------------------------------------------------
# create_index
# ---------------------------------------------------------------------------

def test_create_index_persists_spec(col):
    col.create_index("vec", BRUTE_PARAMS)
    assert col.has_index() is True
    info = col.get_index_info()
    assert info["index_type"] == "BRUTE_FORCE"
    assert info["metric_type"] == "L2"
    assert info["field_name"] == "vec"


def test_create_index_moves_to_released(col):
    col.create_index("vec", HNSW_PARAMS)
    assert col.load_state == "released"


def test_create_index_persists_across_restart(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    c.create_index("vec", HNSW_PARAMS)
    c.close()

    c2 = Collection("t", str(tmp_path / "data"), schema)
    try:
        assert c2.has_index() is True
        info = c2.get_index_info()
        assert info["index_type"] == "HNSW"
        # Restart with index should default to released — must call load.
        assert c2.load_state == "released"
    finally:
        c2.close()


def test_create_index_twice_raises(col):
    col.create_index("vec", HNSW_PARAMS)
    with pytest.raises(IndexAlreadyExistsError):
        col.create_index("vec", BRUTE_PARAMS)


def test_create_index_unknown_field_raises(col):
    with pytest.raises(SchemaValidationError, match="unknown field"):
        col.create_index("ghost", HNSW_PARAMS)


def test_create_index_non_vector_field_raises(col):
    with pytest.raises(SchemaValidationError, match="vector"):
        col.create_index("title", HNSW_PARAMS)


def test_create_index_passthrough_search_params(col):
    col.create_index("vec", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16},
        "search_params": {"ef": 64},
    })
    info = col.get_index_info()
    assert info["search_params"] == {"ef": 64}


# ---------------------------------------------------------------------------
# drop_index
# ---------------------------------------------------------------------------

def test_drop_index_clears_spec(col):
    col.create_index("vec", HNSW_PARAMS)
    col.drop_index("vec")
    assert col.has_index() is False
    assert col.get_index_info() is None


def test_drop_index_moves_back_to_loaded(col):
    col.create_index("vec", HNSW_PARAMS)
    col.drop_index("vec")
    assert col.load_state == "loaded"


def test_drop_index_no_args_drops_existing(col):
    col.create_index("vec", HNSW_PARAMS)
    col.drop_index()
    assert col.has_index() is False


def test_drop_index_with_wrong_field_raises(col):
    col.create_index("vec", HNSW_PARAMS)
    with pytest.raises(IndexNotFoundError):
        col.drop_index("ghost")


def test_drop_index_when_no_index_raises(col):
    with pytest.raises(IndexNotFoundError):
        col.drop_index()


def test_drop_index_persists_across_restart(tmp_path, schema):
    c = Collection("t", str(tmp_path / "data"), schema)
    c.create_index("vec", HNSW_PARAMS)
    c.drop_index("vec")
    c.close()

    c2 = Collection("t", str(tmp_path / "data"), schema)
    try:
        assert c2.has_index() is False
        # Back to auto-loaded.
        assert c2.load_state == "loaded"
    finally:
        c2.close()


# ---------------------------------------------------------------------------
# load / release / state machine
# ---------------------------------------------------------------------------

def test_load_after_create_index(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    assert col.load_state == "released"
    col.load()
    assert col.load_state == "loaded"


def test_load_attaches_index_to_segments(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    # Pre-load: no segment has an attached index.
    for seg in col._segment_cache.values():
        assert seg.index is None

    col.load()

    # Post-load: every segment has an index.
    for seg in col._segment_cache.values():
        assert seg.index is not None
        assert seg.index.metric == "L2"


def test_load_is_idempotent(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    col.load()
    state1 = col.load_state
    col.load()  # second load should be a no-op
    assert col.load_state == state1


def test_load_when_no_index_is_noop(col):
    """No IndexSpec → already loaded; load() should be idempotent."""
    assert col.load_state == "loaded"
    col.load()
    assert col.load_state == "loaded"


def test_release_drops_segment_indexes(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    col.load()
    col.release()
    assert col.load_state == "released"
    for seg in col._segment_cache.values():
        assert seg.index is None


def test_release_when_no_index_is_noop(col):
    """Without an IndexSpec, release() should not change state away from loaded."""
    col.release()
    assert col.load_state == "loaded"


def test_load_release_cycle(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    for _ in range(3):
        col.load()
        assert col.load_state == "loaded"
        col.release()
        assert col.load_state == "released"


# ---------------------------------------------------------------------------
# search/get/query loaded guard
# ---------------------------------------------------------------------------

def test_search_after_create_index_raises(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.create_index("vec", BRUTE_PARAMS)
    with pytest.raises(CollectionNotLoadedError):
        col.search([_vec(0)], top_k=3)


def test_get_after_create_index_raises(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.create_index("vec", BRUTE_PARAMS)
    with pytest.raises(CollectionNotLoadedError):
        col.get([0, 1, 2])


def test_query_after_create_index_raises(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.create_index("vec", BRUTE_PARAMS)
    with pytest.raises(CollectionNotLoadedError):
        col.query("id >= 0")


def test_search_after_load_works(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    col.load()
    res = col.search([_vec(0)], top_k=3)
    assert len(res[0]) == 3


def test_search_after_release_raises(col):
    col.insert([{"id": i, "vec": _vec(i), "title": "x"} for i in range(5)])
    col.flush()
    col.create_index("vec", BRUTE_PARAMS)
    col.load()
    col.release()
    with pytest.raises(CollectionNotLoadedError):
        col.search([_vec(0)], top_k=3)


def test_insert_delete_work_in_released_state(col):
    """Writes should NOT require loaded state — only reads do."""
    col.create_index("vec", BRUTE_PARAMS)
    assert col.load_state == "released"
    # Both should succeed without raising.
    col.insert([{"id": 99, "vec": _vec(99), "title": "x"}])
    col.delete(pks=[99])


def test_describe_includes_load_state_and_index_spec(col):
    d = col.describe()
    assert d["load_state"] == "loaded"
    assert d["index_specs"] == {}

    col.create_index("vec", HNSW_PARAMS)
    d2 = col.describe()
    assert d2["load_state"] == "released"
    assert "vec" in d2["index_specs"]
    assert d2["index_specs"]["vec"]["index_type"] == "HNSW"
    assert d2["index_specs"]["vec"]["metric_type"] == "COSINE"
