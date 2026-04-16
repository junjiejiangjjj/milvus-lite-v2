"""Verify insert doesn't block on compaction / index build.

Compaction + index build run on a background worker after flush. Insert
returns as soon as data is persisted (WAL + parquet + manifest).
"""

import threading
import time
import numpy as np
import pytest

from litevecdb.engine.collection import Collection
from litevecdb.schema.types import CollectionSchema, DataType, FieldSchema


@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
    ])


def _records(start, n):
    return [{"id": start + i,
             "vec": np.random.default_rng(i).random(8).tolist()}
            for i in range(n)]


def test_insert_returns_before_bg_index_build(tmp_path, schema, monkeypatch):
    """Insert returns quickly even when index build is pending.

    Simulates the real-world case where HNSW_SQ index build on
    large segments takes many seconds. That work must run off the
    user thread.
    """
    monkeypatch.setattr("litevecdb.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)
    # Attach an index spec so the bg worker has index work to do.
    col.create_index("vec", {
        "index_type": "BRUTE_FORCE", "metric_type": "COSINE", "params": {},
    })

    # Make the per-segment index build artificially slow.
    from litevecdb.storage.segment import Segment
    real_build = Segment.build_or_load_index

    def slow_build(self, spec, index_dir):
        time.sleep(0.3)  # 300 ms per segment
        return real_build(self, spec, index_dir)

    monkeypatch.setattr(Segment, "build_or_load_index", slow_build)

    # Trigger several flushes in quick succession. Since the index
    # build runs outside the maintenance lock, inserts must not
    # serialize behind each previous flush's bg build.
    t0 = time.time()
    for batch in range(4):
        col.insert(_records(batch * 5, 5))
    insert_elapsed = time.time() - t0

    # With sync indexing, this would be >= 4 * 0.3 = 1.2s.
    # With async, should be well under 1s (only the sync flush path).
    assert insert_elapsed < 1.0, (
        f"insert took {insert_elapsed:.2f}s — index build appears to block it"
    )

    col._wait_for_bg()
    col.close()


def test_search_concurrent_with_bg_compaction(tmp_path, schema, monkeypatch):
    """Searches from a different thread work while bg compaction runs."""
    monkeypatch.setattr("litevecdb.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)

    # Fill with enough data to have segments.
    for batch in range(4):
        col.insert(_records(batch * 5, 5))

    errors = []

    def search_worker():
        try:
            for _ in range(20):
                res = col.search(
                    [[0.1] * 8], top_k=3, metric_type="COSINE",
                )
                assert len(res) == 1
        except Exception as e:
            errors.append(e)

    def insert_worker():
        try:
            for batch in range(4, 8):
                col.insert(_records(batch * 5, 5))
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=search_worker)
    t2 = threading.Thread(target=insert_worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"concurrent ops raised: {errors}"
    col.close()


def test_close_drains_bg_tasks(tmp_path, schema, monkeypatch):
    """close() waits for pending bg tasks before returning."""
    monkeypatch.setattr("litevecdb.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)

    # Trigger several flushes → multiple bg tasks queued.
    for batch in range(4):
        col.insert(_records(batch * 5, 5))

    col.close()
    # After close, reopen should see committed state — no half-flushed
    # manifest, no missing files.
    col2 = Collection("c", str(tmp_path / "d"), schema)
    col2.load()
    assert col2.num_entities == 20
    col2.close()


def test_wait_for_bg_drains_pending(tmp_path, schema, monkeypatch):
    """_wait_for_bg blocks until all queued tasks finish."""
    monkeypatch.setattr("litevecdb.engine.collection.MEMTABLE_SIZE_LIMIT", 5)

    col = Collection("c", str(tmp_path / "d"), schema)

    for batch in range(6):
        col.insert(_records(batch * 5, 5))

    # After wait, manifest should reflect any compaction outcome.
    col._wait_for_bg()

    # Sanity: we can read everything back.
    col.load()
    for i in range(30):
        res = col.get([i])
        assert len(res) == 1

    col.close()
