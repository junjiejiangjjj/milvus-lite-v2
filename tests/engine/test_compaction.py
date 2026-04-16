"""Tests for engine/compaction.py — bucketing, selection, merge, GC.

These are unit tests on CompactionManager directly. Collection-level
integration (flush triggers compaction) is in test_collection.py.
"""

import os

import pyarrow as pa
import pytest

from litevecdb.constants import (
    COMPACTION_BUCKET_BOUNDARIES,
    COMPACTION_MIN_FILES_PER_BUCKET,
    DEFAULT_PARTITION,
    MAX_DATA_FILES,
)
from litevecdb.engine.compaction import CompactionManager
from litevecdb.schema.arrow_builder import build_data_schema, build_delta_schema
from litevecdb.schema.types import CollectionSchema, DataType, FieldSchema
from litevecdb.storage.data_file import read_data_file, write_data_file
from litevecdb.storage.delta_index import DeltaIndex
from litevecdb.storage.manifest import Manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name="title", dtype=DataType.VARCHAR, nullable=True),
    ])


@pytest.fixture
def harness(tmp_path, schema):
    """Bare-bones (data_dir, manifest, delta_index, mgr) tuple."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    manifest = Manifest(data_dir)
    delta_index = DeltaIndex("id")
    mgr = CompactionManager(data_dir, schema)
    return {
        "data_dir": data_dir,
        "schema": schema,
        "manifest": manifest,
        "delta_index": delta_index,
        "mgr": mgr,
    }


def _write_data_file_with_records(harness, rows, seq_min, seq_max):
    """Helper: write a data Parquet file containing the given rows.

    rows = [(seq, pk, vec, title), ...]

    Returns the relative path stored in the manifest after add_data_file.
    """
    table = pa.Table.from_pydict(
        {
            "_seq": [r[0] for r in rows],
            "id": [r[1] for r in rows],
            "vec": [r[2] for r in rows],
            "title": [r[3] for r in rows],
        },
        schema=build_data_schema(harness["schema"]),
    )
    partition_dir = os.path.join(
        harness["data_dir"], "partitions", DEFAULT_PARTITION
    )
    os.makedirs(partition_dir, exist_ok=True)
    rel = write_data_file(table, partition_dir, seq_min, seq_max)
    harness["manifest"].add_data_file(DEFAULT_PARTITION, rel)
    return rel


# ---------------------------------------------------------------------------
# Bucket selection
# ---------------------------------------------------------------------------

def test_bucket_index_boundaries():
    """Bucket boundaries are [1MB, 10MB, 100MB] → 4 buckets."""
    assert CompactionManager._bucket_index(0) == 0
    assert CompactionManager._bucket_index(500_000) == 0
    assert CompactionManager._bucket_index(1_000_000) == 1  # boundary
    assert CompactionManager._bucket_index(5_000_000) == 1
    assert CompactionManager._bucket_index(10_000_000) == 2
    assert CompactionManager._bucket_index(50_000_000) == 2
    assert CompactionManager._bucket_index(100_000_000) == 3
    assert CompactionManager._bucket_index(1_000_000_000) == 3


def _make_files(harness, n_files, rows_per_file=10):
    """Create *n_files* tiny parquet files and return (partition_dir, names)."""
    names = []
    for i in range(n_files):
        rows = [(i * rows_per_file + j, f"pk_{i}_{j}", [0.1, 0.2], None)
                for j in range(rows_per_file)]
        rel = _write_data_file_with_records(
            harness, rows,
            seq_min=i * rows_per_file,
            seq_max=i * rows_per_file + rows_per_file - 1,
        )
        names.append(rel)
    partition_dir = os.path.join(
        harness["data_dir"], "partitions", DEFAULT_PARTITION
    )
    return partition_dir, names


def test_select_target_no_trigger(harness):
    """No bucket has enough files, total below MAX → no target."""
    partition_dir, names = _make_files(harness, 2)
    buckets = [[(names[0], 100)], [(names[1], 200)], [], []]
    target = CompactionManager._select_target(buckets, total_files=2,
                                              partition_dir=partition_dir)
    assert target is None


def test_select_target_full_bucket(harness):
    """Bucket 0 has 4 files → target is those 4."""
    partition_dir, names = _make_files(harness, 4)
    buckets = [
        [(names[0], 100), (names[1], 200), (names[2], 300), (names[3], 400)],
        [], [], [],
    ]
    target = CompactionManager._select_target(buckets, total_files=4,
                                              partition_dir=partition_dir)
    assert target == names


def test_select_target_force_compact_total_exceeds_max(harness):
    """No bucket fully fills, but total > MAX_DATA_FILES → take all files."""
    partition_dir, names = _make_files(harness, MAX_DATA_FILES + 1)
    # Spread across buckets (not 4+ in any single bucket)
    buckets = [[(n, 100)] for n in names[:4]]
    buckets.extend([[(n, 100)] for n in names[4:]])
    # Flatten into single-file-per-bucket style to avoid triggering the
    # bucket-full path. Use 1 file per bucket (buckets has 4 slots).
    buckets = [[], [], [], [(n, 100) for n in names]]
    target = CompactionManager._select_target(
        buckets, total_files=MAX_DATA_FILES + 1,
        partition_dir=partition_dir,
    )
    assert target is not None
    assert len(target) >= 2


def test_select_target_respects_max_segment_rows(harness, monkeypatch):
    """Merge group should not exceed MAX_SEGMENT_ROWS."""
    import litevecdb.engine.compaction as comp_mod
    # Temporarily lower the cap so we can assert behavior without
    # needing to write millions of rows.
    monkeypatch.setattr(comp_mod, "MAX_SEGMENT_ROWS", 30)
    # 4 files × 10 rows each = 40 rows total. With cap=30, should pick
    # only 3 files (30 rows) — but MIN_FILES_PER_BUCKET=4, so nothing
    # gets merged from this bucket.
    partition_dir, names = _make_files(harness, 4, rows_per_file=10)
    buckets = [
        [(n, 100) for n in names],
        [], [], [],
    ]
    target = comp_mod.CompactionManager._select_target(
        buckets, total_files=4, partition_dir=partition_dir,
    )
    # Only 3 files fit under cap → below MIN_FILES_PER_BUCKET → no merge
    assert target is None


def test_select_target_skips_over_cap_files(harness, monkeypatch):
    """Files already over MAX_SEGMENT_ROWS are skipped entirely."""
    import litevecdb.engine.compaction as comp_mod
    monkeypatch.setattr(comp_mod, "MAX_SEGMENT_ROWS", 15)
    # Each file has 20 rows, all already > cap (15) → nothing to merge.
    partition_dir, names = _make_files(harness, 4, rows_per_file=20)
    buckets = [
        [(n, 100) for n in names],
        [], [], [],
    ]
    target = comp_mod.CompactionManager._select_target(
        buckets, total_files=4, partition_dir=partition_dir,
    )
    assert target is None


# ---------------------------------------------------------------------------
# Trigger conditions on real files
# ---------------------------------------------------------------------------

def test_no_compaction_below_min_files(harness):
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET - 1):
        _write_data_file_with_records(
            harness,
            [(i + 1, f"doc_{i}", [0.5, 0.25], f"t{i}")],
            seq_min=i + 1, seq_max=i + 1,
        )
    compacted = harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    assert compacted is False
    assert len(harness["manifest"].get_data_files(DEFAULT_PARTITION)) == COMPACTION_MIN_FILES_PER_BUCKET - 1


def test_compaction_at_min_files(harness):
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET):
        _write_data_file_with_records(
            harness,
            [(i + 1, f"doc_{i}", [0.5, 0.25], f"t{i}")],
            seq_min=i + 1, seq_max=i + 1,
        )
    assert len(harness["manifest"].get_data_files(DEFAULT_PARTITION)) == COMPACTION_MIN_FILES_PER_BUCKET

    compacted = harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    assert compacted is True
    # 4 input files → 1 merged file
    assert len(harness["manifest"].get_data_files(DEFAULT_PARTITION)) == 1


# ---------------------------------------------------------------------------
# Merge correctness
# ---------------------------------------------------------------------------

def test_merge_concatenates_distinct_pks(harness):
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET):
        _write_data_file_with_records(
            harness,
            [(i + 1, f"doc_{i}", [0.5, 0.25], f"t{i}")],
            seq_min=i + 1, seq_max=i + 1,
        )
    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    [rel] = harness["manifest"].get_data_files(DEFAULT_PARTITION)
    abs_path = os.path.join(
        harness["data_dir"], "partitions", DEFAULT_PARTITION, rel
    )
    table = read_data_file(abs_path)
    assert table.num_rows == 4
    assert sorted(table.column("id").to_pylist()) == ["doc_0", "doc_1", "doc_2", "doc_3"]


def test_merge_dedups_same_pk_keeps_max_seq(harness):
    """Four files all with same pk but increasing seq → merged file
    has one row with the max seq."""
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET):
        seq = i + 1
        _write_data_file_with_records(
            harness,
            [(seq, "X", [float(i), 0.25], f"v{i}")],
            seq_min=seq, seq_max=seq,
        )
    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    [rel] = harness["manifest"].get_data_files(DEFAULT_PARTITION)
    abs_path = os.path.join(
        harness["data_dir"], "partitions", DEFAULT_PARTITION, rel
    )
    table = read_data_file(abs_path)
    assert table.num_rows == 1
    [row] = table.to_pylist()
    assert row["id"] == "X"
    assert row["_seq"] == 4
    assert row["title"] == "v3"


def test_merge_filters_deleted_pks(harness, schema):
    """Pre-populate delta_index with a tombstone, then compact files
    containing the deleted pk. The merged file must NOT contain that pk."""
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET):
        _write_data_file_with_records(
            harness,
            [(i + 1, f"doc_{i}", [0.5, 0.25], f"t{i}")],
            seq_min=i + 1, seq_max=i + 1,
        )

    # Tombstone: doc_1 deleted with seq=100 (newer than its data row).
    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["doc_1"], "_seq": [100]},
        schema=build_delta_schema(schema),
    ))

    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    [rel] = harness["manifest"].get_data_files(DEFAULT_PARTITION)
    table = read_data_file(os.path.join(
        harness["data_dir"], "partitions", DEFAULT_PARTITION, rel
    ))
    pks = set(table.column("id").to_pylist())
    assert "doc_1" not in pks
    assert pks == {"doc_0", "doc_2", "doc_3"}


def test_merge_all_rows_filtered_skips_write(harness, schema):
    """If every row is filtered by deletes, no new file is written."""
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET):
        _write_data_file_with_records(
            harness,
            [(i + 1, f"doc_{i}", [0.5, 0.25], f"t{i}")],
            seq_min=i + 1, seq_max=i + 1,
        )
    # Delete all 4 pks with a large seq.
    harness["delta_index"].add_batch(pa.RecordBatch.from_pydict(
        {"id": ["doc_0", "doc_1", "doc_2", "doc_3"], "_seq": [999, 999, 999, 999]},
        schema=build_delta_schema(schema),
    ))

    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    # All input files removed; no new file written.
    assert harness["manifest"].get_data_files(DEFAULT_PARTITION) == []


def test_old_files_deleted_from_disk(harness):
    """After compaction, the old data files must be removed from disk."""
    paths = []
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET):
        rel = _write_data_file_with_records(
            harness,
            [(i + 1, f"doc_{i}", [0.5, 0.25], f"t{i}")],
            seq_min=i + 1, seq_max=i + 1,
        )
        paths.append(os.path.join(
            harness["data_dir"], "partitions", DEFAULT_PARTITION, rel
        ))
    for p in paths:
        assert os.path.exists(p)

    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )
    for p in paths:
        assert not os.path.exists(p)


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------

def test_manifest_saved_after_compaction(harness):
    for i in range(COMPACTION_MIN_FILES_PER_BUCKET):
        _write_data_file_with_records(
            harness,
            [(i + 1, f"doc_{i}", [0.5, 0.25], f"t{i}")],
            seq_min=i + 1, seq_max=i + 1,
        )
    harness["mgr"].maybe_compact(
        DEFAULT_PARTITION, harness["manifest"], harness["delta_index"]
    )

    reloaded = Manifest.load(harness["data_dir"])
    files = reloaded.get_data_files(DEFAULT_PARTITION)
    assert len(files) == 1
