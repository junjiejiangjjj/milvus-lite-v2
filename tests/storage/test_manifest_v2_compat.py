"""Phase 9.3.2 — Manifest v2 (index_spec) + v1 backward compatibility tests."""

import json
import os

import pytest

from litevecdb.index.spec import IndexSpec
from litevecdb.storage.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT_VERSION,
    Manifest,
)


# ---------------------------------------------------------------------------
# v2 — index_spec round-trip
# ---------------------------------------------------------------------------

def test_format_version_is_2():
    assert MANIFEST_FORMAT_VERSION == 2


def test_fresh_manifest_has_no_index_spec(tmp_path):
    m = Manifest(str(tmp_path))
    assert m.index_spec is None
    assert m.format_version == 2


def test_set_index_spec_then_save_load(tmp_path):
    m = Manifest(str(tmp_path))
    spec = IndexSpec(
        field_name="vec",
        index_type="HNSW",
        metric_type="COSINE",
        build_params={"M": 16, "efConstruction": 200},
        search_params={"ef": 64},
    )
    m.set_index_spec(spec)
    m.save()

    # On-disk JSON should now contain index_spec.
    with open(os.path.join(str(tmp_path), MANIFEST_FILENAME)) as f:
        payload = json.load(f)
    assert payload["manifest_format_version"] == 2
    assert payload["index_spec"] is not None
    assert payload["index_spec"]["index_type"] == "HNSW"
    assert payload["index_spec"]["metric_type"] == "COSINE"

    # Reload and verify.
    m2 = Manifest.load(str(tmp_path))
    assert m2.index_spec == spec


def test_clear_index_spec_round_trip(tmp_path):
    m = Manifest(str(tmp_path))
    m.set_index_spec(IndexSpec(
        field_name="v", index_type="HNSW", metric_type="L2", build_params={},
    ))
    m.save()
    m.set_index_spec(None)
    m.save()

    m2 = Manifest.load(str(tmp_path))
    assert m2.index_spec is None


# ---------------------------------------------------------------------------
# v1 → v2 backward compatibility
# ---------------------------------------------------------------------------

def _write_v1_manifest(data_dir: str, payload_extras: dict = None) -> None:
    """Write a hand-crafted v1 manifest.json (no index_spec field)."""
    os.makedirs(data_dir, exist_ok=True)
    payload = {
        "manifest_format_version": 1,
        "version": 5,
        "current_seq": 100,
        "schema_version": 1,
        "active_wal_number": 3,
        "partitions": {
            "_default": {"data_files": ["data_000001_000050.parquet"], "delta_files": []},
        },
        # NOTE: no index_spec key
    }
    if payload_extras:
        payload.update(payload_extras)
    with open(os.path.join(data_dir, MANIFEST_FILENAME), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def test_v1_manifest_loads_with_none_index_spec(tmp_path):
    _write_v1_manifest(str(tmp_path))
    m = Manifest.load(str(tmp_path))
    assert m.index_spec is None
    assert m.version == 5
    assert m.current_seq == 100
    # State preserved.
    assert m.get_data_files("_default") == ["data_000001_000050.parquet"]


def test_v1_manifest_save_upgrades_to_v2(tmp_path):
    _write_v1_manifest(str(tmp_path))
    m = Manifest.load(str(tmp_path))
    m.save()  # save → v2

    with open(os.path.join(str(tmp_path), MANIFEST_FILENAME)) as f:
        payload = json.load(f)
    assert payload["manifest_format_version"] == 2
    # index_spec should be explicit None in v2.
    assert "index_spec" in payload
    assert payload["index_spec"] is None


def test_v1_manifest_save_then_set_index_spec(tmp_path):
    """Round-trip: load v1 → set spec → save → reload → spec present."""
    _write_v1_manifest(str(tmp_path))
    m = Manifest.load(str(tmp_path))
    spec = IndexSpec(
        field_name="vec", index_type="BRUTE_FORCE", metric_type="L2", build_params={},
    )
    m.set_index_spec(spec)
    m.save()

    m2 = Manifest.load(str(tmp_path))
    assert m2.index_spec == spec
    # Pre-existing v1 state preserved through the upgrade.
    assert m2.current_seq == 100
    assert m2.get_data_files("_default") == ["data_000001_000050.parquet"]


def test_v1_manifest_with_index_spec_null_field(tmp_path):
    """Some implementations may write index_spec=null even in v1.
    Should still load fine."""
    _write_v1_manifest(str(tmp_path), payload_extras={"index_spec": None})
    m = Manifest.load(str(tmp_path))
    assert m.index_spec is None


def test_set_index_spec_does_not_save_until_explicit(tmp_path):
    """set_index_spec is in-memory only — caller must save() to persist."""
    m = Manifest(str(tmp_path))
    m.save()  # baseline v2 with no index_spec

    spec = IndexSpec(
        field_name="vec", index_type="HNSW", metric_type="COSINE", build_params={},
    )
    m.set_index_spec(spec)
    # Don't save.

    m2 = Manifest.load(str(tmp_path))
    # Disk version should still have None.
    assert m2.index_spec is None
