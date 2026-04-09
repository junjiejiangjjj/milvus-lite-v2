"""Size-Tiered Compaction + Tombstone GC.

Per-partition compaction. Trigger conditions (either suffices):
    1. Some size bucket holds >= COMPACTION_MIN_FILES_PER_BUCKET files.
    2. The partition's total data file count exceeds MAX_DATA_FILES.

Compaction flow:
    1. Bucket the partition's data files by size.
    2. Pick a target set:
       - first bucket with >= MIN_FILES_PER_BUCKET, OR
       - all files (if total > MAX_DATA_FILES, force-compact)
    3. Read input files into Arrow tables.
    4. Concat → dedup by pk (keep max _seq) → filter delete tombstones
       via delta_index.is_deleted.
    5. Write the merged table to a new data file (skipped if 0 rows).
    6. Atomic Manifest update: remove old files, add new file.
    7. Delete the old files from disk.
    8. Tombstone GC: delta_index.gc_below(min_active_data_seq).

Delta files are NOT consumed during compaction. A delta entry survives
as long as any data file with seq_min <= delete_seq might still contain
its pk — that condition can only be checked globally (across all
partitions), so it lives in step 8's gc_below call. Phase-6+ optimization
can also delete fully-obsolete delta files from disk.

Crash safety:
    Crash before Step 6 (manifest commit) → orphan new file, manifest
        unchanged → recovery's _cleanup_orphan_files removes it.
    Crash during Step 6 → atomic rename, either old or new manifest.
    Crash after Step 6 mid-Step-7 → some old files orphaned (in disk
        but not in manifest); recovery cleans them.
    Crash in Step 8 → delta_index reset to in-memory rebuild on next
        start, no on-disk impact.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import pyarrow as pa

from litevecdb.constants import (
    COMPACTION_BUCKET_BOUNDARIES,
    COMPACTION_MIN_FILES_PER_BUCKET,
    MAX_DATA_FILES,
)
from litevecdb.schema.arrow_builder import (
    build_data_schema,
    get_primary_field,
)
from litevecdb.constants import DATA_FILE_TEMPLATE, SEQ_FORMAT_WIDTH
from litevecdb.storage.data_file import (
    parse_seq_range,
    read_data_file,
    write_data_file,
)

if TYPE_CHECKING:
    from litevecdb.schema.types import CollectionSchema
    from litevecdb.storage.delta_index import DeltaIndex
    from litevecdb.storage.manifest import Manifest

logger = logging.getLogger(__name__)


class CompactionManager:
    """Per-Collection compaction driver.

    Stateless across calls — every maybe_compact() call inspects the
    current Manifest from scratch. The Collection holds one instance.
    """

    def __init__(self, data_dir: str, schema: "CollectionSchema") -> None:
        self._data_dir = data_dir
        self._schema = schema
        self._pk_name = get_primary_field(schema).name
        self._data_schema = build_data_schema(schema)

    # ── public API ──────────────────────────────────────────────

    def maybe_compact(
        self,
        partition: str,
        manifest: "Manifest",
        delta_index: "DeltaIndex",
    ) -> bool:
        """Check if *partition* needs compaction; if so, run it.

        Returns True if a compaction was actually performed.
        """
        files = manifest.get_data_files(partition)
        if len(files) < COMPACTION_MIN_FILES_PER_BUCKET:
            return False

        partition_dir = os.path.join(self._data_dir, "partitions", partition)
        buckets = self._bucket_files(partition_dir, files)
        target = self._select_target(buckets, len(files))
        if target is None:
            return False

        self._compact_files(partition, partition_dir, target, manifest, delta_index)

        # Tombstone GC after compaction. Uses the GLOBAL min_active_data_seq
        # because a tombstone for pk X may need to filter rows in any
        # partition's data files.
        self._gc_tombstones(manifest, delta_index)
        return True

    # ── bucketing + selection ───────────────────────────────────

    def _bucket_files(
        self,
        partition_dir: str,
        files: List[str],
    ) -> List[List[Tuple[str, int]]]:
        """Bucket *files* by size. Returns one list per size bucket;
        each entry is (filename, byte_size)."""
        n_buckets = len(COMPACTION_BUCKET_BOUNDARIES) + 1
        buckets: List[List[Tuple[str, int]]] = [[] for _ in range(n_buckets)]
        for fn in files:
            abs_path = os.path.join(partition_dir, fn)
            if not os.path.exists(abs_path):
                # Defensive — shouldn't happen if recovery is correct.
                continue
            size = os.path.getsize(abs_path)
            buckets[self._bucket_index(size)].append((fn, size))
        return buckets

    @staticmethod
    def _bucket_index(size: int) -> int:
        for i, boundary in enumerate(COMPACTION_BUCKET_BOUNDARIES):
            if size < boundary:
                return i
        return len(COMPACTION_BUCKET_BOUNDARIES)

    @staticmethod
    def _select_target(
        buckets: List[List[Tuple[str, int]]],
        total_files: int,
    ) -> Optional[List[str]]:
        """Pick the set of files to compact.

        Strategy:
            - First, look for a bucket with >= MIN_FILES_PER_BUCKET.
              If found, return all files in that bucket.
            - Else, if total file count > MAX_DATA_FILES, return ALL files
              (force-compact across buckets).
            - Else None.
        """
        for bucket in buckets:
            if len(bucket) >= COMPACTION_MIN_FILES_PER_BUCKET:
                return [fn for fn, _ in bucket]
        if total_files > MAX_DATA_FILES:
            all_files: List[str] = []
            for bucket in buckets:
                all_files.extend(fn for fn, _ in bucket)
            return all_files
        return None

    # ── core compaction ─────────────────────────────────────────

    def _compact_files(
        self,
        partition: str,
        partition_dir: str,
        files_to_compact: List[str],
        manifest: "Manifest",
        delta_index: "DeltaIndex",
    ) -> None:
        # 1. Read all input files.
        tables: List[pa.Table] = []
        for fn in files_to_compact:
            abs_path = os.path.join(partition_dir, fn)
            tables.append(read_data_file(abs_path))
        if not tables:
            return
        combined = pa.concat_tables(tables)

        # 2. Dedup by pk (keep max _seq).
        deduped = self._dedup_max_seq(combined)

        # 3. Filter rows that have a tombstone with strictly larger seq.
        filtered = self._filter_deleted(deduped, delta_index)

        # 4. Write the merged file (or skip if filtered is empty).
        new_rel: Optional[str] = None
        if filtered.num_rows > 0:
            # Use the CONTENT seq_min for the merged filename. This is
            # the tightest safe lower bound on the smallest seq actually
            # present in the file, which lets _global_min_active_data_seq
            # advance after every compaction and lets tombstone GC make
            # progress. seq_max is an upper bound only and is bumped by
            # _pick_unique_seq_range when the natural name would collide
            # with an existing file (the bump has no GC impact because
            # seq_max is not used in that calculation).
            seqs = filtered.column("_seq").to_pylist()
            content_seq_min = min(seqs)
            content_seq_max = max(seqs)
            unique_min, unique_max = self._pick_unique_seq_range(
                partition_dir, content_seq_min, content_seq_max,
            )
            new_rel = write_data_file(
                filtered, partition_dir, seq_min=unique_min, seq_max=unique_max,
            )

        # 5. Atomic manifest update.
        manifest.remove_data_files(partition, files_to_compact)
        if new_rel is not None:
            manifest.add_data_file(partition, new_rel)
        manifest.save()

        # 6. Delete old files from disk. Past this point a crash leaves
        #    orphan files, which recovery's _cleanup_orphan_files handles.
        for fn in files_to_compact:
            abs_path = os.path.join(partition_dir, fn)
            if os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                except OSError as e:
                    logger.warning("compaction: failed to remove %s: %s", abs_path, e)

    @staticmethod
    def _pick_unique_seq_range(
        partition_dir: str,
        seq_min: int,
        seq_max: int,
    ) -> Tuple[int, int]:
        """Return a (seq_min, seq_max) pair whose corresponding filename
        does not yet exist on disk.

        Edge case: a single input file already covers the merged range
        (e.g. inputs [1,10] + [3,5] → union [1,10] which collides with
        the [1,10] input). In that case we keep seq_min and bump seq_max
        until the filename is free. The seq_max stored in the filename is
        an UPPER BOUND on actual content seqs (the file may contain
        smaller seqs only), so this is always safe.
        """
        rel_dir = "data"
        candidate_max = seq_max
        while True:
            filename = DATA_FILE_TEMPLATE.format(
                min=seq_min, max=candidate_max, w=SEQ_FORMAT_WIDTH
            )
            abs_path = os.path.join(partition_dir, rel_dir, filename)
            if not os.path.exists(abs_path):
                return seq_min, candidate_max
            candidate_max += 1

    def _dedup_max_seq(self, table: pa.Table) -> pa.Table:
        """For each pk, keep only the row with the largest _seq."""
        if table.num_rows == 0:
            return table
        pks = table.column(self._pk_name).to_pylist()
        seqs = table.column("_seq").to_pylist()
        # pk → (seq, row_idx)
        pk_to_best: dict = {}
        for i, pk in enumerate(pks):
            seq = seqs[i]
            existing = pk_to_best.get(pk)
            if existing is None or seq > existing[0]:
                pk_to_best[pk] = (seq, i)
        keep_indices = sorted(b[1] for b in pk_to_best.values())
        return table.take(pa.array(keep_indices, type=pa.int64()))

    def _filter_deleted(
        self,
        table: pa.Table,
        delta_index: "DeltaIndex",
    ) -> pa.Table:
        """Drop rows whose pk has a tombstone with strictly larger seq."""
        if table.num_rows == 0 or len(delta_index) == 0:
            return table
        pks = table.column(self._pk_name).to_pylist()
        seqs = table.column("_seq").to_pylist()
        keep_indices: List[int] = []
        for i, pk in enumerate(pks):
            if not delta_index.is_deleted(pk, seqs[i]):
                keep_indices.append(i)
        if len(keep_indices) == len(pks):
            return table  # nothing filtered
        if not keep_indices:
            # All filtered — return an empty table with the same schema.
            return table.slice(0, 0)
        return table.take(pa.array(keep_indices, type=pa.int64()))

    # ── tombstone GC ────────────────────────────────────────────

    def _gc_tombstones(
        self,
        manifest: "Manifest",
        delta_index: "DeltaIndex",
    ) -> int:
        """Drop delta_index entries below the global min_active_data_seq.

        Returns number of entries removed.

        Correctness (architectural invariant §3): a tombstone (pk,
        delete_seq) is unreachable iff every remaining data file has
        seq_min > delete_seq for every row. The conservative form below
        uses the GLOBAL minimum across all partitions and all data files.
        """
        global_min = self._global_min_active_data_seq(manifest)
        return delta_index.gc_below(global_min)

    @staticmethod
    def _global_min_active_data_seq(manifest: "Manifest") -> int:
        """Smallest seq_min across every data file in every partition.

        If there are no data files, returns sys.maxsize so the entire
        delta_index can be drained safely.
        """
        min_seq = sys.maxsize
        for _partition, files in manifest.get_all_data_files().items():
            for rel in files:
                try:
                    seq_min, _seq_max = parse_seq_range(rel)
                except ValueError:
                    continue
                if seq_min < min_seq:
                    min_seq = seq_min
        return min_seq
