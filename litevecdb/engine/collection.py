"""Collection — engine entry point.

Phase 4 scope:
    - insert(records, partition_name="_default")
    - get(pks, partition_names=None) — reads MemTable + Segments
    - search(query_vectors, top_k, metric_type, partition_names=None)
    - Synchronous flush triggered when MemTable.size() >= MEMTABLE_SIZE_LIMIT
    - Crash recovery on construction (replays WAL, rebuilds delta_index,
      loads all manifest segments)
    - WAL + MemTable + Manifest + DeltaIndex + Segment cache

NOT yet:
    - delete (Phase 5) — Collection.delete is not exposed, but the
      plumbing (DeleteOp dispatch in _apply, MemTable.apply_delete,
      delta_index, bitmap pipeline) is all in place. Phase 5 just adds
      the public method.
    - compaction (Phase 6)
    - partition CRUD (Phase 7)

Layering: Collection sits at the top of the engine layer. It is the
only place that knows about Operation dispatch — storage/wal.py and
storage/memtable.py both still take raw RecordBatches. This keeps the
storage layer free of engine-layer types.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from litevecdb.constants import (
    ALL_PARTITIONS,
    DEFAULT_PARTITION,
    FILTER_CACHE_SIZE,
    MEMTABLE_SIZE_LIMIT,
)
from litevecdb.engine.compaction import CompactionManager
from litevecdb.engine.flush import execute_flush
from litevecdb.engine.operation import DeleteOp, InsertOp, Operation
from litevecdb.engine.recovery import execute_recovery
from litevecdb.exceptions import PartitionNotFoundError
from litevecdb.schema.arrow_builder import (
    build_wal_data_schema,
    build_wal_delta_schema,
    get_primary_field,
    get_vector_field,
)
from litevecdb.schema.types import CollectionSchema
from litevecdb.schema.validation import (
    separate_dynamic_fields,
    validate_record,
    validate_schema,
)
from litevecdb.search.assembler import assemble_candidates
from litevecdb.search.executor import execute_search
from litevecdb.search.executor_indexed import execute_search_with_index
from litevecdb.storage.manifest import Manifest
from litevecdb.storage.memtable import MemTable
from litevecdb.storage.segment import Segment
from litevecdb.storage.wal import WAL

if False:  # TYPE_CHECKING
    from litevecdb.search.filter.semantic import CompiledExpr  # noqa: F401


# Segment cache key: (partition, relative_path) — relative_path is what
# the manifest stores so two segments cannot collide on the same name.
_SegmentKey = Tuple[str, str]


def _row_matches_filter(record: dict, compiled_filter) -> bool:
    """Evaluate a CompiledExpr against a single dict row.

    Used by Collection.get() after a successful pk lookup, when we need
    to filter the single hit row by the user's expression. Builds a
    1-row pa.Table on the fly so we can reuse the existing evaluator.
    For Phase F1 we always go through the python_backend (cheaper than
    constructing an Arrow table for one row).
    """
    from litevecdb.search.filter.eval.python_backend import _eval_row
    result = _eval_row(compiled_filter.ast, record)
    return bool(result) if result is not None else False


class Collection:
    """A single Collection — schema + WAL + MemTable + Manifest + DeltaIndex.

    Construction is non-destructive and crash-tolerant:
        1. Load Manifest (with .prev fallback if current is corrupted).
        2. Run recovery — replay any WAL files, rebuild DeltaIndex,
           clean orphan Parquet files.
        3. Allocate a fresh WAL number = max(found WALs + 1, manifest's
           active_wal_number).

    insert() validates fail-fast (no partial state on failure), allocates
    one _seq per row, writes to WAL then MemTable, and triggers a
    synchronous flush if the MemTable hit the size limit.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        schema: CollectionSchema,
    ) -> None:
        validate_schema(schema)

        self._name = name
        self._data_dir = data_dir
        self._schema = schema
        self._pk_name = get_primary_field(schema).name
        self._vector_name = get_vector_field(schema).name

        self._wal_data_schema = build_wal_data_schema(schema)
        self._wal_delta_schema = build_wal_delta_schema(schema)

        os.makedirs(data_dir, exist_ok=True)

        # ── 1. load manifest ────────────────────────────────────
        self._manifest = Manifest.load(data_dir)

        # ── 2. recovery ─────────────────────────────────────────
        self._memtable, self._delta_index, next_wal_number = execute_recovery(
            data_dir=data_dir,
            schema=schema,
            manifest=self._manifest,
        )

        # ── 3. fresh WAL ────────────────────────────────────────
        # next_seq must clear both manifest's recorded seq AND any seq
        # we just learned from WAL replay.
        self._next_seq = max(
            self._manifest.current_seq, self._memtable.max_seq
        ) + 1

        wal_dir = os.path.join(data_dir, "wal")
        self._wal = WAL(
            wal_dir=wal_dir,
            wal_data_schema=self._wal_data_schema,
            wal_delta_schema=self._wal_delta_schema,
            wal_number=next_wal_number,
        )

        # ── 4. segment cache ────────────────────────────────────
        # Loaded from every data file referenced by the manifest. The
        # cache is keyed by (partition, relative_path) and is refreshed
        # after each flush so the search path always sees the latest
        # set of immutable segments.
        self._segment_cache: Dict[_SegmentKey, Segment] = {}
        self._refresh_segment_cache()

        # ── 5. compaction manager ───────────────────────────────
        self._compaction_mgr = CompactionManager(data_dir, schema)

        # ── 6. filter expression cache (Phase F2c) ──────────────
        # LRU on (expr_string → CompiledExpr) — schema is implicit since
        # the cache is per-Collection. Bounded by FILTER_CACHE_SIZE so
        # adversarial / heavy expression diversity can't OOM.
        from litevecdb.search.filter.cache import LRUCache
        self._filter_cache: LRUCache = LRUCache(maxsize=FILTER_CACHE_SIZE)

    # ── public API ──────────────────────────────────────────────

    def insert(
        self,
        records: List[dict],
        partition_name: str = DEFAULT_PARTITION,
    ) -> List[Any]:
        """Insert records into the collection. Returns the list of pks.

        Each record is validated up-front (fail fast — no partial state
        on validation error). After WAL+MemTable apply, if the MemTable
        has hit MEMTABLE_SIZE_LIMIT, a synchronous flush runs before
        returning.
        """
        if not isinstance(records, list):
            raise TypeError(f"records must be a list, got {type(records).__name__}")
        if not records:
            return []

        if not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # 1. validate every record up-front
        for r in records:
            validate_record(r, self._schema)

        # 2. allocate seqs
        seq_start = self._next_seq
        self._next_seq += len(records)
        seqs = list(range(seq_start, seq_start + len(records)))

        # 3. build wal_data RecordBatch
        batch = self._build_wal_data_batch(records, partition_name, seqs)

        # 4. construct Operation and dispatch
        op = InsertOp(partition=partition_name, batch=batch)
        self._apply(op)

        # 5. trigger flush if we hit the size limit
        if self._memtable.size() >= MEMTABLE_SIZE_LIMIT:
            self._trigger_flush()

        return [r[self._pk_name] for r in records]

    def delete(
        self,
        pks: List[Any],
        partition_name: Optional[str] = None,
    ) -> int:
        """Delete a batch of pks. Returns the number of pks scheduled.

        ``partition_name=None`` is a cross-partition delete: the
        tombstone applies to whichever partition the pk currently lives
        in, and at flush time it is replicated into the delta files of
        every existing partition.

        Phase-5 semantics:
            - The whole batch shares ONE _seq (architectural invariant:
              batch delete is one logical event).
            - Deleting a non-existent pk is NOT an error — it just
              writes a tombstone that will never match anything.
            - This method does not return whether each pk actually
              existed; it returns ``len(pks)`` so the caller can
              distinguish "called with N" from "called with 0".
        """
        if not isinstance(pks, list):
            raise TypeError(f"pks must be a list, got {type(pks).__name__}")
        if not pks:
            return 0

        target_partition = partition_name if partition_name is not None else ALL_PARTITIONS

        # Validate the explicit partition exists. ALL_PARTITIONS is a
        # sentinel and is always valid.
        if partition_name is not None and not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # Allocate ONE seq for the whole batch.
        seq = self._next_seq
        self._next_seq += 1

        batch = self._build_wal_delta_batch(pks, target_partition, seq)
        op = DeleteOp(partition=target_partition, batch=batch)
        self._apply(op)

        if self._memtable.size() >= MEMTABLE_SIZE_LIMIT:
            self._trigger_flush()

        return len(pks)

    def get(
        self,
        pks: List[Any],
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,
    ) -> List[dict]:
        """Point read across MemTable + segments.

        Lookup order per pk:
            1. MemTable._pk_index → live insert (newest possible state)
            2. MemTable._delete_index → live tombstone shadows everything
            3. Segments → scan for the largest seq across all segments
               in the requested partitions; check delta_index for an
               on-disk tombstone with a larger seq.

        If ``expr`` is provided, hit records are additionally filtered
        by the compiled scalar expression.

        Returns records in input pk order; missing pks are skipped
        (NOT padded with None).
        """
        if not isinstance(pks, list):
            raise TypeError(f"pks must be a list, got {type(pks).__name__}")

        partition_filter = set(partition_names) if partition_names else None
        compiled_filter = self._compile_filter(expr) if expr else None

        out: List[dict] = []

        for pk in pks:
            rec: Optional[dict] = None

            # Step 1: live insert in MemTable.
            mt_rec = self._memtable.get(pk)
            if mt_rec is not None:
                rec = mt_rec
            elif self._memtable.is_locally_deleted(pk):
                # Step 2: live tombstone shadows any segment hit.
                continue
            else:
                # Step 3: scan segments for the latest version of pk.
                best_seq = -1
                best_segment: Optional[Segment] = None
                best_row_idx: int = -1
                for segment in self._segment_cache.values():
                    if partition_filter is not None and segment.partition not in partition_filter:
                        continue
                    row_idx = segment.find_row(pk)
                    if row_idx is None:
                        continue
                    seq = int(segment.seqs[row_idx])
                    if seq > best_seq:
                        best_seq = seq
                        best_segment = segment
                        best_row_idx = row_idx

                if best_segment is not None:
                    if not self._delta_index.is_deleted(pk, best_seq):
                        rec = best_segment.row_to_dict(best_row_idx)

            if rec is None:
                continue

            # Apply optional scalar filter to the single hit row.
            if compiled_filter is not None and not _row_matches_filter(rec, compiled_filter):
                continue

            out.append(rec)

        return out

    def search(
        self,
        query_vectors: List[list],
        top_k: int = 10,
        metric_type: str = "COSINE",
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[List[dict]]:
        """Vector top-k search.

        Args:
            query_vectors: list of length nq, each item a list of length dim.
            top_k: requested k.
            metric_type: "COSINE" / "L2" / "IP".
            partition_names: optional partition filter.
            expr: optional Milvus-style scalar filter expression. Hits
                that don't match are excluded before top-k selection.
            output_fields: optional whitelist of fields to include in
                result ``entity``. Phase 9.1 semantics:
                  - None  → all fields except pk and vector (legacy default)
                  - []    → empty entity (only id + distance)
                  - list  → exactly those fields (pk always surfaced as
                            "id"; vector included only if listed)

        Returns:
            List of length nq. Each inner list has up to top_k dicts of
            shape ``{"id": pk, "distance": float, "entity": {field: value, ...}}``,
            sorted by ascending distance.
        """
        if not isinstance(query_vectors, list):
            raise TypeError(
                f"query_vectors must be a list, got {type(query_vectors).__name__}"
            )
        if not query_vectors:
            return []

        # Convert to numpy (nq, dim).
        q_arr = np.asarray(query_vectors, dtype=np.float32)
        if q_arr.ndim != 2:
            raise ValueError(
                f"query_vectors must be a 2-D list, got shape {q_arr.shape}"
            )

        compiled_filter = self._compile_filter(expr) if expr else None

        # Phase 9.2: index-aware path. Each segment uses its attached
        # VectorIndex if present, else an ad-hoc BruteForceIndex; the
        # memtable always uses brute force; results are merged across
        # sources at the end. The differential test in
        # tests/search/test_executor_with_index.py validates that this
        # produces the same top-k as the legacy execute_search path
        # for any (records, expr, partition) combination.
        return execute_search_with_index(
            query_vectors=q_arr,
            segments=self._segment_cache.values(),
            memtable=self._memtable,
            delta_index=self._delta_index,
            top_k=top_k,
            metric_type=metric_type,
            pk_field=self._pk_name,
            vector_field=self._vector_name,
            partition_names=partition_names,
            compiled_filter=compiled_filter,
            output_fields=output_fields,
        )

    def query(
        self,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Pure scalar query — no vector, no distance.

        Returns all records matching the filter expression. Used for
        Milvus-style query() workflows where you just want to find rows
        by their scalar attributes.

        Args:
            expr: required Milvus-style filter expression
            output_fields: subset of fields to include in returned dicts.
                None means all schema fields (with _seq / _partition stripped).
                The pk field is always included.
            partition_names: optional partition filter
            limit: max number of rows to return; None = unbounded

        Returns:
            List of dicts (each a record matching the filter). Order is
            "segments first, then MemTable" — within each source, the
            order is the underlying iteration order. No top-k sort.
        """
        if not isinstance(expr, str) or not expr:
            raise TypeError("query() requires a non-empty filter expression")

        compiled_filter = self._compile_filter(expr)

        all_pks, all_seqs, _all_vectors, all_records, filter_mask = assemble_candidates(
            segments=self._segment_cache.values(),
            memtable=self._memtable,
            vector_field=self._vector_name,
            partition_names=partition_names,
            filter_compiled=compiled_filter,
        )

        if not all_pks:
            return []

        # Combine bitmap (dedup + tombstone) with filter_mask via build_valid_mask.
        from litevecdb.search.bitmap import build_valid_mask
        mask = build_valid_mask(
            all_pks, all_seqs, self._delta_index, filter_mask=filter_mask,
        )

        # Project + limit.
        live_indices = np.flatnonzero(mask)
        out: List[dict] = []
        for i in live_indices:
            rec = all_records[int(i)]
            out.append(self._project_record(rec, output_fields))
            if limit is not None and len(out) >= limit:
                break
        return out

    def _compile_filter(self, expr_str: str) -> "CompiledExpr":
        """Parse + compile a filter expression, with LRU caching.

        The cache is keyed only on the expression string because the
        schema is implicit (this Collection's). Schema is immutable for
        the lifetime of a Collection, so cached entries never go stale.
        """
        cached = self._filter_cache.get(expr_str)
        if cached is not None:
            return cached
        from litevecdb.search.filter import compile_filter
        compiled = compile_filter(expr_str, self._schema)
        self._filter_cache.put(expr_str, compiled)
        return compiled

    def _project_record(
        self,
        record: dict,
        output_fields: Optional[List[str]],
    ) -> dict:
        """Apply output_fields projection to a record dict.

        - None → return the record as-is (with internal fields stripped
          by the upstream code path)
        - list → keep only the named fields, plus the pk field
        """
        if output_fields is None:
            return record
        keep = set(output_fields)
        keep.add(self._pk_name)
        return {k: v for k, v in record.items() if k in keep}

    def flush(self) -> None:
        """Force a synchronous flush of the current MemTable.

        No-op if the MemTable is empty.
        """
        if self._memtable.size() == 0:
            return
        self._trigger_flush()

    # ── partition CRUD (Phase 9.1) ──────────────────────────────

    def create_partition(self, partition_name: str) -> None:
        """Create a new partition.

        - Registers the partition on the manifest (raises
          PartitionAlreadyExistsError if already there).
        - Persists the manifest atomically.
        - Creates the on-disk partition directory so flush can write
          into it later. The dir is empty at this point.
        """
        self._manifest.add_partition(partition_name)
        self._manifest.save()
        partition_dir = os.path.join(
            self._data_dir, "partitions", partition_name
        )
        os.makedirs(partition_dir, exist_ok=True)

    def drop_partition(self, partition_name: str) -> None:
        """Drop a partition and remove all its on-disk files.

        - Forbidden for the default partition (raises
          DefaultPartitionError via manifest).
        - Raises PartitionNotFoundError if the partition doesn't exist.
        - Auto-flushes any pending MemTable rows first so we don't
          lose live writes that target this partition.
        - Removes the partition from the manifest, then deletes the
          on-disk partition directory (which contains data + delta +
          future indexes/).
        - Drops any cached Segments belonging to this partition.

        Tombstones in delta_index for the dropped partition's pks are
        left intact — they will be GC'd by the regular tombstone GC
        once min_active_data_seq advances past them. This is safe
        because dropping a partition means there is no longer any
        live data row those tombstones could shadow.
        """
        # Validate first so we don't trigger an unnecessary flush.
        if not self._manifest.has_partition(partition_name):
            raise PartitionNotFoundError(partition_name)

        # Flush any pending writes so we don't drop in-flight rows
        # silently (the user's "insert then drop" should not lose
        # the inserts).
        if self._memtable.size() > 0:
            self._trigger_flush()

        # remove_partition raises DefaultPartitionError or
        # PartitionNotFoundError as appropriate.
        self._manifest.remove_partition(partition_name)
        self._manifest.save()

        # Drop in-memory segment cache entries for this partition.
        for key in list(self._segment_cache.keys()):
            if key[0] == partition_name:
                del self._segment_cache[key]

        # Remove on-disk partition directory.
        partition_dir = os.path.join(
            self._data_dir, "partitions", partition_name
        )
        if os.path.exists(partition_dir):
            import shutil
            shutil.rmtree(partition_dir, ignore_errors=False)

    def list_partitions(self) -> List[str]:
        """Return all partition names, sorted."""
        return self._manifest.list_partitions()

    def has_partition(self, partition_name: str) -> bool:
        """Check whether a partition exists."""
        return self._manifest.has_partition(partition_name)

    # ── statistics & introspection (Phase 9.1) ──────────────────

    @property
    def name(self) -> str:
        """Collection name (read-only)."""
        return self._name

    @property
    def schema(self) -> CollectionSchema:
        """Collection schema (read-only)."""
        return self._schema

    @property
    def num_entities(self) -> int:
        """Approximate live row count across MemTable + segments.

        Walks pks + seqs only (no record materialization), then runs
        the same bitmap pipeline as search to dedup upserts and apply
        tombstones. O(N) where N is the total candidate row count.

        This is the value pymilvus's get_collection_stats reports as
        ``row_count``.
        """
        pk_chunks: List[List[Any]] = []
        seq_chunks: List[np.ndarray] = []

        for seg in self._segment_cache.values():
            if seg.num_rows == 0:
                continue
            pk_chunks.append(list(seg.pks))
            seq_chunks.append(seg.seqs)

        mt_pks, mt_seqs, _vecs, _records = self._memtable.to_search_arrays(
            vector_field=self._vector_name,
            partition_names=None,
        )
        if mt_pks:
            pk_chunks.append(mt_pks)
            seq_chunks.append(mt_seqs)

        if not pk_chunks:
            return 0

        all_pks: List[Any] = []
        for c in pk_chunks:
            all_pks.extend(c)
        all_seqs = np.concatenate(seq_chunks)

        from litevecdb.search.bitmap import build_valid_mask
        mask = build_valid_mask(all_pks, all_seqs, self._delta_index)
        return int(mask.sum())

    def describe(self) -> dict:
        """Return a dict summarizing the Collection.

        Used by pymilvus's describe_collection mapping in Phase 10.
        Mirrors the shape Milvus returns: collection name + schema +
        partition list + row count. Phase 9.3 will extend this with
        index info and load state.
        """
        return {
            "name": self._name,
            "schema": {
                "fields": [
                    {
                        "name": f.name,
                        "dtype": f.dtype.name,
                        "is_primary": f.is_primary,
                        "nullable": f.nullable,
                        "dim": f.dim,
                        "max_length": f.max_length,
                    }
                    for f in self._schema.fields
                ],
                "enable_dynamic_field": self._schema.enable_dynamic_field,
            },
            "partitions": self.list_partitions(),
            "num_entities": self.num_entities,
        }

    # ── orchestration ───────────────────────────────────────────

    def _apply(self, op: Operation) -> None:
        """Single write entry point.

        Dispatches Operation to WAL and MemTable. Storage layer methods
        take raw batches (no Operation knowledge), so the dispatch is
        explicit here.
        """
        if isinstance(op, InsertOp):
            self._wal.write_insert(op.batch)
            self._memtable.apply_insert(op.batch)
        else:  # DeleteOp
            self._wal.write_delete(op.batch)
            self._memtable.apply_delete(op.batch)

    def _trigger_flush(self) -> None:
        """Step 1 of the flush pipeline + execute_flush for Steps 2-7.

        Step 1 (here): freeze the current (MemTable, WAL), swap in fresh
        ones on the Collection. Then call execute_flush on the frozen
        pair, which handles disk writes, manifest commit, WAL cleanup.
        """
        # ── Step 1: freeze ──────────────────────────────────────
        frozen_memtable = self._memtable
        frozen_wal = self._wal
        new_wal_number = frozen_wal.number + 1

        # Swap in fresh ones BEFORE running execute_flush so that any
        # subsequent insert calls (in case of async future) hit the new
        # MemTable. In sync mode this is order-preserving anyway.
        self._memtable = MemTable(self._schema)
        wal_dir = os.path.join(self._data_dir, "wal")
        self._wal = WAL(
            wal_dir=wal_dir,
            wal_data_schema=self._wal_data_schema,
            wal_delta_schema=self._wal_delta_schema,
            wal_number=new_wal_number,
        )

        # ── Steps 2-7: execute_flush ────────────────────────────
        execute_flush(
            frozen_memtable=frozen_memtable,
            frozen_wal=frozen_wal,
            data_dir=self._data_dir,
            schema=self._schema,
            manifest=self._manifest,
            delta_index=self._delta_index,
            new_wal_number=new_wal_number,
        )

        # ── post-flush: maybe trigger compaction per partition ──
        # maybe_compact() is a no-op for partitions below the trigger
        # threshold, so we can call it on all known partitions cheaply.
        for partition in self._manifest.list_partitions():
            self._compaction_mgr.maybe_compact(
                partition, self._manifest, self._delta_index
            )

        # ── refresh segment cache (picks up flushed + compacted) ─
        self._refresh_segment_cache()

    def _refresh_segment_cache(self) -> None:
        """Reconcile self._segment_cache with the manifest's data files.

        - Adds segments for any newly-written files.
        - Drops segments for files no longer referenced (e.g. after
          compaction in Phase 6).
        - Existing segments stay loaded (the underlying Parquet is
          immutable, so no need to reload).
        """
        current_keys: set = set()
        for partition, rels in self._manifest.get_all_data_files().items():
            for rel in rels:
                key = (partition, rel)
                current_keys.add(key)
                if key in self._segment_cache:
                    continue
                abs_path = os.path.join(
                    self._data_dir, "partitions", partition, rel
                )
                if not os.path.exists(abs_path):
                    # Should have been caught by recovery, but be defensive.
                    continue
                self._segment_cache[key] = Segment.load(
                    file_path=abs_path,
                    partition=partition,
                    pk_field=self._pk_name,
                    vector_field=self._vector_name,
                )

        # Evict segments for files that are no longer in the manifest.
        for key in list(self._segment_cache.keys()):
            if key not in current_keys:
                del self._segment_cache[key]

    # ── batch builders ──────────────────────────────────────────

    def _build_wal_data_batch(
        self,
        records: List[dict],
        partition_name: str,
        seqs: List[int],
    ) -> pa.RecordBatch:
        """Build a RecordBatch matching wal_data_schema.

        Splits dynamic fields into $meta if enable_dynamic_field is set.
        """
        n = len(records)
        cols: dict[str, list] = {
            "_seq": seqs,
            "_partition": [partition_name] * n,
        }

        for f in self._schema.fields:
            cols[f.name] = []

        meta_col: Optional[List[Optional[str]]] = None
        if self._schema.enable_dynamic_field:
            meta_col = []

        for r in records:
            schema_part, meta_json = separate_dynamic_fields(r, self._schema)
            for f in self._schema.fields:
                cols[f.name].append(schema_part.get(f.name))
            if meta_col is not None:
                meta_col.append(meta_json)

        if meta_col is not None:
            cols["$meta"] = meta_col

        return pa.RecordBatch.from_pydict(cols, schema=self._wal_data_schema)

    def _build_wal_delta_batch(
        self,
        pks: List[Any],
        partition_name: str,
        seq: int,
    ) -> pa.RecordBatch:
        """Build a wal_delta RecordBatch for a delete operation.

        All rows share the same _seq and _partition (architectural
        invariant: batch delete is one logical event).
        """
        return pa.RecordBatch.from_pydict(
            {
                self._pk_name: pks,
                "_seq": [seq] * len(pks),
                "_partition": [partition_name] * len(pks),
            },
            schema=self._wal_delta_schema,
        )

    # ── lifecycle ───────────────────────────────────────────────

    def close(self) -> None:
        """Flush any pending state and shut down the WAL.

        Phase-3 close runs a final flush so the on-disk state is
        consistent with whatever insert calls returned successfully.
        """
        if self._memtable.size() > 0:
            self._trigger_flush()
        else:
            # Even an empty MemTable needs WAL cleanup so we don't leave
            # an empty wal file behind.
            self._wal.close_and_delete()

    # ── introspection ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def schema(self) -> CollectionSchema:
        return self._schema

    @property
    def pk_field(self) -> str:
        return self._pk_name

    def count(self) -> int:
        """Number of live records in the MemTable.

        NOTE: this is the in-memory count only. Phase 4 will add a
        full collection count that includes flushed Parquet files.
        """
        return self._memtable.size()
