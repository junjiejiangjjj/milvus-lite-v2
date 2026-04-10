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
from litevecdb.exceptions import (
    CollectionNotLoadedError,
    IndexAlreadyExistsError,
    IndexNotFoundError,
    PartitionNotFoundError,
    SchemaValidationError,
)
from litevecdb.index.brute_force import BruteForceIndex
from litevecdb.index.spec import IndexSpec
from litevecdb.schema.types import DataType, FunctionType
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

        # ── 7. index state machine (Phase 9.3) ──────────────────
        # _index_spec is mirrored from manifest so we don't reload on
        # every access; it's also the canonical "is there an index?" flag.
        # _load_state mirrors Milvus's loaded/released semantics:
        #   - Collections WITHOUT an IndexSpec auto-load on construction
        #     (there is nothing to build, and we don't want to break
        #      backward compatibility for users who never call create_index)
        #   - Collections WITH an IndexSpec start as released; the user
        #     must call load() explicitly, mirroring pymilvus behavior
        self._index_spec: Optional[IndexSpec] = self._manifest.index_spec
        self._load_state: str = "loaded" if self._index_spec is None else "released"

        # ── 8. auto_id support (Phase 15) ──────────────────────────
        pk_field = get_primary_field(schema)
        self._auto_id: bool = pk_field.auto_id
        # _next_auto_id tracks the next ID to assign. We initialize it
        # from the manifest's current_seq to ensure monotonic growth
        # across restarts. This is safe because _seq is always >= any
        # previously assigned auto_id.
        self._next_auto_id: int = self._next_seq

        # ── 9. BM25 function analyzers (Phase 11) ─────────────────
        # Pre-build an Analyzer for each BM25 function so insert()
        # can auto-generate sparse vector fields from text input.
        # _bm25_functions: list of (input_field_name, output_field_name, Analyzer)
        self._bm25_functions: List[Tuple[str, str, Any]] = []
        if schema.functions:
            from litevecdb.analyzer.factory import create_analyzer
            field_by_name = {f.name: f for f in schema.fields}
            for func in schema.functions:
                if func.function_type == FunctionType.BM25:
                    in_name = func.input_field_names[0]
                    out_name = func.output_field_names[0]
                    in_field = field_by_name[in_name]
                    analyzer = create_analyzer(in_field.analyzer_params)
                    self._bm25_functions.append((in_name, out_name, analyzer))

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

        # 1. auto-generate primary key IDs if auto_id is enabled
        if self._auto_id:
            id_start = self._next_auto_id
            self._next_auto_id += len(records)
            for i, r in enumerate(records):
                if self._pk_name not in r or r[self._pk_name] is None:
                    r[self._pk_name] = id_start + i

        # 2. auto-generate BM25 function output fields
        if self._bm25_functions:
            self._apply_bm25_functions(records)

        # 3. validate every record up-front
        for r in records:
            validate_record(r, self._schema)

        # 4. allocate seqs
        seq_start = self._next_seq
        self._next_seq += len(records)
        seqs = list(range(seq_start, seq_start + len(records)))

        # 4. build wal_data RecordBatch
        batch = self._build_wal_data_batch(records, partition_name, seqs)

        # 5. construct Operation and dispatch
        op = InsertOp(partition=partition_name, batch=batch)
        self._apply(op)

        # 6. trigger flush if we hit the size limit
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

        self._require_loaded()

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
        anns_field: Optional[str] = None,
        group_by_field: Optional[str] = None,
        group_size: int = 1,
        strict_group_size: bool = False,
        radius: Optional[float] = None,
        range_filter: Optional[float] = None,
    ) -> List[List[dict]]:
        """Vector top-k search.

        Args:
            query_vectors: list of length nq, each item a list of length dim
                (for FLOAT_VECTOR) or list of dict (for SPARSE_FLOAT_VECTOR).
            top_k: requested k (number of groups when group_by_field is set).
            metric_type: "COSINE" / "L2" / "IP" / "BM25".
            partition_names: optional partition filter.
            expr: optional Milvus-style scalar filter expression.
            output_fields: optional whitelist of fields to include in entity.
            anns_field: name of the vector field to search on.
            group_by_field: optional scalar field to group results by.
            group_size: number of results per group (default 1).
            strict_group_size: if True, discard groups with fewer than
                group_size results.
            radius: optional distance lower bound (exclusive). Only results
                with distance > radius are returned.
            range_filter: optional distance upper bound (inclusive). Only
                results with distance <= range_filter are returned.

        Returns:
            List of length nq. Each inner list has dicts of shape
            ``{"id": pk, "distance": float, "entity": {field: value, ...}}``.
            When group_by_field is set, results are grouped and flattened
            (up to top_k * group_size total hits per query).
        """
        if not isinstance(query_vectors, list):
            raise TypeError(
                f"query_vectors must be a list, got {type(query_vectors).__name__}"
            )
        if not query_vectors:
            return []

        self._require_loaded()

        # Validate group_by_field
        if group_by_field is not None:
            gf = next((f for f in self._schema.fields if f.name == group_by_field), None)
            if gf is None:
                raise SchemaValidationError(
                    f"group_by_field {group_by_field!r} not found in schema"
                )
            _GROUP_BY_ALLOWED = (
                DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
                DataType.BOOL, DataType.VARCHAR,
            )
            if gf.dtype not in _GROUP_BY_ALLOWED:
                raise SchemaValidationError(
                    f"group_by_field {group_by_field!r} has type {gf.dtype.name} "
                    f"which is not supported for group_by"
                )

        # Over-fetch when group_by or range search is active
        effective_top_k = top_k
        if group_by_field is not None:
            effective_top_k = max(top_k * group_size * 3, top_k * 10)
        if radius is not None or range_filter is not None:
            effective_top_k = max(effective_top_k, top_k * 5)

        # Resolve the target vector field
        vector_field = self._resolve_anns_field(anns_field)
        field_schema = next(f for f in self._schema.fields if f.name == vector_field)

        if field_schema.dtype == DataType.SPARSE_FLOAT_VECTOR:
            raw_results = self._search_sparse(
                query_vectors=query_vectors,
                vector_field=vector_field,
                top_k=effective_top_k,
                metric_type=metric_type,
                partition_names=partition_names,
                expr=expr,
                output_fields=output_fields,
            )
        else:
            # Dense float vector search (existing path)
            q_arr = np.asarray(query_vectors, dtype=np.float32)
            if q_arr.ndim != 2:
                raise ValueError(
                    f"query_vectors must be a 2-D list, got shape {q_arr.shape}"
                )
            compiled_filter = self._compile_filter(expr) if expr else None
            raw_results = execute_search_with_index(
            query_vectors=q_arr,
            segments=self._segment_cache.values(),
            memtable=self._memtable,
            delta_index=self._delta_index,
            top_k=effective_top_k,
            metric_type=metric_type,
            pk_field=self._pk_name,
            vector_field=vector_field,
            partition_names=partition_names,
            compiled_filter=compiled_filter,
            output_fields=output_fields,
        )

        # Apply range filter (before group_by)
        if radius is not None or range_filter is not None:
            raw_results = _apply_range_filter(raw_results, radius, range_filter, top_k)

        # Apply group_by post-processing
        if group_by_field is not None:
            raw_results = _apply_group_by(
                raw_results, group_by_field, top_k, group_size, strict_group_size,
            )

        return raw_results

    def _resolve_anns_field(self, anns_field: Optional[str]) -> str:
        """Resolve the anns_field parameter to a concrete field name.

        Returns the first FLOAT_VECTOR field if anns_field is None.
        Validates that the field exists and is a vector type.
        """
        if anns_field is None:
            return self._vector_name

        field = next((f for f in self._schema.fields if f.name == anns_field), None)
        if field is None:
            raise SchemaValidationError(
                f"anns_field {anns_field!r} not found in schema"
            )
        if field.dtype not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR):
            raise SchemaValidationError(
                f"anns_field {anns_field!r} is not a vector field "
                f"(dtype={field.dtype.name})"
            )
        return anns_field

    def _search_sparse(
        self,
        query_vectors: List,
        vector_field: str,
        top_k: int,
        metric_type: str,
        partition_names: Optional[List[str]],
        expr: Optional[str],
        output_fields: Optional[List[str]],
    ) -> List[List[dict]]:
        """Sparse vector search using BM25 scoring.

        Collects sparse vectors from all segments + memtable, builds
        a SparseInvertedIndex on the fly, and returns BM25 top-k results.

        For text search via BM25 Function: query_vectors should be
        sparse dicts (term_hash → weight). The caller (gRPC adapter or
        direct API) is responsible for tokenizing query text into
        sparse vectors before calling search().
        """
        from litevecdb.analyzer.sparse import bytes_to_sparse
        from litevecdb.index.sparse_inverted import SparseInvertedIndex

        # Partition filter
        partition_filter: Optional[set] = None
        if partition_names is not None:
            partition_filter = set(partition_names)

        # Collect all sparse vectors, pks, seqs, and records from segments
        all_pks: list = []
        all_seqs: list = []
        all_sparse: list = []  # list of dict[int, float]
        all_records: list = []

        for seg in self._segment_cache.values():
            if partition_filter is not None:
                # Check if segment belongs to a valid partition
                seg_partition = seg.partition_name if hasattr(seg, 'partition_name') else None
                if seg_partition is not None and seg_partition not in partition_filter:
                    continue

            table = seg.table
            if table is None or len(table) == 0:
                continue

            pk_col = table.column(self._pk_name)
            seq_col = table.column("_seq")
            sparse_col = table.column(vector_field)

            for i in range(len(table)):
                pk = pk_col[i].as_py()
                seq = seq_col[i].as_py()
                raw = sparse_col[i].as_py()
                sv = bytes_to_sparse(raw) if isinstance(raw, bytes) else (raw or {})

                all_pks.append(pk)
                all_seqs.append(seq)
                all_sparse.append(sv)
                # Build record dict for result materialization
                row = {}
                for col_name in table.column_names:
                    if col_name == "_seq":
                        continue
                    row[col_name] = table.column(col_name)[i].as_py()
                all_records.append(row)

        # Collect from memtable
        mt = self._memtable
        for pk, (batch_idx, row_idx, seq) in mt._pk_index.items():
            batch = mt._insert_batches[batch_idx]
            if partition_filter is not None:
                part = batch.column("_partition")[row_idx].as_py()
                if part not in partition_filter:
                    continue

            raw = batch.column(vector_field)[row_idx].as_py()
            sv = bytes_to_sparse(raw) if isinstance(raw, bytes) else (raw or {})

            all_pks.append(pk)
            all_seqs.append(seq)
            all_sparse.append(sv)
            row = {}
            for col_name in batch.column_names:
                if col_name in ("_seq", "_partition"):
                    continue
                row[col_name] = batch.column(col_name)[row_idx].as_py()
            all_records.append(row)

        if not all_pks:
            return [[] for _ in query_vectors]

        # Build valid mask (dedup by max seq per pk + tombstone check)
        pk_max_seq: Dict[Any, int] = {}
        pk_last_idx: Dict[Any, int] = {}
        for i, (pk, seq) in enumerate(zip(all_pks, all_seqs)):
            if pk not in pk_max_seq or seq > pk_max_seq[pk]:
                pk_max_seq[pk] = seq
                pk_last_idx[pk] = i

        n = len(all_pks)
        valid_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            pk = all_pks[i]
            seq = all_seqs[i]
            if seq == pk_max_seq[pk] and i == pk_last_idx[pk]:
                # Check tombstone
                if not self._delta_index.is_deleted(pk, seq):
                    valid_mask[i] = True

        # Apply scalar filter
        compiled_filter = None
        if expr:
            compiled_filter = self._compile_filter(expr)
        if compiled_filter is not None:
            from litevecdb.search.filter.eval.python_backend import _eval_row
            for i in range(n):
                if valid_mask[i]:
                    result = _eval_row(compiled_filter.ast, all_records[i])
                    if not result:
                        valid_mask[i] = False

        # Build inverted index
        bm25_k1 = 1.5
        bm25_b = 0.75
        # Check if there's a sparse IndexSpec with BM25 params
        if self._index_spec and self._index_spec.index_type == "SPARSE_INVERTED_INDEX":
            bm25_k1 = self._index_spec.build_params.get("bm25_k1", 1.5)
            bm25_b = self._index_spec.build_params.get("bm25_b", 0.75)

        idx = SparseInvertedIndex(k1=bm25_k1, b=bm25_b)
        idx.build(all_sparse, valid_mask=valid_mask)

        # Convert query vectors: if they're dicts, use directly
        # If they're strings (text queries), tokenize them
        query_sparse: List[Dict[int, float]] = []
        for qv in query_vectors:
            if isinstance(qv, dict):
                query_sparse.append(qv)
            elif isinstance(qv, str):
                # Text query — tokenize using the BM25 function's analyzer
                analyzer = self._bm25_functions[0][2] if self._bm25_functions else None
                if analyzer is None:
                    raise SchemaValidationError(
                        "Text query requires a BM25 function with an analyzer"
                    )
                from litevecdb.analyzer.sparse import compute_tf
                term_ids = analyzer.analyze(qv)
                query_sparse.append(compute_tf(term_ids))
            else:
                raise SchemaValidationError(
                    f"Sparse search query must be a dict or string, "
                    f"got {type(qv).__name__}"
                )

        # Search
        local_ids, distances = idx.search(query_sparse, top_k, valid_mask=None)
        # Note: valid_mask was already applied in build, so no need to re-apply

        # Materialize results
        nq = len(query_vectors)
        results: List[List[dict]] = []
        for qi in range(nq):
            hits: list = []
            for j in range(top_k):
                lid = int(local_ids[qi, j])
                if lid < 0:
                    break
                dist = float(distances[qi, j])
                pk = all_pks[lid]
                rec = all_records[lid]

                entity = {}
                if output_fields is None:
                    # All fields except pk and vector fields
                    for k, v in rec.items():
                        fschema = next((f for f in self._schema.fields if f.name == k), None)
                        if fschema and (fschema.is_primary or fschema.dtype in (
                            DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR
                        )):
                            continue
                        entity[k] = v
                elif output_fields:
                    for fname in output_fields:
                        if fname == self._pk_name:
                            continue
                        entity[fname] = rec.get(fname)

                hits.append({
                    "id": pk,
                    "distance": dist,
                    "entity": entity,
                })
            results.append(hits)

        return results

    def query(
        self,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Pure scalar query — no vector, no distance.

        Returns all records matching the filter expression. Used for
        Milvus-style query() workflows where you just want to find rows
        by their scalar attributes.

        Args:
            expr: Milvus-style filter expression. None or empty string
                means "return all records" (used by query_iterator).
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
        if expr is not None and not isinstance(expr, str):
            raise TypeError("query() expr must be a string or None")

        self._require_loaded()

        compiled_filter = self._compile_filter(expr) if expr else None

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

    def _index_dir(self, partition: str) -> str:
        """Phase 9.4: canonical path for a partition's index sidecar dir.

        Layout: ``data_dir/partitions/<partition>/indexes/``

        The directory is created on demand by build_or_load_index when
        the first .idx is written.
        """
        return os.path.join(self._data_dir, "partitions", partition, "indexes")

    def _require_loaded(self) -> None:
        """Phase 9.3 guard: search/get/query require loaded state.

        Collections without an IndexSpec are auto-loaded on construction
        (see __init__), so this only fires after explicit create_index +
        no load(), or after explicit release().
        """
        if self._load_state != "loaded":
            raise CollectionNotLoadedError(
                f"Collection {self._name!r} is in state {self._load_state!r}; "
                f"call load() before search/get/query"
            )

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

    # ── index lifecycle (Phase 9.3) ─────────────────────────────

    def create_index(
        self,
        field_name: str,
        index_params: dict,
    ) -> None:
        """Persist an IndexSpec on the manifest. Does NOT build any
        index here — that happens at load() time, mirroring Milvus.

        Args:
            field_name: must be a vector field declared in the schema.
            index_params: dict containing at minimum::

                {
                    "index_type":  "HNSW" | "BRUTE_FORCE" | ...,
                    "metric_type": "COSINE" | "L2" | "IP",
                    "params":      {...},   # optional, build_params
                    "search_params": {...}, # optional, search defaults
                }

        Raises:
            IndexAlreadyExistsError: an index already exists
            SchemaValidationError:   field_name doesn't exist or isn't
                                     a vector field
            ValueError:              metric_type / index_type missing or invalid

        Side effect: collection moves to ``released`` state. The user
        must call ``load()`` to actually build segment indexes and
        re-enable search.
        """
        if self._index_spec is not None:
            raise IndexAlreadyExistsError(
                f"index already exists for field {self._index_spec.field_name!r}; "
                f"call drop_index first"
            )

        # Validate the field is in the schema and is a vector type.
        target = next((f for f in self._schema.fields if f.name == field_name), None)
        if target is None:
            raise SchemaValidationError(
                f"unknown field {field_name!r} for create_index"
            )
        if target.dtype != DataType.FLOAT_VECTOR:
            raise SchemaValidationError(
                f"field {field_name!r} has type {target.dtype.name}; "
                f"create_index only supports vector fields"
            )

        spec = IndexSpec(
            field_name=field_name,
            index_type=index_params["index_type"],
            metric_type=index_params["metric_type"],
            build_params=dict(index_params.get("params") or {}),
            search_params=dict(index_params.get("search_params") or {}),
        )

        self._index_spec = spec
        self._manifest.set_index_spec(spec)
        self._manifest.save()

        # Phase 9.3 semantics: an index now exists but is not built
        # → user must explicitly load() before searching.
        # Drop any in-memory indexes that may have been attached to
        # segments (e.g. left over from a previous load + drop_index).
        for seg in self._segment_cache.values():
            seg.release_index()
        self._load_state = "released"

    def drop_index(self, field_name: Optional[str] = None) -> None:
        """Remove the IndexSpec, release in-memory indexes, and delete
        on-disk .idx files.

        Args:
            field_name: optional; if given, must match the existing
                spec's field_name. None means "drop whatever index is
                there" (matches Milvus's drop_index without args).

        Raises:
            IndexNotFoundError: no index has been created

        Phase 9.4: also walks every partition's ``indexes/`` directory
        and deletes the .idx files matching the dropped index_type.
        Other index_type files (if any — currently impossible since we
        only support one index per Collection) are left alone.
        """
        if self._index_spec is None:
            raise IndexNotFoundError("no index to drop")
        if field_name is not None and field_name != self._index_spec.field_name:
            raise IndexNotFoundError(
                f"no index on field {field_name!r}; "
                f"current index is on {self._index_spec.field_name!r}"
            )

        # Release in-memory indexes.
        for seg in self._segment_cache.values():
            seg.release_index()

        # Phase 9.4: delete on-disk .idx files matching this index_type.
        # We do this BEFORE clearing self._index_spec so we still know
        # the index_type for the path computation.
        suffix = f".{self._index_spec.index_type.lower()}.idx"
        for partition in self._manifest.list_partitions():
            index_dir = self._index_dir(partition)
            if not os.path.exists(index_dir):
                continue
            for entry in os.listdir(index_dir):
                if entry.endswith(suffix):
                    try:
                        os.remove(os.path.join(index_dir, entry))
                    except OSError:
                        pass  # best effort — drop_index should not fail
                              # for filesystem hiccups

        self._index_spec = None
        self._manifest.set_index_spec(None)
        self._manifest.save()

        # No index → search no longer requires loaded state.
        self._load_state = "loaded"

    def has_index(self) -> bool:
        """True iff create_index has been called and not dropped."""
        return self._index_spec is not None

    def get_index_info(self) -> Optional[dict]:
        """Return the IndexSpec as a dict, or None if no index exists."""
        return self._index_spec.to_dict() if self._index_spec is not None else None

    def load(self) -> None:
        """Move to the loaded state. Build or load a VectorIndex per
        segment if an IndexSpec exists; idempotent if already loaded.

        Phase 9.4: indexes are persisted to disk. The first load() after
        a fresh create_index builds them and writes .idx sidecars; every
        subsequent load() (including after process restart) reads them
        back via Segment.build_or_load_index, so cold-start is fast.

        Phase 9.3-9.4 only routes to BruteForceIndex; Phase 9.5 will
        plug in the factory + FaissHnswIndex.

        Raises any exception encountered during build, with the state
        machine rolled back to released.
        """
        if self._load_state == "loaded":
            return
        self._load_state = "loading"
        try:
            if self._index_spec is not None:
                for seg in self._segment_cache.values():
                    if seg.num_rows == 0:
                        continue
                    seg.build_or_load_index(
                        self._index_spec, self._index_dir(seg.partition)
                    )
            self._load_state = "loaded"
        except Exception:
            self._load_state = "released"
            raise

    def release(self) -> None:
        """Drop all in-memory segment indexes; subsequent search() raises
        ``CollectionNotLoadedError`` until load() is called again.

        No-op if there's no IndexSpec (such collections never enter
        the released state — see Collection.__init__ for the rationale).
        """
        if self._index_spec is None:
            return
        for seg in self._segment_cache.values():
            seg.release_index()
        self._load_state = "released"

    @property
    def load_state(self) -> str:
        """Current load state: 'released' | 'loading' | 'loaded'.

        Mirrors Milvus's GetLoadState response. The Phase 10 gRPC
        adapter maps this directly to milvus.LoadState enum.
        """
        return self._load_state

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
        partition list + row count + index info + load state.

        Phase 9.3 added ``load_state`` and ``index_spec``. The
        ``index_spec`` field is None when no index has been created.
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
            "load_state": self._load_state,
            "index_spec": (
                self._index_spec.to_dict() if self._index_spec is not None else None
            ),
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

        # ── Phase 9.4: index hook ─────────────────────────────────
        # If the Collection is loaded, attach an index to any newly
        # created segments. This covers BOTH the flush case (new
        # data parquet → new index) AND the compaction case (new
        # merged segment → new index; the old segments and their
        # .idx files were already evicted in _refresh_segment_cache
        # via _cleanup_orphan_index_files below).
        self._cleanup_orphan_index_files()
        self._ensure_loaded_segments_indexed()

    def _ensure_loaded_segments_indexed(self) -> None:
        """Phase 9.4: post-flush / post-compaction index hook.

        For every segment in the cache that lacks an attached index,
        build/load it. No-op when:
            - Collection has no IndexSpec (nothing to build)
            - Collection is not in 'loaded' state (the user explicitly
              released, so we don't bring it back)
        Already-attached segments are skipped by build_or_load_index.
        """
        if self._load_state != "loaded" or self._index_spec is None:
            return
        for seg in self._segment_cache.values():
            if seg.index is None and seg.num_rows > 0:
                seg.build_or_load_index(
                    self._index_spec, self._index_dir(seg.partition)
                )

    def _cleanup_orphan_index_files(self) -> None:
        """Phase 9.4: delete .idx files whose source segment is gone.

        Called from _trigger_flush after _refresh_segment_cache (which
        evicts compaction-removed segments). The cleanup compares the
        on-disk indexes/ directories against the manifest's data file
        list and removes any .idx whose stem doesn't match a current
        data file.

        This is the architectural safety net for invariant §11
        (index 1:1 bound to data; lifecycles strictly aligned).
        """
        if not self._index_spec:
            # Without an index spec we don't know what suffix to clean.
            # Leftover files (e.g. from a previous create_index +
            # drop_index) are handled by drop_index itself.
            return

        suffix = f".{self._index_spec.index_type.lower()}.idx"
        for partition, data_files in self._manifest.get_all_data_files().items():
            index_dir = self._index_dir(partition)
            if not os.path.exists(index_dir):
                continue
            valid_stems = {
                os.path.splitext(os.path.basename(df))[0] for df in data_files
            }
            for entry in os.listdir(index_dir):
                if not entry.endswith(suffix):
                    continue
                stem = entry[: -len(suffix)]
                if stem not in valid_stems:
                    try:
                        os.remove(os.path.join(index_dir, entry))
                    except OSError:
                        pass

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

    # ── BM25 function auto-generation ────────────────────────────

    def _apply_bm25_functions(self, records: List[dict]) -> None:
        """Auto-generate sparse vector fields for BM25 functions.

        For each BM25 function, tokenize the input text field and compute
        term frequencies, then inject the resulting sparse vector dict
        into each record under the output field name.

        Modifies *records* in place.
        """
        from litevecdb.analyzer.sparse import compute_tf

        for in_name, out_name, analyzer in self._bm25_functions:
            for r in records:
                text = r.get(in_name)
                if text is None or not isinstance(text, str):
                    # Nullable text → empty sparse vector
                    r[out_name] = {}
                else:
                    term_ids = analyzer.analyze(text)
                    r[out_name] = compute_tf(term_ids)

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

        # Serialize SPARSE_FLOAT_VECTOR columns: dict → packed bytes
        for f in self._schema.fields:
            if f.dtype == DataType.SPARSE_FLOAT_VECTOR:
                from litevecdb.analyzer.sparse import sparse_to_bytes
                cols[f.name] = [
                    sparse_to_bytes(v) if isinstance(v, dict) else (v or b"")
                    for v in cols[f.name]
                ]

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


# ── Module-level helpers ──────────────────────────────────────────────

def _apply_group_by(
    results: List[List[dict]],
    group_by_field: str,
    limit: int,
    group_size: int,
    strict_group_size: bool,
) -> List[List[dict]]:
    """Post-process search results to group by a scalar field.

    For each query:
    1. Iterate hits in distance order (already sorted).
    2. Group by group_by_field value.
    3. Each group keeps up to group_size hits.
    4. If strict_group_size, discard groups with < group_size hits.
    5. Take the first `limit` groups.
    6. Flatten groups back into a single list (groups ordered by their
       best hit's distance).

    Returns results in the same format as input but filtered/reordered.
    Each hit gets an extra key ``_group_by_value`` for the gRPC layer
    to build the group_by_field_value FieldData.
    """
    out: List[List[dict]] = []

    for query_hits in results:
        # group_key → list of hits (in distance order)
        groups: dict = {}
        group_order: list = []  # track first-seen order (= best distance)

        for hit in query_hits:
            gval = hit.get("entity", {}).get(group_by_field)
            if gval is None:
                # Try top-level (some code paths put fields at top level)
                gval = hit.get(group_by_field)

            if gval not in groups:
                groups[gval] = []
                group_order.append(gval)

            if len(groups[gval]) < group_size:
                # Attach group value to hit for gRPC layer
                hit_copy = dict(hit)
                hit_copy["_group_by_value"] = gval
                groups[gval].append(hit_copy)

        # Filter by strict_group_size
        if strict_group_size:
            group_order = [g for g in group_order if len(groups[g]) == group_size]

        # Take first `limit` groups, flatten
        selected_groups = group_order[:limit]
        flattened: list = []
        for gval in selected_groups:
            flattened.extend(groups[gval])

        out.append(flattened)

    return out


def _apply_range_filter(
    results: List[List[dict]],
    radius: Optional[float],
    range_filter: Optional[float],
    limit: int,
) -> List[List[dict]]:
    """Filter search results by distance range.

    Keeps hits where ``radius < distance <= range_filter``.
    Either bound can be None (no bound on that side).
    After filtering, truncates to *limit* hits per query.
    """
    out: List[List[dict]] = []
    for query_hits in results:
        filtered = []
        for hit in query_hits:
            d = hit["distance"]
            if radius is not None and not (d > radius):
                continue
            if range_filter is not None and not (d <= range_filter):
                continue
            filtered.append(hit)
        out.append(filtered[:limit])
    return out
