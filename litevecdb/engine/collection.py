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
        _vf = get_vector_field(schema)
        self._vector_name: Optional[str] = _vf.name if _vf is not None else None

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
        # _index_specs mirrors manifest's per-field IndexSpec dict.
        # _load_state mirrors Milvus's loaded/released semantics:
        #   - Collections WITHOUT any IndexSpec auto-load on construction
        #   - Collections WITH IndexSpecs start as released; user must load()
        self._index_specs: Dict[str, IndexSpec] = dict(self._manifest.index_specs)
        self._load_state: str = "loaded" if not self._index_specs else "released"

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
        # _embedding_functions: list of (input_field_name, output_field_name, EmbeddingProvider)
        self._embedding_functions: List[Tuple[str, str, Any]] = []
        # _rerank_functions: list of (input_field_name, RerankProvider) — semantic rerankers
        self._rerank_functions: List[Tuple[str, Any]] = []
        # _decay_functions: list of (input_field_name, DecayReranker) — decay rerankers
        self._decay_functions: List[Tuple[str, Any]] = []
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
                elif func.function_type == FunctionType.TEXT_EMBEDDING:
                    from litevecdb.embedding.factory import create_embedding_provider
                    in_name = func.input_field_names[0]
                    out_name = func.output_field_names[0]
                    provider = create_embedding_provider(func.params)
                    self._embedding_functions.append((in_name, out_name, provider))
                elif func.function_type == FunctionType.RERANK:
                    in_name = func.input_field_names[0]
                    reranker_type = func.params.get("reranker", "").lower()
                    if reranker_type == "decay":
                        from litevecdb.rerank.decay import DecayReranker
                        dr = DecayReranker(
                            function=func.params["function"],
                            origin=func.params["origin"],
                            scale=func.params["scale"],
                            offset=func.params.get("offset", 0.0),
                            decay=func.params.get("decay", 0.5),
                        )
                        self._decay_functions.append((in_name, dr))
                    else:
                        from litevecdb.rerank.factory import create_rerank_provider
                        provider = create_rerank_provider(func.params)
                        self._rerank_functions.append((in_name, provider))

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

        # 2. auto-generate function output fields
        if self._bm25_functions:
            self._apply_bm25_functions(records)
        if self._embedding_functions:
            self._apply_embedding_functions(records)

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
        output_fields: Optional[List[str]] = None,
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

            out.append(self._project_record(rec, output_fields))

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
        offset: int = 0,
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
            radius: optional distance lower bound (exclusive).
            range_filter: optional distance upper bound (inclusive).
            offset: number of results to skip before returning (default 0).

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

        # Save original query texts for reranking (before embedding)
        _query_texts: Optional[List[str]] = None
        if self._rerank_functions and query_vectors and isinstance(query_vectors[0], str):
            _query_texts = list(query_vectors)

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

        # Over-fetch when group_by, range search, or offset is active
        effective_top_k = top_k + offset
        if group_by_field is not None:
            effective_top_k = max((top_k + offset) * group_size * 3, (top_k + offset) * 10)
        if radius is not None or range_filter is not None:
            effective_top_k = max(effective_top_k, (top_k + offset) * 5)

        # Resolve the target vector field
        vector_field = self._resolve_anns_field(anns_field)
        field_schema = next(f for f in self._schema.fields if f.name == vector_field)

        # If reranking is active, ensure the rerank input field is fetched
        _rerank_field_injected = False
        if _query_texts is not None:
            rerank_in_name = self._rerank_functions[0][0]
            if output_fields is not None and rerank_in_name not in output_fields:
                output_fields = list(output_fields) + [rerank_in_name]
                _rerank_field_injected = True

        # If decay reranking is active, ensure the decay input field is fetched
        _decay_field_injected = False
        if self._decay_functions:
            decay_in_name = self._decay_functions[0][0]
            if output_fields is not None and decay_in_name not in output_fields:
                output_fields = list(output_fields) + [decay_in_name]
                _decay_field_injected = True

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
            # Dense float vector search — auto-embed text queries if needed
            query_vectors = self._maybe_embed_queries(query_vectors, vector_field)
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
            raw_results = _apply_range_filter(
                raw_results, radius, range_filter, top_k + offset,
                metric_type=metric_type,
            )

        # Apply reranking (after range_filter, before group_by)
        _scores_replaced = False
        if _query_texts is not None:
            raw_results = self._apply_rerank(
                raw_results, _query_texts, _rerank_field_injected,
            )
            _scores_replaced = True

        # Apply decay reranking
        if self._decay_functions:
            raw_results = self._apply_decay(
                raw_results, metric_type, _decay_field_injected,
                _scores_replaced,
            )
            _scores_replaced = True

        # Apply group_by post-processing
        if group_by_field is not None:
            raw_results = _apply_group_by(
                raw_results, group_by_field, top_k + offset,
                group_size, strict_group_size,
            )

        # Apply offset: skip the first `offset` results per query
        if offset > 0:
            raw_results = [hits[offset:offset + top_k] for hits in raw_results]

        # Convert IP distances to Milvus convention (positive = more similar).
        # Internally we use -dot for sorting; Milvus returns raw dot product.
        # Skip if scores were already replaced by reranking.
        if metric_type == "IP" and not _scores_replaced:
            for hits in raw_results:
                for hit in hits:
                    hit["distance"] = -hit["distance"]

        return raw_results

    def _resolve_anns_field(self, anns_field: Optional[str]) -> str:
        """Resolve the anns_field parameter to a concrete field name.

        Returns the first FLOAT_VECTOR field if anns_field is None.
        Validates that the field exists and is a vector type.
        """
        if anns_field is None:
            if self._vector_name is None:
                # Sparse-only collection — caller must specify anns_field
                # explicitly (e.g. the sparse vector field name).
                raise SchemaValidationError(
                    "collection has no FLOAT_VECTOR field; "
                    "specify anns_field explicitly for sparse search"
                )
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
        """Sparse vector search using per-segment cached BM25 indexes.

        Architecture (Perf-3):
        - Each immutable segment gets a cached SparseInvertedIndex
          (built once, reused across searches).
        - The mutable memtable's index is rebuilt each search (small).
        - Per-source top-k results are merged globally.

        TODO: IDF accuracy — each segment currently uses its own IDF
        statistics, so BM25 scores from different segments have different
        baselines. Fix: aggregate global statistics (doc_count/avgdl/df
        summed across segments) at search time and use global IDF for
        scoring. Similar to Elasticsearch's DFS_QUERY_THEN_FETCH strategy.
        """
        from litevecdb.analyzer.sparse import bytes_to_sparse
        from litevecdb.index.sparse_inverted import SparseInvertedIndex

        partition_filter = set(partition_names) if partition_names else None
        _exclude_fields = {f.name for f in self._schema.fields
                          if f.is_primary or f.dtype in (
                              DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR)}

        # BM25 params
        bm25_k1 = 1.5
        bm25_b = 0.75
        sparse_spec = self._index_specs.get(vector_field)
        if sparse_spec and sparse_spec.index_type == "SPARSE_INVERTED_INDEX":
            bm25_k1 = sparse_spec.build_params.get("bm25_k1", 1.5)
            bm25_b = sparse_spec.build_params.get("bm25_b", 0.75)

        # Convert query vectors upfront
        query_sparse = self._prepare_sparse_queries(query_vectors)
        nq = len(query_sparse)

        # Per-source candidates: (distance, global_pk, source_ref)
        Candidate = Tuple[float, Any, Any]  # (dist, pk, (tbl, row_idx))
        per_query_candidates: List[List[Candidate]] = [[] for _ in range(nq)]

        # ── Per-segment search (cached indexes) ──────────────────
        for seg in self._segment_cache.values():
            if partition_filter is not None:
                seg_part = seg.partition_name if hasattr(seg, 'partition_name') else None
                if seg_part is not None and seg_part not in partition_filter:
                    continue
            table = seg.table
            if table is None or len(table) == 0:
                continue

            # Build or reuse cached sparse index for this segment
            cache_key = f"_sparse_{vector_field}"
            cached_idx = seg.indexes.get(cache_key)
            if cached_idx is None:
                sparse_batch = table.column(vector_field).to_pylist()
                sparse_vecs = [
                    bytes_to_sparse(r) if isinstance(r, bytes) else (r or {})
                    for r in sparse_batch
                ]
                cached_idx = SparseInvertedIndex(k1=bm25_k1, b=bm25_b)
                cached_idx.build(sparse_vecs)  # no valid_mask — full segment
                seg.attach_index(cached_idx, field_name=cache_key)

            # Build valid_mask for this segment (dedup + tombstone + filter)
            pks = seg.pks
            seqs = seg.seqs
            n = len(pks)
            valid_mask = np.ones(n, dtype=bool)
            for i in range(n):
                pk, seq = pks[i], int(seqs[i])
                if self._delta_index.is_deleted(pk, seq):
                    valid_mask[i] = False

            # Apply scalar filter
            if expr:
                compiled = self._compile_filter(expr)
                from litevecdb.search.filter.eval import evaluate as filter_evaluate
                fmask = filter_evaluate(compiled, table).to_numpy(zero_copy_only=False)
                valid_mask = valid_mask & fmask

            if not valid_mask.any():
                continue

            # Search this segment's cached index
            local_ids, dists = cached_idx.search(query_sparse, top_k, valid_mask=valid_mask)
            for qi in range(nq):
                for j in range(top_k):
                    lid = int(local_ids[qi, j])
                    if lid < 0:
                        break
                    per_query_candidates[qi].append(
                        (float(dists[qi, j]), pks[lid], (table, lid))
                    )

        # ── Memtable search (rebuilt each time — small + mutable) ─
        mt = self._memtable
        mt_pks: list = []
        mt_sparse: list = []
        mt_refs: list = []
        for pk, (batch_idx, row_idx, seq) in mt._pk_index.items():
            batch = mt._insert_batches[batch_idx]
            if partition_filter is not None:
                part = batch.column("_partition")[row_idx].as_py()
                if part not in partition_filter:
                    continue
            raw = batch.column(vector_field)[row_idx].as_py()
            mt_pks.append(pk)
            mt_sparse.append(
                bytes_to_sparse(raw) if isinstance(raw, bytes) else (raw or {})
            )
            mt_refs.append((batch, row_idx))

        if mt_pks:
            # Check tombstone for memtable rows
            mt_valid = np.ones(len(mt_pks), dtype=bool)
            for i, pk in enumerate(mt_pks):
                # Memtable rows: check if overridden by a segment with higher seq
                # (simplified — memtable rows always have highest seq for their pk)
                if self._delta_index.is_deleted(pk, 0):
                    # Use seq=0 as sentinel; is_deleted checks delete_seq > row_seq
                    pass  # memtable rows have latest seq, typically not deleted via delta

            if expr:
                compiled = self._compile_filter(expr)
                from litevecdb.search.filter.eval.python_backend import _eval_row
                for i in range(len(mt_pks)):
                    if mt_valid[i]:
                        tbl, row_i = mt_refs[i]
                        rec = {
                            col: tbl.column(col)[row_i].as_py()
                            for col in tbl.column_names
                            if col not in ("_seq", "_partition")
                        }
                        if not _eval_row(compiled.ast, rec):
                            mt_valid[i] = False

            mt_idx = SparseInvertedIndex(k1=bm25_k1, b=bm25_b)
            mt_idx.build(mt_sparse, valid_mask=mt_valid)
            local_ids, dists = mt_idx.search(query_sparse, top_k)
            for qi in range(nq):
                for j in range(top_k):
                    lid = int(local_ids[qi, j])
                    if lid < 0:
                        break
                    per_query_candidates[qi].append(
                        (float(dists[qi, j]), mt_pks[lid], mt_refs[lid])
                    )

        # ── Global merge + dedup + materialize ────────────────────
        # Dedup: if same pk appears from segment + memtable, keep
        # the one with better (smaller) distance.
        results: List[List[dict]] = []
        for qi in range(nq):
            candidates = per_query_candidates[qi]
            # Sort by distance ascending (smaller = better)
            candidates.sort(key=lambda c: c[0])
            seen_pks: set = set()
            hits: list = []
            for dist, pk, (tbl, row_i) in candidates:
                if pk in seen_pks:
                    continue
                seen_pks.add(pk)
                # Deferred materialization
                entity = {}
                if output_fields is None:
                    for col in tbl.column_names:
                        if col in ("_seq", "_partition") or col in _exclude_fields:
                            continue
                        entity[col] = tbl.column(col)[row_i].as_py()
                elif output_fields:
                    for fname in output_fields:
                        if fname == self._pk_name:
                            continue
                        entity[fname] = tbl.column(fname)[row_i].as_py()
                hits.append({"id": pk, "distance": dist, "entity": entity})
                if len(hits) >= top_k:
                    break
            results.append(hits)

        return results

    def _maybe_embed_queries(self, query_vectors: List, vector_field: str) -> List:
        """If query_vectors contains strings and this field has a TEXT_EMBEDDING
        function, auto-embed them. Otherwise return as-is."""
        if not query_vectors or not isinstance(query_vectors[0], str):
            return query_vectors

        # Find the embedding provider for this vector field
        provider = None
        for _in, out, prov in self._embedding_functions:
            if out == vector_field:
                provider = prov
                break

        if provider is None:
            raise SchemaValidationError(
                f"Text query on field {vector_field!r} requires a "
                f"TEXT_EMBEDDING function targeting that field"
            )

        embedded = []
        for qv in query_vectors:
            if isinstance(qv, str):
                embedded.append(provider.embed_query(qv))
            else:
                embedded.append(qv)
        return embedded

    def _apply_rerank(
        self,
        raw_results: List[List[dict]],
        query_texts: List[str],
        strip_rerank_field: bool,
    ) -> List[List[dict]]:
        """Re-score search results using the RERANK function provider.

        For each query, extracts the rerank input field text from each hit's
        entity, calls the reranker, and re-orders hits by relevance_score.

        Args:
            raw_results: per-query hit lists from the vector search.
            query_texts: original query strings (one per query).
            strip_rerank_field: if True, remove the rerank input field from
                entity dicts (it was injected internally, user didn't ask for it).
        """
        in_name, provider = self._rerank_functions[0]

        reranked_results: List[List[dict]] = []
        for qi, hits in enumerate(raw_results):
            if not hits:
                reranked_results.append(hits)
                continue

            query_text = query_texts[qi]

            # Extract document texts from hits; fall back to "" for missing
            doc_texts = [
                hit.get("entity", {}).get(in_name, "") or ""
                for hit in hits
            ]

            rr = provider.rerank(query_text, doc_texts, top_n=len(hits))

            # Rebuild hits in reranked order with reranker score
            new_hits = []
            for r in rr:
                hit = hits[r.index]
                hit["distance"] = r.relevance_score
                new_hits.append(hit)
            reranked_results.append(new_hits)

            # Strip rerank input field from entity if we injected it
            if strip_rerank_field:
                for hit in new_hits:
                    hit.get("entity", {}).pop(in_name, None)

        return reranked_results

    def _apply_decay(
        self,
        raw_results: List[List[dict]],
        metric_type: str,
        strip_decay_field: bool,
        scores_already_replaced: bool,
    ) -> List[List[dict]]:
        """Apply decay reranking based on a numeric field's proximity to origin.

        For each hit, computes a decay factor from the numeric field value,
        multiplies it with the vector relevance score, and re-sorts.
        """
        in_name, reranker = self._decay_functions[0]

        decayed_results: List[List[dict]] = []
        for hits in raw_results:
            if not hits:
                decayed_results.append(hits)
                continue

            for hit in hits:
                field_val = hit.get("entity", {}).get(in_name)
                if field_val is None:
                    hit["distance"] = 0.0
                    continue
                factor = reranker.compute_factor(float(field_val))
                if scores_already_replaced:
                    # distance is already a relevance score (higher = better)
                    hit["distance"] = hit["distance"] * factor
                else:
                    score = _distance_to_score(hit["distance"], metric_type)
                    hit["distance"] = score * factor

            # Sort by final score descending (higher = better)
            hits.sort(key=lambda h: h["distance"], reverse=True)

            if strip_decay_field:
                for hit in hits:
                    hit.get("entity", {}).pop(in_name, None)

            decayed_results.append(hits)

        return decayed_results

    def _prepare_sparse_queries(self, query_vectors: List) -> List[Dict[int, float]]:
        """Convert query vectors to sparse dicts (text → tokenize → TF)."""
        query_sparse: List[Dict[int, float]] = []
        for qv in query_vectors:
            if isinstance(qv, dict):
                query_sparse.append(qv)
            elif isinstance(qv, str):
                analyzer = self._bm25_functions[0][2] if self._bm25_functions else None
                if analyzer is None:
                    raise SchemaValidationError(
                        "Text query requires a BM25 function with an analyzer"
                    )
                from litevecdb.analyzer.sparse import compute_tf
                query_sparse.append(compute_tf(analyzer.analyze(qv)))
            else:
                raise SchemaValidationError(
                    f"Sparse search query must be a dict or string, "
                    f"got {type(qv).__name__}"
                )
        return query_sparse

    def query(
        self,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
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

        all_pks, all_seqs, _all_vectors, all_rec_sources, filter_mask = assemble_candidates(
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
        from litevecdb.search.assembler import materialize_record
        mask = build_valid_mask(
            all_pks, all_seqs, self._delta_index, filter_mask=filter_mask,
        )

        # Deferred materialization: only materialize records that pass the mask.
        effective_limit = (offset + limit) if limit is not None else None
        live_indices = np.flatnonzero(mask)
        out: List[dict] = []
        for i in live_indices:
            rec = materialize_record(all_rec_sources[int(i)])
            out.append(self._project_record(rec, output_fields))
            if effective_limit is not None and len(out) >= effective_limit:
                break
        return out[offset:]

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

        - None → return all fields (stripping internal $meta key)
        - list → keep only the named fields, plus the pk field
        """
        if output_fields is None:
            return {k: v for k, v in record.items() if k != "$meta"}
        keep = set(output_fields)
        keep.add(self._pk_name)
        keep.discard("$meta")
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
        if field_name in self._index_specs:
            raise IndexAlreadyExistsError(
                f"index already exists for field {field_name!r}; "
                f"call drop_index first"
            )

        # Validate the field is in the schema and is a vector type.
        target = next((f for f in self._schema.fields if f.name == field_name), None)
        if target is None:
            raise SchemaValidationError(
                f"unknown field {field_name!r} for create_index"
            )
        if target.dtype not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR):
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

        self._index_specs[field_name] = spec
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
        if not self._index_specs:
            raise IndexNotFoundError("no index to drop")
        if field_name is not None and field_name not in self._index_specs:
            raise IndexNotFoundError(
                f"no index on field {field_name!r}; "
                f"indexed fields: {list(self._index_specs.keys())}"
            )

        # Determine which spec(s) to drop
        if field_name is not None:
            drop_specs = [self._index_specs[field_name]]
        else:
            drop_specs = list(self._index_specs.values())

        # Release in-memory indexes for the affected fields.
        for spec in drop_specs:
            for seg in self._segment_cache.values():
                seg.release_index(field_name=spec.field_name)

        # Delete on-disk .idx files matching the dropped index_type(s).
        for spec in drop_specs:
            suffix = f".{spec.index_type.lower()}.idx"
            for partition in self._manifest.list_partitions():
                index_dir = self._index_dir(partition)
                if not os.path.exists(index_dir):
                    continue
                for entry in os.listdir(index_dir):
                    if entry.endswith(suffix):
                        try:
                            os.remove(os.path.join(index_dir, entry))
                        except OSError:
                            pass

        # Remove from specs
        for spec in drop_specs:
            del self._index_specs[spec.field_name]
            self._manifest.remove_index_spec(spec.field_name)
        self._manifest.save()

        # If no indexes remain, auto-load (backward compat).
        if not self._index_specs:
            self._load_state = "loaded"

    def has_index(self, field_name: Optional[str] = None) -> bool:
        """True iff create_index has been called (and not dropped).
        If field_name is given, checks that specific field."""
        if field_name is not None:
            return field_name in self._index_specs
        return bool(self._index_specs)

    def get_index_info(self, field_name: Optional[str] = None) -> Optional[dict]:
        """Return IndexSpec as dict. If field_name is None, returns first."""
        if field_name is not None:
            spec = self._index_specs.get(field_name)
            return spec.to_dict() if spec else None
        if not self._index_specs:
            return None
        return next(iter(self._index_specs.values())).to_dict()

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
            for spec in self._index_specs.values():
                for seg in self._segment_cache.values():
                    if seg.num_rows == 0:
                        continue
                    seg.build_or_load_index(
                        spec, self._index_dir(seg.partition)
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
        if not self._index_specs:
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
            "index_specs": {
                k: v.to_dict() for k, v in self._index_specs.items()
            },
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
        if self._load_state != "loaded" or not self._index_specs:
            return
        for spec in self._index_specs.values():
            for seg in self._segment_cache.values():
                if spec.field_name not in seg.indexes and seg.num_rows > 0:
                    seg.build_or_load_index(
                        spec, self._index_dir(seg.partition)
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
        if not self._index_specs:
            return

        suffixes = {
            f".{spec.index_type.lower()}.idx"
            for spec in self._index_specs.values()
        }
        for partition, data_files in self._manifest.get_all_data_files().items():
            index_dir = self._index_dir(partition)
            if not os.path.exists(index_dir):
                continue
            valid_stems = {
                os.path.splitext(os.path.basename(df))[0] for df in data_files
            }
            for entry in os.listdir(index_dir):
                if not any(entry.endswith(s) for s in suffixes):
                    continue
                for s in suffixes:
                    if entry.endswith(s):
                        stem = entry[: -len(s)]
                        if stem not in valid_stems:
                            try:
                                os.remove(os.path.join(index_dir, entry))
                            except OSError:
                                pass
                        break

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

    # ── TEXT_EMBEDDING function auto-generation ─────────────────

    def _apply_embedding_functions(self, records: List[dict]) -> None:
        """Auto-generate dense vector fields for TEXT_EMBEDDING functions.

        Calls the embedding provider's batch API to convert text inputs
        into float vectors. Modifies *records* in place.
        """
        for in_name, out_name, provider in self._embedding_functions:
            texts = []
            indices = []
            for i, r in enumerate(records):
                text = r.get(in_name)
                if text is not None and isinstance(text, str) and text:
                    texts.append(text)
                    indices.append(i)
                else:
                    # Nullable text → zero vector
                    r[out_name] = [0.0] * provider.dimension

            if texts:
                vectors = provider.embed_documents(texts)
                for idx, vec in zip(indices, vectors):
                    records[idx][out_name] = vec

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

        # Serialize JSON columns: dict/list → JSON string
        import json as _json
        for f in self._schema.fields:
            if f.dtype == DataType.JSON:
                cols[f.name] = [
                    _json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list)) else v
                    for v in cols[f.name]
                ]

        # Replace None with zero vectors for nullable FLOAT_VECTOR fields.
        # Arrow Parquet doesn't support null in FixedSizeList, so we store
        # zeros and rely on the null info being tracked at read time by
        # checking if the vector is all-zeros (or via valid_data on gRPC).
        for f in self._schema.fields:
            if f.dtype == DataType.FLOAT_VECTOR and f.nullable and f.dim:
                zero = [0.0] * f.dim
                cols[f.name] = [
                    v if v is not None else zero
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


def _distance_to_score(distance: float, metric_type: str) -> float:
    """Convert internal distance to a relevance score (higher = better).

    Internal distances are always "lower = better":
    - COSINE: 1 - cosine_similarity (range [0, 2])
    - L2: L2 distance (range [0, ∞))
    - IP: -dot_product (negated for internal sorting)
    """
    if metric_type == "COSINE":
        return 1.0 - distance  # cosine similarity
    elif metric_type == "IP":
        return -distance  # raw dot product
    elif metric_type == "L2":
        return 1.0 / (1.0 + distance)
    return 1.0 - distance  # fallback


def _apply_range_filter(
    results: List[List[dict]],
    radius: Optional[float],
    range_filter: Optional[float],
    limit: int,
    metric_type: str = "COSINE",
) -> List[List[dict]]:
    """Filter search results by distance range.

    Milvus range search semantics:
        L2/COSINE: radius = max distance (outer), range_filter = min distance (inner)
            Keep: range_filter <= distance <= radius
        IP: radius = min score (inner), range_filter = max score (outer)
            Keep: radius <= distance <= range_filter
            (note: at this point IP distances are still internal -dot form)

    Either bound can be None (no bound on that side).
    After filtering, truncates to *limit* hits per query.
    """
    out: List[List[dict]] = []
    for query_hits in results:
        filtered = []
        for hit in query_hits:
            d = hit["distance"]
            if metric_type == "IP":
                # IP internal convention: -dot (smaller = more similar)
                # radius/range_filter are user-facing (positive dot values)
                # but _apply_range_filter runs BEFORE IP sign flip, so
                # negate the bounds for comparison
                if radius is not None and not (d <= -radius):
                    continue
                if range_filter is not None and not (d >= -range_filter):
                    continue
            else:
                # L2/COSINE: smaller distance = closer
                # radius = outer bound (max), range_filter = inner bound (min)
                if radius is not None and not (d <= radius):
                    continue
                if range_filter is not None and not (d >= range_filter):
                    continue
            filtered.append(hit)
        out.append(filtered[:limit])
    return out
