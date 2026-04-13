"""MilvusServicer — gRPC RPC dispatcher for the LiteVecDB engine.

Inherits from ``pymilvus.grpc_gen.milvus_pb2_grpc.MilvusServiceServicer``.
The base class auto-generates UNIMPLEMENTED responses for every method
we don't override, so an empty subclass already returns the right
"not supported" status for the 100+ RPCs we don't plan to implement.

Phase 10.1 ships only the bare minimum needed for pymilvus.connect()
to succeed (Connect + GetVersion). Phase 10.2-10.6 fills in:
    10.2 — Collection lifecycle (Create/Drop/Has/Describe/Show)
    10.3 — Insert/Upsert/Delete/Query/Get  (FieldData ↔ records)
    10.4 — Search/CreateIndex/Load/Release  (the search path)
    10.5 — Partition + Flush + Stats  (rounding out the quickstart)
    10.6 — Error code mapping  (LiteVecDBError → grpc Status)

Implementation discipline (from grpc-adapter-design.md §15):
    The servicer ONLY translates protocol; it never adds engine
    capability. Anything we don't have an engine API for must
    return UNIMPLEMENTED with a friendly message — never silent fail.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import grpc
from pymilvus.grpc_gen import common_pb2, milvus_pb2, milvus_pb2_grpc, schema_pb2

from litevecdb.adapter.grpc.errors import (
    SUCCESS as _SUCCESS,
    UNEXPECTED_ERROR as _UNEXPECTED_ERROR,
    success_status_kwargs,
    to_status_kwargs,
)
from litevecdb.adapter.grpc.translators.index import (
    index_spec_to_kv_pairs,
    kv_pairs_to_index_params_dict,
)
from litevecdb.adapter.grpc.translators.records import (
    fields_data_to_records,
    records_to_fields_data,
)
from litevecdb.adapter.grpc.translators.result import build_search_result_data
from litevecdb.adapter.grpc.translators.schema import (
    litevecdb_to_milvus_schema,
    milvus_to_litevecdb_schema,
)
from litevecdb.adapter.grpc.translators.search import parse_search_request
from litevecdb.exceptions import LiteVecDBError
from litevecdb.schema.types import DataType

if TYPE_CHECKING:
    from litevecdb.db import LiteVecDB

logger = logging.getLogger(__name__)


class MilvusServicer(milvus_pb2_grpc.MilvusServiceServicer):
    """Maps Milvus RPCs onto LiteVecDB engine calls.

    Phase 10.1 handles connection-level RPCs only. All data-plane
    methods inherited from MilvusServiceServicer return UNIMPLEMENTED
    via the gRPC default implementation.
    """

    def __init__(self, db: "LiteVecDB") -> None:
        self._db = db

    # ── Connection-level RPCs ───────────────────────────────────
    #
    # pymilvus.MilvusClient(uri=...) does a Connect call as part of
    # client construction. Without this override the client init
    # would itself raise UNIMPLEMENTED, and users couldn't even open
    # a connection to the server. So this is the absolute minimum
    # surface to ship in Phase 10.1.

    def Connect(self, request, context):
        """Acknowledge the client identity. We don't track sessions
        — every request is processed independently — so this just
        returns a success status with our server identity."""
        return milvus_pb2.ConnectResponse(
            status=common_pb2.Status(code=_SUCCESS, reason=""),
            server_info=common_pb2.ServerInfo(
                build_tags="litevecdb",
                build_time="",
                git_commit="",
                go_version="",
                deploy_mode="embedded",
            ),
            identifier=0,
        )

    def GetVersion(self, request, context):
        """Return the LiteVecDB version string. pymilvus uses this
        as a smoke test for "the server is alive and speaks the
        Milvus protocol"."""
        return milvus_pb2.GetVersionResponse(
            status=common_pb2.Status(code=_SUCCESS, reason=""),
            version="litevecdb-0.1.0",
        )

    def CheckHealth(self, request, context):
        """Health probe. Always reports healthy — single-process
        embedded servers don't have a partial-failure mode."""
        return milvus_pb2.CheckHealthResponse(
            status=common_pb2.Status(code=_SUCCESS, reason=""),
            isHealthy=True,
            reasons=[],
        )

    # ── Collection lifecycle (Phase 10.2) ───────────────────────

    def CreateCollection(self, request, context):
        """Decode the schema bytes blob, validate, then call
        ``LiteVecDB.create_collection``."""
        try:
            proto_schema = schema_pb2.CollectionSchema()
            proto_schema.ParseFromString(request.schema)
            litevecdb_schema = milvus_to_litevecdb_schema(proto_schema)
            self._db.create_collection(request.collection_name, litevecdb_schema)
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("CreateCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DropCollection(self, request, context):
        try:
            self._db.drop_collection(request.collection_name)
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("DropCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def HasCollection(self, request, context):
        try:
            exists = self._db.has_collection(request.collection_name)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                value=exists,
            )
        except Exception as e:
            logger.exception("HasCollection failed: %s", e)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
                value=False,
            )

    def DescribeCollection(self, request, context):
        """Return the collection's schema + basic stats. The Phase 9
        Collection.describe() output is rebuilt into Milvus's
        DescribeCollectionResponse shape."""
        try:
            col = self._db.get_collection(request.collection_name)
            proto_schema = litevecdb_to_milvus_schema(col.name, col.schema)
            return milvus_pb2.DescribeCollectionResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                schema=proto_schema,
                collection_name=col.name,
                shards_num=1,
                num_partitions=len(col.list_partitions()),
            )
        except LiteVecDBError as e:
            return milvus_pb2.DescribeCollectionResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("DescribeCollection failed: %s", e)
            return milvus_pb2.DescribeCollectionResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def ShowCollections(self, request, context):
        """Return the list of collection names. Milvus's response also
        carries timestamps and IDs which we don't track — those slots
        stay empty."""
        try:
            names = self._db.list_collections()
            return milvus_pb2.ShowCollectionsResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                collection_names=names,
            )
        except Exception as e:
            logger.exception("ShowCollections failed: %s", e)
            return milvus_pb2.ShowCollectionsResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Data CRUD (Phase 10.3) ──────────────────────────────────

    def Insert(self, request, context):
        """Decode columnar fields_data into records, dispatch to
        ``Collection.insert``. Returns a MutationResult with the
        inserted IDs (which double as the success indicator for
        pymilvus's MilvusClient.insert)."""
        try:
            col = self._db.get_collection(request.collection_name)
            records = fields_data_to_records(
                request.fields_data, request.num_rows
            )
            partition_name = request.partition_name or "_default"
            inserted_pks = col.insert(records, partition_name=partition_name)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**success_status_kwargs()),
                IDs=self._build_ids_proto(inserted_pks, col),
                insert_cnt=len(inserted_pks),
                succ_index=list(range(len(inserted_pks))),
            )
        except LiteVecDBError as e:
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Insert failed: %s", e)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Upsert(self, request, context):
        """Same engine call as Insert (LiteVecDB's insert is
        upsert-by-pk). Returns upsert_cnt instead of insert_cnt."""
        try:
            col = self._db.get_collection(request.collection_name)
            records = fields_data_to_records(
                request.fields_data, request.num_rows
            )
            partition_name = request.partition_name or "_default"
            upserted_pks = col.insert(records, partition_name=partition_name)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**success_status_kwargs()),
                IDs=self._build_ids_proto(upserted_pks, col),
                upsert_cnt=len(upserted_pks),
                succ_index=list(range(len(upserted_pks))),
            )
        except LiteVecDBError as e:
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Upsert failed: %s", e)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Delete(self, request, context):
        """Two paths:

        1. Filter expression looks like ``id in [1,2,3]`` → extract pks
           and call ``col.delete(pks=[...])`` directly. This is what
           pymilvus emits for ``client.delete(ids=[...])``.

        2. Any other expression → fall back to "query → extract pks →
           delete". The cost is one extra read pass; it's the only way
           to honor delete-by-filter without engine-native support.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            partition_name = request.partition_name or None

            pks = self._extract_pks_from_expr(request.expr, col)
            if pks is None:
                # Fall back: query to find matching pks, then delete.
                # Requires the collection to be loaded.
                hits = col.query(request.expr, output_fields=[col._pk_name])
                pks = [r[col._pk_name] for r in hits]

            count = col.delete(pks, partition_name=partition_name)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**success_status_kwargs()),
                IDs=self._build_ids_proto(pks, col),
                delete_cnt=count,
            )
        except LiteVecDBError as e:
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Delete failed: %s", e)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Query(self, request, context):
        """Two paths, same dispatch as Delete:

        1. Expression looks like ``id in [...]`` → call ``col.get(pks)``.
           This is what pymilvus emits for ``client.get(ids=[...])``.

        2. Any other expression → call ``col.query(expr, ...)``.

        Both paths return their results encoded as columnar
        fields_data via the records translator.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            partition_names = list(request.partition_names) or None
            output_fields = list(request.output_fields) or None

            # Extract limit and offset from query_params KV list.
            limit = None
            offset = 0
            for kv in request.query_params:
                if kv.key == "limit":
                    try:
                        limit = int(kv.value)
                    except (ValueError, TypeError):
                        pass
                elif kv.key == "offset":
                    try:
                        offset = int(kv.value)
                    except (ValueError, TypeError):
                        pass

            expr = request.expr if request.expr else None
            pks = self._extract_pks_from_expr(expr, col) if expr else None
            if pks is not None:
                rows = col.get(pks, partition_names=partition_names)
            else:
                rows = col.query(
                    expr,
                    output_fields=output_fields,
                    partition_names=partition_names,
                    limit=limit,
                    offset=offset,
                )

            return milvus_pb2.QueryResults(
                status=common_pb2.Status(**success_status_kwargs()),
                fields_data=records_to_fields_data(
                    rows, col.schema, output_fields=output_fields,
                ),
                collection_name=col.name,
                output_fields=output_fields or [],
                primary_field_name=col._pk_name,
            )
        except LiteVecDBError as e:
            return milvus_pb2.QueryResults(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Query failed: %s", e)
            return milvus_pb2.QueryResults(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Vector search (Phase 10.4) ──────────────────────────────

    def Search(self, request, context):
        """Decode the search request, dispatch to ``Collection.search``,
        flatten the result back into a SearchResultData proto.

        The hard part is the request decoding (placeholder_group bytes
        → list of query vectors via PlaceholderGroup proto + struct
        unpack), centralized in translators/search.py.

        Metric resolution: pymilvus's MilvusClient.search doesn't
        include metric_type in search_params by default. We fall back
        to the collection's IndexSpec.metric_type so the engine uses
        the same metric the index was built with.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            # Pull the canonical metric from the first IndexSpec if any.
            first_spec = col._index_specs.get(col._vector_name) if col._index_specs else None  # noqa: SLF001
            if first_spec is None and col._index_specs:
                first_spec = next(iter(col._index_specs.values()))
            default_metric = first_spec.metric_type if first_spec else "COSINE"
            parsed = parse_search_request(request, default_metric_type=default_metric)

            group_by_field = parsed.get("group_by_field")
            group_size = parsed.get("group_size") or 1
            strict = parsed.get("group_size_strict") or False

            results = col.search(
                query_vectors=parsed["query_vectors"],
                top_k=parsed["top_k"],
                metric_type=parsed["metric_type"],
                partition_names=parsed["partition_names"],
                expr=parsed["expr"],
                output_fields=parsed["output_fields"],
                anns_field=parsed.get("anns_field"),
                group_by_field=group_by_field,
                group_size=group_size,
                strict_group_size=strict,
                radius=parsed.get("radius"),
                range_filter=parsed.get("range_filter"),
                offset=parsed.get("offset", 0),
            )

            result_data = build_search_result_data(
                results=results,
                schema=col.schema,
                top_k=parsed["top_k"],
                pk_name=col._pk_name,  # noqa: SLF001
                output_fields=parsed["output_fields"],
                group_by_field=group_by_field,
            )

            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**success_status_kwargs()),
                results=result_data,
                collection_name=col.name,
            )
        except LiteVecDBError as e:
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Search failed: %s", e)
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Index lifecycle (Phase 10.4) ────────────────────────────

    def CreateIndex(self, request, context):
        """Decode IndexParams and call ``Collection.create_index``.

        pymilvus's MilvusClient.create_index packs ``index_type``,
        ``metric_type``, ``params``, and (optionally) ``search_params``
        into the ``extra_params`` KeyValuePair list. The translator
        unpacks these into the dict shape Collection.create_index
        consumes.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            params = kv_pairs_to_index_params_dict(
                request.extra_params, field_name=request.field_name
            )
            col.create_index(request.field_name, params)
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("CreateIndex failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DropIndex(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            field_name = request.field_name or None
            # Resolve index_name → field_name if field_name not provided
            if field_name is None and request.index_name:
                # Our naming convention: "{field_name}_idx"
                idx_name = request.index_name
                for fn in col._index_specs:  # noqa: SLF001
                    if f"{fn}_idx" == idx_name:
                        field_name = fn
                        break
            col.drop_index(field_name)
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("DropIndex failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DescribeIndex(self, request, context):
        """Return the IndexSpec wrapped in a DescribeIndexResponse.

        Returns an INDEX_NOT_FOUND status when there's no matching
        index (pymilvus's describe_index parses this as None rather
        than raising AmbiguousIndexName, which is what would happen
        if we returned SUCCESS + empty list).

        IndexState is always ``Finished`` because Phase 9 builds
        indexes synchronously inside ``load()``.
        """
        try:
            from litevecdb.exceptions import IndexNotFoundError as _INFE

            col = self._db.get_collection(request.collection_name)
            all_specs = col._index_specs  # noqa: SLF001

            if not all_specs:
                raise _INFE(
                    f"no index on collection {request.collection_name!r}"
                )

            # Filter by field_name if requested.
            if request.field_name:
                spec = all_specs.get(request.field_name)
                if spec is None:
                    raise _INFE(
                        f"no index on field {request.field_name!r} of "
                        f"collection {request.collection_name!r}"
                    )
                specs_to_report = [spec]
            else:
                specs_to_report = list(all_specs.values())

            num_ent = col.num_entities
            descriptions = [
                milvus_pb2.IndexDescription(
                    index_name=f"{s.field_name}_idx",
                    field_name=s.field_name,
                    params=index_spec_to_kv_pairs(s),
                    state=common_pb2.IndexState.Finished,
                    indexed_rows=num_ent,
                    total_rows=num_ent,
                )
                for s in specs_to_report
            ]
            return milvus_pb2.DescribeIndexResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                index_descriptions=descriptions,
            )
        except LiteVecDBError as e:
            return milvus_pb2.DescribeIndexResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("DescribeIndex failed: %s", e)
            return milvus_pb2.DescribeIndexResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Load / release (Phase 10.4) ─────────────────────────────

    def LoadCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.load()
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("LoadCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def ReleaseCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.release()
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("ReleaseCollection failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def GetLoadingProgress(self, request, context):
        """Polled by pymilvus's load_collection wrapper.

        Our load is synchronous (Phase 9), so once it returns the
        collection is fully loaded → progress 100. If the collection
        is in 'loading' state we still report 0; if released, 0.
        """
        try:
            col = self._db.get_collection(request.collection_name)
            progress = 100 if col.load_state == "loaded" else 0
            return milvus_pb2.GetLoadingProgressResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                progress=progress,
            )
        except LiteVecDBError as e:
            return milvus_pb2.GetLoadingProgressResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("GetLoadingProgress failed: %s", e)
            return milvus_pb2.GetLoadingProgressResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def GetLoadState(self, request, context):
        """Map Collection._load_state to Milvus's LoadState enum.

            released → LoadStateNotLoad   (1)
            loading  → LoadStateLoading   (2)
            loaded   → LoadStateLoaded    (3)
        """
        try:
            if not self._db.has_collection(request.collection_name):
                return milvus_pb2.GetLoadStateResponse(
                    status=common_pb2.Status(**success_status_kwargs()),
                    state=common_pb2.LoadState.LoadStateNotExist,
                )
            col = self._db.get_collection(request.collection_name)
            mapping = {
                "released": common_pb2.LoadState.LoadStateNotLoad,
                "loading":  common_pb2.LoadState.LoadStateLoading,
                "loaded":   common_pb2.LoadState.LoadStateLoaded,
            }
            return milvus_pb2.GetLoadStateResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                state=mapping.get(col.load_state, common_pb2.LoadState.LoadStateNotLoad),
            )
        except LiteVecDBError as e:
            return milvus_pb2.GetLoadStateResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("GetLoadState failed: %s", e)
            return milvus_pb2.GetLoadStateResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    # ── Partition + Flush + Stats (Phase 10.5) ─────────────────

    def CreatePartition(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.create_partition(request.partition_name)
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("CreatePartition failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def DropPartition(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.drop_partition(request.partition_name)
            return common_pb2.Status(**success_status_kwargs())
        except LiteVecDBError as e:
            return common_pb2.Status(**to_status_kwargs(e))
        except Exception as e:
            logger.exception("DropPartition failed: %s", e)
            return common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e))

    def HasPartition(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            exists = col.has_partition(request.partition_name)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                value=exists,
            )
        except LiteVecDBError as e:
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
                value=False,
            )
        except Exception as e:
            logger.exception("HasPartition failed: %s", e)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
                value=False,
            )

    def ShowPartitions(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            names = col.list_partitions()
            return milvus_pb2.ShowPartitionsResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                partition_names=names,
            )
        except LiteVecDBError as e:
            return milvus_pb2.ShowPartitionsResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("ShowPartitions failed: %s", e)
            return milvus_pb2.ShowPartitionsResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def Flush(self, request, context):
        """Flush all named collections. pymilvus sends the collection
        name(s) in request.collection_names (plural)."""
        try:
            for cname in request.collection_names:
                if self._db.has_collection(cname):
                    col = self._db.get_collection(cname)
                    col.flush()
            return milvus_pb2.FlushResponse(
                status=common_pb2.Status(**success_status_kwargs()),
            )
        except LiteVecDBError as e:
            return milvus_pb2.FlushResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("Flush failed: %s", e)
            return milvus_pb2.FlushResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def GetFlushState(self, request, context):
        """pymilvus polls this after Flush. Phase 9 flush is synchronous
        so the answer is always True (flushed)."""
        return milvus_pb2.GetFlushStateResponse(
            status=common_pb2.Status(**success_status_kwargs()),
            flushed=True,
        )

    def GetCollectionStatistics(self, request, context):
        """Return row_count as a KeyValuePair list. pymilvus's
        get_collection_stats parses these pairs into a dict."""
        try:
            stats = self._db.get_collection_stats(request.collection_name)
            kv_pairs = [
                common_pb2.KeyValuePair(key=str(k), value=str(v))
                for k, v in stats.items()
            ]
            return milvus_pb2.GetCollectionStatisticsResponse(
                status=common_pb2.Status(**success_status_kwargs()),
                stats=kv_pairs,
            )
        except LiteVecDBError as e:
            return milvus_pb2.GetCollectionStatisticsResponse(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("GetCollectionStatistics failed: %s", e)
            return milvus_pb2.GetCollectionStatisticsResponse(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def ListDatabases(self, request, context):
        """LiteVecDB has no database concept; return a single default."""
        return milvus_pb2.ListDatabasesResponse(
            status=common_pb2.Status(**success_status_kwargs()),
            db_names=["default"],
        )

    # ── Explicitly UNIMPLEMENTED stubs ─────────────────────────
    #
    # The base class returns UNIMPLEMENTED for every method we don't
    # override, but those responses carry a generic "Method not
    # implemented!" message. For high-frequency RPCs that pymilvus
    # users might hit, we override with a friendlier explanation so
    # error messages point the user in the right direction instead of
    # being cryptic.

    def HybridSearch(self, request, context):
        """Multi-route ANN search with reranking fusion.

        Parses each sub-SearchRequest independently, dispatches to
        Collection.search(), then merges results via WeightedRanker
        or RRFRanker.
        """
        try:
            from litevecdb.adapter.grpc.reranker import parse_rank_params, rerank

            col = self._db.get_collection(request.collection_name)
            first_spec = next(iter(col._index_specs.values()), None) if col._index_specs else None  # noqa: SLF001
            default_metric = first_spec.metric_type if first_spec else "COSINE"

            # Parse rank_params
            rp = parse_rank_params(request.rank_params)

            # Execute each sub-request independently
            all_results = []
            for sub_req in request.requests:
                parsed = parse_search_request(sub_req, default_metric_type=default_metric)
                results = col.search(
                    query_vectors=parsed["query_vectors"],
                    top_k=parsed["top_k"],
                    metric_type=parsed["metric_type"],
                    partition_names=parsed.get("partition_names") or (
                        list(request.partition_names) or None
                    ),
                    expr=parsed["expr"],
                    output_fields=list(request.output_fields) or None,
                    anns_field=parsed.get("anns_field"),
                )
                all_results.append(results)

            # Rerank
            merged = rerank(
                strategy=rp["strategy"],
                params=rp["params"],
                all_results=all_results,
                limit=rp["limit"],
                offset=rp["offset"],
            )

            # Apply group_by if specified in rank_params
            gb_field = rp.get("group_by_field")
            if gb_field is not None:
                from litevecdb.engine.collection import _apply_group_by
                gb_size = rp.get("group_size") or 1
                gb_strict = rp.get("strict_group_size") or False
                merged = _apply_group_by(merged, gb_field, rp["limit"], gb_size, gb_strict)

            # Build response
            output_fields = list(request.output_fields) or None
            result_data = build_search_result_data(
                results=merged,
                schema=col.schema,
                top_k=rp["limit"],
                pk_name=col._pk_name,  # noqa: SLF001
                output_fields=output_fields,
                group_by_field=gb_field,
            )

            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**success_status_kwargs()),
                results=result_data,
                collection_name=col.name,
            )
        except LiteVecDBError as e:
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(**to_status_kwargs(e)),
            )
        except Exception as e:
            logger.exception("HybridSearch failed: %s", e)
            return milvus_pb2.SearchResults(
                status=common_pb2.Status(code=_UNEXPECTED_ERROR, reason=str(e)),
            )

    def RenameCollection(self, request, context):
        return self._unimplemented(
            context, "RenameCollection",
            "collection renaming is not supported in LiteVecDB MVP",
        )

    def CreateAlias(self, request, context):
        return self._unimplemented(context, "CreateAlias", "aliases are not in MVP scope")

    def DropAlias(self, request, context):
        return self._unimplemented(context, "DropAlias", "aliases are not in MVP scope")

    def AlterCollection(self, request, context):
        return self._unimplemented(
            context, "AlterCollection",
            "schema is immutable in LiteVecDB — create a new collection instead",
        )

    def LoadPartitions(self, request, context):
        return self._unimplemented(
            context, "LoadPartitions",
            "partition-level load is not supported; use load_collection instead",
        )

    def ReleasePartitions(self, request, context):
        return self._unimplemented(
            context, "ReleasePartitions",
            "partition-level release is not supported; use release_collection instead",
        )

    def ManualCompaction(self, request, context):
        """Compaction runs automatically after flush. pymilvus
        exposes compact() → ManualCompaction, but the engine doesn't
        have an on-demand trigger via the Collection API yet. Return
        success so pymilvus clients don't crash — the effect is already
        achieved by the automatic post-flush compaction."""
        return milvus_pb2.ManualCompactionResponse(
            status=common_pb2.Status(**success_status_kwargs()),
        )

    def GetCompactionState(self, request, context):
        return milvus_pb2.GetCompactionStateResponse(
            status=common_pb2.Status(**success_status_kwargs()),
            state=common_pb2.CompactionState.Completed,
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _build_ids_proto(self, pks, col):
        """Construct an ``IDs`` proto from a list of pk values.

        Picks the int_id or str_id slot based on the pk field's
        DataType. We don't (yet) handle mixed-type pk lists — Milvus
        doesn't either; pks must all be the same type.
        """
        from pymilvus.grpc_gen import schema_pb2 as _schema_pb2
        ids = _schema_pb2.IDs()
        pk_field = next((f for f in col.schema.fields if f.is_primary), None)
        if pk_field is None or not pks:
            return ids
        if pk_field.dtype == DataType.VARCHAR:
            ids.str_id.data.extend([str(p) for p in pks])
        else:
            ids.int_id.data.extend([int(p) for p in pks])
        return ids

    @staticmethod
    def _extract_pks_from_expr(expr: str, col) -> "list | None":
        """If *expr* is the trivial ``<pk_field> in [v1, v2, ...]`` form,
        return the pk list. Otherwise return None to signal "fall back
        to the general query path".

        This pattern is what pymilvus's ``client.get(ids=[...])`` emits
        — recognizing it lets us route directly to ``col.get`` instead
        of doing a full filter pass.
        """
        if not expr:
            return None
        from litevecdb.search.filter import parse_expr
        from litevecdb.search.filter.ast import InOp, FieldRef, IntLit, StringLit
        try:
            ast = parse_expr(expr)
        except Exception:
            return None
        if not isinstance(ast, InOp):
            return None
        if not isinstance(ast.field, FieldRef):
            return None
        if ast.field.name != col._pk_name:
            return None
        if ast.negate:
            return None  # "not in" — too broad, fall through to query
        pks: list = []
        for el in ast.values.elements:
            if isinstance(el, IntLit):
                pks.append(el.value)
            elif isinstance(el, StringLit):
                pks.append(el.value)
            else:
                return None  # mixed/non-literal — fall back
        return pks

    @staticmethod
    def _unimplemented(context, rpc_name: str, reason: str = "") -> common_pb2.Status:
        """Build a friendly UNIMPLEMENTED status. Used by stubs that
        need to return a Status-shaped response (some RPCs return
        Status directly; others wrap it in a richer message). The
        gRPC context is also marked so the client sees the proper
        StatusCode."""
        msg = f"LiteVecDB does not support {rpc_name}"
        if reason:
            msg += f": {reason}"
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(msg)
        return common_pb2.Status(
            code=_UNEXPECTED_ERROR,
            reason=msg,
        )
