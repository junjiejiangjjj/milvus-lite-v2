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
from litevecdb.adapter.grpc.translators.records import (
    fields_data_to_records,
    records_to_fields_data,
)
from litevecdb.adapter.grpc.translators.schema import (
    litevecdb_to_milvus_schema,
    milvus_to_litevecdb_schema,
)
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

            pks = self._extract_pks_from_expr(request.expr, col)
            if pks is not None:
                rows = col.get(pks, partition_names=partition_names)
            else:
                rows = col.query(
                    request.expr,
                    output_fields=output_fields,
                    partition_names=partition_names,
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

    # ── Helpers (used by Phase 10.2+ implementations) ───────────

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
