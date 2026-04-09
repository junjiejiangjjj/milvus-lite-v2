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
from litevecdb.adapter.grpc.translators.schema import (
    litevecdb_to_milvus_schema,
    milvus_to_litevecdb_schema,
)
from litevecdb.exceptions import LiteVecDBError

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

    # ── Helpers (used by Phase 10.2+ implementations) ───────────

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
