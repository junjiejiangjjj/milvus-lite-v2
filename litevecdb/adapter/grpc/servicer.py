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

from typing import TYPE_CHECKING

import grpc
from pymilvus.grpc_gen import common_pb2, milvus_pb2, milvus_pb2_grpc

if TYPE_CHECKING:
    from litevecdb.db import LiteVecDB


# Milvus ErrorCode constants we use directly (the full enum lives in
# pymilvus.grpc_gen.common_pb2.ErrorCode but we only need a few here).
_SUCCESS = 0
_UNEXPECTED_ERROR = 1


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
