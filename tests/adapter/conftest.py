"""Shared fixtures for the gRPC adapter test suite.

Skipped automatically when pymilvus / grpcio is not installed, so the
suite doesn't break CI on bare environments.
"""

import pytest

# Probe pymilvus + grpcio at import time. If either is missing, skip
# the entire adapter test directory — the design treats them as an
# optional [grpc] extra, not a hard dependency.
pymilvus = pytest.importorskip("pymilvus")
grpc = pytest.importorskip("grpc")


@pytest.fixture
def grpc_server(tmp_path):
    """Start a LiteVecDB gRPC server in the current process on a
    random free port. Yields ``(port, db)``; tears down after the
    test."""
    from litevecdb.adapter.grpc.server import start_server_in_thread

    server, db, port = start_server_in_thread(str(tmp_path / "data"))
    try:
        yield port, db
    finally:
        server.stop(grace=1)
        db.close()


@pytest.fixture
def milvus_client(grpc_server):
    """A pymilvus MilvusClient connected to the local server."""
    from pymilvus import MilvusClient
    port, _db = grpc_server
    client = MilvusClient(uri=f"http://127.0.0.1:{port}")
    try:
        yield client
    finally:
        client.close()
