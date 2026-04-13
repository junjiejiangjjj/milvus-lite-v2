"""milvus-lite compatibility layer.

Drop-in replacement for the original milvus-lite package. When pymilvus
detects a `.db` URI, it does::

    from milvus_lite.server_manager import server_manager_instance
    local_uri = server_manager_instance.start_and_get_uri(uri)

This module provides that interface, backed by LiteVecDB's pure-Python
engine and gRPC adapter running in-process.
"""

from milvus_lite.server_manager import server_manager_instance

__all__ = ["server_manager_instance"]
