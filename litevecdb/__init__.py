"""LiteVecDB — local embedded vector database.

Public API:

    from litevecdb import LiteVecDB, CollectionSchema, FieldSchema, DataType

    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128),
    ])

    with LiteVecDB("/path/to/data") as db:
        col = db.create_collection("docs", schema)
        col.insert([{"id": "doc1", "vec": [...]}, ...])
        results = col.search([[query_vector]], top_k=10)
"""

from litevecdb.db import LiteVecDB
from litevecdb.engine.collection import Collection
from litevecdb.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DataDirLockedError,
    DefaultPartitionError,
    LiteVecDBError,
    ManifestCorruptedError,
    PartitionAlreadyExistsError,
    PartitionNotFoundError,
    SchemaValidationError,
    WALCorruptedError,
)
from litevecdb.schema.types import CollectionSchema, DataType, FieldSchema
from litevecdb.search.filter import (
    FilterError,
    FilterFieldError,
    FilterParseError,
    FilterTypeError,
)

__all__ = [
    # Top-level entry point
    "LiteVecDB",
    # Collection (returned from db.get_collection / db.create_collection)
    "Collection",
    # Schema types
    "CollectionSchema",
    "FieldSchema",
    "DataType",
    # Exception hierarchy
    "LiteVecDBError",
    "SchemaValidationError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "PartitionNotFoundError",
    "PartitionAlreadyExistsError",
    "DefaultPartitionError",
    "WALCorruptedError",
    "ManifestCorruptedError",
    "DataDirLockedError",
    # Filter expression errors (Phase 8)
    "FilterError",
    "FilterParseError",
    "FilterFieldError",
    "FilterTypeError",
]
