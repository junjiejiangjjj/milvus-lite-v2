class LiteVecDBError(Exception):
    """Base exception for LiteVecDB."""


class SchemaValidationError(LiteVecDBError):
    """Schema definition or record validation failed."""


class CollectionNotFoundError(LiteVecDBError):
    """Collection does not exist."""


class CollectionAlreadyExistsError(LiteVecDBError):
    """Collection already exists."""


class PartitionNotFoundError(LiteVecDBError):
    """Partition does not exist."""


class PartitionAlreadyExistsError(LiteVecDBError):
    """Partition already exists."""


class DefaultPartitionError(LiteVecDBError):
    """Illegal operation on the default partition."""


class WALCorruptedError(LiteVecDBError):
    """WAL file is corrupted and cannot be fully recovered."""


class ManifestCorruptedError(LiteVecDBError):
    """Both manifest.json and manifest.json.prev failed to load."""


class DataDirLockedError(LiteVecDBError):
    """Another process holds the data_dir LOCK file."""


# ── Phase 9.3 — index lifecycle ─────────────────────────────────────


class CollectionNotLoadedError(LiteVecDBError):
    """search / get / query was called on a Collection that is not in
    the 'loaded' state. Mirrors Milvus behavior — the user must call
    Collection.load() (or pymilvus client.load_collection) first.
    Caused by either having never called load(), or having explicitly
    called release()."""


class IndexAlreadyExistsError(LiteVecDBError):
    """create_index was called on a Collection that already has an
    index. Drop the existing one with drop_index first."""


class IndexNotFoundError(LiteVecDBError):
    """drop_index / describe_index was called for a field that has no
    index attached."""


class IndexBackendUnavailableError(LiteVecDBError):
    """Requested index_type requires an optional dependency that is
    not installed (e.g. faiss-cpu for HNSW). Install the
    [faiss] extra: ``pip install litevecdb[faiss]``."""
