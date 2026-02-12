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
