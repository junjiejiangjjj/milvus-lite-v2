"""DB layer — multi-Collection lifecycle management.

LiteVecDB is the top-level entry point. It manages multiple Collections
under one ``data_dir`` and ensures only one process can hold the dir at
a time via an advisory file lock.

Layout:

    data_dir/
    ├── LOCK                      # advisory flock — held while DB is open
    └── collections/
        ├── col_a/
        │   ├── schema.json       # Collection schema
        │   ├── manifest.json     # Manifest (single source of truth)
        │   ├── manifest.json.prev
        │   ├── wal/              # WAL files
        │   └── partitions/       # data + delta Parquet by partition
        └── col_b/
            └── ...

Collection instances are cached so that calling ``get_collection`` twice
with the same name returns the same object (and the same in-memory
MemTable / WAL state).
"""

from __future__ import annotations

import os
import sys
import shutil
from typing import Any, Dict, List, Optional

from litevecdb.engine.collection import Collection
from litevecdb.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DataDirLockedError,
)
from litevecdb.schema.persistence import load_schema, save_schema
from litevecdb.schema.types import CollectionSchema
from litevecdb.schema.validation import validate_schema


COLLECTIONS_DIRNAME = "collections"
LOCK_FILENAME = "LOCK"
SCHEMA_FILENAME = "schema.json"


class LiteVecDB:
    """Top-level entry point. Open one of these per process per data_dir.

    Usage:

        db = LiteVecDB("/path/to/data")
        col = db.create_collection("docs", schema)
        col.insert([...])
        results = col.search([...])
        db.close()

    Or as a context manager:

        with LiteVecDB("/path/to/data") as db:
            col = db.get_collection("docs")
            ...

    Multi-process safety: ``__init__`` acquires an advisory ``flock`` on
    ``{data_dir}/LOCK``. If another process already holds the lock,
    construction raises ``DataDirLockedError``. The lock is released by
    ``close()`` (or by process exit, since the OS reclaims it).
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self._collections_root(), exist_ok=True)

        self._lock_path = os.path.join(data_dir, LOCK_FILENAME)
        self._lock_fd: Optional[int] = None
        self._acquire_lock()

        # Cache of opened Collections, keyed by name. Collections are
        # only created when explicitly requested (lazy load on get).
        self._collections: Dict[str, Collection] = {}
        self._closed = False

    # ── public API ──────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        schema: CollectionSchema,
    ) -> Collection:
        """Create a new Collection. Raises if a Collection with this
        name already exists."""
        self._check_open()
        self._validate_name(name)

        # Validate the schema BEFORE touching disk so we don't leave a
        # half-initialized collection directory if validation fails.
        validate_schema(schema)

        if self.has_collection(name):
            raise CollectionAlreadyExistsError(
                f"collection {name!r} already exists"
            )

        col_dir = self._collection_dir(name)
        os.makedirs(col_dir, exist_ok=False)

        # Persist schema first — this is the marker that the Collection
        # exists. has_collection() looks for it.
        save_schema(schema, name, os.path.join(col_dir, SCHEMA_FILENAME))

        col = Collection(name, col_dir, schema)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        """Open an existing Collection. Subsequent calls return the same
        cached instance."""
        self._check_open()
        if name in self._collections:
            return self._collections[name]
        if not self.has_collection(name):
            raise CollectionNotFoundError(
                f"collection {name!r} does not exist"
            )

        col_dir = self._collection_dir(name)
        _name, schema = load_schema(os.path.join(col_dir, SCHEMA_FILENAME))
        col = Collection(name, col_dir, schema)
        self._collections[name] = col
        return col

    def drop_collection(self, name: str) -> None:
        """Close and delete a Collection. Raises if it does not exist."""
        self._check_open()
        if not self.has_collection(name):
            raise CollectionNotFoundError(
                f"collection {name!r} does not exist"
            )

        # Close the cached instance first so its WAL writers release
        # any open file handles before we rmtree the directory.
        if name in self._collections:
            self._collections[name].close()
            del self._collections[name]

        col_dir = self._collection_dir(name)
        shutil.rmtree(col_dir, ignore_errors=False)

    def has_collection(self, name: str) -> bool:
        """True iff a Collection with this name exists on disk."""
        return os.path.exists(
            os.path.join(self._collection_dir(name), SCHEMA_FILENAME)
        )

    def list_collections(self) -> List[str]:
        """Return all Collection names, sorted."""
        root = self._collections_root()
        if not os.path.exists(root):
            return []
        names: List[str] = []
        for entry in os.listdir(root):
            sub = os.path.join(root, entry)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, SCHEMA_FILENAME)):
                names.append(entry)
        return sorted(names)

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """Phase 9.1: Return basic stats for a collection.

        Currently returns ``{"row_count": int}``. The Phase 10 gRPC
        adapter maps this directly into Milvus's
        ``GetCollectionStatistics`` response (a list of KeyValuePair
        with the single ``row_count`` entry).

        Loads the Collection if not already cached. Raises
        ``CollectionNotFoundError`` if it doesn't exist.
        """
        col = self.get_collection(name)
        return {"row_count": col.num_entities}

    def close(self) -> None:
        """Close every cached Collection and release the LOCK.

        Idempotent. After close(), this DB instance is unusable; create
        a new one to reopen the data_dir.
        """
        if self._closed:
            return
        for name in list(self._collections.keys()):
            try:
                self._collections[name].close()
            except Exception:
                pass
        self._collections.clear()
        self._release_lock()
        self._closed = True

    # ── context manager ─────────────────────────────────────────

    def __enter__(self) -> "LiteVecDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── properties ──────────────────────────────────────────────

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def closed(self) -> bool:
        return self._closed

    # ── internals ───────────────────────────────────────────────

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("LiteVecDB is closed")

    def _collections_root(self) -> str:
        return os.path.join(self._data_dir, COLLECTIONS_DIRNAME)

    def _collection_dir(self, name: str) -> str:
        return os.path.join(self._collections_root(), name)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"collection name must be a string, got {type(name).__name__}")
        if not name:
            raise ValueError("collection name must not be empty")
        # Forbid path separators and dot segments — names map directly to
        # filesystem paths, so leakage outside collections/ is unsafe.
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError(f"invalid collection name: {name!r}")

    def _acquire_lock(self) -> None:
        """Acquire an exclusive non-blocking lock on LOCK_FILENAME.

        Uses fcntl.flock on Unix and msvcrt.locking on Windows.
        Raises DataDirLockedError if another process holds it.
        """
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as e:
            os.close(fd)
            raise DataDirLockedError(
                f"another process holds the lock on {self._data_dir!r}: {e}"
            ) from e
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass
        self._lock_fd = None
