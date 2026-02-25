"""Write-Ahead Log — Arrow IPC Streaming, dual-file (data + delta).

Each WAL round corresponds to a pair of files (wal_data_{N}.arrow + wal_delta_{N}.arrow).
After a successful flush the pair is deleted.  Writers are lazily initialised on
first write so that unused files are never created.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import pyarrow as pa

from litevecdb.constants import SEQ_FORMAT_WIDTH, WAL_DATA_TEMPLATE, WAL_DELTA_TEMPLATE


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    """Read a single WAL file and return its RecordBatch list.

    * File does not exist → []
    * File is complete   → all batches
    * File is truncated  → batches read before the truncation point
    """
    if not os.path.exists(path):
        return []

    batches: list[pa.RecordBatch] = []
    try:
        source = pa.OSFile(path, "rb")
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            batches.append(batch)
    except pa.ArrowInvalid:
        # Truncated RecordBatch — keep whatever was successfully read.
        pass
    except Exception:
        # Schema unreadable / severely corrupted — return empty.
        pass

    return batches


def _cleanup_old_wals(wal_dir: str, up_to_number: int) -> None:
    """Delete all WAL files whose number is <= *up_to_number*."""
    for n in WAL.find_wal_files(wal_dir):
        if n <= up_to_number:
            data_path = os.path.join(
                wal_dir,
                WAL_DATA_TEMPLATE.format(n=n, w=SEQ_FORMAT_WIDTH),
            )
            delta_path = os.path.join(
                wal_dir,
                WAL_DELTA_TEMPLATE.format(n=n, w=SEQ_FORMAT_WIDTH),
            )
            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(delta_path):
                os.remove(delta_path)


# ---------------------------------------------------------------------------
# WAL class
# ---------------------------------------------------------------------------

class WAL:
    """Write-Ahead Log with lazy dual-file initialisation."""

    def __init__(
        self,
        wal_dir: str,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
        wal_number: int,
    ) -> None:
        self.wal_dir = wal_dir
        self._wal_data_schema = wal_data_schema
        self._wal_delta_schema = wal_delta_schema
        self._number = wal_number

        self._data_writer: Optional[pa.ipc.RecordBatchStreamWriter] = None
        self._delta_writer: Optional[pa.ipc.RecordBatchStreamWriter] = None
        self._data_sink: Optional[pa.OSFile] = None
        self._delta_sink: Optional[pa.OSFile] = None
        self._closed = False

        os.makedirs(wal_dir, exist_ok=True)

    # ── properties ──────────────────────────────────────────────

    @property
    def number(self) -> int:
        return self._number

    @property
    def data_path(self) -> Optional[str]:
        if self._data_writer is None:
            return None
        return os.path.join(
            self.wal_dir,
            WAL_DATA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )

    @property
    def delta_path(self) -> Optional[str]:
        if self._delta_writer is None:
            return None
        return os.path.join(
            self.wal_dir,
            WAL_DELTA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )

    # ── write ───────────────────────────────────────────────────

    def write_insert(self, record_batch: pa.RecordBatch) -> None:
        """Append *record_batch* to the wal_data file (lazy init)."""
        assert not self._closed, "WAL already closed"

        if self._data_writer is None:
            path = os.path.join(
                self.wal_dir,
                WAL_DATA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
            )
            self._data_sink = pa.OSFile(path, "wb")
            self._data_writer = pa.ipc.new_stream(self._data_sink, self._wal_data_schema)

        self._data_writer.write_batch(record_batch)

    def write_delete(self, record_batch: pa.RecordBatch) -> None:
        """Append *record_batch* to the wal_delta file (lazy init)."""
        assert not self._closed, "WAL already closed"

        if self._delta_writer is None:
            path = os.path.join(
                self.wal_dir,
                WAL_DELTA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
            )
            self._delta_sink = pa.OSFile(path, "wb")
            self._delta_writer = pa.ipc.new_stream(self._delta_sink, self._wal_delta_schema)

        self._delta_writer.write_batch(record_batch)

    # ── lifecycle ───────────────────────────────────────────────

    def close_and_delete(self) -> None:
        """Close writers and delete both WAL files.  Idempotent."""
        if self._closed:
            return

        # Close writers
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_sink.close()
        if self._delta_writer is not None:
            self._delta_writer.close()
            self._delta_sink.close()

        # Delete files
        data_path = os.path.join(
            self.wal_dir,
            WAL_DATA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )
        delta_path = os.path.join(
            self.wal_dir,
            WAL_DELTA_TEMPLATE.format(n=self._number, w=SEQ_FORMAT_WIDTH),
        )
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists(delta_path):
            os.remove(delta_path)

        self._closed = True

    # ── static helpers ──────────────────────────────────────────

    @staticmethod
    def find_wal_files(wal_dir: str) -> List[int]:
        """Scan *wal_dir* and return a sorted list of WAL numbers found."""
        if not os.path.exists(wal_dir):
            return []

        numbers: set[int] = set()
        pattern = re.compile(r"^wal_(data|delta)_(\d{6})\.arrow$")
        for filename in os.listdir(wal_dir):
            m = pattern.match(filename)
            if m:
                numbers.add(int(m.group(2)))

        return sorted(numbers)

    @staticmethod
    def recover(
        wal_dir: str,
        wal_number: int,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
    ) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """Read WAL files for *wal_number* and return (data_batches, delta_batches)."""
        data_path = os.path.join(
            wal_dir,
            WAL_DATA_TEMPLATE.format(n=wal_number, w=SEQ_FORMAT_WIDTH),
        )
        delta_path = os.path.join(
            wal_dir,
            WAL_DELTA_TEMPLATE.format(n=wal_number, w=SEQ_FORMAT_WIDTH),
        )

        data_batches = _read_wal_file(data_path)
        delta_batches = _read_wal_file(delta_path)

        return data_batches, delta_batches
