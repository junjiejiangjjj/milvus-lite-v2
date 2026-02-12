# ── MemTable ──
MEMTABLE_SIZE_LIMIT = 10_000

# ── Compaction ──
MAX_DATA_FILES = 32
COMPACTION_MIN_FILES_PER_BUCKET = 4
COMPACTION_BUCKET_BOUNDARIES = [1_000_000, 10_000_000, 100_000_000]  # bytes

# ── 文件命名 ──
SEQ_FORMAT_WIDTH = 6
DATA_FILE_TEMPLATE = "data_{min:0{w}d}_{max:0{w}d}.parquet"
DELTA_FILE_TEMPLATE = "delta_{min:0{w}d}_{max:0{w}d}.parquet"
WAL_DATA_TEMPLATE = "wal_data_{n:0{w}d}.arrow"
WAL_DELTA_TEMPLATE = "wal_delta_{n:0{w}d}.arrow"

# ── Partition ──
DEFAULT_PARTITION = "_default"
ALL_PARTITIONS = "_all"
