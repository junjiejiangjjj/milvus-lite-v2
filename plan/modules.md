# LiteVecDB 代码模块设计

## 1. 顶层 Package 划分

```
litevecdb/
├── schema/          # 数据模型与类型系统
├── storage/         # 存储层：持久化 + 内存缓冲
├── engine/          # 引擎层：核心逻辑编排
├── search/          # 搜索层：向量检索
├── db.py            # DB 层：多 Collection 生命周期管理
├── constants.py     # 全局常量
├── exceptions.py    # 异常层级
└── __init__.py      # 公共 API 导出
```

## 2. 完整代码结构

```
lite-v2/
├── MVP.md
├── write-pipeline.md
├── research.md
├── modules.md
├── CLAUDE.md
│
├── litevecdb/
│   ├── __init__.py                 # 公共 API 导出
│   ├── constants.py                # 全局常量
│   ├── exceptions.py               # 异常层级
│   │
│   ├── schema/                     # ══ 数据模型与类型系统 ══
│   │   ├── __init__.py             #   导出: DataType, FieldSchema, CollectionSchema
│   │   ├── types.py                #   DataType enum, FieldSchema, CollectionSchema 类定义
│   │   ├── validation.py           #   validate_schema(), validate_record(), separate_dynamic_fields()
│   │   ├── arrow_builder.py        #   4 个 Arrow Schema 构建器 (data/delta/wal_data/wal_delta)
│   │   └── persistence.py          #   schema.json 读写 (save_schema / load_schema)
│   │
│   ├── storage/                    # ══ 存储层 ══
│   │   ├── __init__.py             #   导出: WAL, MemTable, DataFile, DeltaLog, Manifest
│   │   ├── wal.py                  #   WAL (Arrow IPC Streaming, 双文件, write/recover/delete)
│   │   ├── memtable.py             #   MemTable (insert_buf + delete_buf, _partition 路由, flush 拆分)
│   │   ├── data_file.py            #   数据 Parquet 文件 (读写, 命名, seq 范围解析)
│   │   ├── delta_log.py            #   Delta Log (Parquet 读写 + 内存 deleted_map)
│   │   └── manifest.py             #   Manifest (JSON, 原子更新, Partition 级文件列表)
│   │
│   ├── engine/                     # ══ 引擎层 ══
│   │   ├── __init__.py             #   导出: Collection
│   │   ├── collection.py           #   Collection 核心 (入口, _seq 分配, insert/delete/get/search, Partition CRUD)
│   │   ├── flush.py                #   Flush 管线 (7 步: 冻结→拆分→写Parquet→更新deleted_map→更新Manifest→删WAL→Compaction)
│   │   ├── recovery.py             #   崩溃恢复 (5 步: 加载Manifest→重放WAL→校验文件→清理孤儿→重建deleted_map)
│   │   └── compaction.py           #   Compaction Manager (Size-Tiered, 按 Partition 独立, 文件分桶/合并/去重/过滤)
│   │
│   ├── search/                     # ══ 搜索层 ══
│   │   ├── __init__.py             #   导出: search 函数
│   │   ├── bitmap.py               #   bitmap 管线 (去重 + 删除过滤, 将来扩展标量过滤)
│   │   ├── distance.py             #   距离计算 (cosine / L2 / inner product, NumPy 实现)
│   │   └── executor.py             #   搜索执行器 (收集数据 + bitmap + 向量检索 + top-k, 将来替换为 FAISS)
│   │
│   └── db.py                       # ══ DB 层 ══
│                                    #   LiteVecDB 类 (create/get/drop/list_collection, close)
│
├── tests/
│   ├── conftest.py                 # 共享 fixtures: 临时目录, 示例 Schema, 随机向量生成器
│   │
│   ├── schema/
│   │   ├── test_types.py           #   DataType / FieldSchema / CollectionSchema 定义
│   │   ├── test_validation.py      #   Schema 校验规则, 记录校验, 动态字段分离
│   │   ├── test_arrow_builder.py   #   4 个 Arrow Schema 构建, TYPE_MAP 映射
│   │   └── test_persistence.py     #   schema.json 序列化/反序列化往返
│   │
│   ├── storage/
│   │   ├── test_wal.py             #   WAL 写入/恢复往返, 双文件生命周期, 损坏处理
│   │   ├── test_memtable.py        #   put/delete/get 语义, upsert, flush Partition 拆分
│   │   ├── test_data_file.py       #   Parquet 读写往返, 文件命名, seq 范围解析
│   │   ├── test_delta_log.py       #   load_all/add/is_deleted, deleted_map 正确性
│   │   └── test_manifest.py        #   加载/保存原子性, Partition 文件列表管理
│   │
│   ├── engine/
│   │   ├── test_collection.py      #   Collection 级 E2E: insert/delete/get/search, Partition CRUD, upsert
│   │   ├── test_flush.py           #   Flush 管线端到端, 6 个崩溃点场景, 恢复正确性
│   │   ├── test_recovery.py        #   崩溃恢复 5 步, WAL 重放, 孤儿文件清理
│   │   └── test_compaction.py      #   文件分桶, 合并去重, 删除过滤, Manifest 更新
│   │
│   ├── search/
│   │   ├── test_bitmap.py          #   bitmap 构建: 去重 + 删除过滤
│   │   ├── test_distance.py        #   cosine / L2 / IP 距离正确性
│   │   └── test_executor.py        #   搜索端到端, top-k, Partition Pruning
│   │
│   └── test_db.py                  #   多 Collection 生命周期, close/cleanup
│
├── pyproject.toml
└── requirements.txt
```

## 3. 四大 Package 详解

### 3.1 schema/ — 数据模型与类型系统

**职责边界**：定义数据是什么样的，不关心数据存在哪里、怎么流转。

| 子模块 | 职责 | 核心类/函数 |
|--------|------|------------|
| `types.py` | 类型定义 | `DataType` enum, `FieldSchema` dataclass, `CollectionSchema` dataclass, `TYPE_MAP` (DataType→Arrow) |
| `validation.py` | 校验逻辑 | `validate_schema(schema)` — 主键/向量约束; `validate_record(record, schema)` — 字段类型/非空/维度; `separate_dynamic_fields(record, schema)` — Schema 内外字段分离→$meta |
| `arrow_builder.py` | Arrow Schema 构建 | `build_data_schema()`, `build_delta_schema()`, `build_wal_data_schema()`, `build_wal_delta_schema()`, `get_primary_field()`, `get_vector_field()` |
| `persistence.py` | 持久化 | `save_schema_json(schema, collection_name, path)`, `load_schema_json(path)` |

```python
# schema/__init__.py
from litevecdb.schema.types import DataType, FieldSchema, CollectionSchema
```

### 3.2 storage/ — 存储层

**职责边界**：管理数据的物理存储（磁盘文件 + 内存缓冲），提供读写原语。不理解业务流程（flush 编排、recovery 编排由 engine 负责）。

| 子模块 | 职责 | 核心类/方法 |
|--------|------|------------|
| `wal.py` | WAL 持久化 | `WAL(wal_dir, wal_data_schema, wal_delta_schema)` — `write_insert(rb)`, `write_delete(rb)`, `recover()→(data_batches, delta_batches)`, `close_and_delete()` |
| `memtable.py` | 内存缓冲 | `MemTable(schema)` — `put(_seq, _partition, **fields)`, `delete(pk, _seq, _partition)`, `get(pk)`, `flush()→Dict[partition, (data_table, delta_table)]`, `size()` |
| `data_file.py` | 数据 Parquet | `write_data_file(table, partition_dir, seq_min, seq_max)→path`, `read_data_file(path)→pa.Table`, `parse_seq_range(filename)→(min, max)`, `get_file_size(path)→int` |
| `delta_log.py` | 删除记录 | `DeltaLog(pk_name)` — `load_all(partition_files)`, `add(delta_table, path)`, `is_deleted(pk, data_seq)→bool`, `remove_files(files)` |
| `manifest.py` | 全局状态 | `Manifest(data_dir)` — `load()`, `save()` (原子), `add/remove_data_file(partition, file)`, `add/remove_delta_file(partition, file)`, `add/remove_partition(name)` |

```python
# storage/__init__.py
from litevecdb.storage.wal import WAL
from litevecdb.storage.memtable import MemTable
from litevecdb.storage.data_file import write_data_file, read_data_file
from litevecdb.storage.delta_log import DeltaLog
from litevecdb.storage.manifest import Manifest
```

### 3.3 engine/ — 引擎层

**职责边界**：编排 storage 组件，实现业务流程（写入→flush→落盘、崩溃恢复、compaction）。是 storage 和 search 的上层调用者。

| 子模块 | 职责 | 核心内容 |
|--------|------|---------|
| `collection.py` | 引擎核心 | `Collection(name, data_dir, schema)` — `insert()`, `delete()`, `get()`, `search()`, `create/drop/list_partitions()`, `add_field()`, `flush()`, `close()`, `_alloc_seq()` |
| `flush.py` | Flush 管线 | `execute_flush(frozen_memtable, frozen_wal, manifest, delta_log, compaction_mgr)` — 7 步流程 |
| `recovery.py` | 崩溃恢复 | `execute_recovery(data_dir, manifest, wal, delta_log)` — 5 步流程 |
| `compaction.py` | Compaction | `CompactionManager(data_dir, schema)` — `maybe_compact(partition, data_files, delta_log, manifest)`, `_bucket_files()`, `_merge_and_dedup()` |

```python
# engine/__init__.py
from litevecdb.engine.collection import Collection
```

**flush.py 与 collection.py 的关系**：
- `collection.py` 在 `insert()`/`delete()` 中检测 MemTable 满了时，调用 `flush.execute_flush()`
- `flush.py` 接收冻结的 MemTable/WAL + 各 storage 组件引用，执行 7 步管线
- flush 不持有 Collection 引用，只操作传入的 storage 组件 → **无循环依赖**

**recovery.py 与 collection.py 的关系**：
- `collection.py` 在 `__init__()` 中调用 `recovery.execute_recovery()`
- recovery 接收 data_dir + storage 组件引用，返回恢复后的状态
- 同样无循环依赖

### 3.4 search/ — 搜索层

**职责边界**：向量检索，输入是数据 + 查询，输出是 top-k 结果。不关心数据从哪来。

| 子模块 | 职责 | 核心函数 |
|--------|------|---------|
| `bitmap.py` | 有效性过滤 | `build_valid_mask(all_pks, all_seqs, delta_log)→np.ndarray[bool]` — 去重(同PK保留max_seq) + 删除过滤(delta_log.is_deleted) + 将来扩展标量过滤 |
| `distance.py` | 距离计算 | `cosine_distance(q, candidates)→np.ndarray`, `l2_distance(...)`, `ip_distance(...)`, `compute_distances(q, candidates, metric_type)` — 纯数学，无状态 |
| `executor.py` | 搜索编排 | `execute_search(query_vectors, all_pks, all_seqs, all_vectors, delta_log, top_k, metric_type)→List[List[dict]]` — 调用 bitmap + distance + top-k 选择 |

```python
# search/__init__.py
from litevecdb.search.executor import execute_search
```

**为什么 search 独立为 package**：
- bitmap 管线将来要扩展标量过滤，逻辑会增长
- distance 是纯数学模块，将来替换 FAISS 时只需替换 executor.py
- 与 engine 解耦：engine.collection 调用 `search.execute_search()`，传入收集好的数据

## 4. 依赖图

```
                        litevecdb.__init__
                              │
                           db.py
                              │
                    engine/collection.py
                     /        |        \
                    /         |         \
        engine/flush.py  engine/recovery.py  engine/compaction.py
               \         |         /              |
                \        |        /               |
                 storage/*                  search/executor.py
                 ├── wal.py                    /          \
                 ├── memtable.py      search/bitmap.py  search/distance.py
                 ├── data_file.py          |
                 ├── delta_log.py          |
                 └── manifest.py           |
                        \                  |
                     schema/*              |
                     ├── types.py          |
                     ├── validation.py     |
                     ├── arrow_builder.py  |
                     └── persistence.py    |
                            \             /
                         constants.py + exceptions.py
```

**依赖方向**（严格向下，无循环）：

```
Level 0:  constants.py, exceptions.py           ← 无内部依赖
Level 1:  schema/*                              ← 依赖 L0
Level 2:  storage/*                             ← 依赖 L0, L1
Level 3:  search/bitmap.py, search/distance.py  ← 依赖 L0, L2(delta_log)
Level 4:  search/executor.py                    ← 依赖 L3
Level 5:  engine/flush.py, recovery.py, compaction.py ← 依赖 L0-L4
Level 6:  engine/collection.py                  ← 依赖 L0-L5
Level 7:  db.py                                 ← 依赖 L6
```

## 5. 公共 API 导出

```python
# litevecdb/__init__.py
from litevecdb.schema import DataType, FieldSchema, CollectionSchema
from litevecdb.db import LiteVecDB
from litevecdb.exceptions import LiteVecDBError

__all__ = ["LiteVecDB", "CollectionSchema", "FieldSchema", "DataType", "LiteVecDBError"]
```

用户只需 `from litevecdb import LiteVecDB, CollectionSchema, FieldSchema, DataType`。Collection 通过 `db.get_collection()` 获得，不直接导出。

## 6. constants.py 内容

```python
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
```

## 7. exceptions.py 内容

```python
class LiteVecDBError(Exception): ...

class SchemaValidationError(LiteVecDBError): ...
class CollectionNotFoundError(LiteVecDBError): ...
class CollectionAlreadyExistsError(LiteVecDBError): ...
class PartitionNotFoundError(LiteVecDBError): ...
class PartitionAlreadyExistsError(LiteVecDBError): ...
class DefaultPartitionError(LiteVecDBError): ...
class WALCorruptedError(LiteVecDBError): ...
```

## 8. 与实现优先级对齐

| P# | 实现模块 | 测试 |
|----|---------|------|
| P0 | `constants`, `exceptions`, `schema/*` (全部) | `tests/schema/*` |
| P1 | `storage/manifest` | `tests/storage/test_manifest` |
| P2 | `storage/memtable`, `storage/wal` | `tests/storage/test_memtable`, `test_wal` |
| P3 | `storage/data_file`, `storage/delta_log`, `engine/flush` | `tests/storage/test_data_file`, `test_delta_log`, `tests/engine/test_flush` |
| P4 | `engine/collection` (partition 方法) | `tests/engine/test_collection` (部分) |
| P5 | `engine/collection` (insert/delete/get E2E) | `tests/engine/test_collection` |
| P6 | `search/*` (全部) | `tests/search/*` |
| P7 | `engine/compaction` | `tests/engine/test_compaction` |
| P8 | `db.py` | `tests/test_db` |
| P9 | `engine/recovery` | `tests/engine/test_recovery` |
| P10 | `engine/collection` (add_field) | `tests/engine/test_collection` |
| P11 | 端到端测试 | 全部 |

## 9. 模块接口详解

按依赖层级自底向上设计，每个模块只列对外接口（public），不列内部实现细节。

---

### 9.0 constants.py

```python
# ── MemTable ──
MEMTABLE_SIZE_LIMIT: int = 10_000          # insert_buf + delete_buf 合计阈值

# ── Compaction ──
MAX_DATA_FILES: int = 32                    # 单 Partition 最大数据文件数
COMPACTION_MIN_FILES_PER_BUCKET: int = 4    # 同一桶内文件数触发阈值
COMPACTION_BUCKET_BOUNDARIES: List[int] = [1_000_000, 10_000_000, 100_000_000]  # bytes

# ── 文件命名 ──
SEQ_FORMAT_WIDTH: int = 6
DATA_FILE_TEMPLATE: str = "data_{min:0{w}d}_{max:0{w}d}.parquet"
DELTA_FILE_TEMPLATE: str = "delta_{min:0{w}d}_{max:0{w}d}.parquet"
WAL_DATA_TEMPLATE: str = "wal_data_{n:0{w}d}.arrow"
WAL_DELTA_TEMPLATE: str = "wal_delta_{n:0{w}d}.arrow"

# ── Partition ──
DEFAULT_PARTITION: str = "_default"
ALL_PARTITIONS: str = "_all"               # 跨 Partition 删除时的内部标记
```

### 9.1 exceptions.py

```python
class LiteVecDBError(Exception):
    """所有 LiteVecDB 异常的基类"""

class SchemaValidationError(LiteVecDBError):
    """Schema 定义不合法 或 记录不符合 Schema"""

class CollectionNotFoundError(LiteVecDBError):
    """Collection 不存在"""

class CollectionAlreadyExistsError(LiteVecDBError):
    """Collection 已存在"""

class PartitionNotFoundError(LiteVecDBError):
    """Partition 不存在"""

class PartitionAlreadyExistsError(LiteVecDBError):
    """Partition 已存在"""

class DefaultPartitionError(LiteVecDBError):
    """试图删除 _default Partition"""

class WALCorruptedError(LiteVecDBError):
    """WAL 文件损坏，无法恢复"""
```

---

### 9.2 schema/types.py

```python
class DataType(Enum):
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    DOUBLE = "double"
    VARCHAR = "varchar"
    JSON = "json"
    FLOAT_VECTOR = "float_vector"


@dataclass
class FieldSchema:
    name: str
    dtype: DataType
    is_primary: bool = False
    dim: Optional[int] = None           # 仅 FLOAT_VECTOR 需要
    max_length: Optional[int] = None    # 仅 VARCHAR 需要
    nullable: bool = False
    default_value: Any = None


@dataclass
class CollectionSchema:
    fields: List[FieldSchema]
    version: int = 1                    # Schema 版本号，每次变更 +1
    enable_dynamic_field: bool = False  # 是否启用 $meta 动态字段


# DataType → PyArrow 类型映射
TYPE_MAP: Dict[DataType, Any] = {
    DataType.BOOL:         pa.bool_(),
    DataType.INT8:         pa.int8(),
    DataType.INT16:        pa.int16(),
    DataType.INT32:        pa.int32(),
    DataType.INT64:        pa.int64(),
    DataType.FLOAT:        pa.float32(),
    DataType.DOUBLE:       pa.float64(),
    DataType.VARCHAR:      pa.string(),
    DataType.JSON:         pa.string(),
    DataType.FLOAT_VECTOR: None,  # 需要 dim，运行时通过 lambda dim: pa.list_(pa.float32(), dim) 生成
}
```

### 9.3 schema/validation.py

```python
def validate_schema(schema: CollectionSchema) -> None:
    """校验 CollectionSchema 定义合法性。

    检查规则：
    - 有且仅有一个 is_primary=True 字段，类型为 VARCHAR 或 INT64
    - 有且仅有一个 FLOAT_VECTOR 字段（MVP 限制）
    - 向量字段必须指定 dim > 0
    - 主键字段不可 nullable
    - 字段名不可重复
    - 字段名不可使用保留名（_seq, _partition, $meta）

    Raises:
        SchemaValidationError
    """


def validate_record(record: dict, schema: CollectionSchema) -> None:
    """校验单条记录是否符合 Schema。

    检查规则：
    - 主键字段存在且非 None
    - 向量字段存在且维度 == schema.dim
    - 已定义字段的类型匹配
    - non-nullable 字段不为 None
    - enable_dynamic_field=False 时不允许 Schema 外字段

    Raises:
        SchemaValidationError
    """


def separate_dynamic_fields(
    record: dict, schema: CollectionSchema
) -> Tuple[dict, Optional[str]]:
    """将 record 拆分为 Schema 内字段 + $meta JSON。

    Args:
        record: 用户传入的原始记录
        schema: CollectionSchema

    Returns:
        (schema_fields, meta_json)
        - schema_fields: 只含 Schema 定义的字段（含默认值填充）
        - meta_json: 动态字段序列化后的 JSON 字符串，无动态字段时为 None

    Raises:
        SchemaValidationError: enable_dynamic_field=False 但有 Schema 外字段
    """
```

### 9.4 schema/arrow_builder.py

```python
def get_primary_field(schema: CollectionSchema) -> FieldSchema:
    """返回主键 FieldSchema。"""


def get_vector_field(schema: CollectionSchema) -> FieldSchema:
    """返回向量 FieldSchema。"""


def build_data_schema(schema: CollectionSchema) -> pa.Schema:
    """数据 Parquet 文件的 Arrow Schema。
    列顺序: _seq(uint64) + 用户字段 + [$meta(string)]
    不含 _partition。"""


def build_delta_schema(schema: CollectionSchema) -> pa.Schema:
    """Delta Parquet 文件的 Arrow Schema。
    列: {pk_name}(主键类型) + _seq(uint64)
    不含 _partition。"""


def build_wal_data_schema(schema: CollectionSchema) -> pa.Schema:
    """WAL 数据文件的 Arrow Schema。
    列顺序: _seq(uint64) + _partition(string) + 用户字段 + [$meta(string)]
    比 data_schema 多 _partition 列。"""


def build_wal_delta_schema(schema: CollectionSchema) -> pa.Schema:
    """WAL 删除文件的 Arrow Schema。
    列: {pk_name}(主键类型) + _seq(uint64) + _partition(string)
    比 delta_schema 多 _partition 列。"""
```

### 9.5 schema/persistence.py

```python
def save_schema(
    schema: CollectionSchema,
    collection_name: str,
    path: str,
) -> None:
    """将 Schema 序列化为 JSON 写入 path。
    JSON 结构包含 collection_name（自描述）+ version + fields + enable_dynamic_field。
    使用 write-tmp + rename 原子写入。"""


def load_schema(path: str) -> Tuple[str, CollectionSchema]:
    """从 JSON 文件加载 Schema。
    Returns: (collection_name, schema)
    Raises: FileNotFoundError, SchemaValidationError（JSON 格式非法）"""
```

---

### 9.6 storage/wal.py

```python
class WAL:
    """Write-Ahead Log，Arrow IPC Streaming 格式，双文件（data + delta）。

    每轮写入对应一对 WAL 文件（wal_data_{N}.arrow + wal_delta_{N}.arrow），
    flush 成功后整个删除。Writer 延迟初始化（首次写入时创建文件）。
    """

    def __init__(
        self,
        wal_dir: str,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
        wal_number: int,
    ) -> None:
        """
        Args:
            wal_dir: WAL 文件所在目录
            wal_data_schema: wal_data 文件使用的 Arrow Schema（含 _partition）
            wal_delta_schema: wal_delta 文件使用的 Arrow Schema（含 _partition）
            wal_number: 本轮 WAL 编号（文件名中的 N）
        """

    def write_insert(self, record_batch: pa.RecordBatch) -> None:
        """追加写入 wal_data 文件。首次调用时创建文件和 writer。
        record_batch schema 必须匹配 wal_data_schema。"""

    def write_delete(self, record_batch: pa.RecordBatch) -> None:
        """追加写入 wal_delta 文件。首次调用时创建文件和 writer。
        record_batch schema 必须匹配 wal_delta_schema。"""

    def close_and_delete(self) -> None:
        """关闭 writer，删除两个 WAL 文件。flush 成功后调用。"""

    @staticmethod
    def find_wal_files(wal_dir: str) -> List[int]:
        """扫描 wal_dir，返回存在的 WAL 编号列表（用于 recovery 判断）。"""

    @staticmethod
    def recover(
        wal_dir: str,
        wal_number: int,
        wal_data_schema: pa.Schema,
        wal_delta_schema: pa.Schema,
    ) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """读取指定编号的 WAL 文件，返回 (data_batches, delta_batches)。
        每个 RecordBatch 含 _partition 列，恢复时按 _partition 路由到 MemTable。
        Raises: WALCorruptedError"""

    @property
    def number(self) -> int:
        """当前 WAL 编号。"""

    @property
    def data_path(self) -> Optional[str]:
        """wal_data 文件路径（未创建时为 None）。"""

    @property
    def delta_path(self) -> Optional[str]:
        """wal_delta 文件路径（未创建时为 None）。"""
```

### 9.7 storage/memtable.py

```python
class MemTable:
    """Collection 级共享的内存缓冲区。

    内部维护两个独立缓冲区：
    - insert_buf: dict[pk_value → record_dict]（含 _seq, _partition, 用户字段）
    - delete_buf: dict[pk_value → (_seq, _partition)]
    天然 upsert 语义：相同 PK 直接覆盖。
    """

    def __init__(self, schema: CollectionSchema) -> None:
        """
        Args:
            schema: CollectionSchema，用于确定 pk_name 和字段信息
        """

    def put(self, _seq: int, _partition: str, **fields) -> None:
        """写入一条插入记录到 insert_buf。
        - 相同 PK 直接覆盖（upsert 语义）
        - 清除同 PK 的 delete 记录（最后一次操作获胜）
        fields 中必须包含主键字段。"""

    def delete(self, pk_value: Any, _seq: int, _partition: str) -> None:
        """写入一条删除记录到 delete_buf。
        - 记录 (_seq, _partition)
        - 清除同 PK 的 insert 记录（最后一次操作获胜）"""

    def get(self, pk_value: Any) -> Optional[dict]:
        """点查单条记录。
        处理 insert/delete 冲突：比较 _seq，返回较新的状态。
        返回 record dict（不含 _partition）或 None。"""

    def flush(self) -> Dict[str, Tuple[Optional[pa.Table], Optional[pa.Table]]]:
        """按 Partition 拆分输出 Arrow Table。
        Returns: {partition_name: (data_table, delta_table)}
            - data_table: 使用 data_schema（不含 _partition），可为 None
            - delta_table: 使用 delta_schema（不含 _partition），可为 None
        调用后 insert_buf 和 delete_buf 不清空（由调用方冻结后丢弃整个 MemTable）。"""

    def size(self) -> int:
        """返回 len(insert_buf) + len(delete_buf)。"""

    def get_active_records(
        self, partition_names: Optional[List[str]] = None
    ) -> List[dict]:
        """返回 insert_buf 中未被 delete_buf 覆盖的活跃记录。
        用于 search/get 时读取 MemTable 层数据。
        partition_names 不为 None 时，只返回匹配 Partition 的记录。
        返回的 dict 不含 _partition 和 _seq（对外干净）。"""
```

### 9.8 storage/data_file.py

```python
def write_data_file(
    table: pa.Table,
    partition_dir: str,
    seq_min: int,
    seq_max: int,
) -> str:
    """将 Arrow Table 写入数据 Parquet 文件。
    文件路径: {partition_dir}/data/data_{seq_min:06d}_{seq_max:06d}.parquet
    自动创建 data/ 子目录（如不存在）。
    Returns: 写入的文件相对路径（相对于 Collection data_dir）。"""


def read_data_file(path: str) -> pa.Table:
    """读取数据 Parquet 文件，返回完整 Arrow Table。"""


def parse_seq_range(filename: str) -> Tuple[int, int]:
    """从文件名解析 seq 范围。
    'data_000001_000500.parquet' → (1, 500)
    'delta_000501_000503.parquet' → (501, 503)"""


def get_file_size(path: str) -> int:
    """返回文件字节大小，用于 Compaction 分桶。"""
```

### 9.9 storage/delta_log.py

```python
class DeltaLog:
    """Delta Log 管理：Parquet 文件读写 + 内存 deleted_map。

    deleted_map 常驻内存，启动时从所有 delta 文件重建，
    运行时随 flush/compaction 增量更新。
    """

    def __init__(self, pk_name: str) -> None:
        """
        Args:
            pk_name: 主键字段名（用于从 Parquet 中提取 PK 列）
        """

    def load_all(self, partition_delta_files: Dict[str, List[str]]) -> None:
        """从所有 Partition 的 delta 文件重建 deleted_map。
        Args:
            partition_delta_files: {partition_name: [绝对路径列表]}
        启动时调用一次。"""

    def write_and_update(self, delta_table: pa.Table, path: str) -> None:
        """写入新 delta Parquet 文件，同时更新内存 deleted_map。
        flush 步骤 3（写文件）+ 步骤 4（更新内存）合并。"""

    def update_memory(self, delta_table: pa.Table) -> None:
        """只更新内存 deleted_map，不写文件。
        用于 WAL 重放等场景。"""

    def is_deleted(self, pk_value: Any, data_seq: int) -> bool:
        """判断某条数据记录是否已被删除。
        规则: deleted_map[pk] > data_seq → 已删除。"""

    def remove_files(self, files: List[str]) -> None:
        """删除已消费的 delta 文件（Compaction 后调用）。"""

    def remove_entries(self, pk_values: List[Any]) -> None:
        """从 deleted_map 中移除指定 PK（Compaction 已物化删除后调用）。"""

    @property
    def deleted_map(self) -> Dict[Any, int]:
        """只读访问: pk_value → max_delete_seq。"""
```

### 9.10 storage/manifest.py

```python
class Manifest:
    """全局状态快照文件，通过原子替换（write-tmp + rename）更新。

    记录当前 _seq、Schema 版本、各 Partition 的文件列表、活跃 WAL。
    是系统唯一的 truth source。
    """

    def __init__(self, data_dir: str) -> None: ...

    # ── 持久化 ──

    def save(self) -> None:
        """原子更新 manifest.json：write-tmp + os.rename。version 自动 +1。"""

    @classmethod
    def load(cls, data_dir: str) -> "Manifest":
        """加载 manifest.json。文件不存在则返回初始状态（version=0, _default Partition）。"""

    # ── 文件管理（per Partition）──

    def add_data_file(self, partition: str, filename: str) -> None:
        """向指定 Partition 添加数据文件。"""

    def add_delta_file(self, partition: str, filename: str) -> None:
        """向指定 Partition 添加 delta 文件。"""

    def remove_data_files(self, partition: str, filenames: List[str]) -> None:
        """从指定 Partition 移除数据文件（Compaction 后）。"""

    def remove_delta_files(self, partition: str, filenames: List[str]) -> None:
        """从指定 Partition 移除 delta 文件（Compaction 后）。"""

    def get_data_files(self, partition: str) -> List[str]:
        """返回指定 Partition 的数据文件列表。"""

    def get_delta_files(self, partition: str) -> List[str]:
        """返回指定 Partition 的 delta 文件列表。"""

    def get_all_data_files(self) -> Dict[str, List[str]]:
        """返回所有 Partition 的数据文件: {partition: [files]}。"""

    def get_all_delta_files(self) -> Dict[str, List[str]]:
        """返回所有 Partition 的 delta 文件: {partition: [files]}。"""

    # ── Partition 管理 ──

    def add_partition(self, name: str) -> None:
        """添加新 Partition（初始化空文件列表）。
        Raises: PartitionAlreadyExistsError"""

    def remove_partition(self, name: str) -> None:
        """移除 Partition。
        Raises: DefaultPartitionError（不允许删除 _default）, PartitionNotFoundError"""

    def list_partitions(self) -> List[str]:
        """返回所有 Partition 名称列表。"""

    def has_partition(self, name: str) -> bool:
        """检查 Partition 是否存在。"""

    # ── 属性 ──

    @property
    def version(self) -> int:
        """Manifest 版本号（每次 save +1）。"""

    @property
    def current_seq(self) -> int:
        """当前最大 _seq，启动时恢复计数器。"""

    @current_seq.setter
    def current_seq(self, value: int) -> None: ...

    @property
    def schema_version(self) -> int:
        """当前 Schema 版本号。"""

    @schema_version.setter
    def schema_version(self, value: int) -> None: ...

    @property
    def active_wal_number(self) -> Optional[int]:
        """当前活跃 WAL 编号。"""

    @active_wal_number.setter
    def active_wal_number(self, value: int) -> None: ...
```

---

### 9.11 search/bitmap.py

```python
def build_valid_mask(
    all_pks: np.ndarray,
    all_seqs: np.ndarray,
    delta_log: "DeltaLog",
) -> np.ndarray:
    """构建有效行 bitmap（np.ndarray[bool]，True=有效）。

    两步过滤：
    1. 去重：同一 PK 出现多次时，只保留 _seq 最大的行，其余标记 False
    2. 删除过滤：调用 delta_log.is_deleted(pk, seq)，已删除标记 False

    将来扩展：
    3. 标量过滤（filter expression）

    Args:
        all_pks: shape=(N,) 所有行的主键值
        all_seqs: shape=(N,) 所有行的 _seq
        delta_log: DeltaLog 实例，提供 is_deleted() 查询

    Returns:
        np.ndarray[bool] shape=(N,)
    """
```

### 9.12 search/distance.py

```python
def cosine_distance(
    query: np.ndarray,          # shape=(dim,)
    candidates: np.ndarray,     # shape=(n, dim)
) -> np.ndarray:                # shape=(n,)
    """余弦距离 = 1 - cosine_similarity。值域 [0, 2]，越小越相似。"""


def l2_distance(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """L2（欧氏）距离。越小越相似。"""


def ip_distance(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """内积距离 = -dot(q, c)。取负使得越小越相似（与 cosine/L2 统一方向）。"""


def compute_distances(
    query: np.ndarray,
    candidates: np.ndarray,
    metric_type: str,           # "COSINE" | "L2" | "IP"
) -> np.ndarray:
    """根据 metric_type 分派到对应距离函数。
    Raises: ValueError（不支持的 metric_type）"""
```

### 9.13 search/executor.py

```python
def execute_search(
    query_vectors: np.ndarray,      # shape=(nq, dim)，nq 个查询向量
    all_pks: np.ndarray,            # shape=(N,)，所有候选行的 PK
    all_seqs: np.ndarray,           # shape=(N,)，所有候选行的 _seq
    all_vectors: np.ndarray,        # shape=(N, dim)，所有候选行的向量
    all_records: List[dict],        # 所有候选行的完整记录（用于返回 entity 字段）
    delta_log: "DeltaLog",
    top_k: int,
    metric_type: str,
) -> List[List[dict]]:
    """执行向量搜索。

    流程：
    1. build_valid_mask() → 有效行 bitmap
    2. 对每个 query_vector：
       a. 用 bitmap 过滤候选向量
       b. compute_distances() → 距离数组
       c. argpartition 取 top-k 最小距离
       d. 组装结果
    3. 返回结果

    Returns:
        外层 List = 每个查询向量的结果
        内层 List = top-K 结果，按 distance 升序
        每条结果: {"id": pk_value, "distance": float, "entity": {field: value}}
    """
```

---

### 9.14 engine/flush.py

```python
def execute_flush(
    frozen_memtable: "MemTable",
    frozen_wal: "WAL",
    data_dir: str,
    schema: "CollectionSchema",
    manifest: "Manifest",
    delta_log: "DeltaLog",
    compaction_mgr: "CompactionManager",
) -> None:
    """执行 Flush 管线（7 步）。

    前置条件：调用方已完成 Step 1（冻结旧 MemTable/WAL，创建新的）。

    Step 2: frozen_memtable.flush() → {partition: (data_table, delta_table)}
    Step 3: 写 Parquet 文件到各 Partition 目录
    Step 4: 更新 delta_log 内存（delta_log.update_memory）
    Step 5: 原子更新 Manifest（新增文件 + 更新 current_seq + 切换 active_wal）
    Step 6: 删除旧 WAL（frozen_wal.close_and_delete）
    Step 7: 按 Partition 触发 compaction_mgr.maybe_compact()

    崩溃安全：
    - Step 3 崩溃 → Manifest 未更新，Parquet 成为孤儿文件，recovery 清理
    - Step 5 前崩溃 → WAL 完整，重放恢复
    - Step 5 后崩溃 → Manifest 已更新，WAL 重放产生重复但 _seq 去重保证正确
    """
```

### 9.15 engine/recovery.py

```python
def execute_recovery(
    data_dir: str,
    schema: "CollectionSchema",
    manifest: "Manifest",
) -> Tuple["MemTable", "DeltaLog", int]:
    """执行崩溃恢复（5 步）。

    前置条件：调用方已加载 Manifest（Step 1）。

    Step 2: 扫描 wal/ 目录，有未清理的 WAL → 重放到新建 MemTable
    Step 3: 校验 Manifest 中的文件是否实际存在（处理 Compaction 中途崩溃）
    Step 4: 清理孤儿文件（在磁盘但不在 Manifest 中的 Parquet）
    Step 5: 从所有 Partition 的 delta_files 重建 DeltaLog.deleted_map

    Returns:
        (memtable, delta_log, next_wal_number)
        - memtable: 重放 WAL 后的 MemTable（无 WAL 则为空 MemTable）
        - delta_log: 重建后的 DeltaLog
        - next_wal_number: 下一轮 WAL 应使用的编号
    """
```

### 9.16 engine/compaction.py

```python
class CompactionManager:
    """Size-Tiered Compaction Manager，按 Partition 独立执行。"""

    def __init__(self, data_dir: str, schema: CollectionSchema) -> None:
        """
        Args:
            data_dir: Collection 数据目录
            schema: CollectionSchema（用于读取 Parquet 时确定主键和 schema）
        """

    def maybe_compact(
        self,
        partition: str,
        manifest: "Manifest",
        delta_log: "DeltaLog",
    ) -> None:
        """检查指定 Partition 是否需要 Compaction，满足条件则执行。

        触发条件（满足任一）：
        - 同一大小桶内文件数 >= COMPACTION_MIN_FILES_PER_BUCKET
        - 该 Partition 文件总数 > MAX_DATA_FILES

        Compaction 流程：
        1. 对文件按大小分桶
        2. 选择目标桶中的文件
        3. 读取并合并 Arrow Tables
        4. 按主键去重（保留 max _seq）
        5. 用 delta_log.is_deleted() 过滤已删除记录
        6. 写入新 Parquet 文件
        7. 原子更新 Manifest（移除旧文件 + 新增新文件 + 移除已消费 delta 文件）
        8. 删除旧文件和已消费的 delta 文件
        """
```

### 9.17 engine/collection.py

```python
class Collection:
    """Collection 层：引擎核心，管理 WAL / MemTable / Manifest / Compaction。

    持有全部 storage 组件实例，编排读写路径。
    __init__ 时自动执行 recovery。
    """

    def __init__(self, name: str, data_dir: str, schema: CollectionSchema) -> None:
        """初始化 Collection，加载 Manifest，执行 recovery。
        Args:
            name: Collection 名称
            data_dir: Collection 数据目录（包含 manifest.json, schema.json, wal/, partitions/）
            schema: CollectionSchema
        """

    # ─── 写操作（partition_name: 单数 str）───

    def insert(
        self,
        records: List[dict],
        partition_name: str = "_default",
    ) -> List:
        """批量写入，天然 upsert 语义（PK 存在则覆盖）。

        流程: 校验 → 分配 _seq → WAL → MemTable → (满了触发 flush)
        每条记录分配独立 _seq。

        Args:
            records: List[dict]，每个 dict 包含用户字段
            partition_name: 目标 Partition

        Returns:
            写入的 PK 列表

        Raises:
            SchemaValidationError: 记录不符合 Schema
            PartitionNotFoundError: 指定 Partition 不存在
        """

    def delete(
        self,
        pks: List,
        partition_name: Optional[str] = None,
    ) -> int:
        """批量删除，不检查记录是否存在。

        流程: 分配共享 _seq → WAL → MemTable → (满了触发 flush)
        多条 PK 共享同一个 _seq。

        Args:
            pks: PK 值列表
            partition_name: 目标 Partition，None 则跨所有 Partition 删除

        Returns:
            处理的 PK 数量

        Raises:
            PartitionNotFoundError: 指定 Partition 不存在
        """

    # ─── 读操作（partition_names: 复数 List[str]）───

    def get(
        self,
        pks: List,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """按 PK 批量查询。

        流程: MemTable 查 → 数据文件查（从新到旧）→ deleted_map 过滤
        未找到的 PK 不在返回列表中。

        Args:
            pks: PK 值列表
            output_fields: 返回字段列表，None 则返回所有字段
            partition_names: 搜索范围，None 则搜索所有 Partition

        Returns:
            List[dict]，每个 dict 为一条记录
        """

    def search(
        self,
        vectors: List[list],
        top_k: int = 10,
        metric_type: str = "COSINE",
        partition_names: Optional[List[str]] = None,
    ) -> List[List[dict]]:
        """向量检索。

        流程: Partition 剪枝 → 收集数据(MemTable + Parquet) → bitmap → 向量检索 → top-k

        Args:
            vectors: 查询向量列表，每个元素是 list[float]
            top_k: 返回最近邻数量
            metric_type: 距离度量 "COSINE" | "L2" | "IP"
            partition_names: 搜索范围，None 则搜索所有 Partition

        Returns:
            外层 List = 每个查询向量
            内层 List = top-K 结果
            每条: {"id": pk, "distance": float, "entity": {field: value}}
        """

    # ─── Partition 管理 ───

    def create_partition(self, partition_name: str) -> None:
        """创建新 Partition。创建目录 + 更新 Manifest。
        Raises: PartitionAlreadyExistsError"""

    def drop_partition(self, partition_name: str) -> None:
        """删除 Partition。删除目录 + 更新 Manifest。
        Raises: DefaultPartitionError, PartitionNotFoundError"""

    def list_partitions(self) -> List[str]:
        """返回所有 Partition 名称列表。"""

    # ─── Schema 变更 ───

    def add_field(self, field: FieldSchema) -> None:
        """新增字段到 Schema。schema_version +1，持久化更新。
        利用 Parquet 天然 schema evolution：旧文件缺失列自动填 null。
        不支持删除字段和修改字段类型。"""

    # ─── 生命周期 ───

    def flush(self) -> None:
        """手动触发 flush（不等 MemTable 满）。
        MemTable 为空时无操作。"""

    def close(self) -> None:
        """关闭 Collection：flush 残留数据 → 关闭 WAL writer。"""
```

### 9.18 db.py

```python
class LiteVecDB:
    """DB 层：管理多个 Collection 的生命周期。

    对应磁盘上一个根目录，每个 Collection 是根目录下的一个子目录。
    """

    def __init__(self, root_dir: str) -> None:
        """初始化 DB，扫描 root_dir 发现已有 Collection（延迟加载）。
        Args:
            root_dir: 数据库根目录，不存在则自动创建
        """

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
    ) -> "Collection":
        """创建新 Collection。
        创建子目录 + 写入 schema.json + 初始化 Manifest + 创建 _default Partition。
        Returns: Collection 实例
        Raises: CollectionAlreadyExistsError"""

    def get_collection(self, collection_name: str) -> "Collection":
        """获取已有 Collection（首次获取时加载并执行 recovery）。
        Raises: CollectionNotFoundError"""

    def drop_collection(self, collection_name: str) -> None:
        """删除 Collection。关闭 → 删除整个子目录。
        Raises: CollectionNotFoundError"""

    def list_collections(self) -> List[str]:
        """返回所有 Collection 名称列表。"""

    def close(self) -> None:
        """关闭所有已加载的 Collection。"""
