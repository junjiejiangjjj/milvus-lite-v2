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
│   │   ├── __init__.py             #   导出: WAL, MemTable, DataFile, DeltaFile, DeltaIndex, Manifest
│   │   ├── wal.py                  #   WAL (Arrow IPC Streaming, 双文件, write/read_operations/close, fsync)
│   │   ├── memtable.py             #   MemTable (RecordBatch list + pk_index + delete_index, seq-aware)
│   │   ├── data_file.py            #   数据 Parquet 无状态函数 (读写, 命名, seq 范围解析)
│   │   ├── delta_file.py           #   delta Parquet 无状态函数 (读写)
│   │   ├── delta_index.py          #   DeltaIndex (内存 pk→max_delete_seq, gc_below)
│   │   └── manifest.py             #   Manifest (JSON, tmp+rename 原子, .prev 备份, Partition 文件列表)
│   │
│   ├── engine/                     # ══ 引擎层 ══
│   │   ├── __init__.py             #   导出: Collection
│   │   ├── operation.py            #   InsertOp / DeleteOp / Operation Union (写编排抽象层)
│   │   ├── collection.py           #   Collection 核心 (入口, _seq 分配, insert/delete/get/search, _apply 统一路径, Partition CRUD)
│   │   ├── flush.py                #   Flush 管线 (7 步, 同步阻塞)
│   │   ├── recovery.py             #   崩溃恢复 (5 步, 经 WAL.read_operations 按 seq 回放)
│   │   └── compaction.py           #   Compaction Manager (Size-Tiered + tombstone GC)
│   │
│   ├── search/                     # ══ 搜索层 ══
│   │   ├── __init__.py             #   导出: execute_search
│   │   ├── bitmap.py               #   bitmap 管线 (去重 + 删除过滤 + 可选 filter_mask)
│   │   ├── distance.py             #   距离计算 (cosine / L2 / inner product, NumPy 实现)
│   │   ├── assembler.py            #   候选集拼装 (segments + memtable → numpy + 可选 filter_mask)
│   │   ├── executor.py             #   搜索执行器 (收集数据 + bitmap + 向量检索 + top-k)
│   │   └── filter/                 # ══ 标量过滤子系统 (Phase 8) ══
│   │       ├── __init__.py         #   导出: parse_expr, compile_expr, evaluate, FilterError
│   │       ├── exceptions.py       #   FilterError / FilterParseError / FilterFieldError / FilterTypeError
│   │       ├── tokens.py           #   TokenKind enum + Token + tokenize()
│   │       ├── ast.py              #   11 个 frozen AST 节点 + Expr Union
│   │       ├── parser.py           #   Pratt parser (借鉴 Milvus Plan.g4)
│   │       ├── semantic.py         #   compile_expr + 类型推断 + 字段绑定 + backend 选择
│   │       └── eval/
│   │           ├── __init__.py     #   evaluate() backend dispatcher
│   │           ├── arrow_backend.py #   pyarrow.compute 后端 (主)
│   │           └── python_backend.py#   row-wise Python 后端 (兜底 + 差分测试基准)
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
│   │   ├── test_wal.py             #   WAL 写入/恢复往返, 双文件生命周期, 损坏处理, fsync
│   │   ├── test_memtable.py        #   apply_insert/apply_delete/get 语义, upsert, seq-aware 乱序反例, flush Partition 拆分
│   │   ├── test_data_file.py       #   Parquet 读写往返, 文件命名, seq 范围解析
│   │   ├── test_delta_file.py      #   delta Parquet 读写往返
│   │   ├── test_delta_index.py     #   add_batch / is_deleted / gc_below / rebuild_from
│   │   └── test_manifest.py        #   加载/保存原子性, .prev 备份与 fallback, Partition 文件列表管理
│   │
│   ├── engine/
│   │   ├── test_operation.py       #   InsertOp / DeleteOp 构造, 属性 (seq_min/seq_max/num_rows)
│   │   ├── test_collection.py      #   Collection 级 E2E: insert/delete/get/search, Partition CRUD, upsert
│   │   ├── test_flush.py           #   Flush 管线端到端, 7 个崩溃点 (含 fsync), 恢复正确性
│   │   ├── test_recovery.py        #   崩溃恢复 5 步, WAL 按 seq 重放, 孤儿文件清理
│   │   └── test_compaction.py      #   文件分桶, 合并去重, 删除过滤, Manifest 更新, tombstone GC
│   │
│   ├── search/
│   │   ├── test_bitmap.py          #   bitmap 构建: 去重 + 删除过滤 + filter_mask
│   │   ├── test_distance.py        #   cosine / L2 / IP 距离正确性
│   │   ├── test_executor.py        #   搜索端到端, top-k, Partition Pruning
│   │   └── filter/                 #   ── 过滤子系统单元测试 ──
│   │       ├── test_tokens.py      #   各 literal 词法 + 关键字大小写 + 错误位置
│   │       ├── test_parser.py      #   Pratt 优先级 + 括号 + 错误恢复
│   │       ├── test_semantic.py    #   字段不存在 / 类型不匹配 / did-you-mean
│   │       ├── test_arrow_backend.py    #   每个 AST 节点 → 正确 BooleanArray
│   │       ├── test_python_backend.py   #   同上, 行级实现对照
│   │       └── test_e2e.py         #   差分测试: arrow == python
│   │
│   ├── test_db.py                  #   多 Collection 生命周期, close/cleanup
│   └── test_smoke_e2e.py           #   走公开 API 的端到端冒烟
│
├── pyproject.toml
└── requirements.txt
```

## 架构不变量（核心约束）

下列约束贯穿所有模块，是设计共识，不在每个 §9.x 接口里重复。任何模块的实现违反这里任一条都视为 bug。

**正确性 / 数据一致性：**

1. **`_seq` 是操作的全序**。所有"覆盖 / 丢弃 / 去重"判断必须比较 `_seq`，不依赖调用顺序或文件物理顺序。这条让 recovery 乱序回放、compaction 重排、未来引入并发都不会破坏正确性。
2. **MemTable cross-clear 必须 seq-aware**。put 与 delete 互相清除对方 buffer 同 pk 条目前，**必须**先比较 `_seq`，只有当前操作 `_seq` 更大时才覆盖；否则当前操作直接丢弃。详见 §9.7。
3. **Tombstone GC 规则**：`delta_index` 中 `pk → delete_seq` 条目可以丢弃，当且仅当不存在 `seq_min ≤ delete_seq` 且包含该 pk 的 data 文件。MVP 用保守版本：全局 `min_active_data_seq` 之下的所有 tombstone 均可丢。详见 §9.9 / §9.16。
4. **文件不可变**。所有磁盘文件（data Parquet、delta Parquet、WAL Arrow、Manifest）一旦写完只能整体删除，不允许就地修改。这是 LSM 路线的基础。
5. **Manifest 是单一真相源**，原子更新（write-tmp + rename），且保留 `manifest.json.prev` 兜底一次序列化事故。详见 §9.10。

**并发模型：**

6. **MVP 同步 flush**。`Collection.insert/delete` 检测到 MemTable 满后**阻塞执行** flush，flush 完成才返回。异步/后台 flush 列入 future，不进 MVP。
   - 这条决定影响 MemTable / Collection / Search 三层接口形态。要改成异步必须先开新文档讨论锁/快照/RCU 边界。
7. **单 writer per Collection**。Collection 不做内部加锁；同一进程多线程并发写同一 Collection 是 undefined behavior。
8. **单进程 per data_dir**。`db.py` 启动时用 `fcntl.flock(data_dir/LOCK)` 抢占；被占用直接报错退出，不等待。

**Schema / 演化：**

9. **Schema 不可变**。MVP 不支持 alter table；schema 变更只能通过新建 Collection + reindex。这是简化设计的关键前提：4 套 Arrow Schema、所有历史 Parquet 文件都不需要考虑兼容。

**WAL / 持久性：**

10. **WAL 默认 `sync_mode="close"`**：在 `close_and_delete` 前对 sink 做一次 `os.fsync`。覆盖容器 OOM-kill 后立即接管的崩溃场景。详见 wal-design.md §8。

---

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
| `wal.py` | WAL 持久化 | `WAL(wal_dir, wal_data_schema, wal_delta_schema, wal_number, sync_mode="close")` — `write(op)`, `read_operations(...)→Iterator[Operation]`, `close_and_delete()`（含 fsync） |
| `memtable.py` | 内存缓冲 | `MemTable(schema)` — `apply_insert(batch)`, `apply_delete(batch)`, `get(pk)`, `flush()→Dict[partition, (data_table, delta_table)]`, `size()`。**内部表示：append-only RecordBatch list + pk_index + delete_index；cross-clear 必须 seq-aware**。 |
| `data_file.py` | 数据 Parquet（无状态） | `write_data_file(table, partition_dir, seq_min, seq_max)→path`, `read_data_file(path)→pa.Table`, `parse_seq_range(filename)→(min, max)`, `get_file_size(path)→int` |
| `delta_file.py` | delta Parquet（无状态） | `write_delta_file(...)→path`, `read_delta_file(path)→pa.Table` |
| `delta_index.py` | 内存删除索引 | `DeltaIndex(pk_name)` — `add_batch(batch)`, `is_deleted(pk, seq)→bool`, `gc_below(min_active_seq)→int`, `rebuild_from(...)` |
| `manifest.py` | 全局状态 | `Manifest(data_dir)` — `load()`, `save()` (原子+`.prev`), `add/remove_data_file(partition, file)`, `add/remove_delta_file(partition, file)`, `add/remove_partition(name)` |

```python
# storage/__init__.py
from litevecdb.storage.wal import WAL
from litevecdb.storage.memtable import MemTable
from litevecdb.storage.data_file import write_data_file, read_data_file
from litevecdb.storage.delta_file import write_delta_file, read_delta_file
from litevecdb.storage.delta_index import DeltaIndex
from litevecdb.storage.manifest import Manifest
```

### 3.3 engine/ — 引擎层

**职责边界**：编排 storage 组件，实现业务流程（写入→flush→落盘、崩溃恢复、compaction）。是 storage 和 search 的上层调用者。

| 子模块 | 职责 | 核心内容 |
|--------|------|---------|
| `operation.py` | 写编排抽象 | `InsertOp`, `DeleteOp`, `Operation = Union[InsertOp, DeleteOp]` — frozen dataclass + Arrow batch，纯描述无行为 |
| `collection.py` | 引擎核心 | `Collection(name, data_dir, schema)` — `insert()`, `delete()`, `get()`, `search()`, `_apply(op)`（统一写入路径），`create/drop/list_partitions()`, `flush()`, `close()`, `_alloc_seq()` |
| `flush.py` | Flush 管线 | `execute_flush(frozen_memtable, frozen_wal, manifest, delta_index, compaction_mgr)` — 7 步流程，**同步阻塞** |
| `recovery.py` | 崩溃恢复 | `execute_recovery(data_dir, manifest)` — 5 步流程，经由 `WAL.read_operations()` 按 seq 回放 |
| `compaction.py` | Compaction | `CompactionManager(data_dir, schema)` — `maybe_compact(partition, manifest, delta_index)`，含 tombstone GC |

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

**职责边界**：向量检索，输入是数据 + 查询，输出是 top-k 结果。不关心数据从哪来。**含可选的标量过滤子系统** `search/filter/`（Phase 8 新增），见 §9.19-9.25。

| 子模块 | 职责 | 核心函数 |
|--------|------|---------|
| `bitmap.py` | 有效性过滤 | `build_valid_mask(all_pks, all_seqs, delta_index, filter_mask=None)→np.ndarray[bool]` — 去重 + 删除过滤 + 可选标量过滤 mask |
| `distance.py` | 距离计算 | `cosine_distance(q, candidates)→np.ndarray`, `l2_distance(...)`, `ip_distance(...)`, `compute_distances(q, candidates, metric_type)` — 纯数学，无状态 |
| `assembler.py` | 候选拼装 | `assemble_candidates(segments, memtable, vector_field, partition_names=None, filter_compiled=None)` — 把各源数据拼成统一 numpy + 可选 filter mask |
| `executor.py` | 搜索编排 | `execute_search(query_vectors, all_pks, all_seqs, all_vectors, all_records, delta_index, top_k, metric_type, ...)→List[List[dict]]` — bitmap + distance + top-k 选择 |
| `filter/` | 标量过滤子系统 | `parse_expr(s) → compile_expr(expr, schema) → evaluate(compiled, table) → BooleanArray`，详见 §9.19-9.25 |

```python
# search/__init__.py
from litevecdb.search.executor import execute_search
from litevecdb.search.assembler import assemble_candidates
from litevecdb.search.filter import (
    parse_expr, compile_expr, evaluate,
    FilterError, FilterParseError, FilterFieldError, FilterTypeError,
)
```

**为什么 search 独立为 package**：
- bitmap 管线扩展标量过滤后逻辑增长（filter_mask）
- distance 是纯数学模块，将来替换 FAISS 时只需替换 executor.py
- filter 子包是相对独立的"小型 DSL"，单独成 package 便于测试和未来切换 parser 实现
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

**内部表示**：MemTable 不持有 `dict[pk → record_dict]`。Python dict + record dict 的内存开销是 Arrow 表示的 10-100 倍，且 flush 时还要做一次 Python → Arrow 转换。MemTable 内部维护 **append-only 的 RecordBatch list + 两个轻量索引**：

```
_insert_batches: list[pa.RecordBatch]              # append-only，每次 apply_insert 追加一个 batch
_pk_index:       dict[pk → (batch_idx, row_idx)]   # 指向 _insert_batches 中的最新位置
_delete_index:   dict[pk → delete_seq]             # 删除水位
```

`_pk_index` 是 lazy 的——同一 pk 的旧版本在 `_insert_batches` 中物理保留，直到 flush 时一次性 dedup 取活跃行。这样：

- **写入零拷贝**：apply_insert 直接 `append(batch)` + 更新 index
- **flush 几乎零成本**：`pa.Table.from_batches` + 按 `_pk_index.values()` take 活跃行
- **search 拿到的是 Arrow 列**：不再二次转换

**Cross-clear 必须 seq-aware**（架构不变量 §2）。put / delete 之前先比较 `_seq`，只有当前操作 `_seq` 更大才生效；否则当前操作直接丢弃。**这条不变量决定了 recovery / 未来并发写入都不会触发数据损坏。**

```python
class MemTable:
    """Collection 级共享的内存缓冲区。

    内部表示：
    - _insert_batches: list[pa.RecordBatch]，append-only
    - _pk_index: dict[pk → (batch_idx, row_idx)]，最新位置索引
    - _delete_index: dict[pk → delete_seq]，删除水位

    cross-clear 语义：
    - apply_insert(seq=S, pk=P) 时，若 _delete_index[P] >= S 则操作丢弃（更新的 delete 已存在）；
      否则插入并清除 _delete_index[P]。
    - apply_delete(seq=S, pk=P) 时，若 _insert_batches 中 P 的最新 _seq >= S 则操作丢弃；
      否则更新 _delete_index[P]=S 并清除 _pk_index[P]。
    """

    def __init__(self, schema: CollectionSchema) -> None:
        """
        Args:
            schema: CollectionSchema，用于确定 pk_name 和字段信息
        """

    def apply_insert(self, batch: pa.RecordBatch) -> None:
        """追加一个 wal_data schema 的 RecordBatch。
        - batch 的每行必须含 _seq, _partition, pk_field 和所有用户字段
        - 内部按 seq-aware 规则更新 _pk_index 和清除 _delete_index 中过时条目
        - batch 物理追加到 _insert_batches，旧版本不删除（flush 时 dedup）
        """

    def apply_delete(self, batch: pa.RecordBatch) -> None:
        """处理一个 wal_delta schema 的 RecordBatch。
        - batch 共享同一 _seq；内部按 seq-aware 规则更新 _delete_index
        - 不持有 batch 引用（只取 pk + seq 写进 _delete_index）
        - 处理 _partition='_all' 的跨 partition 删除
        """

    def get(self, pk_value: Any) -> Optional[dict]:
        """点查单条记录。
        通过 _pk_index 定位最新位置，再检查 _delete_index 是否覆盖。
        返回 record dict（不含 _partition, _seq）或 None。"""

    def flush(self) -> Dict[str, Tuple[Optional[pa.Table], Optional[pa.Table]]]:
        """按 Partition 拆分输出 Arrow Table。

        实现：
        1. 把 _insert_batches concat 成一个大 Table
        2. 按 _pk_index.values() 取活跃行（dedup 在这里发生）
        3. 按 _partition 列拆分成各 partition 的 data_table
        4. 把 _delete_index 物化成各 partition 的 delta_table

        Returns: {partition_name: (data_table, delta_table)}
            - data_table: 使用 data_schema（不含 _partition），可为 None
            - delta_table: 使用 delta_schema（不含 _partition），可为 None
        调用后内部状态不清空（由调用方冻结后丢弃整个 MemTable）。"""

    def size(self) -> int:
        """返回 len(_pk_index) + len(_delete_index)。
        注意是"活跃 pk 数"，不是 _insert_batches 总行数——
        flush 触发阈值用这个，避免同 pk 反复 upsert 把内存撑爆。"""

    def get_active_records(
        self, partition_names: Optional[List[str]] = None
    ) -> List[dict]:
        """返回 _pk_index 中未被 _delete_index 覆盖的活跃记录。
        用于 search/get 时读取 MemTable 层数据。
        partition_names 不为 None 时，只返回匹配 Partition 的记录。
        返回的 dict 不含 _partition 和 _seq（对外干净）。"""
```

**关键测试点**（实现时必须覆盖）：

```python
# 验证 seq-aware cross-clear: 乱序 apply 仍得到正确终态
mt = MemTable(schema)
mt.apply_insert(batch_with_pk_X_seq_7)   # 先来个 seq=7
mt.apply_insert(batch_with_pk_X_seq_5)   # 再来个 seq=5（应丢弃）
mt.apply_delete(batch_with_pk_X_seq_6)   # 再来个 delete seq=6（应丢弃，因 seq=7 更新）
assert mt.get("X")["_seq"] == 7          # ← seq=7 必须保留
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

### 9.9 storage/delta_file.py + storage/delta_index.py

**DeltaLog 拆成两个模块**，与 `data_file.py` 对称：

- `delta_file.py` —— 纯 IO 函数，无状态
- `delta_index.py` —— 内存 `DeltaIndex` 类，独立可测

**为什么拆**：原 `DeltaLog` 同时管 Parquet IO + 内存索引 + 查询，6 个方法 3 个职责，难单测；拆开后 IO 函数和 `data_file.py` 形成对称结构，内存索引可以脱离磁盘单测，未来想换实现（numpy / pyarrow dict array）也只动一处。

#### 9.9.a delta_file.py（无状态 IO）

```python
def write_delta_file(
    delta_table: pa.Table,
    partition_dir: str,
    seq_min: int,
    seq_max: int,
) -> str:
    """将 delta Arrow Table 写入 delta Parquet 文件。
    文件路径: {partition_dir}/delta/delta_{seq_min:06d}_{seq_max:06d}.parquet
    自动创建 delta/ 子目录。
    Returns: 相对路径（相对于 Collection data_dir）。"""


def read_delta_file(path: str) -> pa.Table:
    """读取 delta Parquet 文件，返回 Arrow Table（pk_field + _seq 两列）。"""
```

#### 9.9.b delta_index.py（内存索引）

```python
class DeltaIndex:
    """内存中的 delete 水位索引：pk → max_delete_seq。

    启动时通过 rebuild_from() 从所有 delta 文件重建；
    运行时通过 add_batch() 增量更新；
    通过 gc_below() 在 Compaction 后回收老 tombstone（架构不变量 §3）。
    """

    def __init__(self, pk_name: str) -> None:
        """
        Args:
            pk_name: 主键字段名（用于从 batch 中提取 pk 列）
        """

    def add_batch(self, delta_batch: pa.RecordBatch) -> None:
        """把 delta batch 中的 (pk, _seq) 合并进内存索引。
        - 对每个 pk 取 max(已有 seq, 新 seq)
        - 不写磁盘
        - 用于 flush 后更新 + WAL 重放
        """

    def is_deleted(self, pk_value: Any, data_seq: int) -> bool:
        """判断某条数据记录是否被删除。
        规则: _map.get(pk, -1) > data_seq → 已删除。"""

    def gc_below(self, min_active_data_seq: int) -> int:
        """回收 delete_seq < min_active_data_seq 的所有 tombstone。

        Args:
            min_active_data_seq: 当前 manifest 中所有 data 文件 seq_min 的最小值
        Returns:
            被回收的条目数

        正确性：任何 delete_seq < min_active_data_seq 的 tombstone，对应的所有
        data 行都已经被 compaction 物理消化掉，不存在残留 data 行需要它过滤，
        因此可以安全丢弃。详见架构不变量 §3。
        """

    @classmethod
    def rebuild_from(
        cls,
        pk_name: str,
        partition_delta_files: Dict[str, List[str]],
    ) -> "DeltaIndex":
        """启动时一次性重建。
        Args:
            partition_delta_files: {partition_name: [绝对路径列表]}
        Returns: 完整 DeltaIndex 实例
        """

    def __len__(self) -> int:
        """当前活跃 tombstone 条目数（监控/测试用）。"""

    @property
    def snapshot(self) -> Dict[Any, int]:
        """只读快照: pk_value → max_delete_seq（拷贝，不持有内部引用）。"""
```

**关键点**：
- `add_batch` 接受 `pa.RecordBatch` 而不是 `pa.Table`，与 WAL / MemTable 的颗粒度统一
- `gc_below` 是 Compaction 调用的入口，封装架构不变量 §3 的 GC 规则
- 没有 `remove_files` 方法——文件管理是 Manifest 的事，DeltaIndex 不持有文件路径

### 9.10 storage/manifest.py

```python
class Manifest:
    """全局状态快照文件，通过原子替换（write-tmp + rename）更新。

    记录当前 _seq、Schema 版本、各 Partition 的文件列表、活跃 WAL。
    是系统唯一的 truth source（架构不变量 §5）。

    持久化布局：
        data_dir/
          ├── manifest.json          # 当前版本
          └── manifest.json.prev     # 上一版本备份（兜底一次序列化事故）
    """

    def __init__(self, data_dir: str) -> None: ...

    # ── 持久化 ──

    def save(self) -> None:
        """原子更新 manifest.json，version 自动 +1。

        步骤：
        1. 序列化到 manifest.json.tmp
        2. 若 manifest.json 存在，cp 它 → manifest.json.prev（覆盖旧 .prev）
        3. os.rename(manifest.json.tmp, manifest.json)  ← 原子切换
        4. fsync data_dir 目录确保 rename 持久化

        失败语义：步骤 1 失败 → tmp 文件孤立，无影响；
                  步骤 2 失败 → 抛异常，磁盘上仍是上一次成功 save 的 manifest；
                  步骤 3 是原子的，要么成功要么失败。
        """

    @classmethod
    def load(cls, data_dir: str) -> "Manifest":
        """加载 manifest.json。

        加载策略：
        1. 尝试 manifest.json
           - 文件不存在 → 返回初始状态（version=0, _default Partition），不报错
           - 加载成功 → 返回
           - 加载失败（JSON 损坏、字段缺失）→ 警告日志 + 走第 2 步
        2. 尝试 manifest.json.prev
           - 加载成功 → 警告日志（"using prev manifest, last save likely corrupted"）+ 返回
           - 加载失败 → 抛 ManifestCorruptedError
        """

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
    all_pks: List[Any],
    all_seqs: np.ndarray,
    delta_index: "DeltaIndex",
    filter_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """构建有效行 bitmap（np.ndarray[bool]，True=有效）。

    三步过滤（按顺序）：
    1. 去重：同一 PK 出现多次时，只保留 _seq 最大的行，其余标记 False
    2. 删除过滤：调用 delta_index.is_deleted(pk, seq)，已删除标记 False
    3. 标量过滤：若 filter_mask 不为 None，按位 AND 进最终 mask

    Args:
        all_pks: 长度 N 的 pk 列表
        all_seqs: shape=(N,) 所有行的 _seq
        delta_index: DeltaIndex 实例，提供 is_deleted() 查询
        filter_mask: 可选，长度 N 的 bool array，由 search/filter 子系统对
            assemble_candidates 中各源的 pa.Table 求值后拼接得到

    Returns:
        np.ndarray[bool] shape=(N,)

    Raises:
        ValueError: filter_mask 长度不等于 all_pks 长度
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
    all_pks: List[Any],             # 长度 N，所有候选行的 PK
    all_seqs: np.ndarray,           # shape=(N,)，所有候选行的 _seq
    all_vectors: np.ndarray,        # shape=(N, dim)，所有候选行的向量
    all_records: List[dict],        # 所有候选行的完整记录（用于返回 entity 字段）
    delta_index: "DeltaIndex",
    top_k: int,
    metric_type: str,
    pk_field: str,
    vector_field: str,
    filter_mask: Optional[np.ndarray] = None,    # Phase 8: 标量过滤
) -> List[List[dict]]:
    """执行向量搜索。

    流程：
    1. build_valid_mask(filter_mask=filter_mask) → 有效行 bitmap（含去重 +
       删除过滤 + 标量过滤）
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

### 9.13.5 search/assembler.py

```python
def assemble_candidates(
    segments: Iterable["Segment"],
    memtable: "MemTable",
    vector_field: str,
    partition_names: Optional[List[str]] = None,
    filter_compiled: Optional["CompiledExpr"] = None,
) -> Tuple[
    List[Any],         # all_pks
    np.ndarray,        # all_seqs (uint64)
    np.ndarray,        # all_vectors (float32, shape=(N, dim))
    List[dict],        # all_records (entity dicts)
    Optional[np.ndarray],  # filter_mask (bool, length N) or None
]:
    """把 segments 和 MemTable 拼成统一的候选数组。

    顺序：先 segments（按迭代顺序），然后 MemTable。这个顺序决定 filter_mask
    的拼接顺序，bitmap 阶段按相同顺序使用。

    若 filter_compiled 非 None：
        - 对每个进入候选集的 segment，调 evaluator 求出 BooleanArray
        - 对 MemTable 的活跃数据，构造临时 pa.Table 后调 evaluator
        - 各源的 mask 按 candidate 顺序 concat 成单一 numpy array
    """
```

`assembler` 是 search 子系统中**唯一同时知道 storage 类型 (Segment, MemTable) 和 filter 子系统**的模块——其他 search 文件都是 storage-agnostic 的。

---

### 9.14 engine/flush.py

```python
def execute_flush(
    frozen_memtable: "MemTable",
    frozen_wal: "WAL",
    data_dir: str,
    schema: "CollectionSchema",
    manifest: "Manifest",
    delta_index: "DeltaIndex",
    compaction_mgr: "CompactionManager",
) -> None:
    """执行 Flush 管线（7 步，**同步阻塞**——架构不变量 §6）。

    前置条件：调用方已完成 Step 1（冻结旧 MemTable/WAL，创建新的）。

    Step 2: frozen_memtable.flush() → {partition: (data_table, delta_table)}
    Step 3: 写 Parquet 文件到各 Partition 目录（含 delta Parquet）
    Step 4: 更新 delta_index 内存（add_batch 每个 delta_table 的 RecordBatch）
    Step 5: 原子更新 Manifest（新增文件 + 更新 current_seq + 切换 active_wal + .prev 备份）
    Step 6: 删除旧 WAL（frozen_wal.close_and_delete，含 fsync）
    Step 7: 按 Partition 触发 compaction_mgr.maybe_compact()（其中含 tombstone GC）

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
) -> Tuple["MemTable", "DeltaIndex", int]:
    """执行崩溃恢复（5 步）。

    前置条件：调用方已加载 Manifest（Step 1）。

    Step 2: 扫描 wal/ 目录，有未清理的 WAL → 按 _seq 顺序重放 Operations 到新建 MemTable
            （重放经由 WAL.read_operations() Iterator[Operation]，详见 §9.x operation）
    Step 3: 校验 Manifest 中的文件是否实际存在（处理 Compaction 中途崩溃）
    Step 4: 清理孤儿文件（在磁盘但不在 Manifest 中的 Parquet）
    Step 5: DeltaIndex.rebuild_from(所有 Partition 的 delta_files)

    Returns:
        (memtable, delta_index, next_wal_number)
        - memtable: 重放 WAL 后的 MemTable（无 WAL 则为空 MemTable）
        - delta_index: 重建后的 DeltaIndex
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
        delta_index: "DeltaIndex",
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
        5. 用 delta_index.is_deleted() 过滤已删除记录
        6. 写入新 Parquet 文件
        7. 原子更新 Manifest（移除旧文件 + 新增新文件 + 移除已消费 delta 文件）
        8. 删除旧文件和已消费的 delta 文件
        9. **Tombstone GC**: 调用 delta_index.gc_below(min_active_data_seq)
           其中 min_active_data_seq 来自 Manifest 中所有 partition 的 data 文件 seq_min 的全局最小值
        """

    def _global_min_active_data_seq(self, manifest: "Manifest") -> int:
        """计算所有 partition 中 data 文件 seq_min 的最小值。

        用于 tombstone GC 触发——任何 delete_seq 小于此值的 tombstone 都
        不可能再过滤到任何 data 行（因为所有 seq < 此值的 data 都已被
        compaction 物理消化掉）。

        Returns:
            全局最小 seq_min；若无 data 文件，返回 sys.maxsize（GC 会清空 delta_index）
        """
```

**Tombstone GC 不变量**（架构不变量 §3 的实现说明）：

```
对任意 delete tombstone (pk, delete_seq):
  存在残留 data 行需要它过滤  ⟺  ∃ 某个 data 文件含 pk 且 seq_min ≤ delete_seq

保守 GC 规则（MVP 用）:
  if delete_seq < min(所有 data 文件的 seq_min):
      drop tombstone(pk, delete_seq)

正确性证明：
  delete_seq < min_seq_min ⟹ 不存在 seq_min ≤ delete_seq 的 data 文件
                         ⟹ 不存在残留 data 行需要它过滤
                         ⟹ 安全可丢
```

### 9.16.5 engine/operation.py（写编排抽象层）

**目的**：为 insert / delete 流水线提供统一的编排入口。**统一 orchestration，保留 representation**——schema、buffer、parquet 文件类型该分仍然分，只在编排层（Collection / WAL / MemTable / recovery）抽象一层。

**为什么要有这一层**：

1. Collection 的 `_apply` 是单一写入路径——任何写操作的唯一入口，未来加新操作（schema migration、bulk import）只在一处加 dispatch
2. WAL.write / MemTable.apply 都接受 Operation，不需要为每种操作各开一个方法
3. recovery 路径变成 5 行 `for op in WAL.read_operations(): memtable.apply(op)`，告别 row-by-row 的嵌套循环
4. Operation 是冻结 dataclass + Arrow batch，不持有 Collection / WAL 引用——纯描述，不带行为

```python
# engine/operation.py

from dataclasses import dataclass
from typing import Union
import pyarrow as pa


@dataclass(frozen=True)
class InsertOp:
    """一次 insert 调用的事务描述。

    batch 的 schema = wal_data_schema（含 _seq, _partition, 用户字段, $meta），
    每行都已经分配独立的 _seq。
    """
    partition: str          # 单个 partition 名
    batch: pa.RecordBatch   # 含 _seq, _partition, 用户字段, $meta

    @property
    def seq_min(self) -> int:
        """batch 中最小 _seq。"""

    @property
    def seq_max(self) -> int:
        """batch 中最大 _seq。"""

    @property
    def num_rows(self) -> int:
        """batch 行数。"""


@dataclass(frozen=True)
class DeleteOp:
    """一次 delete 调用的事务描述。

    batch 的 schema = wal_delta_schema（含 pk, _seq, _partition），
    整个 batch 共享同一个 _seq。partition 可以是 '_all'（跨所有 partition 删除）。
    """
    partition: str          # 可以是 ALL_PARTITIONS = "_all"
    batch: pa.RecordBatch   # 含 pk, _seq, _partition

    @property
    def seq(self) -> int:
        """batch 共享的 _seq。"""

    @property
    def num_rows(self) -> int:
        """batch 行数（被删除的 pk 数）。"""


Operation = Union[InsertOp, DeleteOp]
```

**Collection 入口的样子**（§9.17 会重写 insert/delete，这里先示意）：

```python
def insert(self, records, partition_name="_default"):
    self._validate_records(records)
    seq_start = self._alloc_seq(len(records))
    batch = self._build_wal_data_batch(records, partition_name, seq_start)
    op = InsertOp(partition=partition_name, batch=batch)
    self._apply(op)
    return [r[self.pk_field] for r in records]

def delete(self, pks, partition_name=None):
    seq = self._alloc_seq(1)
    partition = partition_name or ALL_PARTITIONS
    batch = self._build_wal_delta_batch(pks, partition, seq)
    op = DeleteOp(partition=partition, batch=batch)
    self._apply(op)
    return len(pks)

def _apply(self, op: Operation) -> None:
    """单一写入路径——dispatch 到 raw batch 接口。

    Operation 抽象只活在 engine 层。WAL / MemTable 不知道它，
    所以 dispatch 必须在这里显式做。
    """
    if isinstance(op, InsertOp):
        self.wal.write_insert(op.batch)
        self.memtable.apply_insert(op.batch)
    else:  # DeleteOp
        self.wal.write_delete(op.batch)
        self.memtable.apply_delete(op.batch)
    if self.memtable.size() >= MEMTABLE_SIZE_LIMIT:
        self._trigger_flush()
```

**WAL / MemTable 不知道 Operation**——它们仍然只接受 raw `pa.RecordBatch`，dispatch 在 Collection.\_apply 里完成。

**为什么这样设计**：依赖层级。`storage/` 在 Level 2，`engine/` 在 Level 5/6，让 storage 反向 import engine 的 `Operation` 类型会破坏层级。Operation 是事务编排概念，本来就属于 engine 层。

```python
class WAL:
    # 仍然是 raw batch 接口（Phase 1 已落地）
    def write_insert(self, record_batch: pa.RecordBatch) -> None: ...
    def write_delete(self, record_batch: pa.RecordBatch) -> None: ...

    # Recovery 路径在 engine/recovery.py 里包装：
    # raw WAL.recover() → 拼成 Iterator[Operation]
    @staticmethod
    def recover(wal_dir, wal_number) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        ...


class MemTable:
    # 仍然是 raw batch 接口
    def apply_insert(self, batch: pa.RecordBatch) -> None: ...
    def apply_delete(self, batch: pa.RecordBatch) -> None: ...
```

**Operation 的统一回放在 engine/recovery.py 里实现**（Phase 3 落地）：

```python
# engine/recovery.py
def replay_wal_operations(
    wal_dir: str, wal_number: int, pk_field: str,
) -> Iterator[Operation]:
    """读 WAL 文件，按 _seq 顺序 yield Operation。

    实现：
    1. WAL.recover(wal_dir, wal_number) → (data_batches, delta_batches)
    2. 把每个 batch 包成 InsertOp(partition=..., batch=b) / DeleteOp(...)
    3. 按起始 _seq 归并排序后 yield

    按 _seq 顺序回放是为了让 recovery 后 MemTable 的 max observed seq
    天然等于最后一次 yield 的 op.seq——并非正确性必需（MemTable 已 seq-aware），
    而是让 next_seq 推导更干净。
    """
```

**Recovery 受益**：

```python
def execute_recovery(...):
    memtable = MemTable(schema)
    max_seq = manifest.current_seq
    for n in WAL.find_wal_files(wal_dir):
        for op in WAL.read_operations(wal_dir, n, pk_field):
            memtable.apply(op)
            if isinstance(op, InsertOp):
                max_seq = max(max_seq, op.seq_max)
            else:
                max_seq = max(max_seq, op.seq)
    return memtable, delta_index, max_seq + 1
```

**保持不动的部分**（不要被"统一"诱惑）：

- `insert_buf` 与 `delete_buf` 内部表示**不合并**（语义不同：覆盖 vs max 累积）
- `wal_data_schema` 与 `wal_delta_schema` **不合并**（schema 是数据契约，不是编排概念）
- `InsertOp` 与 `DeleteOp` **不继承共同 base class**——用 `Union` + `isinstance` dispatch
- **不**给 Operation 加 `execute(collection)` 方法——会变成 god object
- **不**让 storage 层 import Operation——dispatch 留在 Collection.\_apply

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
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,                  # Phase 8
    ) -> List[dict]:
        """按 PK 批量查询。

        流程：MemTable 查 → segment 查（取 max-seq 版本）→ delta_index 过滤 →
              （若 expr 给出）调 filter.evaluate 在命中行上再过滤一遍。

        Args:
            pks: PK 值列表
            partition_names: 搜索范围，None 则搜索所有 Partition
            expr: 可选 Milvus-style 过滤表达式（详见 §9.19-9.25）。命中
                的 pk 必须额外满足该表达式才会出现在结果里。

        Returns:
            List[dict]，每个 dict 为一条记录（不在返回列表中 = pk 不存在
            或被过滤掉）
        """

    def search(
        self,
        query_vectors: List[list],
        top_k: int = 10,
        metric_type: str = "COSINE",
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,                  # Phase 8
    ) -> List[List[dict]]:
        """向量检索。

        流程：(若 expr 给出) parse_expr → compile_expr →
              assemble_candidates(filter_compiled=...) →
              execute_search(filter_mask=...)

        Args:
            query_vectors: 查询向量列表，每个元素是 list[float]
            top_k: 返回最近邻数量
            metric_type: "COSINE" | "L2" | "IP"
            partition_names: 搜索范围，None 则搜索所有 Partition
            expr: 可选 Milvus-style 过滤表达式（详见 §9.19-9.25）

        Returns:
            外层 List = 每个查询向量
            内层 List = top-K 结果
            每条: {"id": pk, "distance": float, "entity": {field: value}}
        """

    def query(                                       # Phase 8 新方法
        self,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """纯标量查询（无向量、无 distance）。

        流程：parse_expr → compile_expr → assemble_candidates(无 query) →
              build_valid_mask(filter_mask=...) → 取所有 True 行 →
              project output_fields → 截断 limit。

        Args:
            expr: 必填，过滤表达式（详见 §9.19-9.25）
            output_fields: 返回字段列表，None 返回所有字段（不含 _seq, _partition）
            partition_names: 搜索范围
            limit: 最多返回条数（None = 不限）

        Returns:
            List[dict]，每个 dict 是一条匹配的记录
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
```

---

## Phase 8: search/filter 子系统接口详解

**目标**：让 `Collection.search` / `get` / `query` 接受 Milvus-style 标量过滤表达式。

**架构**：三阶段编译 + 双 backend dispatcher。

```
source string  ──parse_expr()──▶  Expr (raw AST, schema 无关)
                                      │
                                      │ compile_expr(expr, schema)
                                      ▼
                              CompiledExpr (字段绑定 + 类型检查 + backend 标记)
                                      │
                                      │ evaluate(compiled, table)
                                      ▼
                              pa.BooleanArray (length == table.num_rows)
```

**架构不变量补充（写进顶部"架构不变量"区段）**：

11. **Filter parser 与 evaluator 通过 AST 解耦**——AST 是稳定接口，未来切换 parser 实现（例如 ANTLR）不影响 type checker / backends。
12. **Filter backend 在 compile 时静态决定**——不在 evaluate 热路径上 dispatch；F1 始终选 arrow，未来 F2b 引入 `$meta` 时遇到含动态字段的 ref 才升级到 python。

### 9.19 search/filter/exceptions.py

```python
from litevecdb.exceptions import LiteVecDBError

class FilterError(LiteVecDBError):
    """Base class for filter expression errors."""

class FilterParseError(FilterError):
    """Lexing or parsing failed.

    Carries (source, pos) for caret-style rendering:

        FilterParseError: unexpected token '>' at column 5
          age >> 18
              ^
    """
    def __init__(self, message: str, source: str, pos: int) -> None: ...

class FilterFieldError(FilterError):
    """Reference to a field that does not exist in the schema.

    Includes did-you-mean suggestion via difflib:

        FilterFieldError: unknown field 'agg' at column 1
          agg > 18
          ^^^
        did you mean 'age'?
    """

class FilterTypeError(FilterError):
    """Type mismatch in expression operands.

        FilterTypeError: type mismatch at column 7
          age > 'eighteen'
                ^^^^^^^^^
        left side is int (field 'age'), right side is string
    """
```

### 9.20 search/filter/tokens.py

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, List

class TokenKind(Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOL = "BOOL"
    IDENT = "IDENT"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "AND"     # and / AND / &&
    OR = "OR"       # or / OR / ||
    NOT = "NOT"     # not / NOT / !
    IN = "IN"       # in / IN
    SUB = "-"       # 一元负号（不识别为 INT 字面量的一部分）
    EOF = "EOF"

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str       # original source slice
    pos: int        # column in source
    value: Any      # parsed literal value (None for non-literals)

def tokenize(source: str) -> List[Token]:
    """Single-pass lexer.

    Behaviour (matches Milvus Plan.g4 where applicable):
        - Whitespace ' \\t \\r \\n' skipped (no comments)
        - Identifiers case-sensitive: [a-zA-Z_][a-zA-Z_0-9]*
        - Keywords case-insensitive: 'and'/'AND'/'&&', 'or'/'OR'/'||',
          'not'/'NOT'/'!', 'in'/'IN'
        - Booleans: 'true'/'True'/'TRUE', 'false'/'False'/'FALSE'
          (only these 6 forms — 'tRuE' rejected with did-you-mean)
        - Strings: "..." or '...' with escapes \\" \\' \\\\ \\n \\t \\r
        - Numbers: decimal int + decimal float + scientific notation
          (1, 3.14, 1e3, 1.5e-2). Negative sign is unary, not part of literal.
        - '==' is the equality operator; '=' alone raises FilterParseError
          with hint "did you mean '=='?".

    Raises:
        FilterParseError: on lex errors. Always carries source + pos.
    """
```

### 9.21 search/filter/ast.py

```python
from dataclasses import dataclass
from typing import Tuple, Union

# ── Literals ────────────────────────────────────────────────

@dataclass(frozen=True)
class IntLit:
    value: int
    pos: int

@dataclass(frozen=True)
class FloatLit:
    value: float
    pos: int

@dataclass(frozen=True)
class StringLit:
    value: str
    pos: int

@dataclass(frozen=True)
class BoolLit:
    value: bool
    pos: int

@dataclass(frozen=True)
class ListLit:
    """Homogeneous literal list, used inside `in [...]`."""
    elements: Tuple["Literal", ...]
    pos: int

# ── Reference ───────────────────────────────────────────────

@dataclass(frozen=True)
class FieldRef:
    name: str
    pos: int

# ── Operations ──────────────────────────────────────────────

@dataclass(frozen=True)
class CmpOp:
    op: str          # "==", "!=", "<", "<=", ">", ">="
    left: "Expr"
    right: "Expr"
    pos: int

@dataclass(frozen=True)
class InOp:
    field: FieldRef
    values: ListLit
    negate: bool     # True for "not in"
    pos: int

@dataclass(frozen=True)
class And:
    operands: Tuple["Expr", ...]
    pos: int

@dataclass(frozen=True)
class Or:
    operands: Tuple["Expr", ...]
    pos: int

@dataclass(frozen=True)
class Not:
    operand: "Expr"
    pos: int

# ── Type aliases ────────────────────────────────────────────

Literal = Union[IntLit, FloatLit, StringLit, BoolLit]

Expr = Union[
    Literal, ListLit, FieldRef,
    CmpOp, InOp, And, Or, Not,
]
```

**关键设计**：
- 11 个 frozen dataclass，全部值语义、可哈希、自动 `__eq__`
- 用 `tuple` 不用 `list`（frozen 友好）
- 没有共同 base class — 用 `Union` + `isinstance` dispatch（与 Operation 一致）
- 没有方法 — 行为在 backend 里
- 每节点带 `pos` 用于错误信息溯源
- 节点命名比 Milvus 简化：单一 `CmpOp`（带 op 字段）替代 Milvus 的 `Equality`/`Relational`

### 9.22 search/filter/parser.py

```python
class Parser:
    def __init__(self, tokens: List[Token], source: str) -> None: ...

    def parse(self) -> Expr:
        """Parse one expression and verify EOF."""

    # Pratt-style descent (low → high precedence)
    def parse_or(self) -> Expr: ...      # prec 1: a or b or c
    def parse_and(self) -> Expr: ...     # prec 2: a and b and c
    def parse_not(self) -> Expr: ...     # prec 3: not a
    def parse_cmp(self) -> Expr: ...     # prec 4: a == b, a in [...]
    def parse_primary(self) -> Expr: ... # literal | ident | ( expr )

def parse_expr(source: str) -> Expr:
    """Public entry. Lex + parse."""
```

**优先级表**（与 Milvus Plan.g4 对齐）：

| Prec | Operator | Associativity |
|---|---|---|
| 1 | `or`, `OR`, `\|\|` | left |
| 2 | `and`, `AND`, `&&` | left |
| 3 | `not`, `NOT`, `!` (前缀) | right |
| 4 | `==, !=, <, <=, >, >=` | left（链式比较 parse 接受、semantic 拒绝）|
| 4 | `in [...]`, `not in [...]` | non-assoc |
| 5 | unary `-`（前缀） | right |
| 6 | literal / ident / `(...)` | — |

**与 Milvus 收紧的部分**：
- `in` 的 RHS 必须是字面量数组（Milvus 接受任意 expr，但实际只用字面量数组）
- 数组字面量元素必须是字面量（Milvus 接受 expr，F1 只接 literal）
- F1 不支持算术 / `like` / `is null` / `exists` / `$meta` / 函数调用 — 这些 token 会被 lex 或 parse 阶段拒绝并给出 "Phase F2/F3 will support" 提示

### 9.23 search/filter/semantic.py

```python
@dataclass(frozen=True)
class FieldInfo:
    name: str
    dtype: DataType
    nullable: bool

@dataclass(frozen=True)
class CompiledExpr:
    """Type-checked, schema-bound expression ready for evaluation."""
    ast: Expr
    fields: Dict[str, FieldInfo]   # all field names referenced
    backend: str                    # "arrow" | "python"

def compile_expr(expr: Expr, schema: CollectionSchema) -> CompiledExpr:
    """Bind field references, check types, choose backend.

    Steps:
        1. Walk AST, collect all FieldRef
        2. For each: lookup in schema; reject reserved (_seq, _partition,
           $meta) or vector fields; produce did-you-mean on miss
        3. Walk again, infer + check types (cmp operands compat,
           list elements homogeneous, bool combinators bool operands)
        4. Choose backend:
           - F1: always "arrow"
           - F2b+: "python" if expression contains $meta access
           - F3+: "python" if expression contains UDF call
        5. Wrap in CompiledExpr

    Raises:
        FilterFieldError: unknown field reference
        FilterTypeError:  operand type mismatch
    """
```

**类型推断规则**：

| 节点 | 推断类型 | 校验 |
|---|---|---|
| `IntLit` | `int` | — |
| `FloatLit` | `float` | — |
| `StringLit` | `string` | — |
| `BoolLit` | `bool` | — |
| `ListLit` | `list[T]` | 元素 mutually compatible |
| `FieldRef` | schema 声明类型 | 必须存在、非保留、非 vector |
| `CmpOp` | `bool` | 左右类型 compatible |
| `InOp` | `bool` | field type ≈ list 元素类型 |
| `And/Or/Not` | `bool` | operand 是 bool |

**Compatible types**：
- `int ≈ int` ✓
- `int ≈ float` ✓（晋升）
- `string ≈ string` ✓
- `bool ≈ bool` ✓
- 其他 ✗

### 9.24 search/filter/eval/arrow_backend.py

```python
import functools
import pyarrow as pa
import pyarrow.compute as pc

_CMP_KERNELS = {
    "==": pc.equal, "!=": pc.not_equal,
    "<":  pc.less,  "<=": pc.less_equal,
    ">":  pc.greater, ">=": pc.greater_equal,
}

def evaluate_arrow(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Walk the AST, translating each node into pyarrow.compute calls.

    NULL handling: top-level result is fill_null(False) so any null
    in operand chain becomes "no row matches" rather than three-valued
    result. AND/OR use the Kleene variants (and_kleene, or_kleene).
    """
```

**Dispatch 表**（每个 AST 节点 → pyarrow.compute 调用）：

| AST | pyarrow operation |
|---|---|
| `IntLit / FloatLit / StringLit / BoolLit` | `pa.scalar(value)` |
| `FieldRef` | `table.column(name)` |
| `CmpOp(op, l, r)` | `_CMP_KERNELS[op](_eval(l), _eval(r))` |
| `InOp(field, values, negate)` | `pc.is_in(col, value_set=values)`，`negate` 时 `pc.invert` |
| `And(operands)` | `functools.reduce(pc.and_kleene, masks)` |
| `Or(operands)` | `functools.reduce(pc.or_kleene, masks)` |
| `Not(operand)` | `pc.invert(_eval(operand))` |

### 9.25 search/filter/eval/python_backend.py

```python
def evaluate_python(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Row-wise interpreter. Slow but flexible.

    F1 用途：差分测试基准（arrow_backend 的输出必须与之相等）。
    F2b+ 用途：评估含 $meta 的表达式。
    F3+ 用途：评估含 UDF 的表达式。

    NULL 语义：用 Kleene 三值逻辑实现 AND/OR/NOT，最终结果 None → False。
    """
```

**Dispatch**：与 arrow_backend 镜像，但每个节点接受 row dict 返回 Python 值：

| AST | Python operation |
|---|---|
| `IntLit` 等 | `node.value` |
| `FieldRef` | `row.get(node.name)` |
| `CmpOp(op, l, r)` | `_CMP_OPS[op](_eval(l, row), _eval(r, row))`，None 传染 |
| `InOp` | 集合查找 + 可选否定 |
| `And/Or` | Kleene 三值短路 |
| `Not` | None 传染 |

### 9.26 search/filter/eval/__init__.py

```python
def evaluate(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Backend dispatcher.

    F1 始终走 arrow_backend；F2b+ 根据 compiled.backend 字段 dispatch。
    在 evaluate 热路径上**不做**重复 backend 决策——决策已在 compile_expr
    完成、固化在 CompiledExpr.backend 字段。
    """
    if compiled.backend == "arrow":
        return evaluate_arrow(compiled, data)
    elif compiled.backend == "python":
        return evaluate_python(compiled, data)
    raise ValueError(f"unknown filter backend: {compiled.backend!r}")
```

### 9.27 Phase 8 实施分阶段

| Phase | 目标 grammar | Backend |
|---|---|---|
| **F1** | Tier 1：`==/!=/<.../in/and/or/not` + 字面量 + 字段引用 + 括号 | 仅 arrow_backend（python_backend 用作差分测试基准） |
| **F2a** | + `like` + 算术 (`+ - * / %`) + `is null` | 仍 arrow_backend |
| **F2b** | + `$meta["key"]` 动态字段 | 引入 python_backend dispatch（含 $meta 的表达式自动降级） |
| **F2c** | + filter 缓存 + `query()` 公开方法（如果 F1 没做） | 与 backend 无关 |
| **F3** | + `json_contains` / `array_contains` / UDF / 严格 Milvus 兼容 | 扩展 python_backend；可选 ANTLR parser swap |
| **F3+** | 性能优化：per-batch JSON 预处理 / DuckDB opt-in | 引入 hybrid_backend |

### 9.28 Phase 8 设计参考

- **Milvus Plan.g4**: `internal/parser/planparserv2/Plan.g4`（master 分支）
- **操作符优先级、关键字大小写、字面量语法均对齐 Milvus**
- **AST 节点形态借鉴 Milvus PlanNode 概念**，但简化（`CmpOp` 替代 `Equality`/`Relational`）
- **F1 不追 binary 兼容**——文档化我们的子集，未来 F3 才考虑
- **F1 选手写 Pratt parser** 而非 ANTLR：F1 grammar 小、错误信息更友好、零依赖。AST 是稳定接口，F3 切 ANTLR 不影响 type checker / backends
