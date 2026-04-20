# LiteVecDB - 本地向量数据库 MVP 设计方案

## 1. 概述

基于 LSM-Tree 思想的本地向量数据库，内存层使用 PyArrow，持久化层使用 Parquet 格式。采用 **DB → Collection → Partition** 三层数据组织（对齐 Milvus），支持 Collection Schema 模型（类型化字段、Schema 版本管理、动态字段）和 Partition 级数据隔离。插入数据与删除记录分离存储，支持向量的实时增删改查与相似度检索。

## 2. 核心设计原则

### 落盘文件一律不可变

这是整个系统最基本的约束：

- **数据文件 (Parquet)**：写入后永远不修改，只有 Compaction 时整个删除
- **Delta Log (Parquet)**：写入后永远不修改，Compaction 消费后整个删除
- **WAL 文件**：只做追加写入，flush 成功后整个删除
- **Manifest**：唯一通过原子替换（write-tmp + rename）更新的状态文件
- **没有任何可变的辅助文件**（不用 bitmap、不用 sidecar 文件）

### 插入数据与删除记录分离

参考 Milvus 的设计，Insert 和 Delete 走**两条独立的数据流**，落到不同的文件中：

```
Insert("doc_1", vec)  →  数据文件 (Parquet)    # 包含向量和元数据
Delete("doc_1")       →  Delta Log (Parquet)   # 只记录 (_id, _seq)
```

好处：
- 数据文件内不含删除标记，将来可直接对文件构建向量索引
- Delta Log 体积小（只有 id + seq），可全量加载到内存
- 搜索时用内存中的 deleted set 过滤，不需要修改任何磁盘文件

### 扁平文件组织，不分 Level

传统 LSM-Tree 的 Level 结构是为了优化点查（L1+ 层内 key 不重叠，可跳过无关文件）。但向量数据库的核心操作是**全量扫描**，Level 不提供额外收益，反而增加复杂度。因此：

- 所有数据文件放在同一个**扁平目录**
- 点查通过文件名中的 seq 范围从新到旧扫描，配合 `deleted_map` 判断有效性
- Compaction 采用 **Size-Tiered** 策略（合并大小相近的小文件），而非 Leveled

### 代价与取舍

- **读放大**：搜索需要读所有数据文件，依赖 Compaction 控制文件数量
- **空间放大**：被删除/更新的旧版本在 Compaction 前仍占空间
- **写放大**：Compaction 会重写数据，但换来更少的文件数和更好的读性能
- **换来的好处**：实现简单、崩溃安全、并发友好、天然支持未来的向量索引

## 3. 整体架构

### 3.1 数据层级

```
DB ("my_app")                        ← 命名空间，对应一个根目录
  ├── Collection ("documents")       ← Schema 定义在这层，共享 WAL / MemTable / _seq
  │     ├── Partition ("2024_Q1")    ← 数据文件按 Partition 隔离
  │     ├── Partition ("2024_Q2")
  │     └── Partition ("_default")   ← 默认 Partition，不可删除
  └── Collection ("images")
        └── Partition ("_default")
```

- **DB**：纯命名空间，对应磁盘上一个根目录，无存储逻辑
- **Collection**：Schema 的所有者，拥有独立的 WAL、MemTable、Manifest、`_seq` 计数器
- **Partition**：Collection 内的数据分片，共享 Schema，独立的数据文件和 Delta Log 目录

### 3.2 组件架构

```
                    ┌──────────────────────────┐
                    │       Client API          │
                    │  insert / delete / update │
                    │  search / get             │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │        DB Engine          │
                    │  (管理多 DB / Collection)   │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │    Collection Engine      │
                    │  (全局 _seq 分配、调度)     │
                    └──┬──────────────────────┬┘
                       │                      │
            ┌──────────▼──────────┐  ┌────────▼─────────┐
            │    Write Path       │  │    Read Path      │
            │                     │  │                    │
            │ WAL → MemTable      │  │ MemTable + Data   │
            │ → Flush(按Partition) │  │ Files - Delta Set │
            └──────────┬──────────┘  └──────────────────┘
                       │
         ┌─────────────▼──────────────────────┐
         │     Storage Layer (per Collection)  │
         │                                     │
         │  ┌─────────┐  ┌──────────────────┐ │
         │  │  WAL    │  │  Partition "Q1"  │ │
         │  │(Arrow   │  │  ┌─────┐┌─────┐ │ │
         │  │  IPC)   │  │  │Data ││Delta│ │ │
         │  │(共享)    │  │  │Files││Logs │ │ │
         │  └─────────┘  │  └─────┘└─────┘ │ │
         │               ├──────────────────┤ │
         │  ┌──────────┐ │  Partition "Q2"  │ │
         │  │ Manifest │ │  ┌─────┐┌─────┐ │ │
         │  │(全局状态)  │ │  │Data ││Delta│ │ │
         │  └──────────┘ │  │Files││Logs │ │ │
         │               │  └─────┘└─────┘ │ │
         │  ┌──────────┐ └──────────────────┘ │
         │  │Compaction│                      │
         │  │ Manager  │                      │
         │  └──────────┘                      │
         └─────────────────────────────────────┘
```

## 4. 数据模型

### 4.1 Collection Schema（对齐 Milvus）

参考 Milvus 的 Schema 设计，采用 **Collection Schema + Field Schema** 模型。用户创建 Collection 时定义 Schema，包含主键字段、向量字段、标量字段等。

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

class FieldSchema:
    name: str               # 字段名
    dtype: DataType         # 字段类型
    is_primary: bool        # 是否为主键（有且仅有一个）
    dim: Optional[int]      # 向量维度（仅向量字段需要）
    max_length: Optional[int]  # VARCHAR 最大长度
    nullable: bool          # 是否允许 null
    default_value: Any      # 默认值

class CollectionSchema:
    fields: List[FieldSchema]
    version: int            # Schema 版本号（每次变更 +1）
    enable_dynamic_field: bool  # 是否启用动态字段（$meta JSON 列）
```

#### Schema 约束

- 必须有且仅有一个 `is_primary=True` 字段，类型为 `VARCHAR` 或 `INT64`
- 必须有且仅有一个 `FLOAT_VECTOR` 字段（MVP 限制，将来可放开）
- 主键字段不可为 null
- Schema 持久化为 `data_dir/schema.json`

#### DataType → Arrow Type 映射

```python
TYPE_MAP = {
    DataType.BOOL:         pa.bool_(),
    DataType.INT8:         pa.int8(),
    DataType.INT16:        pa.int16(),
    DataType.INT32:        pa.int32(),
    DataType.INT64:        pa.int64(),
    DataType.FLOAT:        pa.float32(),
    DataType.DOUBLE:       pa.float64(),
    DataType.VARCHAR:      pa.string(),
    DataType.JSON:         pa.string(),       # JSON 序列化为字符串
    DataType.FLOAT_VECTOR: lambda dim: pa.list_(pa.float32(), list_size=dim),
}
```

### 4.2 Schema 体系（由 Collection Schema 自动生成）

系统内部有 **四套 Arrow Schema**，全部从 Collection Schema 自动推导：

```python
def _build_user_fields(collection_schema: CollectionSchema) -> list:
    """提取用户定义字段的 Arrow 类型"""
    fields = []
    for f in collection_schema.fields:
        arrow_type = TYPE_MAP[f.dtype]
        if callable(arrow_type):
            arrow_type = arrow_type(f.dim)
        fields.append((f.name, arrow_type))
    if collection_schema.enable_dynamic_field:
        fields.append(("$meta", pa.string()))
    return fields

def build_data_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """数据 Parquet 文件的 Schema（不含 _partition）"""
    fields = [("_seq", pa.uint64())]
    fields += _build_user_fields(collection_schema)
    return pa.schema(fields)

def build_delta_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """Delta Parquet 文件的 Schema（不含 _partition）"""
    pk = get_primary_field(collection_schema)
    return pa.schema([(pk.name, TYPE_MAP[pk.dtype]), ("_seq", pa.uint64())])

def build_wal_data_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """WAL 数据文件的 Schema（比 data_schema 多一列 _partition）"""
    fields = [("_seq", pa.uint64()), ("_partition", pa.string())]
    fields += _build_user_fields(collection_schema)
    return pa.schema(fields)

def build_wal_delta_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """WAL 删除文件的 Schema（比 delta_schema 多一列 _partition）"""
    pk = get_primary_field(collection_schema)
    return pa.schema([(pk.name, TYPE_MAP[pk.dtype]), ("_seq", pa.uint64()), ("_partition", pa.string())])
```

#### 四套 Schema 的关系

| Schema | 用途 | 包含 `_partition` | 文件格式 |
|--------|------|:-:|------|
| `wal_data_schema` | WAL 数据文件 | **是** | Arrow IPC |
| `wal_delta_schema` | WAL 删除文件 | **是** | Arrow IPC |
| `data_schema` | 数据 Parquet 文件 | 否 | Parquet |
| `delta_schema` | Delta Parquet 文件 | 否 | Parquet |

**为什么 WAL 需要 `_partition` 而 Parquet 不需要？**
- WAL 是 Collection 级共享的单个文件，恢复时需要知道每条记录属于哪个 Partition
- Parquet 文件已经按 Partition 目录隔离，文件路径本身就体现了归属，无需冗余列
```

示例（以 documents Collection 为例）：

```python
schema = CollectionSchema(
    fields=[
        FieldSchema("doc_id", DataType.VARCHAR, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128),
        FieldSchema("source", DataType.VARCHAR, nullable=True),
        FieldSchema("score", DataType.FLOAT, nullable=True),
    ],
    enable_dynamic_field=True,
)

# data_schema（数据 Parquet）            wal_data_schema（WAL 数据）
# ─────────────────────────              ─────────────────────────
# _seq:      uint64                      _seq:       uint64
#                                        _partition: string        ← WAL 独有
# doc_id:    string                      doc_id:     string
# embedding: list<f32, 128>              embedding:  list<f32, 128>
# source:    string                      source:     string
# score:     float32                     score:      float32
# $meta:     string                      $meta:      string

# delta_schema（Delta Parquet）          wal_delta_schema（WAL 删除）
# ─────────────────────────              ─────────────────────────
# doc_id:    string                      doc_id:     string
# _seq:      uint64                      _seq:       uint64
#                                        _partition: string        ← WAL 独有
```

注意：
- `_seq` 是系统内部字段，不出现在用户 Schema 中
- `_partition` 只存在于 WAL 中（Parquet 通过目录隔离体现 Partition）
- 主键字段由用户定义（替代原来的硬编码 `_id`）
- 标量字段拥有明确的类型，Parquet 可做谓词下推

### 4.3 Schema 版本管理

Schema 变更（如新增字段）采用 **Parquet 天然的 schema evolution**：

- 新增字段：新文件使用新 Schema 写入，旧文件读取时缺失的列自动填充 null
- 删除字段：不支持（只做加法）
- 修改字段类型：不支持

```json
// data_dir/schema.json（name 来自 create_collection 的参数，持久化到 schema.json 供自描述）
{
    "collection_name": "documents",
    "version": 2,
    "fields": [
        {"name": "doc_id", "dtype": "varchar", "is_primary": true},
        {"name": "embedding", "dtype": "float_vector", "dim": 128},
        {"name": "source", "dtype": "varchar", "nullable": true},
        {"name": "score", "dtype": "float", "nullable": true},
        {"name": "category", "dtype": "varchar", "nullable": true}  // ← v2 新增
    ],
    "enable_dynamic_field": true
}
```

### 4.4 动态字段

当 `enable_dynamic_field=True` 时，数据文件中会多一列 `$meta`（JSON 字符串），用于存储不在 Schema 中定义的字段：

```python
# 插入时，Schema 外的字段自动归入 $meta
db.insert(doc_id="doc_1", embedding=[...], source="wiki",
          category="science", tags=["ml", "ai"])
# → source 进入 source 列（Schema 内）
# → category, tags 进入 $meta: '{"category": "science", "tags": ["ml", "ai"]}'
```

### 4.5 Delta Log Schema（存储删除记录，不变）

| 字段       | 类型     | 说明                         |
| ---------- | -------- | ---------------------------- |
| `{pk_name}`| 主键类型 | 被删除记录的主键              |
| `_seq`     | `uint64` | 删除操作的序号，用于和数据版本比较 |

```python
def build_delta_schema(collection_schema: CollectionSchema) -> pa.Schema:
    """从 Collection Schema 生成 Delta Log 的 Arrow Schema"""
    pk = get_primary_field(collection_schema)
    pk_type = TYPE_MAP[pk.dtype]
    return pa.schema([
        (pk.name, pk_type),
        ("_seq", pa.uint64()),
    ])
```

### 4.6 版本判定规则（不变）

对于同一个主键，可能同时存在数据记录和删除记录。判定逻辑：

```
data_seq  = 数据文件中该主键最大的 _seq
delta_seq = delta log 中该主键最大的 _seq

if delta_seq > data_seq  → 该记录已被删除
if data_seq > delta_seq  → 该记录有效（删除后又重新插入）
if 只有 data_seq         → 该记录有效
if 只有 delta_seq        → 该记录已被删除（可忽略）
```

## 5. 核心组件设计

### 5.1 WAL (Write-Ahead Log)

采用 **Arrow IPC Streaming** 格式，二进制直接写磁盘，避免 JSONL 文本编码对向量数据的 3 倍写放大。

插入和删除使用**两个独立的 WAL 文件**，与 MemTable 的双缓冲区一一对应：

- **`wal_data_{N}.arrow`**：记录插入/更新操作，使用 `wal_data_schema`（比 `data_schema` 多 `_partition` 列）
- **`wal_delta_{N}.arrow`**：记录删除操作，使用 `wal_delta_schema`（比 `delta_schema` 多 `_partition` 列）

```python
class WAL:
    def __init__(self, wal_dir, wal_data_schema, wal_delta_schema):
        self.data_writer = None   # pa.ipc.RecordBatchStreamWriter
        self.delta_writer = None  # pa.ipc.RecordBatchStreamWriter

    def write_insert(self, record_batch: pa.RecordBatch):
        """追加写入 wal_data 文件（RecordBatch 含 _partition 列）"""
        if self.data_writer is None:
            self.data_writer = pa.ipc.new_stream(data_path, self.wal_data_schema)
        self.data_writer.write_batch(record_batch)

    def write_delete(self, record_batch: pa.RecordBatch):
        """追加写入 wal_delta 文件（RecordBatch 含 _partition 列）"""
        if self.delta_writer is None:
            self.delta_writer = pa.ipc.new_stream(delta_path, self.wal_delta_schema)
        self.delta_writer.write_batch(record_batch)

    def close_and_delete(self):
        """flush 成功后关闭并删除两个 WAL 文件"""

    def recover(self) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """启动时读取未清理的 WAL 文件，返回 (data_batches, delta_batches)
        每个 RecordBatch 含 _partition 列，恢复时按 _partition 路由到 MemTable"""
```

- **生命周期**：MemTable 成功 flush 后，对应的两个 WAL 文件整个删除
- **恢复**：启动时读取未清理的 WAL 文件，重建 MemTable 的两个缓冲区

```
data_dir/
  wal/
    wal_data_000001.arrow
    wal_delta_000001.arrow
```

### 5.2 MemTable

MemTable 是 **Collection 级共享**的（与 Milvus 一致：WAL/MemTable 在 Collection 级，不按 Partition 拆分），内部维护两个独立的缓冲区，字段结构由 Collection Schema 驱动。每条记录携带 `_partition` 标记，flush 时按 Partition 拆分输出。

**Upsert 语义**：`insert_buf` 使用 dict[pk → record]，相同 PK 直接覆盖——PK 存在则更新，不存在则新增。内部引擎只提供 `insert()` 一个写入方法，天然具备 upsert 语义。Collection 级保证 PK 唯一。

```python
class MemTable:
    def __init__(self, schema: CollectionSchema):
        self.schema = schema
        self.pk_name = get_primary_field(schema).name
        self.insert_buf = {}   # pk_value -> {_partition: str, field_name: value, ...}
        self.delete_buf = {}   # pk_value -> (_seq, _partition)
        self.lock = threading.Lock()

    def put(self, _seq: int, _partition: str, **fields):
        """写入插入记录到 insert_buf"""
        pk = fields[self.pk_name]
        record = {"_seq": _seq, "_partition": _partition, **fields}
        self.insert_buf[pk] = record
        self.delete_buf.pop(pk, None)

    def delete(self, pk_value, _seq: int, _partition: str):
        """写入删除记录到 delete_buf"""
        self.delete_buf[pk_value] = (_seq, _partition)
        self.insert_buf.pop(pk_value, None)

    def get(self, pk_value) -> Optional[dict]:
        """点查：先检查 delete_buf，再检查 insert_buf"""
        if pk_value in self.delete_buf:
            del_seq, _ = self.delete_buf[pk_value]
            if pk_value in self.insert_buf and self.insert_buf[pk_value]["_seq"] > del_seq:
                return self.insert_buf[pk_value]
            return None
        return self.insert_buf.get(pk_value)

    def flush(self) -> Dict[str, Tuple[Optional[pa.Table], Optional[pa.Table]]]:
        """按 Partition 拆分输出 {partition_name: (data_table, delta_table)}"""
        result = {}
        # 按 _partition 分组 insert_buf
        for pk, record in self.insert_buf.items():
            part = record["_partition"]
            result.setdefault(part, ([], []))
            result[part][0].append(record)
        # 按 _partition 分组 delete_buf
        for pk, (seq, part) in self.delete_buf.items():
            result.setdefault(part, ([], []))
            result[part][1].append({self.pk_name: pk, "_seq": seq})
        # 转为 Arrow Table
        return {
            part: (
                build_arrow_table(inserts, self.schema) if inserts else None,
                build_delta_table(deletes, self.schema) if deletes else None,
            )
            for part, (inserts, deletes) in result.items()
        }

    def size(self) -> int:
        return len(self.insert_buf) + len(self.delete_buf)
```

- **大小限制**：默认 `MEMTABLE_SIZE_LIMIT = 10000` 条（insert + delete 合计，跨 Partition）
- **Flush 触发**：达到阈值时冻结当前 MemTable，创建新 MemTable 接收写入
- **Flush 输出**：按 Partition 拆分，每个 Partition 独立的 data Parquet + delta Parquet
- **动态字段处理**：`put()` 时将 Schema 外的字段序列化为 JSON 存入 `$meta`
- **`_partition` 不写入 Parquet**：`_partition` 只用于 flush 时路由，Parquet 文件本身不含此列（由目录隔离体现）

### 5.3 数据文件 (Parquet)

存储插入/更新的向量记录，**不包含任何删除标记**。

- **文件组织**：扁平目录，所有数据文件在同一层
- **文件命名**：`data_{seq_min}_{seq_max}.parquet`
- **排序**：每个文件内按 `_id` 排序，便于合并
- **不可变**：写入后不再修改，Compaction 时整个文件删除
- **自描述**：利用 Parquet 自带的列统计信息（min/max）获取 id 范围和 seq 范围

```
data_dir/
  data/
    data_000001_000500.parquet
    data_000501_001000.parquet
    data_000001_002000.parquet    ← compaction 合并产出
```

### 5.4 Delta Log (Parquet)

存储删除操作记录，与数据文件完全分离。

- **文件组织**：扁平目录，每次 flush 生成一个新文件
- **文件命名**：`delta_{seq_min}_{seq_max}.parquet`
- **内容**：只有 `(_id, _seq)` 两列，体积远小于数据文件
- **不可变**：写入后不再修改，Compaction 消费后整个删除
- **内存缓存**：启动时全量加载所有 delta log 到内存，构建 `deleted_map: dict[str, int]`（_id → 最大 delete _seq）

```
data_dir/
  deltas/
    delta_000501_000503.parquet
    delta_001001_001002.parquet
```

```python
class DeltaLog:
    def __init__(self, data_dir, pk_name: str):
        self.data_dir = data_dir
        self.pk_name = pk_name
        self.deleted_map = {}  # pk_value -> max delete _seq (内存常驻)

    def load_all(self):
        """启动时从所有 delta parquet 文件重建 deleted_map"""
        for f in glob(deltas/*.parquet):
            table = pq.read_table(f)
            for pk, _seq in zip(table[self.pk_name], table["_seq"]):
                cur = self.deleted_map.get(pk.as_py(), 0)
                self.deleted_map[pk.as_py()] = max(cur, _seq.as_py())

    def add(self, delta_table: pa.Table):
        """flush 时写入新的 delta 文件并更新内存"""
        pq.write_table(delta_table, path)
        # 同时更新 deleted_map

    def is_deleted(self, pk_value, data_seq: int) -> bool:
        """判断某条数据记录是否已被删除"""
        del_seq = self.deleted_map.get(pk_value, 0)
        return del_seq > data_seq

    def remove_files(self, files: List[str]):
        """Compaction 后删除已消费的 delta 文件，并清理 deleted_map 中对应的条目"""
```

### 5.5 Manifest（借鉴 LanceDB）

Manifest 是数据库的**全局状态快照文件**，记录当前有哪些文件、当前 _seq 等关键状态。借鉴 LanceDB 的 Manifest 设计，解决三个问题：

1. **快速启动**：不用扫描目录，直接从 manifest 获取文件列表和 _seq
2. **原子状态变更**：Flush / Compaction 后通过原子替换 manifest 保证一致性
3. **Snapshot 基础**：快照 = 某个时刻的 manifest 副本

#### Manifest 内容

Manifest 按 Partition 组织文件列表，WAL 和 `_seq` 在 Collection 级共享：

```json
// data_dir/manifest.json
{
    "version": 42,
    "current_seq": 15023,
    "schema_version": 2,
    "partitions": {
        "_default": {
            "data_files": ["_default/data_000001_005000.parquet"],
            "delta_files": ["_default/delta_005001_005003.parquet"]
        },
        "2024_Q1": {
            "data_files": [
                "2024_Q1/data_000001_003000.parquet",
                "2024_Q1/data_003001_008000.parquet"
            ],
            "delta_files": ["2024_Q1/delta_008001_008002.parquet"]
        },
        "2024_Q2": {
            "data_files": ["2024_Q2/data_005001_010000.parquet"],
            "delta_files": []
        }
    },
    "active_wal": {
        "data": "wal_data_000003.arrow",
        "delta": "wal_delta_000003.arrow"
    }
}
```

| 字段 | 说明 |
|------|------|
| `version` | Manifest 版本号，每次更新 +1 |
| `current_seq` | 当前最大 _seq，启动时恢复计数器（Collection 级） |
| `schema_version` | 当前 Collection Schema 版本号 |
| `partitions` | 按 Partition 名组织的文件列表，每个 Partition 有独立的 `data_files` 和 `delta_files` |
| `active_wal` | 当前活跃的 WAL 文件路径（Collection 级共享） |

#### 原子更新

通过 **write-tmp + rename** 保证原子性：

```python
class Manifest:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "manifest.json")
        self.version = 0
        self.current_seq = 0
        self.schema_version = 1
        self.partitions = {"_default": {"data_files": [], "delta_files": []}}
        self.active_wal = {"data": None, "delta": None}

    def save(self):
        """原子更新：写临时文件 → rename"""
        self.version += 1
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, self.path)  # 原子操作

    @classmethod
    def load(cls, data_dir: str) -> "Manifest":
        """启动时加载 manifest"""
        path = os.path.join(data_dir, "manifest.json")
        if os.path.exists(path):
            return cls.from_dict(json.load(open(path)))
        return cls(data_dir)  # 首次创建
```

`os.rename` 在 POSIX 文件系统上是原子操作——要么看到旧 manifest，要么看到新的，不会看到半写的状态。

#### 更新时机

| 事件 | Manifest 变更 |
|------|--------------|
| **Flush** | 对应 Partition 的 `data_files` / `delta_files` 加入新文件，`current_seq` 更新，`active_wal` 切换 |
| **Compaction** | 对应 Partition 的 `data_files` 移除旧文件 + 加入新文件，`delta_files` 移除已消费的文件 |
| **Create Partition** | `partitions` 新增一个 key，初始化空文件列表 |
| **Drop Partition** | `partitions` 移除该 key（文件随后异步删除） |
| **Schema 变更** | `schema_version` +1 |
| **WAL 轮转** | `active_wal` 更新 |

#### 崩溃恢复

启动时的恢复逻辑：

```
1. 读取 manifest.json
   ├─ 存在 → 获取文件列表和 current_seq
   └─ 不存在 → fallback 到目录扫描（首次启动或 manifest 损坏）
2. 检查 wal/ 目录是否有未清理的 WAL 文件
   ├─ 无 → 正常启动
   └─ 有 → 重放 WAL 到 MemTable（manifest 中的状态是 WAL 之前的一致性快照）
3. 校验 manifest 中的文件是否实际存在（防止 compaction 中途崩溃）
4. 从 delta_files 重建 deleted_map
5. 恢复完成，保存新的 manifest
```

关键：manifest 总是在 WAL 删除**之前**更新。如果 flush 完成了（数据写入 + manifest 更新）但 WAL 未删除，重放 WAL 会产生重复数据，但由于 `_seq` 去重，不会影响正确性。

### 5.6 Compaction Manager

采用 **Size-Tiered Compaction** 策略：不分 Level，按文件大小分组合并。**Compaction 按 Partition 独立执行**——每个 Partition 的文件互不干扰。

#### 触发条件

| 条件 | 阈值 | 说明 |
| ---- | ---- | ---- |
| 小文件过多 | 同一数量级大小的文件 >= 4 个 | 合并为一个更大的文件 |
| 文件总数过多 | 总文件数 > `MAX_DATA_FILES`（默认 32） | 选择最小的几个文件合并 |

#### 文件大小分组

将文件按大小分桶（对数刻度）：
- 桶 0：< 1MB
- 桶 1：1MB ~ 10MB
- 桶 2：10MB ~ 100MB
- ...

同一个桶内文件数 >= 4 个时，触发合并。

#### Compaction 流程

1. 对某个 Partition，选择同一大小桶中的多个文件
2. 读取并合并 Arrow Tables
3. 按主键去重（保留最大 `_seq` 的版本）
4. 用 `deleted_map` 过滤已删除的记录
5. 写入一个新的 Parquet 文件（到该 Partition 的 data/ 目录）
6. **原子更新 Manifest**（该 Partition 内：移除旧文件 + 加入新文件 + 移除已消费的 delta 文件）
7. 删除旧的数据文件和已消费的 delta log 文件
8. MVP 阶段在主线程同步执行，按 Partition 逐个检查

注意步骤 6-7 的顺序：先更新 Manifest 再删除旧文件。如果删除旧文件前崩溃，启动时 Manifest 已指向新文件，旧文件变成孤儿文件可安全清理。

```python
class CompactionManager:
    def __init__(self, data_dir, delta_log): ...

    def maybe_compact(self):
        # 1. 按大小对数据文件分桶
        # 2. 找到文件数 >= 4 的桶
        # 3. 合并该桶中的文件
        #    a. 读取所有 Arrow Tables
        #    b. 按 _id 去重（保留 max _seq）
        #    c. 用 delta_log.is_deleted() 过滤
        #    d. 写新文件，删旧文件
        # 4. 清理可回收的 delta log 文件

    def merge_tables(self, tables: List[pa.Table]) -> pa.Table: ...
```

### 5.7 Vector Search

搜索采用**先构建 bitmap，再执行向量检索**的管线。bitmap 统一处理删除过滤（MVP）和标量过滤（将来），向量检索只看 bitmap 中有效的行。

#### 搜索管线

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1. 收集数据   │ ──→ │ 2. 构建 bitmap│ ──→ │ 3. 向量检索   │
│ MemTable +   │     │ 去重 + 删除   │     │ 只计算有效行  │
│ 所有数据文件  │     │ (+ 标量过滤)  │     │ 返回 top_k   │
└──────────────┘     └──────────────┘     └──────────────┘
```

#### bitmap 构建规则

```python
# valid_mask[i] = True 表示第 i 行参与向量检索
valid_mask = np.ones(n, dtype=bool)

# 1. 去重：同一主键只保留最大 _seq 的行，其余标记为 False
for duplicate_rows in group_by_pk(records):
    keep only max _seq, mark others False

# 2. 删除过滤：检查 deleted_map
for i, (pk, _seq) in enumerate(records):
    if delta_log.is_deleted(pk, _seq):
        valid_mask[i] = False

# 3. 标量过滤（将来）：
# if filter_expr:
#     scalar_mask = evaluate_filter(typed_columns, filter_expr)  # Parquet 谓词下推
#     valid_mask &= scalar_mask
```

#### MVP 实现

MVP 阶段使用 NumPy 暴力扫描 + bitmap mask：

```python
def search(self, vectors: List[list], top_k: int = 10,
           metric_type: str = "COSINE",
           partition_names: List[str] = None) -> List[List[dict]]:
    # 1. 收集数据：MemTable 活跃记录 + 目标 Partition 的数据 Parquet 文件
    all_pks, all_seqs, all_vectors = collect_all(self.schema, partition_names)

    # 2. 构建 bitmap
    valid_mask = build_valid_mask(all_pks, all_seqs, self.delta_log)

    # 3. 向量检索（只对有效行计算距离）
    results = []
    for query_vector in vectors:
        valid_vectors = all_vectors[valid_mask]
        distances = compute_distances(query_vector, valid_vectors, metric_type)
        top_indices = np.argpartition(distances, top_k)[:top_k]
        results.append(build_results(top_indices, ...))
    return results  # List[List[{"id": pk, "distance": float, "entity": {...}}]]
```

#### Phase 9：FAISS HNSW 已接入（per-segment 索引）

Phase 9 把检索路径从"NumPy 暴力扫描"升级为"FAISS HNSW per-segment"。同一套 bitmap pipeline 被复用——`valid_mask` 直接喂给 FAISS 的 `IDSelectorBitmap` 做 pre-filter：

```python
# Phase 9 实际实现（litevecdb/index/faiss_hnsw.py）
import faiss, numpy as np
mask_packed = np.packbits(valid_mask, bitorder='little')
sel = faiss.IDSelectorBitmap(num_vectors, faiss.swig_ptr(mask_packed))
params = faiss.SearchParametersHNSW(sel=sel, efSearch=ef)
distances, ids = index.search(query, top_k, params=params)
```

**架构决策**（详见 `plan/index-design.md`）：
- 索引绑定 **segment-level**（每个 data parquet 一个 .idx 文件，1:1 绑定）
- 索引库选 **FAISS-cpu**（IDSelectorBitmap 与 bitmap pipeline 同构；索引家族对齐 Milvus）
- **BruteForceIndex 长期保留** 作为差分基准 + 无 faiss 时的 fallback
- **load/release 状态机** 与 Milvus 行为对齐——重启后默认 released，必须显式 load

存储层无任何改动——这正是 LSM 不可变架构 + bitmap pipeline 在 Phase 0 就预留的红利。

- 支持的距离度量：`cosine`（默认）、`l2`、`ip`（内积）

## 6. 读写路径

### 6.1 写入路径 (Insert)

内部引擎只有 `insert()` 一个写入方法，天然具备 upsert 语义（PK 存在则覆盖）。输入始终是 `List[dict]`，参数规范化由上层（gRPC 适配层）处理。

```
Collection.insert(records=[{"doc_id": "doc_1", ...}], partition_name="_default")
  │
  ├─ 1. 解析目标 Partition（partition_name 未指定则使用 "_default"）
  ├─ 2. Schema 校验（字段类型、主键非空、向量维度）
  ├─ 3. 分离 Schema 内字段 vs 动态字段（→ $meta JSON）
  ├─ 4. 分配全局递增 _seq（Collection 级，每条记录独立 _seq）
  ├─ 5. 写入 WAL (wal_data): Arrow IPC RecordBatch（含 _partition 标记）
  ├─ 6. 写入 MemTable.insert_buf（携带 _partition，相同 PK 直接覆盖 → upsert 语义）
  ├─ 7. 检查 MemTable 大小
  │     ├─ 未满 → 返回
  │     └─ 已满 → 冻结当前 MemTable
  │               ├─ 创建新 MemTable + 新 WAL
  │               ├─ 按 Partition 拆分 Flush：
  │               │   ├─ Partition A: insert_buf → data Parquet, delete_buf → delta Parquet
  │               │   └─ Partition B: insert_buf → data Parquet, delete_buf → delta Parquet
  │               ├─ 更新内存 deleted_map
  │               ├─ 原子更新 Manifest（各 Partition 新增文件 + 更新 current_seq + 切换 active_wal）
  │               ├─ 删除旧 WAL (wal_data + wal_delta)
  │               └─ 触发 Compaction 检查（按 Partition 独立）
  └─ 返回写入的 PK 列表
```

- **Upsert 语义**：相同主键再次 insert，分配更高 `_seq`，MemTable 中直接覆盖旧版本

### 6.2 删除路径 (Delete)

输入始终是 `List[pk]`，多条 PK 共享同一个 `_seq`。参数规范化（单值→列表）由上层处理。

```
Collection.delete(pks=["doc_1"])                                   # partition_name=None → 跨所有 Partition 删除
Collection.delete(pks=["doc_1", "doc_2"], partition_name="2024_Q1") # 指定 partition → 只在该 Partition 删除
  │
  ├─ 1. 解析目标 Partition：
  │     ├─ 指定 partition_name → 使用该 Partition
  │     └─ 未指定（None）→ 不绑定 Partition（跨所有 Partition 删除）
  ├─ 2. 分配全局递增 _seq（整批共享同一个 _seq）
  ├─ 3. 写入 WAL (wal_delta): Arrow IPC RecordBatch（含 _partition 标记）
  ├─ 4. 写入 MemTable.delete_buf（携带 _partition）
  ├─ 5. 同样的 flush 逻辑（达到阈值时一起 flush，按 Partition 拆分输出）
  └─ 返回处理的 PK 数量
```

注意：删除不需要确认记录是否存在。写入即可，如果主键不存在，delta log 里的这条记录在 compaction 时自然被清理。

### 6.3 读取路径 (Get by PK)

输入始终是 `List[pk]`。

```
Collection.get(pks=["doc_1"], partition_names=None)
Collection.get(pks=["doc_1", "doc_2"], output_fields=["doc_id", "source"], partition_names=["2024_Q1"])
  │
  ├─ 1. 对每个 PK：
  │     ├─ 查 MemTable → 内部已处理 insert/delete 冲突，命中则加入结果
  │     ├─ 查冻结中的 MemTable（如有）
  │     ├─ 确定搜索范围：
  │     │     ├─ 指定 partition_names → 只扫描指定 Partition 的文件
  │     │     └─ 未指定 → 扫描所有 Partition 的文件
  │     ├─ 扫描数据文件（从 Manifest 获取文件列表，按 seq 从新到旧）
  │     │     找到数据记录后 → 检查 deleted_map：
  │     │       delta_seq > data_seq → 已删除，跳过
  │     │       否则 → 加入结果（按 output_fields 裁剪返回字段）
  │     └─ 未找到 → 该 PK 不出现在结果中
  └─ 返回 List[dict]（每个 dict 为一条记录，未找到的 PK 不在列表中）
```

由于 deleted_map 常驻内存，判断是否删除只是一次 dict 查找，O(1)。指定 partition_names 可减少文件扫描范围。

### 6.4 向量检索路径 (Search)

```
Collection.search(vectors=[[0.1, 0.2, ...]], top_k=10, metric_type="COSINE", partition_names=None)
  │
  │  ── 阶段 0：Partition 剪枝 ──
  ├─ 0. 确定搜索范围：
  │     ├─ 指定 partition_names → 只搜索指定 Partition 的文件（Partition Pruning）
  │     └─ 未指定 → 搜索所有 Partition
  │
  │  ── 阶段 1：收集数据 ──
  ├─ 1. 从 MemTable 收集活跃记录（按目标 Partition 过滤）
  ├─ 2. 从 Manifest 获取目标 Partition 的文件列表，读取数据 Parquet 文件
  │
  │  ── 阶段 2：构建 bitmap ──
  ├─ 3. 按主键去重（保留最大 _seq），标记重复行为无效
  ├─ 4. 用 deleted_map 标记已删除行为无效
  ├─ 5.（将来）标量过滤（bitmap 管线预留扩展点）
  │
  │  ── 阶段 3：向量检索 ──
  ├─ 6. 只对 bitmap 中有效的行计算距离
  └─ 7. 返回 List[List[dict]]（外层=每个查询向量，内层=top-K 结果）
       每条结果: {"id": pk_value, "distance": float, "entity": {field: value, ...}}
```

## 7. 磁盘文件结构

```
root_dir/                                   # DB 根目录
  my_app/                                   # DB 名称
    documents/                              # Collection 名称
      manifest.json                         # 全局状态快照（原子替换更新）
      schema.json                           # Collection Schema 定义（含版本号）
      wal/                                  # WAL（Collection 级共享）
        wal_data_000001.arrow               #   数据 WAL (Arrow IPC)，flush 后整个删除
        wal_delta_000001.arrow              #   删除 WAL (Arrow IPC)，flush 后整个删除
      _default/                             # Partition: _default（默认，不可删除）
        data/
          data_000001_000500.parquet        #   数据文件，写一次不可变
          data_000501_001000.parquet
        deltas/
          delta_000501_000503.parquet       #   Delta Log，写一次不可变
      2024_Q1/                              # Partition: 2024_Q1
        data/
          data_000001_003000.parquet
          data_003001_008000.parquet
        deltas/
          delta_008001_008002.parquet
      2024_Q2/                              # Partition: 2024_Q2
        data/
          data_005001_010000.parquet
        deltas/
          (空)
    images/                                 # 另一个 Collection
      manifest.json
      schema.json
      wal/
      _default/
        data/
        deltas/
```

**层级规则**：
- **DB** → 目录，纯命名空间
- **Collection** → 目录，包含 `manifest.json` + `schema.json` + `wal/` + 各 Partition 子目录
- **Partition** → 目录，包含 `data/` + `deltas/`，Drop Partition = 删除整个子目录 + 更新 Manifest
- **WAL** → Collection 级共享，不按 Partition 拆分

数据文件和 Delta Log 的生命周期：**创建 → 不可变 → 整个删除**，无例外。
`manifest.json` 和 `schema.json` 通过 **write-tmp + rename** 原子替换更新。

## 8. 代码目录结构

```
lite-v2/
├── MVP.md
├── litevecdb/
│   ├── __init__.py
│   ├── db.py               # DB 层：管理多个 Collection 的生命周期
│   ├── collection.py       # Collection 层：_seq 分配、WAL/MemTable 调度、Partition 管理
│   ├── schema.py           # CollectionSchema / FieldSchema / DataType
│   │                       # Schema 校验、Arrow Schema 生成、schema.json 读写
│   ├── manifest.py         # Manifest 管理（加载、原子保存、Partition 级文件列表）
│   ├── memtable.py         # MemTable（insert_buf + delete_buf，含 _partition 路由）
│   ├── wal.py              # WAL 实现（Arrow IPC Streaming，双文件，Collection 级共享）
│   ├── sstable.py          # 数据文件 (Parquet) 读写
│   ├── delta_log.py        # Delta Log 管理（读写 + 内存 deleted_map）
│   ├── compaction.py       # Compaction Manager（Size-Tiered，按 Partition 独立执行）
│   └── search.py           # 向量检索（暴力扫描 + bitmap 管线 + Partition Pruning）
├── tests/
│   ├── test_schema.py      # Schema 定义、校验、Arrow 转换
│   ├── test_manifest.py    # Manifest 加载、保存、Partition 级文件管理
│   ├── test_memtable.py
│   ├── test_wal.py
│   ├── test_sstable.py
│   ├── test_delta_log.py
│   ├── test_compaction.py
│   ├── test_search.py
│   ├── test_collection.py  # Collection 级端到端测试
│   └── test_db.py          # 多 DB / 多 Collection 测试
├── pyproject.toml
└── requirements.txt
```

## 9. 内部引擎 API

内部引擎 API 面向实现，输入已规范化（始终 List）。后续会在引擎之上加 gRPC 适配层，该层负责 Milvus 协议兼容（参数规范化、表达式解析、返回值包装），使 pymilvus 可直接连接。

### 9.1 API 总览

```python
class LiteVecDB:
    """DB 层：管理多个 Collection 的生命周期"""

    def __init__(self, root_dir: str): ...
    def create_collection(self, collection_name: str, schema: CollectionSchema) -> Collection: ...
    def get_collection(self, collection_name: str) -> Collection: ...
    def drop_collection(self, collection_name: str): ...
    def list_collections(self) -> List[str]: ...
    def close(self): ...


class Collection:
    """Collection 层：引擎核心，管理 WAL / MemTable / Manifest / Compaction"""

    # ─── 写操作（partition_name: 单数 str，写入一个 Partition）───
    def insert(self, records: List[dict], partition_name: str = "_default") -> List:
        """批量写入。PK 已存在则覆盖（upsert 语义）。返回写入的 PK 列表。"""

    def delete(self, pks: List, partition_name: str = None) -> int:
        """批量删除。partition_name=None 则跨所有 Partition。返回处理的 PK 数。
        多条 PK 共享同一个 _seq。"""

    # ─── 读操作（partition_names: 复数 List[str]，可跨多个 Partition）───
    def get(self, pks: List, output_fields: List[str] = None,
            partition_names: List[str] = None) -> List[dict]:
        """按 PK 批量查询。未找到的 PK 不在返回列表中。"""

    def search(self, vectors: List[list], top_k: int = 10,
               metric_type: str = "COSINE",
               partition_names: List[str] = None) -> List[List[dict]]:
        """向量检索。返回外层=每个查询向量，内层=top-K 结果。
        每条结果: {"id": pk, "distance": float, "entity": {field: value}}"""

    # ─── Partition 管理 ───
    def create_partition(self, partition_name: str): ...
    def drop_partition(self, partition_name: str): ...
    def list_partitions(self) -> List[str]: ...

    # ─── Schema 变更 ───
    def add_field(self, field: FieldSchema): ...
```

### 9.2 使用示例

```python
from litevecdb import LiteVecDB, CollectionSchema, FieldSchema, DataType

# ═══ DB ═══
db = LiteVecDB(root_dir="./my_data")

# ═══ Collection ═══
schema = CollectionSchema(
    fields=[
        FieldSchema("doc_id", DataType.VARCHAR, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128),
        FieldSchema("source", DataType.VARCHAR, nullable=True),
        FieldSchema("score", DataType.FLOAT, nullable=True),
    ],
    enable_dynamic_field=True,
)
col = db.create_collection(collection_name="documents", schema=schema)

# ═══ Partition ═══
col.create_partition("2024_Q1")

# ═══ Insert（输入始终是 List[dict]）═══
col.insert(
    records=[{"doc_id": "doc_1", "embedding": [0.1, 0.2, ...], "source": "wiki"}],
)
col.insert(
    records=[
        {"doc_id": "doc_3", "embedding": [...], "source": "arxiv", "score": 0.95},
        {"doc_id": "doc_4", "embedding": [...], "source": "web"},
    ],
    partition_name="2024_Q1",
)
# 动态字段（Schema 外的字段自动存入 $meta）
col.insert(records=[{"doc_id": "doc_5", "embedding": [...], "source": "wiki",
                     "category": "science", "tags": ["ml", "ai"]}])
# Upsert：相同 PK 再次 insert 即覆盖
col.insert(records=[{"doc_id": "doc_1", "embedding": [0.3, 0.4, ...], "source": "updated"}])

# ═══ Delete（输入始终是 List[pk]）═══
col.delete(pks=["doc_1"])                                    # partition=None → 跨所有 Partition
col.delete(pks=["doc_2", "doc_3"], partition_name="2024_Q1") # 指定 Partition

# ═══ Get ═══
records = col.get(pks=["doc_1"])
records = col.get(pks=["doc_1", "doc_2"], output_fields=["doc_id", "source"],
                  partition_names=["2024_Q1"])

# ═══ Search ═══
results = col.search(vectors=[[0.1, 0.2, ...]], top_k=5, metric_type="COSINE")
results = col.search(vectors=[[0.1, 0.2, ...]], top_k=5, metric_type="L2",
                     partition_names=["2024_Q1"])

# ═══ 关闭 ═══
db.close()
```

### 9.3 写操作 vs 读操作的 Partition 参数约定

| 操作类型 | 参数名 | 类型 | 语义 |
|---------|--------|------|------|
| **写操作** (insert/delete) | `partition_name` | `Optional[str]` | 目标 Partition（单数，写入一个 Partition） |
| **读操作** (get/search) | `partition_names` | `Optional[List[str]]` | 搜索范围（复数，可跨多个 Partition） |

### 9.4 gRPC 适配层（Phase 10 已落地）

Phase 10 在内部引擎之上构造 gRPC 服务层，让 pymilvus 客户端无需修改代码即可连接：

```
pymilvus ──gRPC──→ [ litevecdb/adapter/grpc/ ] ──→ [ 内部引擎 ]

适配层职责（详见 plan/grpc-adapter-design.md）：
├─ Milvus Insert/Upsert RPC  →  engine.insert(records, partition_name)
├─ Milvus Delete(ids=) RPC   →  engine.delete(pks, partition_name)
├─ Milvus Delete(filter=) RPC→  query(filter) → 提取 PK → engine.delete(pks)
├─ Milvus Get RPC            →  engine.get(pks, ...)
├─ Milvus Query RPC          →  engine.query(filter, output_fields, limit)
├─ Milvus Search RPC         →  解析 search_params → engine.search(vectors, top_k, expr, output_fields)
├─ Milvus CreateIndex RPC    →  engine.create_index(field, params)
├─ Milvus LoadCollection RPC →  engine.load()
├─ Milvus ReleaseCollection  →  engine.release()
├─ FieldData ↔ records 列行转置（translators/records.py）
├─ 错误码翻译（LiteVecDBError → grpc Status code）
└─ 不支持的 RPC 返回 UNIMPLEMENTED + 友好消息（不 silent fail）

启动方式：
$ python -m litevecdb.adapter.grpc --data-dir ./data --port 19530
```

**架构原则**：适配层只做协议翻译，不增加 engine 能力。pymilvus 兼容性边界见 §10。

## 10. MVP 边界与限制

**包含：**
- **DB → Collection → Partition** 三层数据组织
- **内部引擎 API**：`insert(records)` / `delete(pks)` / `get(pks)` / `search(vectors)`，输入已规范化（始终 List）
- **Insert 天然 upsert 语义**：PK 存在则覆盖（Collection 级 PK 唯一）
- **Delete 支持全局删除**：`partition_name=None` 时跨所有 Partition
- Collection Schema 模型（类型化字段、主键约束、动态字段，name 不在 Schema 内）
- Partition 支持（创建 / 删除 / 列出，`_default` 不可删除）
- Partition Pruning（搜索时跳过无关 Partition 的文件）
- Schema 版本管理与持久化
- Arrow IPC Streaming WAL（双文件：wal_data + wal_delta，Collection 级共享）
- Manifest 全局状态管理（Partition 级文件列表、原子更新、崩溃恢复）
- 单进程、单线程安全的读写（`threading.Lock`）
- WAL + Manifest 保证崩溃恢复
- 插入数据与删除记录分离（数据文件 + Delta Log）
- 扁平文件组织 + Size-Tiered Compaction（按 Partition 独立执行）
- Bitmap 管线（去重 + 删除过滤 → 向量检索）
- Cosine / L2 / Inner-Product 距离
- ✅ **标量过滤**（Phase 8）—— Milvus-style 表达式（比较 / IN / AND / OR / NOT / LIKE / 算术 / IS NULL / `$meta` 动态字段）+ filter LRU cache + `query()` 公开方法 + hybrid backend 优化
- ✅ **FAISS HNSW 向量索引**（Phase 9）—— per-segment 索引 + IDSelectorBitmap pre-filter + load/release 状态机 + index 持久化；fallback 到 BruteForceIndex
- ✅ **gRPC 适配层**（Phase 10）—— pymilvus 客户端无需改代码即可连接 LiteVecDB

**不包含（后续迭代）：**
- Auto ID（自动生成主键）
- 表达式过滤删除（pymilvus `delete(filter=)` MVP 走 query→delete 间接路径）
- Partition Key（自动哈希分区，当前只支持手动指定 Partition）
- IVF / IVF-PQ / OPQ 等量化向量索引（Phase 9 只做 HNSW）
- Sparse / Binary / Float16 / BFloat16 向量类型
- 多向量字段（一个 Collection 多个 vector 列）
- Hybrid Search（多向量召回）
- Snapshot 快照（见下方迭代规划）
- Bloom Filter 加速点查定位
- 多线程后台 Compaction
- 异步 index build（Phase 9 是 flush 同步内联）
- 多进程并发访问
- 分布式支持
- 认证 / RBAC（嵌入式默认无）
- Backup / Restore RPC

### 存储层为后续特性预留的基础

当前存储设计已为以下特性提供底层支撑，无需改动文件格式：

| 特性 | 依赖的存储层基础 | 状态 |
|---------|----------------|---|
| FAISS 索引 | 数据文件不含删除标记，可直接对文件建索引；segment-level 索引与 immutable 架构天然匹配 | ✅ Phase 9 |
| 标量过滤 | 类型化字段 + bitmap 管线 filter_mask 扩展点，Parquet 可做谓词下推 | ✅ Phase 8 |
| gRPC 适配层 | engine API 输入已规范化（List[dict] / List[pk]），翻译层只做协议包装 | ✅ Phase 10 |
| Snapshot | `_seq` 提供时间锚点，Manifest 提供状态快照，文件不可变可被多快照共享 | TODO |
| Schema 变更 | Parquet schema evolution，旧文件缺失列自动填 null | TODO |
| Partition Key | Manifest 已按 Partition 组织文件，只需加哈希路由逻辑 | TODO |
| IVF / IVF-PQ 量化索引 | Phase 9 的 VectorIndex protocol 已为多种 FAISS index_type 留好接口 | TODO |
| 多向量字段 | Schema 改造 + 多 .idx 文件命名约定 | TODO |

### pymilvus 兼容性边界（Phase 10）

| pymilvus 调用 | 状态 | 备注 |
|---|---|---|
| `connect / disconnect` | ✅ | gRPC server 模式 |
| `create_collection / drop_collection / has_collection / describe_collection / list_collections` | ✅ | 直接映射 |
| `create_partition / drop_partition / has_partition / list_partitions` | ✅ | 直接映射 |
| `insert / upsert` | ✅ | engine insert 已是 upsert 语义，两个 RPC 共享同一实现 |
| `delete(ids=...)` | ✅ | 直接映射 |
| `delete(filter=...)` | ⚠️ | servicer 内部 query → delete 间接实现 |
| `get / query` | ✅ | 直接映射 |
| `search(filter, output_fields, top_k)` | ✅ | output_fields 完整支持；filter 透传到 Phase 8 |
| `create_index / drop_index / describe_index` | ✅ | HNSW + BruteForce |
| `load_collection / release_collection / get_load_state` | ✅ | 完整状态机 |
| `flush / compact` | ✅ | 直接映射 |
| `get_collection_stats` | ✅ | row_count |
| `hybrid_search`（多向量） | ❌ UNIMPLEMENTED | engine 不支持多向量字段 |
| `search_iterator / pagination` | ❌ UNIMPLEMENTED | engine 暂无 offset 支持 |
| Aliases (`create_alias` 等) | ❌ UNIMPLEMENTED | |
| Backup / Restore | ❌ UNIMPLEMENTED | |
| RBAC / User / Role / Privilege | ⚠️ stub OK | 嵌入式默认单用户，stub 返回成功避免 pymilvus crash |
| Resource Group / Replica | ❌ UNIMPLEMENTED | 单进程不需要 |
| `list_databases / using_database` | ⚠️ stub | 永远是 `default` |
| Sparse / Binary / Float16 / BFloat16 vector | ❌ | engine 只支持 FLOAT_VECTOR |
| `json_contains / array_contains / text_match` | ❌ FilterUnsupportedError | Phase F3 todo |

## 11. 后续迭代规划：Snapshot

### 概述

Snapshot 是数据库在某个 `_seq` 时刻的一致性只读视图。基于 Manifest 设计，Snapshot **不需要拷贝数据**，本质上就是**保存一份当时的 Manifest**。

### 实现方式

有了 Manifest，Snapshot 的实现变得非常自然：

**创建快照** = 复制当前 Manifest 并标记 `_seq` 上界：

```python
# 持久化为 data_dir/snapshots/{snap_name}.manifest.json
{
    "name": "snap_001",
    "seq": 1500,                    # 快照时刻的 _seq 上界
    "manifest_version": 42,          # 创建时的 Manifest 版本
    "partitions": {                  # 从当时的 Manifest 复制（按 Partition 组织）
        "_default": {
            "data_files": ["_default/data_000001_000500.parquet"],
            "delta_files": ["_default/delta_000501_000503.parquet"]
        },
        "2024_Q1": {
            "data_files": ["2024_Q1/data_000001_003000.parquet"],
            "delta_files": []
        }
    }
}
```

**文件引用计数**：

- 快照创建时：对引用的文件引用计数 +1
- 快照释放时：引用计数 -1
- Compaction 删除旧文件前：检查引用计数，> 0 则跳过

**Snapshot 读取路径**：

```
snapshot.search(query_vector, top_k)
  │
  ├─ 1. 只读该快照记录的 data_files（来自快照 Manifest）
  ├─ 2. 每个文件内按 _seq <= snapshot_seq 过滤
  ├─ 3. 从 delta_files 按 _seq <= snapshot_seq 构建临时 deleted_map
  ├─ 4. 构建 bitmap → 向量检索（复用同一套管线）
  └─ 5. 返回结果
```

不使用全局 `deleted_map`（它只反映最新状态），而是从 delta 文件按时间窗口重建。

**Compaction 适配**：

- 有活跃快照时，被快照引用的文件不可删除
- Compaction 产出新文件后，旧文件仅在引用计数归零时才物理删除

### 对 MVP 存储层的要求（已满足）

| 要求 | 当前状态 |
|------|---------|
| `_seq` 全局递增，可作为时间锚点 | 已有 |
| Manifest 记录完整文件列表，可作为快照基础 | 已有 |
| 文件不可变，可被多快照共享 | 已有 |
| Delta Log 与数据文件分离，可独立按 _seq 过滤 | 已有 |
| Parquet 支持 `_seq <= S` 谓词下推 | 已有 |
| Compaction 可选择性跳过文件 | 需加引用计数，不影响文件格式 |

## 12. 依赖

```
pyarrow >= 14.0
numpy >= 1.24
```

## 13. 实现优先级

| 阶段 | 内容                                          |
| ---- | --------------------------------------------- |
| P0   | Collection Schema（定义、校验、Arrow Schema 生成、持久化） |
| P1   | Manifest（加载、原子保存、Partition 级文件列表管理）  |
| P2   | MemTable（Schema 驱动、含 _partition 路由）、WAL（Arrow IPC 双文件）|
| P3   | 数据文件读写、Delta Log 读写、Flush 流程（按 Partition 拆分 + Manifest 更新）|
| P4   | Partition 管理（create / drop / list，_default 自动创建）|
| P5   | Insert / Delete / Get 端到端打通（内部引擎 API）        |
| P6   | 向量检索 (Brute-Force + bitmap 管线 + Partition Pruning)|
| P7   | Compaction Manager（Size-Tiered，按 Partition 独立 + Manifest 更新）|
| P8   | DB 层 + Collection 层（多 DB、多 Collection 管理）|
| P9   | 崩溃恢复（Manifest 加载 + WAL 重放 + deleted_map 重建）|
| P10  | Schema 变更（add_field + 版本号递增）            |
| P11  | 端到端测试与边界用例                             |
