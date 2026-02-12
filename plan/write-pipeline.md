# LiteVecDB 写链路详解

## 1. 概览

写链路是 LiteVecDB 的核心数据通路，所有数据变更（Insert / Update / Delete）都经过同一条流水线。

```
用户调用                    WAL                MemTable              Flush                    磁盘文件
────────  →  规范化+校验+seq  →  持久化(Arrow IPC)  →  写内存缓冲区  →  (满了触发)  →  按Partition拆分 → Parquet + Manifest
            + 解析partition_name (Collection级共享)     (携带_partition)                (每个Partition独立目录)
```

### 写链路涉及的组件

| 组件 | 职责 | 生命周期 |
|------|------|---------|
| **SeqAllocator** | 分配全局递增 `_seq` | 常驻内存，从 Manifest 恢复 |
| **Schema Validator** | 校验字段类型、主键非空、向量维度 | 常驻内存 |
| **WAL** | 持久化未 flush 的写操作（崩溃恢复用） | 每轮 flush 后删除，新建下一个 |
| **MemTable** | 内存缓冲区，积攒写入直到 flush | 每轮 flush 后冻结，新建下一个 |
| **Flusher** | 将 MemTable 转为 Parquet 落盘 | flush 时创建，完成后销毁 |
| **Manifest** | 全局状态快照，原子记录文件列表 | 常驻磁盘，原子替换更新 |
| **DeltaLog** | 管理删除记录和内存 deleted_map | 常驻内存 + 磁盘 |

## 2. 写操作类型

```
insert(records=[{...}, {...}])         →  数据通路  →  wal_data + insert_buf → 数据 Parquet
delete(pks=["doc_1", "doc_2"])         →  删除通路  →  wal_delta + delete_buf → Delta Parquet
```

**关键设计**：
- Insert 和 Delete 走两条独立的数据流，落盘到不同文件
- **内部引擎只有 `insert()` 和 `delete()` 两个写方法**，输入始终是 List（参数规范化由上层 gRPC 适配层处理）
- `insert()` 天然具备 upsert 语义：PK 存在则覆盖（MemTable dict 天然去重）
- 批量 delete 内所有 PK **共享一个 _seq**
- 每条 insert 记录分配**独立 _seq**（每条数据需要独立的版本号用于去重）

## 3. Insert 完整流程

内部引擎只有 `insert()` 一个写入方法，天然具备 upsert 语义。输入始终是 `List[dict]`。

```python
def insert(self, records: List[dict], partition_name: str = "_default"):
    """
    批量写入。PK 已存在时直接覆盖（upsert 语义，MemTable dict 天然去重）。
    records: 已规范化的 list of dicts（上层负责 dict→list 转换）。
    返回写入的 PK 列表。
    """
```

### Step 1: 解析目标 Partition

```
partition_name = "_default"（未指定时）或显式传入的 Partition 名
├─ 检查 Partition 是否存在于 Manifest
│   → 不存在 → 抛异常
└─ 存在 → 继续
```

### Step 2: Schema 校验（对 records 中每条记录）

```
输入: record = {doc_id: "doc_1", embedding: [0.1, 0.2, ...], source: "wiki", tags: ["ml"]}
                                                                              ↑ 动态字段
├─ 检查主键字段存在且非空
│   → record["doc_id"] 存在 ✓, 不为 None ✓
├─ 检查向量字段存在且维度正确
│   → record["embedding"] 存在 ✓, len == 128 ✓
├─ 检查标量字段类型匹配
│   → record["source"] is str ✓
└─ 分离动态字段（Schema 外的字段 → $meta JSON）
    → schema_fields = {doc_id, embedding, source}
    → dynamic_fields = {tags: ["ml"]}
    → $meta = '{"tags": ["ml"]}'
```

**校验失败**：立即抛异常，不分配 _seq，不写 WAL，不修改任何状态。

### Step 3: 分配 _seq

```python
with self.lock:
    self._seq += 1
    seq = self._seq
```

- `_seq` 是全局严格递增的 uint64 计数器
- 启动时从 `Manifest.current_seq` 恢复
- 在锁内分配，保证单调递增

### Step 4: 写 WAL (wal_data)

```
数据格式：Arrow IPC RecordBatch（批量时 N 行，一次 IO）

RecordBatch = {
    _seq:       uint64    → [3001, 3002]                 ← 每条记录独立 _seq
    _partition: string    → ["_default", "_default"]     ← 标记目标 Partition
    doc_id:     string    → ["doc_1", "doc_2"]
    embedding:  list<f32> → [[0.1, ...], [0.3, ...]]
    source:     string    → ["wiki", "arxiv"]
    $meta:      string    → ['{"tags": ["ml"]}', '{}']
}

写入方式：
  writer = pa.ipc.new_stream(wal_data_path, wal_data_schema)
  writer.write_batch(record_batch)    # 批量：1 个 RecordBatch 含 N 行（减少 IO）
  # 文件追加写，不 fsync 每条（批量 fsync 在 flush 前）
```

**WAL 写入是持久化保障点**：数据到达 WAL 后，即使进程崩溃也可恢复。
**`_partition` 只存在于 WAL 和 MemTable 中**，不写入最终 Parquet 文件（由目录隔离体现）。

### Step 5: 写 MemTable (insert_buf)

```python
# MemTable 内部结构（携带 _partition）
insert_buf = {
    "doc_1": {"_seq": 3001, "_partition": "_default", "doc_id": "doc_1", "embedding": [...], ...},
    "doc_2": {"_seq": 3002, "_partition": "2024_Q1", ...},
    ...
}

# 写入逻辑（逐条写入，维护 PK 去重语义 → 天然实现 upsert）
def put(self, _seq, _partition, **fields):
    pk = fields[self.pk_name]         # "doc_1"
    record = {"_seq": _seq, "_partition": _partition, **fields}
    self.insert_buf[pk] = record      # 相同 PK 直接覆盖（upsert 语义）
    self.delete_buf.pop(pk, None)     # 如果之前有 delete，清除（Insert 优先级 > Delete）
```

**MemTable 特性**：
- dict 按 PK 去重，天然实现 upsert（后写覆盖先写，PK Collection 级唯一）
- Insert 后清除同 PK 的 Delete 记录（最后一次操作获胜）
- 每条记录携带 `_partition`，flush 时按 Partition 拆分输出

### Step 6: 检查是否需要 Flush

```python
if self.memtable.size() >= MEMTABLE_SIZE_LIMIT:  # 默认 10000（跨 Partition 合计）
    self._trigger_flush()
```

- `size() = len(insert_buf) + len(delete_buf)`（跨所有 Partition 合计）
- 未达阈值 → 直接返回，写入完成
- 达到阈值 → 进入 Flush 流程（见 Section 5），按 Partition 拆分输出

### 完整时序图

```
Caller          Collection      WAL             MemTable        Disk
  │                │              │                │              │
  │──insert(records)─→            │                │              │
  │                │─validate──→  │                │              │
  │                │  (schema)    │                │              │
  │                │              │                │              │
  │                │─alloc_seq──→ │                │              │
  │                │ (per record) │                │              │
  │                │              │                │              │
  │                │──write_insert()──→            │              │
  │                │              │ 1 batch (N行)  │              │
  │                │              │                │              │
  │                │──────for rec in records────→  │              │
  │                │              │                │ buf[pk]=rec  │
  │                │              │                │ (upsert语义) │
  │                │              │                │              │
  │                │─check_size── │                │              │
  │                │  (< 10000)   │                │              │
  │ ←── pk_list ───│              │                │              │
  │                │              │                │              │
```

## 4. Delete 完整流程

输入始终是 `List[pk]`（参数规范化由上层处理）。

```python
def delete(self, pks: List, partition_name: str = None) -> int:
    """
    批量删除。
    - pks: PK 列表（已规范化，始终是 list）
    - partition_name: 可选，None 则跨所有 Partition 删除
    - 多条 PK 共享同一个 _seq
    返回处理的 PK 数量。
    """
```

**注意**：Delete 不做"记录是否存在"的检查。PK 不存在也能 delete，delta log 里的无效记录在 compaction 时自然清理。

### Step 1: 分配共享 _seq

```python
with self.lock:
    self._seq += 1
    shared_seq = self._seq   # 不管 1 条还是 N 条，整批共享一个 _seq
```

**为什么共享 _seq？**
- 语义上：一次 delete 调用是一个原子操作，所有 PK 在同一时刻被删除
- 正确性：版本判定 `delta_seq > data_seq` 按 PK 独立比较，共享 _seq 不影响正确性
- 效率：N 条删除只消耗 1 个 _seq，不浪费序号空间
- 对齐 Milvus：Milvus 同一批次的实体共享同一 TSO 时间戳

### Step 2: 解析目标 Partition

```
partition_name 处理:
├─ 指定 partition_name → 使用该 Partition（delta 文件写入该 Partition 目录）
└─ 未指定（None）→ 不绑定 Partition
    → MemTable 中标记为特殊值（如 "_all"），flush 时分发到全局 delta
    → 搜索/读取时 deleted_map 全局匹配，不受 Partition 限制
```

### Step 3: 写 WAL (wal_delta)

```
数据格式：Arrow IPC RecordBatch (N 行，一次 IO)

RecordBatch = {
    doc_id:     string → ["doc_1", "doc_2", "doc_3"]   # N 个主键
    _seq:       uint64 → [3005,    3005,    3005]       # 共享同一个 _seq
    _partition: string → ["2024_Q1","2024_Q1","2024_Q1"]# 目标 Partition（可为 "_all"）
}

写入 wal_delta 文件（追加，单个 RecordBatch）
```

### Step 4: 写 MemTable (delete_buf)

```python
# MemTable 内部结构
delete_buf = {
    "doc_1": (3005, "2024_Q1"),   # pk_value → (_seq, _partition)
    "doc_3": (3007, "_all"),      # _all 表示跨所有 Partition
}

# 逐条写入 MemTable（维护 PK 去重语义）
for pk in pks:
    self.memtable.delete(pk, shared_seq, partition_name or "_all")
    # delete_buf[pk] = (shared_seq, partition)
    # insert_buf.pop(pk, None)  # 清除同 PK 的 insert（Delete 优先级 > Insert）
```

### Step 5: 检查是否需要 Flush

同 Insert 逻辑。`size()` 增加了 N（每个 PK 一条 delete_buf 条目）。

### 4.1 _seq 分配策略对比

```
操作                               _seq 分配          Delta Log 记录               _seq 消耗
──────────────────────────────────────────────────────────────────────────────────────────
delete(pks=["A"])                  seq=100           (A, 100)                     1
delete(pks=["B"])                  seq=101           (B, 101)                     1

delete(pks=["A","B","C"])          shared_seq=102    (A, 102), (B, 102), (C, 102) 1（共享）

insert(records=[rec1, rec2])       各自独立 seq      每行独立 _seq                 N
                                   seq=103,104
```

**注意 Insert 和 Delete 的不对称**：
- `insert(records=list)`：每条记录分配独立 _seq（因为每条数据需要独立的版本号用于去重）
- `delete(pks=list)`：共享 _seq（删除只需要 > 被删数据的 _seq 即可，不需要互相区分）

### Delete 时序图

```
Caller          Collection      WAL             MemTable
  │                │              │                │
  │──delete(pks)──→│              │                │
  │                │              │                │
  │                │─alloc_seq──→ │                │
  │                │ (shared=3005)│                │
  │                │              │                │
  │                │──write_delete()──→            │
  │                │              │ 1 batch (N行)  │
  │                │              │                │
  │                │──────────for pk in pks─────→  │
  │                │              │                │ buf[A]=3005
  │                │              │                │ buf[B]=3005
  │                │              │                │ buf[C]=3005
  │                │              │                │
  │ ←── count: N ──│              │                │
```

## 5. Flush 流程（核心）

Flush 是写链路中最复杂的环节：将内存数据转为不可变磁盘文件，同时保证崩溃安全。

### 触发条件

```
MemTable.size() >= MEMTABLE_SIZE_LIMIT (10000)
```

### Flush 完整步骤

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Flush Pipeline                               │
│                                                                     │
│  Step 1: 冻结 MemTable                                              │
│  ─────────────────                                                  │
│  frozen_mt = self.memtable          # 冻结当前 MemTable             │
│  frozen_wal = self.wal              # 冻结当前 WAL                  │
│  self.memtable = MemTable(schema)   # 创建新 MemTable 接收后续写入   │
│  self.wal = WAL(new_wal_paths)      # 创建新 WAL                    │
│                                                                     │
│  Step 2: MemTable → 按 Partition 拆分 Arrow Table                    │
│  ─────────────────────────────────────────                          │
│  partition_tables = frozen_mt.flush()                               │
│  # 返回: {partition_name: (data_table, delta_table)}                │
│  # 例如: {"_default": (table_100rows, table_3rows),                │
│  #        "2024_Q1": (table_50rows, None)}                         │
│                                                                     │
│  Step 3: Arrow Table → Parquet 文件（按 Partition 写入对应目录）       │
│  ──────────────────────────────────────────────                     │
│  for partition, (data_table, delta_table) in partition_tables:      │
│      if data_table:                                                 │
│          path = f"{partition}/data/{seq_min}_{seq_max}.parquet"     │
│          pq.write_table(data_table, path)                           │
│      if delta_table:                                                │
│          path = f"{partition}/deltas/{seq_min}_{seq_max}.parquet"   │
│          pq.write_table(delta_table, path)                          │
│                                                                     │
│  Step 4: 更新内存 deleted_map                                        │
│  ────────────────────────                                           │
│  for partition, (_, delta_table) in partition_tables:               │
│      if delta_table:                                                │
│          for pk, seq in delta_table:                                │
│              deleted_map[pk] = max(deleted_map.get(pk, 0), seq)     │
│                                                                     │
│  Step 5: 原子更新 Manifest                                          │
│  ────────────────────────                                           │
│  for partition, files in new_files:                                 │
│      manifest.partitions[partition].data_files.append(...)          │
│      manifest.partitions[partition].delta_files.append(...)         │
│  manifest.current_seq = new_max_seq          # 更新 _seq 水位       │
│  manifest.active_wal = new_wal_paths         # 切换到新 WAL         │
│  manifest.save()  # write-tmp + rename 原子替换                     │
│                                                                     │
│  Step 6: 删除旧 WAL 文件                                            │
│  ────────────────────                                               │
│  frozen_wal.close_and_delete()               # 删除 wal_data + wal_delta │
│                                                                     │
│  Step 7: 触发 Compaction 检查（按 Partition 独立）                    │
│  ──────────────────────────────────────                             │
│  for partition in partition_tables:                                 │
│      compaction_manager.maybe_compact(partition)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Flush 时序图

```
DB Engine       MemTable(frozen)    Parquet Writer    Manifest       WAL(old)     Compaction
  │                  │                   │               │              │             │
  │──freeze()──→     │                   │               │              │             │
  │  (swap MT+WAL)   │                   │               │              │             │
  │                  │                   │               │              │             │
  │──flush()───→     │                   │               │              │             │
  │         data_table, delta_table      │               │              │             │
  │  ←───────────────│                   │               │              │             │
  │                  │                   │               │              │             │
  │──────────write_table(data)──────→    │               │              │             │
  │──────────write_table(delta)─────→    │               │              │             │
  │                  │              (fsync)              │              │             │
  │  ←──────────────done─────────────    │               │              │             │
  │                  │                   │               │              │             │
  │──update deleted_map──                │               │              │             │
  │                  │                   │               │              │             │
  │──────────────────────────────────────────save()────→ │              │             │
  │                  │                   │          (atomic rename)     │             │
  │  ←──────────────────────────────────────done────     │              │             │
  │                  │                   │               │              │             │
  │──────────────────────────────────────────────close_and_delete()──→  │             │
  │                  │                   │               │              │             │
  │──────────────────────────────────────────────────────────maybe_compact()────→     │
  │                  │                   │               │              │             │
```

### 数据格式在 Flush 各阶段的变化

```
阶段            格式                    存储位置        示例
──────────────────────────────────────────────────────────────────

用户调用        Python dict             -               {"doc_id": "doc_1", "embedding": [...]}

Schema 校验后   Python dict + $meta     -               {"doc_id": "doc_1", ..., "$meta": '{"tags":["ml"]}'}

WAL            Arrow IPC RecordBatch   wal_data.arrow  二进制 Arrow 流式格式

MemTable       Python dict (by PK)     内存            insert_buf["doc_1"] = {_seq: 3001, ...}

flush()输出    PyArrow Table           内存            pa.Table with N rows, schema-driven columns

Parquet        Parquet 列存            data/*.parquet  不可变，含列统计信息
```

## 6. Flush 崩溃安全分析

Flush 过程中的任意时刻都可能发生崩溃。逐步分析每个阶段崩溃的影响：

### 崩溃点分析

```
Flush Step          崩溃后果                              恢复方式
────────────────────────────────────────────────────────────────────────────────

Step 1: 冻结MT      新 MT 还没写入任何数据，旧 WAL 仍在     从旧 WAL 恢复 → 重建 MemTable
        (swap)

Step 2: MT→Arrow    Arrow Table 还在内存，未落盘             同上，WAL 完整可恢复

Step 3: 写 Parquet  Parquet 文件可能只写了一半               Manifest 未更新，不可见
                    → 变成孤儿文件                          启动时可清理（不在 Manifest 中的文件）

Step 4: 更新        deleted_map 内存状态丢失                 从 delta_files 重建
        deleted_map

Step 5: 更新        两种情况：
        Manifest    a) rename 前崩溃 → Manifest 仍是旧版本   → WAL 重放（数据文件成为孤儿）
                    b) rename 后崩溃 → Manifest 已更新       → Manifest 指向新文件，WAL 重放
                                                              产生重复数据，但 _seq 去重保证正确

Step 6: 删除 WAL    Manifest 已更新，新文件可见               旧 WAL 未删除 → 重放产生重复
                    旧 WAL 残留                              _seq 去重保证正确

Step 7: Compaction  Flush 已完全成功                         Compaction 本身有独立的崩溃安全
```

### 崩溃恢复流程

```
启动
  │
  ├─ 1. 加载 Manifest
  │     ├─ 存在 → 获取 current_seq, 文件列表
  │     └─ 不存在 → 目录扫描（首次或损坏）
  │
  ├─ 2. 扫描 wal/ 目录
  │     ├─ 无 WAL → 正常启动
  │     └─ 有 WAL → 重放 WAL 到 MemTable
  │                  ├─ 读 wal_data → insert_buf
  │                  └─ 读 wal_delta → delete_buf
  │
  ├─ 3. 校验 Manifest 中的文件是否实际存在
  │     └─ 缺失文件 → 从 Manifest 移除（compaction 中途崩溃场景）
  │
  ├─ 4. 清理孤儿文件（在磁盘但不在 Manifest 中的 Parquet）
  │
  ├─ 5. 从 delta_files 重建 deleted_map
  │
  └─ 6. 保存新 Manifest
```

**核心不变量**：Manifest 是唯一的 truth source。不在 Manifest 中的文件可安全删除，在 Manifest 中的文件必须存在。

## 7. 并发控制

### MVP：单锁串行

```python
class Collection:
    def __init__(self):
        self.lock = threading.Lock()

    def insert(self, records: List[dict], partition_name: str = "_default"):
        with self.lock:
            for record in records:
                self._validate(record)
            # 每条记录分配独立 _seq
            for record in records:
                record["_seq"] = self._alloc_seq()
            # WAL: 一次写入整个 RecordBatch（N 行，1 次 IO）
            rb = self._build_insert_batch(records, partition_name)
            self.wal.write_insert(rb)
            # MemTable: 逐条写入（维护 PK 去重 → upsert 语义）
            for record in records:
                self.memtable.put(record["_seq"], partition_name, **record)
            if self.memtable.size() >= LIMIT:
                self._flush()

    def delete(self, pks: List, partition_name: str = None):
        with self.lock:
            shared_seq = self._alloc_seq()             # 整批共享一个 _seq
            # WAL: 一次写入整个 RecordBatch（N 行，1 次 IO）
            part = partition_name or "_all"
            rb = self._build_delete_batch(pks, shared_seq, part)
            self.wal.write_delete(rb)
            # MemTable: 逐条写入（维护 PK 去重）
            for pk in pks:
                self.memtable.delete(pk, shared_seq, part)
            if self.memtable.size() >= LIMIT:
                self._flush()
```

**整个写入操作在锁内完成**：校验 → _seq 分配 → WAL 写入 → MemTable 写入 → (可能触发 Flush)。

### 为什么 Flush 也在锁内？

MVP 选择 Flush 在锁内同步执行（阻塞后续写入），原因：
1. **简单可靠**：无需处理"正在 flush 的冻结 MemTable"和"新 MemTable"的并发读问题
2. **MVP 规模小**：10000 条数据 flush 到 Parquet 通常在毫秒级完成
3. **无并发写入需求**：单线程嵌入式场景，阻塞可接受

### 将来优化方向（不在 MVP）

```
优化                        描述
───────────────────────────────────────────────────
异步 Flush                  冻结 MemTable 后释放锁，后台线程写 Parquet
双 MemTable                 一个 Active 接收写入，一个 Frozen 正在 flush
WAL group commit            积攒多个写入后批量 fsync，减少 IO 次数
```

## 8. 批量写入的 IO 优势

内部引擎 `insert()` 和 `delete()` 始终接收 List，天然支持批量。一次调用 N 条 vs N 次调用各 1 条的 IO 差异：

### Insert: N 条 vs N × 1 条

```
维度             N 次 insert([1条])        1 次 insert([N条])
─────────────────────────────────────────────────────────────────────
锁获取/释放       N 次                     1 次
WAL IO           N 个 RecordBatch         1 个 RecordBatch (N 行)
MemTable 写入    N 次 dict 操作            N 次 dict 操作（相同）
Schema 校验      N 次                     N 次（相同）
_seq 消耗        N 个                     N 个（每条独立 _seq）
```

### Delete: N 条 vs N × 1 条

```
维度             N 次 delete([1个PK])      1 次 delete([N个PK])
─────────────────────────────────────────────────────────────────────
锁获取/释放       N 次                     1 次
WAL IO           N 个 RecordBatch         1 个 RecordBatch (N 行)
MemTable 写入    N 次 dict 操作            N 次 dict 操作（相同）
_seq 消耗        N 个                     1 个（共享 _seq）
```

## 9. WAL 细节

### 文件格式：Arrow IPC Streaming

```
选择 Arrow IPC 而非 JSONL 的原因：

  JSONL:  向量 [0.1, 0.2, ..., 0.1] → 文本编码 → 约 3x 体积膨胀
  Arrow:  向量 [0.1, 0.2, ..., 0.1] → 二进制直写 → 1x 体积，零解析开销
```

### 双 WAL 文件

```
wal/
  wal_data_000001.arrow     # Insert/Update 操作
  wal_delta_000001.arrow    # Delete 操作
```

与 MemTable 的双缓冲区一一对应：
- `wal_data` ↔ `insert_buf`
- `wal_delta` ↔ `delete_buf`

### WAL 生命周期

```
创建                    追加写入                   Flush 成功              删除
────────               ────────                  ────────              ────────
new_stream()     →     write_batch() × N    →    Manifest 更新   →    close_and_delete()
(写入 Schema header)   (追加 RecordBatch)         (记录新 WAL path)    (删除两个 WAL 文件)
```

### WAL 编号轮转

```
第 1 轮写入:  wal_data_000001.arrow + wal_delta_000001.arrow
  ↓ flush 成功，删除
第 2 轮写入:  wal_data_000002.arrow + wal_delta_000002.arrow
  ↓ flush 成功，删除
第 3 轮写入:  wal_data_000003.arrow + wal_delta_000003.arrow
  ...
```

编号从 Manifest 中恢复，确保不会重复。

## 10. MemTable 内存语义

### 操作交织的正确性

```
场景 1: Insert → Delete（同一 PK）
  insert(doc_1, seq=100) → insert_buf[doc_1] = {seq:100, ...}
  delete(doc_1, seq=101) → delete_buf[doc_1] = 101, insert_buf.pop(doc_1)
  结果: doc_1 被删除 ✓

场景 2: Delete → Insert（同一 PK）
  delete(doc_1, seq=100) → delete_buf[doc_1] = 100
  insert(doc_1, seq=101) → insert_buf[doc_1] = {seq:101, ...}, delete_buf.pop(doc_1)
  结果: doc_1 存在，版本 101 ✓

场景 3: Insert → Insert（同一 PK，即 Update）
  insert(doc_1, seq=100) → insert_buf[doc_1] = {seq:100, ...}
  insert(doc_1, seq=101) → insert_buf[doc_1] = {seq:101, ...}  # 覆盖
  结果: doc_1 存在，版本 101 ✓

场景 4: Insert → Delete → Insert（同一 PK）
  insert(doc_1, seq=100) → insert_buf[doc_1] = {seq:100}
  delete(doc_1, seq=101) → delete_buf[doc_1] = 101, insert_buf.pop(doc_1)
  insert(doc_1, seq=102) → insert_buf[doc_1] = {seq:102}, delete_buf.pop(doc_1)
  结果: doc_1 存在，版本 102 ✓
```

```
场景 5: 批量 delete 与 insert 交织
  insert([{doc_1}], seq=100) → insert_buf[doc_1] = {seq:100}
  insert([{doc_2}], seq=101) → insert_buf[doc_2] = {seq:101}
  delete(pks=[doc_1, doc_2], shared_seq=102)
    → delete_buf[doc_1] = 102, insert_buf.pop(doc_1)
    → delete_buf[doc_2] = 102, insert_buf.pop(doc_2)
  结果: doc_1, doc_2 都被删除 ✓

场景 6: 批量 delete 后再 insert 其中一个
  delete(pks=[doc_1, doc_2], shared_seq=100)
    → delete_buf[doc_1] = 100
    → delete_buf[doc_2] = 100
  insert([{doc_1}], seq=101) → insert_buf[doc_1] = {seq:101}, delete_buf.pop(doc_1)
  结果: doc_1 存在(v101), doc_2 被删除 ✓
```

**核心规则**：最后一次操作获胜（Last Write Wins），由 _seq 单调递增保证。批量 delete 内的所有 PK 共享同一 _seq，不影响此规则。

### Flush 输出

```python
def flush(self) -> Tuple[Optional[pa.Table], Optional[pa.Table]]:
    # insert_buf → data_table
    if self.insert_buf:
        rows = list(self.insert_buf.values())
        data_table = pa.Table.from_pylist(rows, schema=self.data_schema)
    else:
        data_table = None

    # delete_buf → delta_table
    if self.delete_buf:
        rows = [{"pk": k, "_seq": v} for k, v in self.delete_buf.items()]
        delta_table = pa.Table.from_pylist(rows, schema=self.delta_schema)
    else:
        delta_table = None

    return data_table, delta_table
```

**输出可能的组合**：

| data_table | delta_table | 含义 |
|------------|-------------|------|
| 有 | 有 | 该轮既有 insert 又有 delete |
| 有 | None | 该轮只有 insert |
| None | 有 | 该轮只有 delete |
| None | None | 不可能（size > 0 才触发 flush） |

## 11. 文件命名与 Manifest 更新

### 文件命名规则

```
数据文件:  {partition}/data/data_{seq_min:06d}_{seq_max:06d}.parquet
Delta 文件: {partition}/deltas/delta_{seq_min:06d}_{seq_max:06d}.parquet
WAL 文件:  wal/wal_data_{N:06d}.arrow / wal/wal_delta_{N:06d}.arrow （Collection 级共享）

其中:
  partition = Partition 名称（如 "_default", "2024_Q1"）
  seq_min   = 该文件中最小的 _seq
  seq_max   = 该文件中最大的 _seq
  N         = WAL 轮次编号（单调递增）
```

### Flush 后 Manifest 变更示例

```
Flush 前 Manifest:
{
    "version": 5,
    "current_seq": 3000,
    "partitions": {
        "_default": {
            "data_files": ["_default/data/data_000001_001000.parquet"],
            "delta_files": []
        },
        "2024_Q1": {
            "data_files": ["2024_Q1/data/data_001001_002000.parquet"],
            "delta_files": ["2024_Q1/deltas/delta_001501_001503.parquet"]
        }
    },
    "active_wal": {"data": "wal_data_000003.arrow", "delta": "wal_delta_000003.arrow"}
}

Flush 后 Manifest (_default 有 300 条 insert，2024_Q1 有 200 条 insert + 3 条 delete):
{
    "version": 6,                                                         ← +1
    "current_seq": 3503,                                                  ← 更新到最新 seq
    "partitions": {
        "_default": {
            "data_files": [
                "_default/data/data_000001_001000.parquet",
                "_default/data/data_002001_002300.parquet"                ← 新增
            ],
            "delta_files": []
        },
        "2024_Q1": {
            "data_files": [
                "2024_Q1/data/data_001001_002000.parquet",
                "2024_Q1/data/data_002301_003500.parquet"                 ← 新增
            ],
            "delta_files": [
                "2024_Q1/deltas/delta_001501_001503.parquet",
                "2024_Q1/deltas/delta_003501_003503.parquet"              ← 新增
            ]
        }
    },
    "active_wal": {"data": "wal_data_000004.arrow", "delta": "wal_delta_000004.arrow"}
}                                                                         ↑ 切换到新 WAL
```

## 12. 端到端示例

```python
# 初始状态：空数据库，_default Partition 自动创建
# Manifest: {version:0, current_seq:0, partitions:{"_default":{data_files:[], delta_files:[]}}}

# === 创建额外 Partition ===
col.create_partition("2024_Q1")

# === 写入 3 条数据（分属不同 Partition） ===
col.insert(records=[{"doc_id": "A", "embedding": [...], "source": "wiki"}])                        # _seq=1, _default
col.insert(records=[{"doc_id": "B", "embedding": [...], "source": "arxiv"}], partition_name="2024_Q1")  # _seq=2, 2024_Q1
col.insert(records=[{"doc_id": "C", "embedding": [...], "source": "web"}],   partition_name="2024_Q1")  # _seq=3, 2024_Q1

# 状态:
# WAL:       wal_data_000001.arrow 有 3 个 RecordBatch
# MemTable:  insert_buf = {A: {seq:1, part:_default}, B: {seq:2, part:2024_Q1}, C: {seq:3, part:2024_Q1}}
# Manifest:  未变 (current_seq 仍为 0)

# === 删除 B 和 C ===
col.delete(pks=["B", "C"], partition_name="2024_Q1")  # shared _seq=4

# 状态:
# MemTable:  insert_buf = {A: {seq:1, part:_default}},
#            delete_buf = {B: (4, 2024_Q1), C: (4, 2024_Q1)}

# === 更新 A（insert 天然 upsert 语义：PK 存在则覆盖） ===
col.insert(records=[{"doc_id": "A", "embedding": [...], "source": "updated"}])  # _seq=5, _default

# 状态:
# MemTable:  insert_buf = {A: {seq:5, part:_default}},
#            delete_buf = {B: (4, 2024_Q1), C: (4, 2024_Q1)}

# === 假设此时触发 Flush (实际由 size >= 10000 触发) ===

# Flush 按 Partition 拆分输出:
# _default:
#   data_table → 1 行: A(seq=5) → _default/data/data_000001_000005.parquet
#   delta_table → None
# 2024_Q1:
#   data_table → None（B、C 已从 insert_buf 清除）
#   delta_table → 2 行: B(seq=4), C(seq=4) → 2024_Q1/deltas/delta_000004_000004.parquet

# Flush 后:
# WAL:         wal_data_000001 + wal_delta_000001 已删除
# MemTable:    新建空 MemTable
# Manifest:    {version:1, current_seq:5,
#               partitions: {
#                 "_default": {data_files:[_default/data/data_000001_000005.parquet], delta_files:[]},
#                 "2024_Q1": {data_files:[], delta_files:[2024_Q1/deltas/delta_000004_000004.parquet]}
#               }}
# deleted_map: {B: 4, C: 4}
```

## 13. 写链路关键不变量（Invariants）

| # | 不变量 | 保证方式 |
|---|--------|---------|
| 1 | `_seq` 严格单调递增 | 锁内分配，原子 +1 |
| 2 | WAL 写入先于 MemTable 写入 | 代码顺序保证 |
| 3 | Parquet 落盘先于 Manifest 更新 | Flush 步骤顺序保证 |
| 4 | Manifest 更新先于 WAL 删除 | Flush 步骤顺序保证 |
| 5 | 落盘文件不可变 | Parquet 只写一次，不 append/modify |
| 6 | Manifest 原子更新 | write-tmp + os.rename |
| 7 | 任意时刻崩溃，数据不丢不错 | WAL + Manifest + _seq 去重 |
| 8 | 同 PK 最后一次操作获胜（upsert 语义） | _seq 单调递增 + Last Write Wins + MemTable dict 去重 |
| 9 | `delete(pks=list)` 内所有 PK 共享同一 _seq | 一次 alloc_seq，N 条记录复用 |
| 10 | Flush 按 Partition 拆分输出，文件落入对应 Partition 目录 | MemTable 携带 _partition，flush() 按 partition 分组 |
| 11 | WAL 和 _seq 在 Collection 级共享，不按 Partition 拆分 | 与 Milvus 一致：Channel 是 Collection 级的 |
| 12 | `insert()` 天然 upsert：PK Collection 级唯一 | MemTable dict 按 PK 去重，相同 PK 直接覆盖 |
| 13 | `delete(partition_name=None)` 跨所有 Partition 删除 | 内部使用 "_all" 标记，deleted_map 全局匹配 |
