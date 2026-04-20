# 深入设计：WAL · Segment · 搜索架构

## 1. 概述

WAL 是 MilvusLite 崩溃安全的核心保障。任何写入在进入 MemTable 之前，必须先持久化到 WAL。
系统崩溃后，通过重放 WAL 可以恢复 MemTable 中未 flush 的数据。

**设计目标**：
- **持久性**：进程崩溃后数据不丢失（OS 级崩溃见 §8 fsync 讨论）
- **低写放大**：Arrow IPC 二进制格式，避免文本编码对向量的 3x 膨胀
- **简单可靠**：只追加写入，不修改已写入内容，文件整个删除
- **快速恢复**：顺序读取，Arrow 零拷贝反序列化

---

## 2. Arrow IPC Streaming 格式

### 2.1 为什么选 Arrow IPC

| | JSONL | Arrow IPC Streaming |
|--|-------|---------------------|
| 向量编码 | 文本 `[0.1, 0.2, ...]` ≈ 3x 膨胀 | 二进制直写，1x |
| 解析开销 | JSON parse + float convert | 零拷贝 mmap / 直接反序列化 |
| Schema 校验 | 无（运行时才发现类型错误） | 文件头自带 Schema，读取时自动校验 |
| 批量 IO | 每条记录一行，N 次 IO | 一个 RecordBatch = 一次 IO |
| 部分写入恢复 | 可按行截断 | 需按 RecordBatch 边界截断（见 §9） |

### 2.2 文件内部结构

```
Arrow IPC Streaming 文件布局：

┌─────────────────────────────────┐
│  Schema Message                 │  ← 文件创建时写入（new_stream）
│  (字段名/类型/元数据)              │
├─────────────────────────────────┤
│  RecordBatch Message #1         │  ← 第 1 次 write_batch
│  (metadata + body)              │
├─────────────────────────────────┤
│  RecordBatch Message #2         │  ← 第 2 次 write_batch
│  (metadata + body)              │
├─────────────────────────────────┤
│  ...                            │
├─────────────────────────────────┤
│  RecordBatch Message #N         │  ← 第 N 次 write_batch
├─────────────────────────────────┤
│  EOS (End-of-Stream) Marker     │  ← close() 时写入
│  (4 bytes: 0x00000000)          │
└─────────────────────────────────┘
```

- **Schema Message**：包含完整的 Arrow Schema（字段名、类型、nullable、metadata）
- **RecordBatch Message**：包含一个 batch 的所有列数据，二进制紧凑排列
- **EOS Marker**：4 字节全零，标记流的正常结束
- **无 Footer**：与 Arrow IPC File 格式不同，Streaming 格式没有 Footer，不支持随机访问

### 2.3 PyArrow 核心 API

```python
import pyarrow as pa

# ── 写入 ──
sink = pa.OSFile(path, "wb")                           # 打开文件
writer = pa.ipc.new_stream(sink, schema)               # 写 Schema Message
writer.write_batch(record_batch)                       # 追加 RecordBatch
writer.close()                                         # 写 EOS + 关闭文件

# ── 读取 ──
source = pa.OSFile(path, "rb")                         # 打开文件
reader = pa.ipc.open_stream(source)                    # 读 Schema Message
for batch in reader:                                   # 逐个读 RecordBatch
    process(batch)
# 或一次性读取：
table = reader.read_all()                              # 所有 batch 合并为 Table
```

---

## 3. 双文件结构

### 3.1 文件分离

```
wal/
  wal_data_000001.arrow     # Insert/Update 操作 → 对应 MemTable.insert_buf
  wal_delta_000001.arrow    # Delete 操作        → 对应 MemTable.delete_buf
```

**为什么分两个文件而不是一个？**

1. **Schema 不同**：data 文件含所有用户字段 + `$meta` + `_seq` + `_partition`；delta 文件只含主键 + `_seq` + `_partition`
2. **延迟初始化**：如果一轮只有 insert 没有 delete，delta 文件不会被创建（反之亦然）
3. **与 MemTable 对称**：MemTable 内部也是 insert_buf / delete_buf 两个独立缓冲区，恢复时一一对应
4. **与 Parquet 对称**：flush 后产出 data Parquet 和 delta Parquet 两种文件

### 3.2 WAL Schema

**wal_data_schema**（插入/更新）：

```
字段           类型                   说明
────           ────                   ────
_seq           uint64                 每条记录独立分配的序列号
_partition     utf8                   目标 Partition 名（恢复时路由用）
{pk_field}     (由 Schema 决定)       主键
{vector_field} list<float32>          向量
{field_1}      ...                    用户自定义字段
{field_N}      ...
$meta          utf8 (nullable)        动态字段 JSON 序列化
```

**wal_delta_schema**（删除）：

```
字段           类型                   说明
────           ────                   ────
{pk_field}     (由 Schema 决定)       主键
_seq           uint64                 批量删除共享同一个 _seq
_partition     utf8                   目标 Partition（"_all" 表示跨所有 Partition）
```

**注意**：WAL schema 比 Parquet schema 多一个 `_partition` 列。因为 WAL 是 Collection 级共享的，
需要记录每条数据属于哪个 Partition，恢复时才能正确路由到 MemTable。Parquet 文件已经按 Partition 目录隔离，所以不需要。

---

## 4. WAL 内部状态模型

### 4.1 实例属性

```python
class WAL:
    wal_dir: str                           # WAL 目录路径
    _wal_data_schema: pa.Schema            # data 文件的 Arrow Schema
    _wal_delta_schema: pa.Schema           # delta 文件的 Arrow Schema
    _number: int                           # 本轮 WAL 编号 N
    _data_writer: Optional[pa.ipc.RecordBatchStreamWriter]   # 延迟初始化
    _delta_writer: Optional[pa.ipc.RecordBatchStreamWriter]  # 延迟初始化
    _data_sink: Optional[pa.OSFile]        # data 文件句柄
    _delta_sink: Optional[pa.OSFile]       # delta 文件句柄
    _closed: bool                          # 是否已关闭
```

### 4.2 状态转换

data_writer 和 delta_writer 是**独立**的，各自有三个状态：

```
         首次 write_insert()            close_and_delete()
  None ──────────────────────→ Active ──────────────────────→ Closed
  (未创建文件)                  (文件已创建, writer 可写)       (文件已关闭+删除)
```

组合状态矩阵：

```
                        delta_writer
                  None      Active     Closed
            ┌──────────┬──────────┬──────────┐
    None    │ 初始态    │ 仅 delete │    -     │
data_       ├──────────┼──────────┼──────────┤
writer Active│ 仅 insert │ 两者都有  │    -     │
            ├──────────┼──────────┼──────────┤
    Closed  │    -     │    -     │ 已关闭    │
            └──────────┴──────────┴──────────┘

注：close_and_delete() 一次性将两个 writer 都转为 Closed
    不存在 data=Active + delta=Closed 的中间状态
```

### 4.3 文件路径

```python
@property
def data_path(self) -> Optional[str]:
    if self._data_writer is None:
        return None
    return os.path.join(self.wal_dir, f"wal_data_{self._number:06d}.arrow")

@property
def delta_path(self) -> Optional[str]:
    if self._delta_writer is None:
        return None
    return os.path.join(self.wal_dir, f"wal_delta_{self._number:06d}.arrow")
```

---

## 5. 核心方法实现

### 5.1 `__init__`

```python
def __init__(self, wal_dir, wal_data_schema, wal_delta_schema, wal_number):
    self.wal_dir = wal_dir
    self._wal_data_schema = wal_data_schema
    self._wal_delta_schema = wal_delta_schema
    self._number = wal_number
    self._data_writer = None
    self._delta_writer = None
    self._data_sink = None
    self._delta_sink = None
    self._closed = False

    os.makedirs(wal_dir, exist_ok=True)
```

- **不创建文件**：延迟到首次写入
- **创建目录**：确保 wal_dir 存在

### 5.2 `write_insert`

```python
def write_insert(self, record_batch: pa.RecordBatch) -> None:
    """
    流程：
    1. 检查状态：已关闭则报错
    2. 延迟初始化：首次调用时创建文件 + writer
    3. 写入 RecordBatch
    """
    assert not self._closed, "WAL already closed"

    # ── 延迟初始化 ──
    if self._data_writer is None:
        path = os.path.join(self.wal_dir, f"wal_data_{self._number:06d}.arrow")
        self._data_sink = pa.OSFile(path, "wb")
        self._data_writer = pa.ipc.new_stream(self._data_sink, self._wal_data_schema)

    # ── 写入 ──
    self._data_writer.write_batch(record_batch)
```

**关键设计点**：

- **不做 Schema 校验**：`write_batch()` 内部已校验，schema 不匹配会抛 `ArrowInvalid`
- **不 flush/fsync**：依赖 OS buffer cache（见 §8 讨论）
- **一个 RecordBatch 对应一次 insert() 调用**：批量 insert N 条 = 1 个含 N 行的 RecordBatch

### 5.3 `write_delete`

```python
def write_delete(self, record_batch: pa.RecordBatch) -> None:
    """与 write_insert 对称，写入 wal_delta 文件。"""
    assert not self._closed, "WAL already closed"

    if self._delta_writer is None:
        path = os.path.join(self.wal_dir, f"wal_delta_{self._number:06d}.arrow")
        self._delta_sink = pa.OSFile(path, "wb")
        self._delta_writer = pa.ipc.new_stream(self._delta_sink, self._wal_delta_schema)

    self._delta_writer.write_batch(record_batch)
```

### 5.4 `close_and_delete`

```python
def close_and_delete(self) -> None:
    """
    调用时机：flush 成功后。
    流程：
    1. 关闭 data_writer（写 EOS + 关闭文件句柄）
    2. 关闭 delta_writer（写 EOS + 关闭文件句柄）
    3. 删除 data 文件
    4. 删除 delta 文件
    5. 标记为 closed
    """
    if self._closed:
        return  # 幂等

    # ── 关闭 writer ──
    if self._data_writer is not None:
        self._data_writer.close()      # 写 EOS marker
        self._data_sink.close()        # 关闭文件句柄
    if self._delta_writer is not None:
        self._delta_writer.close()
        self._delta_sink.close()

    # ── 删除文件 ──
    data_path = os.path.join(self.wal_dir, f"wal_data_{self._number:06d}.arrow")
    delta_path = os.path.join(self.wal_dir, f"wal_delta_{self._number:06d}.arrow")

    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists(delta_path):
        os.remove(delta_path)

    self._closed = True
```

**幂等性**：多次调用 `close_and_delete()` 安全无副作用。

**部分成功**：如果 data 文件删除成功但 delta 文件删除失败（极端情况），下次 recovery 时会发现孤儿 delta 文件，重放它不会造成数据不一致（_seq 去重保证）。

### 5.5 `find_wal_files`

```python
@staticmethod
def find_wal_files(wal_dir: str) -> List[int]:
    """
    扫描 wal_dir，找出所有存在的 WAL 编号。
    通过匹配 wal_data_NNNNNN.arrow 和 wal_delta_NNNNNN.arrow 文件名提取编号。
    返回去重排序的编号列表。
    """
    if not os.path.exists(wal_dir):
        return []

    numbers = set()
    pattern = re.compile(r"^wal_(data|delta)_(\d{6})\.arrow$")
    for filename in os.listdir(wal_dir):
        m = pattern.match(filename)
        if m:
            numbers.add(int(m.group(2)))

    return sorted(numbers)
```

**为什么要匹配两种文件名？** 因为 data 和 delta 文件独立存在：
- 可能只有 wal_data（只做了 insert，没有 delete）
- 可能只有 wal_delta（只做了 delete，没有 insert）
- 也可能两者都有

只要任一文件存在，该编号就需要被恢复。

### 5.6 `recover`

```python
@staticmethod
def recover(wal_dir, wal_number, wal_data_schema, wal_delta_schema):
    """
    读取指定编号的 WAL 文件，返回 (data_batches, delta_batches)。

    流程：
    1. 构造文件路径
    2. 分别读取 data 和 delta 文件
    3. 每个文件：存在 → 读取所有 RecordBatch；不存在 → 返回空列表
    4. 截断的 RecordBatch → 丢弃，返回截断前的完整 batch（见 §9 错误处理）
    """
    data_path = os.path.join(wal_dir, f"wal_data_{wal_number:06d}.arrow")
    delta_path = os.path.join(wal_dir, f"wal_delta_{wal_number:06d}.arrow")

    data_batches = _read_wal_file(data_path)
    delta_batches = _read_wal_file(delta_path)

    return data_batches, delta_batches


def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    """
    读取单个 WAL 文件，返回 RecordBatch 列表。
    处理三种情况：
    - 文件不存在 → []
    - 文件完整 → 所有 batch
    - 文件截断 → 截断前的完整 batch
    """
    if not os.path.exists(path):
        return []

    batches = []
    try:
        source = pa.OSFile(path, "rb")
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            batches.append(batch)
    except pa.ArrowInvalid:
        # 文件截断：Schema 之后的某个 RecordBatch 不完整
        # 已读取的 batch 是完整的，丢弃不完整的部分
        pass
    except Exception:
        # Schema 都读不出来 → 文件严重损坏
        # 返回空，让上层决定如何处理
        # （不抛 WALCorruptedError，因为 recovery 应尽量恢复可恢复的部分）
        pass

    return batches
```

**设计决策——截断处理**：

Arrow IPC Streaming 的特性：`pa.ipc.open_stream()` 成功 → Schema 完整。之后逐个读 RecordBatch，
某个 batch 读到一半文件结束 → 抛 `ArrowInvalid`。已经成功读取的 batch 都是完整的。

我们选择**尽量恢复**：
- 能读多少读多少，截断的部分丢弃
- 这比 "文件有任何损坏就全部丢弃" 更好
- 丢弃的是最后一次 write_batch 的数据（崩溃时正在写入的那个 batch）

---

## 6. WAL 编号与生命周期管理

### 6.1 编号分配规则

```
WAL 编号 N 是单调递增的整数，从 1 开始。
每次 flush 后 N += 1。

来源：Manifest.active_wal_number
  - Collection 首次创建：N = 1
  - 每次 flush：Manifest 更新 active_wal_number = N + 1
  - Recovery：从 Manifest 读取，结合 wal/ 目录扫描结果，取 max
```

### 6.2 完整生命周期

```
                   Collection 初始化
                         │
                         ▼
              ┌─────────────────────┐
              │  WAL(wal_dir, ...,  │
              │       number=N)     │
              │  状态: 两个 writer   │
              │        都是 None    │
              └─────────┬───────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              │              ▼
   write_insert()       │        write_delete()
   首次: 创建文件+writer  │        首次: 创建文件+writer
   后续: 追加 batch      │        后续: 追加 batch
         │              │              │
         └──────────────┼──────────────┘
                        │
                  MemTable 满了
                        │
                        ▼
              ┌─────────────────────┐
              │  Flush 触发          │
              │  1. 冻结当前 WAL(N)  │
              │  2. 创建新 WAL(N+1)  │
              └─────────┬───────────┘
                        │
        ┌───────────────┤
        ▼               ▼
   新 WAL(N+1)     冻结的 WAL(N) 进入 flush 管线
   继续接收写入          │
                        ▼
              ┌─────────────────────┐
              │  Flush 管线          │
              │  Step 1-4: 写 Parquet│
              │  Step 5: Manifest   │
              │    active_wal=N+1   │
              │    manifest.save()  │
              │  Step 6:            │
              │    WAL(N).close_    │
              │    and_delete()     │
              └─────────────────────┘
```

### 6.3 与 Manifest 的同步

```
时间线        WAL 状态                    Manifest.active_wal_number
──────        ────────                    ──────────────────────────
T0            WAL(1) 创建                 1
T1            WAL(1) 写入中...            1
T2            WAL(1) 冻结, WAL(2) 创建    1  ← 还没更新
T3            Flush: Parquet 写入完成     1
T4            Flush: Manifest 更新        2  ← 此刻更新
T5            WAL(1) 删除                 2
T6            WAL(2) 写入中...            2
```

**关键不变量**：Manifest 更新（T4）先于 WAL 删除（T5）。
这保证了：如果在 T4 和 T5 之间崩溃，WAL(1) 残留在磁盘上，
但 Manifest 已指向新数据文件 → 重放 WAL(1) 产生的重复数据通过 _seq 去重消除。

---

## 7. 崩溃恢复详解

### 7.1 恢复时的 WAL 状态分析

Recovery 启动时，可能在 wal/ 目录下发现以下情况：

```
场景 A: 无 WAL 文件
  原因：上次正常关闭，或 flush 完成后正常退出
  处理：无需重放，直接启动

场景 B: 只有 WAL(N)，N == Manifest.active_wal_number
  原因：正常写入过程中崩溃（flush 未触发或未完成 manifest 更新）
  处理：重放 WAL(N) → MemTable

场景 C: WAL(N) + WAL(N+1)，Manifest.active_wal_number == N+1
  原因：Flush 完成了 Manifest 更新（active_wal=N+1），但 WAL(N) 未删除
  处理：重放 WAL(N)（数据已在 Parquet 中，_seq 去重消除重复）
        重放 WAL(N+1)（新写入的数据）

场景 D: WAL(N) + WAL(N+1)，Manifest.active_wal_number == N
  原因：Flush 进行中，新 WAL 已创建但 Manifest 未更新就崩溃了
  处理：重放 WAL(N)（完整数据）
        重放 WAL(N+1)（新写入的数据）
        孤儿 Parquet 由 recovery 清理

场景 E: 只有 WAL 的 data 文件或 delta 文件（不成对）
  原因：该轮只做了 insert 或只做了 delete
  处理：正常，缺失的文件视为空（_read_wal_file 返回 []）
```

### 7.2 Recovery 中 WAL 的处理流程

```python
def execute_recovery(data_dir, schema, manifest):
    wal_dir = os.path.join(data_dir, "wal")
    memtable = MemTable(schema)

    # ── Step 1: 发现 WAL 文件 ──
    wal_numbers = WAL.find_wal_files(wal_dir)
    if not wal_numbers:
        # 场景 A: 无需恢复
        return memtable, ...

    # ── Step 2: 按编号升序重放所有 WAL ──
    max_seq = manifest.current_seq
    for n in sorted(wal_numbers):
        data_batches, delta_batches = WAL.recover(
            wal_dir, n, wal_data_schema, wal_delta_schema
        )
        for batch in data_batches:
            for row in range(batch.num_rows):
                seq = batch.column("_seq")[row].as_py()
                partition = batch.column("_partition")[row].as_py()
                fields = {col: batch.column(col)[row].as_py()
                          for col in batch.schema.names
                          if col not in ("_seq", "_partition")}
                memtable.put(seq, partition, **fields)
                max_seq = max(max_seq, seq)

        for batch in delta_batches:
            for row in range(batch.num_rows):
                pk = batch.column(pk_name)[row].as_py()
                seq = batch.column("_seq")[row].as_py()
                partition = batch.column("_partition")[row].as_py()
                memtable.delete(pk, seq, partition)
                max_seq = max(max_seq, seq)

    # ── Step 3: 不删除 WAL 文件（见 §7.3） ──
    next_seq = max_seq + 1
    return memtable, delta_log, next_seq
```

### 7.3 恢复后的 WAL 文件处理策略

**核心问题**：Recovery 把 WAL 重放到 MemTable 后，旧 WAL 文件要不要立刻删除？

**方案对比**：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A. 立刻删除 | 重放后 `os.remove()` | 磁盘干净 | 如果恢复后再次崩溃（flush 前），数据永久丢失 |
| B. 保留不删 | 不动，等 flush 时清理 | 二次崩溃安全 | 需要处理"多轮 WAL + 当前 WAL"的清理 |

**选择方案 B**：保留旧 WAL 文件，由 flush 统一清理。

**实现**：
- Recovery 后，Collection 知道 wal/ 目录下可能残留旧 WAL 文件
- Collection 创建新 WAL（编号 = max(所有已有编号) + 1）
- 当 MemTable flush 时，flush 管线在 Step 6 不仅删除 frozen WAL，
  还清理所有编号 < 当前 active_wal_number 的旧 WAL 文件

```python
# flush.py Step 6 增强
def _cleanup_wal_files(wal_dir: str, max_number_to_delete: int) -> None:
    """删除所有编号 <= max_number_to_delete 的 WAL 文件。"""
    for n in WAL.find_wal_files(wal_dir):
        if n <= max_number_to_delete:
            data_path = os.path.join(wal_dir, f"wal_data_{n:06d}.arrow")
            delta_path = os.path.join(wal_dir, f"wal_delta_{n:06d}.arrow")
            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(delta_path):
                os.remove(delta_path)
```

### 7.4 Recovery 后的 WAL 编号

```python
# 恢复后新 WAL 编号的确定
found_numbers = WAL.find_wal_files(wal_dir)       # 磁盘上残留的
manifest_number = manifest.active_wal_number       # Manifest 记录的

new_wal_number = max(
    manifest_number,
    max(found_numbers) + 1 if found_numbers else 1
)
```

**这保证了新 WAL 编号不会与任何已存在或曾经存在的 WAL 文件冲突。**

---

## 8. fsync 与持久性

### 8.1 崩溃类型

| 崩溃类型 | 示例 | OS buffer cache | 不开 fsync 的数据丢失风险 |
|---------|------|----------------|------------|
| 进程崩溃（同 OS 内立即接管） | SIGKILL, 异常退出 | 保留（仍由 OS 持有） | 低——但仍有 |
| 容器/进程被 kill 后立刻被新进程接管 | OOM-kill → restart | 仍在 cache，新进程 read 命中 | **高**——见 §8.2 反例 |
| OS 崩溃 | 内核 panic, 断电 | 丢失 | 有 |

### 8.2 默认 `sync_mode="close"`：在 close 时 fsync 一次

```python
class WAL:
    def __init__(self, ..., sync_mode: str = "close"):
        """
        sync_mode:
          "none"   - 完全不 fsync（仅供测试 / 性能基准）
          "close"  - 默认。在 close_and_delete 前对 sink 做一次 os.fsync
          "batch"  - 每次 write_batch 后 fsync（最强一致性，最慢）
        """
```

**为什么不是"none"作为默认**：

考虑这个反例：

```
T0: WAL.write_insert(batch_X)   # batch X 进入 OS buffer cache，未刷盘
T1: Collection.insert 返回成功，client 收到成功响应
T2: 容器被 OOM-killed
T3: 编排系统立刻拉起新容器，挂同一卷
T4: 新进程 Collection.__init__ → recovery → 读 WAL
```

在 T4 那一刻：
- 老进程持有的 OS buffer cache 已经随老进程消失（容器隔离）
- batch_X 还没刷到磁盘
- 新进程 read 看到的是**没有 batch_X 的 WAL 文件**
- 但 client 已经被告知 "成功"——**数据丢失，违反持久性承诺**

`sync_mode="close"` 在 `close_and_delete` 前调一次 `os.fsync(sink.fileno())` 就能堵掉这个窗口：
- WAL 在 flush 触发时被 close，close 前 fsync 把整个文件持久化
- 频率 = flush 频率 = 每 `MEMTABLE_SIZE_LIMIT` 行一次，很稀疏，性能影响可忽略
- 关键路径："WAL 切换 → Manifest 更新"这条 commit 路径上的耐久性补齐了

**`sync_mode="batch"` 的成本**：

每次 `write_batch` 都 fsync。对于嵌入式向量库（vector 体积大）这个成本明显，但作为可选项保留给追求最强持久性的用户。**默认不开**。

### 8.3 实现细节

```python
def write_insert(self, record_batch):
    assert not self._closed
    if self._data_writer is None:
        # ... lazy init
    self._data_writer.write_batch(record_batch)
    if self._sync_mode == "batch":
        os.fsync(self._data_sink.fileno())

def close_and_delete(self):
    if self._closed:
        return
    try:
        if self._sync_mode in ("close", "batch") and self._data_writer is not None:
            self._data_writer.close()                      # 写 EOS marker
            os.fsync(self._data_sink.fileno())             # ← 强刷盘
            self._data_sink.close()
        # ... 同上 _delta_writer
    finally:
        self._closed = True
        # ... 删文件
```

**注意**：fsync 必须在 `_data_writer.close()` 之后、`_data_sink.close()` 之前。先 close writer 把 EOS marker 写进 OS buffer，再 fsync 把整个文件含 EOS 持久化。

---

## 9. 错误处理

### 9.1 写入时的错误

| 错误类型 | 触发场景 | 处理方式 |
|---------|---------|---------|
| `ArrowInvalid` | RecordBatch schema 不匹配 | 直接抛出（调用方 bug，不应到达 WAL 层） |
| `OSError` | 磁盘满、权限不足 | 直接抛出，Collection 层处理（写入失败，MemTable 不更新） |
| `AssertionError` | 对已关闭的 WAL 写入 | 编程错误，不应发生 |

**WAL 写入失败时的影响**：

```
Collection.insert() 流程：
  1. validate + allocate _seq     ← 成功
  2. build RecordBatch            ← 成功
  3. WAL.write_insert(batch)      ← 失败！抛异常
  4. MemTable.put(...)            ← 不会执行

结果：_seq 被浪费（有间隔），但数据一致性不受影响。
WAL 和 MemTable 保持同步——都没有写入这条数据。
```

### 9.2 恢复时的错误

| 错误类型 | 触发场景 | 处理方式 |
|---------|---------|---------|
| 文件不存在 | 只有 data 没有 delta（或反之） | 正常，缺失的视为空 |
| Schema 读取失败 | 文件严重损坏（只写了几个字节） | 跳过该文件，日志警告 |
| RecordBatch 截断 | 写入过程中崩溃 | 返回截断前的完整 batch |
| 数据校验失败 | 磁盘位翻转等极端情况 | 跳过该文件，日志警告 |

**恢复原则：尽力恢复，不因部分损坏而放弃全部数据。**

### 9.3 _seq 间隔的安全性

WAL 写入失败会导致 _seq 出现间隔（gap），例如：

```
正常:    _seq = [1, 2, 3, 4, 5]
有间隔:  _seq = [1, 2, 4, 5]      ← 3 因 WAL 写入失败而跳过
```

这是安全的，因为：
- _seq 只用于"同 PK 取 max_seq"的去重，不依赖连续性
- delta_log 的 `is_deleted(pk, data_seq)` 比较的是 "delete_seq > data_seq"，不依赖连续性

---

## 10. 与上层组件的交互

### 10.1 Insert 流程

```
Collection.insert(records, partition_name)
  │
  ├─ 1. validate_record(record, schema)       # 校验每条记录
  ├─ 2. separate_dynamic_fields(record)       # 分离动态字段 → $meta
  ├─ 3. seq = self._alloc_seq()               # 为每条记录分配独立 _seq
  ├─ 4. 构建 RecordBatch (含 _seq, _partition)
  │
  ├─ 5. self.wal.write_insert(batch)          ← WAL 先写
  │
  ├─ 6. for each record:                      ← 然后 MemTable
  │       self.memtable.put(seq, partition, **fields)
  │
  └─ 7. if self.memtable.size() >= LIMIT:     ← 检查是否需要 flush
           self._trigger_flush()
```

### 10.2 Delete 流程

```
Collection.delete(pks, partition_name)
  │
  ├─ 1. shared_seq = self._alloc_seq()        # 批量共享一个 _seq
  ├─ 2. partition = partition_name or "_all"
  ├─ 3. 构建 RecordBatch (pk_values, shared_seq, partition)
  │
  ├─ 4. self.wal.write_delete(batch)          ← WAL 先写
  │
  ├─ 5. for pk in pks:                        ← 然后 MemTable
  │       self.memtable.delete(pk, shared_seq, partition)
  │
  └─ 6. if self.memtable.size() >= LIMIT:
           self._trigger_flush()
```

### 10.3 Flush 时的 WAL 切换

```
Collection._trigger_flush()
  │
  ├─ 1. frozen_memtable = self.memtable       # 冻结
  ├─ 2. frozen_wal = self.wal                 # 冻结
  │
  ├─ 3. self.memtable = MemTable(schema)      # 新建空 MemTable
  ├─ 4. new_number = frozen_wal.number + 1
  ├─ 5. self.wal = WAL(wal_dir, ..., new_number)  # 新建空 WAL
  │
  └─ 6. execute_flush(                        # 后台执行 flush 管线
  │       frozen_memtable,
  │       frozen_wal,                         # ← 传入冻结的 WAL
  │       ...
  │     )
  │
  └─ flush 管线内部：
       Step 1-4: 写 Parquet 文件
       Step 5:   manifest.active_wal_number = new_number
                 manifest.save()              # 原子更新
       Step 6:   _cleanup_wal_files(wal_dir, frozen_wal.number)
                                              # 删除冻结 WAL + 更早的残留 WAL
```

### 10.4 Recovery 流程

```
Collection.__init__(name, data_dir, schema)
  │
  ├─ 1. manifest = Manifest.load(data_dir)
  │
  ├─ 2. memtable, delta_log, next_seq = execute_recovery(
  │       data_dir, schema, manifest          # Recovery 内部:
  │     )                                     #   find_wal_files → recover → 重放到 memtable
  │
  ├─ 3. self.memtable = memtable              # 使用恢复后的 MemTable
  ├─ 4. self._seq_counter = next_seq          # 恢复 _seq 计数器
  │
  ├─ 5. new_wal_number = ...                  # 见 §7.4 编号计算
  └─ 6. self.wal = WAL(wal_dir, ..., new_wal_number)  # 新 WAL 用于后续写入
                                              # 旧 WAL 文件留在磁盘，等 flush 清理
```

---

## 11. 完整接口（最终版）

综合以上设计，更新 WAL 的完整接口：

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
        sync_mode: str = "close",
    ) -> None:
        """初始化 WAL（不创建文件，延迟到首次写入）。

        Args:
            wal_dir: WAL 文件所在目录（不存在则创建）
            wal_data_schema: wal_data 文件的 Arrow Schema（含 _seq, _partition, 用户字段, $meta）
            wal_delta_schema: wal_delta 文件的 Arrow Schema（含 pk, _seq, _partition）
            wal_number: 本轮 WAL 编号 N（文件名中的 N，从 Manifest 或 recovery 推算）
            sync_mode: 持久性策略，详见 §8。
                - "none"  完全不 fsync（仅供测试 / 性能基准）
                - "close" 默认。close_and_delete 前 fsync 一次
                - "batch" 每次 write_batch 后 fsync
        """

    def write_insert(self, record_batch: pa.RecordBatch) -> None:
        """追加一个 RecordBatch 到 wal_data 文件。

        首次调用时创建文件和 StreamWriter（延迟初始化）。
        record_batch 的 schema 必须与 wal_data_schema 一致（PyArrow 内部校验）。

        sync_mode="batch" 时，write_batch 后 fsync 一次。

        注：WAL 不知道 Operation 类型——它接受 raw RecordBatch。
        Operation dispatch 在 Collection._apply 里完成（依赖层级原因，
        见 modules.md §9.16.5）。

        Raises:
            ArrowInvalid: schema 不匹配
            OSError: 磁盘满或权限不足
            AssertionError: WAL 已关闭
        """

    def write_delete(self, record_batch: pa.RecordBatch) -> None:
        """追加一个 RecordBatch 到 wal_delta 文件。语义与 write_insert 对称。"""

    def close_and_delete(self) -> None:
        """关闭 writer 并删除两个 WAL 文件。幂等操作。

        调用时机：flush Step 6（Manifest 已更新之后）。

        流程（每个 writer 各自包 try/finally，互不影响）：
            for writer in (data_writer, delta_writer):
                writer.close()                              # 写 EOS marker
                if sync_mode in ("close", "batch"):
                    os.fsync(sink.fileno())                 # 强刷盘
                sink.close()
            for file in (data_path, delta_path):
                if exists: os.remove(file)
            self._closed = True

        幂等性：第二次调用直接 return；任一 writer.close 失败都不影响另一个。
        文件不存在时静默跳过删除步骤。
        """

    @staticmethod
    def find_wal_files(wal_dir: str) -> List[int]:
        """扫描目录，返回所有 WAL 编号（去重、升序）。

        匹配 wal_data_NNNNNN.arrow 和 wal_delta_NNNNNN.arrow，
        只要任一文件存在就包含该编号。

        Returns:
            升序排列的 WAL 编号列表，目录不存在则返回 []
        """

    @staticmethod
    def recover(
        wal_dir: str,
        wal_number: int,
    ) -> Tuple[List[pa.RecordBatch], List[pa.RecordBatch]]:
        """读取指定编号的 WAL 文件，返回 (data_batches, delta_batches)。

        - 文件不存在 → 对应列表为空
        - 文件截断 → 返回截断前的完整 batch，丢弃不完整部分
        - 文件严重损坏（Schema 不可读）→ 对应列表为空，日志警告

        Operation 包装在 engine/recovery.py 的 replay_wal_operations() 里完成
        （依赖层级原因，详见 modules.md §9.16.5）。
        """

    @property
    def number(self) -> int:
        """当前 WAL 编号 N。"""

    @property
    def data_path(self) -> Optional[str]:
        """wal_data 文件路径。延迟初始化前（未写入过 insert）返回 None。"""

    @property
    def delta_path(self) -> Optional[str]:
        """wal_delta 文件路径。延迟初始化前（未写入过 delete）返回 None。"""


# ── 模块级辅助函数（不导出） ──

def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    """读取单个 WAL 文件，容错处理截断和损坏。"""

def _cleanup_old_wals(wal_dir: str, up_to_number: int) -> None:
    """删除所有编号 <= up_to_number 的 WAL 文件。flush Step 6 调用。"""
```

---

## 12. WAL 设计决策汇总

| 决策 | 选择 | 理由 |
|------|------|------|
| 文件格式 | Arrow IPC Streaming | 向量无写放大，零拷贝解析，Schema 内建 |
| 文件结构 | 双文件（data + delta） | Schema 不同，延迟初始化，与 MemTable 对称 |
| Writer 初始化 | 延迟（首次写入时创建） | 避免空文件 |
| **fsync** | **默认 `sync_mode="close"`，close 前 fsync 一次** | 覆盖容器 OOM-kill 后立即接管的崩溃场景；频率 = flush 频率，开销可忽略。详见 §8 |
| 截断处理 | 尽力恢复（读到哪算哪） | 最大化数据恢复 |
| Recovery 后旧 WAL | 保留，等 flush 清理 | 二次崩溃安全 |
| WAL 编号 | 单调递增，从 Manifest 恢复 | 不重复，可追溯 |
| close_and_delete | 幂等，每个 writer 独立 try/finally | 崩溃重试安全；一个 writer 关闭失败不影响另一个 |
| 清理范围 | flush 时清理所有 <= frozen 编号的 WAL | 统一处理残留 + 冻结 |
| **写入入口** | **raw `write_insert / write_delete`** | WAL 不知道 Operation；dispatch 在 Collection.\_apply 里做（依赖层级原因，见 modules.md §9.16.5） |
| **读取入口** | **`recover() → (data_batches, delta_batches)`** | engine/recovery.py 的 `replay_wal_operations` 把它包装成按 _seq 排序的 Operation 流 |

---

# Part II: Segment 与搜索架构

---

## 13. Upsert 与 WAL 的关系

### 13.1 Upsert 不是 Delete + Insert

传统数据库的 upsert 通常实现为"先删后插"，涉及两步写入。
MilvusLite 采用 LSM-Tree 风格，**upsert 纯粹是一次 insert，只写 `wal_data`，不碰 `wal_delta`**。

```
Insert("doc_1", new_data)   ← PK "doc_1" 已存在于磁盘 Parquet 中

WAL:       wal_data 追加一条 RecordBatch（_seq=新值）  ✅ 只写一个文件
MemTable:  put("doc_1", ...) → dict 按 PK 覆盖旧条目    ✅ 内存覆盖
磁盘旧数据: 仍在 Parquet 中，不动                       ✅ 不需要显式删除
```

"覆盖"通过 **_seq 去重**隐式实现：

```
磁盘 Parquet:  doc_1, _seq=100, embedding=[0.1, ...]    ← 旧版本
MemTable:      doc_1, _seq=500, embedding=[0.9, ...]    ← 新版本

搜索时 bitmap 去重：同 PK 保留 max_seq → _seq=500 胜出 → 旧版本不可见
Compaction 时：旧记录 (_seq=100) 被物理清除
```

### 13.2 每个 API 操作涉及的 WAL 文件

| 操作 | wal_data | wal_delta | 原子性风险 |
|------|----------|-----------|-----------|
| insert（新 PK） | 写 | - | 无（单文件） |
| insert（已有 PK = upsert） | 写 | - | 无（单文件） |
| delete | - | 写 | 无（单文件） |

**没有任何单个 API 调用需要同时写两个 WAL 文件**，因此不存在"双文件半写"的一致性问题。

### 13.3 LSM-Tree 风格的 trade-off

写入简单（只追加）→ 但搜索时需要额外过滤过期数据 → 由 bitmap 去重解决 → 由 Compaction 最终清理物理数据。

---

## 14. Segment：数据文件的内存缓存

### 14.1 问题：每次搜索都读磁盘不可接受

MVP 暴力搜索需要读取所有 Parquet 文件的全量数据。如果每次 search 都从磁盘读取：

```
100 万条 × 128 维 × 4 bytes = 488 MB
每次 search 都读 488 MB → 延迟数秒 → 不可接受
```

### 14.2 Segment 概念

每个 Parquet 数据文件在内存中对应一个 **Segment**（封存段）。
Parquet 文件不可变 → Segment 加载后永不失效，无需缓存淘汰策略。

```python
class Segment:
    """一个 Parquet 数据文件的内存缓存。文件不可变 → 缓存永不失效。"""

    file_path: str              # 源文件路径
    partition: str              # 所属 Partition

    # ── 预提取的 NumPy 数组（搜索热路径用） ──
    pks: np.ndarray             # 主键数组，bitmap 去重用
    seqs: np.ndarray            # _seq 数组，bitmap 去重用
    vectors: np.ndarray         # (N, dim) float32 矩阵，距离计算用

    # ── 完整记录（返回搜索结果用） ──
    table: pa.Table             # 原始 Arrow Table，取 top-k 对应行返回

    # ── 未来 ──
    # faiss_index: faiss.Index  # FAISS 索引
```

### 14.3 为什么要预提取 NumPy 数组

```
方案 A: 每次搜索时从 pa.Table 提取
  pa.Table → .column("vec") → .to_numpy() → 有转换开销
  向量列是 list<float32>（变长），需 stack 成 (N, dim) 连续数组
  每次搜索都做 → 重复浪费

方案 B: 加载时一次性预提取（选择此方案）
  Segment 创建时 → 提取 pks / seqs / vectors 到 NumPy 数组
  搜索时 → 直接使用，零额外开销
```

### 14.4 Segment 完整接口

```python
# storage/segment.py

class Segment:
    """不可变数据文件的内存表示。加载时预提取搜索热路径所需的 NumPy 数组。"""

    def __init__(self, file_path: str, partition: str, table: pa.Table,
                 pk_field: str, vector_field: str):
        self.file_path = file_path
        self.partition = partition
        self.table = table

        # 一次性预提取
        self.pks = table.column(pk_field).to_numpy()
        self.seqs = table.column("_seq").to_numpy()
        self.vectors = self._extract_vectors(table, vector_field)

    @staticmethod
    def load(file_path: str, partition: str,
             pk_field: str, vector_field: str) -> "Segment":
        """从 Parquet 文件加载为 Segment。"""
        table = read_data_file(file_path)
        return Segment(file_path, partition, table, pk_field, vector_field)

    @staticmethod
    def from_table(file_path: str, partition: str, table: pa.Table,
                   pk_field: str, vector_field: str) -> "Segment":
        """从已有 Arrow Table 构建 Segment（Flush 时跳过磁盘读取）。"""
        return Segment(file_path, partition, table, pk_field, vector_field)

    def search(self, query_vector: np.ndarray, valid_mask: np.ndarray,
               top_k: int, metric_type: str) -> List["Hit"]:
        """在本 Segment 内搜索，只对 valid_mask=True 的行计算距离。

        Returns:
            按距离升序排列的 Hit 列表（最多 top_k 个）
        """
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_vectors = self.vectors[valid_indices]
        distances = compute_distances(query_vector, valid_vectors, metric_type)

        k = min(top_k, len(distances))
        top_k_pos = np.argpartition(distances, k - 1)[:k]
        top_k_pos = top_k_pos[np.argsort(distances[top_k_pos])]

        return [
            Hit(pk=self.pks[valid_indices[i]],
                distance=float(distances[i]),
                row_index=int(valid_indices[i]),
                segment=self)
            for i in top_k_pos
        ]

    def get_record(self, row_index: int) -> dict:
        """按行索引取完整记录（top-k 结果回查用）。"""
        return {col: self.table.column(col)[row_index].as_py()
                for col in self.table.schema.names
                if col != "_seq"}

    @staticmethod
    def _extract_vectors(table: pa.Table, vector_field: str) -> np.ndarray:
        """list<float32> 列 → (N, dim) float32 连续数组。"""
        vec_col = table.column(vector_field)
        return np.stack([v.as_py() for v in vec_col]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.table)
```

### 14.5 缓存生命周期

Segment 的生死完全跟随 Manifest，不需要 LRU 等淘汰策略：

```
事件                          缓存操作
────                          ────────
Collection 启动 / Recovery    加载 Manifest 中所有 data 文件 → Segment 列表
Flush 完成                    新 Parquet → 创建新 Segment 加入缓存
Compaction 完成               旧文件删除 → 驱逐旧 Segment；新文件 → 创建新 Segment
Collection.close()            释放所有 Segment
```

```python
# engine/collection.py 中的 Segment 管理

class Collection:
    _segments: Dict[str, List[Segment]]   # partition_name → [Segment, ...]

    def __init__(self, name, data_dir, schema):
        # 启动时加载所有 Segment
        self._segments = {}
        for partition in manifest.list_partitions():
            self._segments[partition] = [
                Segment.load(f, partition, pk_field, vec_field)
                for f in manifest.get_data_files(partition)
            ]

    def _on_flush_complete(self, partition, new_file, table):
        """Flush 回调：用内存中的 table 直接构建 Segment（跳过磁盘读取）"""
        seg = Segment.from_table(new_file, partition, table, pk_field, vec_field)
        self._segments.setdefault(partition, []).append(seg)

    def _on_compaction_complete(self, partition, old_files, new_file):
        """Compaction 回调：驱逐旧 Segment，加载新 Segment"""
        old_set = set(old_files)
        self._segments[partition] = [
            s for s in self._segments[partition] if s.file_path not in old_set
        ]
        seg = Segment.load(new_file, partition, pk_field, vec_field)
        self._segments[partition].append(seg)
```

### 14.6 Flush 优化：零拷贝建 Segment

Flush 时数据已经在内存中（冻结的 MemTable → Arrow Table）。
写入 Parquet 后，**直接用这个 Table 构建 Segment，不需要再从磁盘读回来**：

```
普通路径：  MemTable → pa.Table → write_data_file() → read_data_file() → Segment
                                        写磁盘              读磁盘（浪费）

优化路径：  MemTable → pa.Table ──┬── write_data_file()     写磁盘
                                 └── Segment.from_table()  直接复用内存
```

### 14.7 内存预算

对于嵌入式本地数据库，MVP 假设数据集能放进内存：

```
100 万条 × 128 维 float32 = 488 MB（向量）
加上 PK / _seq / 其他字段 ≈ 600-800 MB 总计

对本地机器（8-16 GB 内存）完全可接受
```

未来如果数据量超出内存，可按 Segment 粒度加 LRU 驱逐——但 MVP 不需要。

---

## 15. 分段搜索架构

### 15.1 两类 Segment

```
Sealed Segment    从 Parquet 文件加载，不可变，可建 FAISS 索引
                  数据来源：storage/segment.py

Growing Segment   MemTable 中未落盘的数据，持续变化，只能暴力搜索
                  数据来源：storage/memtable.py
```

搜索时两者都要参与，行为一致：接收 valid_mask，返回 local top-k。

### 15.2 核心矛盾：去重是全局的，搜索是分段的

同一个 PK 可能存在于多个 Segment 中（upsert 导致）：

```
Segment A (旧 Parquet):  doc_1  _seq=100  vec=[0.1, ...]
Segment B (新 Parquet):  doc_1  _seq=500  vec=[0.9, ...]   ← upsert 后 flush
MemTable:                doc_1  _seq=800  vec=[0.5, ...]   ← 又一次 upsert
```

如果每个 Segment 独立去重，Segment A 不知道 doc_1 已被覆盖。
因此必须：**先全局去重，再分段搜索**。

### 15.3 完整搜索流程

```
Step 1: 全局去重 → 生成 per-segment mask（轻量，只用预提取的 pk/seq 数组）
──────────────────────────────────────────────────────────────────────────────

  segments:
    seg_A.pks  = [doc_1, doc_2]     seg_A.seqs = [100, 101]
    seg_B.pks  = [doc_1, doc_3]     seg_B.seqs = [500, 501]
    memtable   = [doc_1, doc_4]     seqs       = [800, 802]

  全局去重：
    doc_1 → seg_A(100) vs seg_B(500) vs memtable(800) → memtable 胜
    doc_2 → seg_A(101) 唯一 → 有效
    doc_3 → seg_B(501) 唯一 → 有效
    doc_4 → memtable(802) 唯一 → 有效

  删除过滤：delta_log.is_deleted(pk, seq)

  输出 per-segment mask：
    seg_A_mask = [False, True ]     ← doc_1 被覆盖，doc_2 有效
    seg_B_mask = [False, True ]     ← doc_1 被覆盖，doc_3 有效
    mem_mask   = [True,  True ]     ← doc_1 最新版有效，doc_4 有效

Step 2: 每个 Segment 独立搜索（只对 valid 的行计算距离）
──────────────────────────────────────────────────────────────────────────────

  seg_A.search(query, seg_A_mask, top_k) → [(doc_2, dist=0.3)]
  seg_B.search(query, seg_B_mask, top_k) → [(doc_3, dist=0.5)]
  memtable.search(query, mem_mask, top_k) → [(doc_1, dist=0.1), (doc_4, dist=0.7)]

Step 3: 合并所有 local top-k → global top-k
──────────────────────────────────────────────────────────────────────────────

  all_hits = [(doc_1, 0.1), (doc_2, 0.3), (doc_3, 0.5), (doc_4, 0.7)]
  sort by distance → 取前 top_k 个

Step 4: 取完整记录（只查 top-k 条）
──────────────────────────────────────────────────────────────────────────────

  for hit in global_top_k:
      hit.segment.get_record(hit.row_index)   ← 从对应 Segment 回查
```

### 15.4 全局去重实现

```python
# search/bitmap.py

def build_segment_masks(
    segments: List[Segment],
    memtable: MemTable,
    delta_log: DeltaLog,
    pk_field: str,
) -> List[np.ndarray]:
    """全局去重 + 删除过滤，输出 per-segment mask。

    返回 N+1 个 boolean mask：前 N 个对应 segments，最后 1 个对应 memtable。
    每个 mask 长度等于对应段的行数。
    """
    # ── 全局去重：PK → (max_seq, segment_idx, row_idx) ──
    best = {}  # pk → (max_seq, seg_idx, row_idx)

    for seg_idx, seg in enumerate(segments):
        for row_idx in range(len(seg.pks)):
            pk = seg.pks[row_idx]
            seq = int(seg.seqs[row_idx])
            if pk not in best or seq > best[pk][0]:
                best[pk] = (seq, seg_idx, row_idx)

    # MemTable 作为最后一个"段"
    mem_idx = len(segments)
    mem_pks, mem_seqs = memtable.get_pk_seq_arrays()
    for row_idx in range(len(mem_pks)):
        pk = mem_pks[row_idx]
        seq = int(mem_seqs[row_idx])
        if pk not in best or seq > best[pk][0]:
            best[pk] = (seq, mem_idx, row_idx)

    # ── 构建 per-segment mask ──
    masks = [np.zeros(len(seg.pks), dtype=bool) for seg in segments]
    masks.append(np.zeros(len(mem_pks), dtype=bool))  # MemTable mask

    for pk, (seq, seg_idx, row_idx) in best.items():
        if not delta_log.is_deleted(pk, seq):
            masks[seg_idx][row_idx] = True

    return masks
```

### 15.5 MemTable 的搜索支持

MemTable 是 Growing Segment，需要提供与 Sealed Segment 对称的搜索接口：

```python
# storage/memtable.py 新增方法

class MemTable:
    def get_pk_seq_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 insert_buf 中所有活跃记录的 (pks, seqs) 数组。
        用于 bitmap 全局去重。"""

    def get_vectors(self, vector_field: str) -> np.ndarray:
        """返回 insert_buf 中所有活跃记录的向量矩阵 (N, dim)。
        用于暴力搜索。"""

    def search(self, query_vector: np.ndarray, valid_mask: np.ndarray,
               top_k: int, metric_type: str) -> List["Hit"]:
        """暴力搜索 insert_buf 中 valid_mask=True 的记录。
        逻辑与 Segment.search() 相同。"""

    def get_record(self, row_index: int) -> dict:
        """按行索引取完整记录（top-k 回查用）。"""
```

### 15.6 合并策略

```python
# search/merge.py

@dataclass
class Hit:
    """单条搜索结果。"""
    pk: Any
    distance: float
    row_index: int
    segment: Any        # Segment 或 MemTable 引用，用于回查完整记录

    def to_record(self) -> dict:
        """回查完整记录并附上距离。"""
        record = self.segment.get_record(self.row_index)
        record["_distance"] = self.distance
        return record

def merge_results(segment_hits: List[List[Hit]], top_k: int) -> List[Hit]:
    """合并多个段的 local top-k，取 global top-k。

    MVP: 简单拼接 + 排序（段数量少，足够快）
    未来: k-way merge with heap（段多时更高效）
    """
    all_hits = []
    for hits in segment_hits:
        all_hits.extend(hits)
    all_hits.sort(key=lambda h: h.distance)
    return all_hits[:top_k]
```

### 15.7 更新后的 executor

```python
# search/executor.py

def execute_search(
    query_vectors: List[np.ndarray],
    segments: List[Segment],
    memtable: MemTable,
    delta_log: DeltaLog,
    top_k: int,
    metric_type: str,
    pk_field: str,
) -> List[List[dict]]:
    """搜索入口。全局去重 → 分段搜索 → 合并 → 回查记录。"""

    results = []
    for query in query_vectors:
        # ── Step 1: 全局去重 → per-segment mask ──
        masks = build_segment_masks(segments, memtable, delta_log, pk_field)

        # ── Step 2: 分段搜索 ──
        all_hits = []
        for seg, mask in zip(segments, masks[:-1]):
            all_hits.append(seg.search(query, mask, top_k, metric_type))

        mem_mask = masks[-1]
        all_hits.append(memtable.search(query, mem_mask, top_k, metric_type))

        # ── Step 3: 合并 ──
        merged = merge_results(all_hits, top_k)

        # ── Step 4: 回查完整记录 ──
        results.append([hit.to_record() for hit in merged])

    return results
```

---

## 16. data_file.py 接口扩展

当前 `data_file.py` 只有全量读取。为搜索优化和未来 FAISS 支持，新增列投影和行选择：

```python
# storage/data_file.py

# ── 已有 ──
def write_data_file(table, partition_dir, seq_min, seq_max) -> str:
    """写 Parquet 数据文件。"""

def read_data_file(path) -> pa.Table:
    """全量读取 Parquet 文件。Segment.load() 调用。"""

def parse_seq_range(filename) -> Tuple[int, int]:
    """从文件名解析 seq 范围。"""

def get_file_size(path) -> int:
    """获取文件大小（Compaction 分桶用）。"""

# ── 新增 ──
def read_columns(path: str, columns: List[str]) -> pa.Table:
    """列投影读取。只读指定列，跳过其余列（尤其是向量列）。

    Parquet 列式存储下，跳过向量列可节省 90%+ 的 IO。
    用途：未来 FAISS 场景下，bitmap 构建只需 PK + _seq。

    示例: read_columns(path, ["doc_id", "_seq"])
    """
    return pq.read_table(path, columns=columns)

def read_rows(path: str, row_indices: List[int],
              columns: Optional[List[str]] = None) -> pa.Table:
    """行选择读取。只读指定行（可选列投影）。

    用途：未来 FAISS 返回 top-k 的 row_index 后，
    只读这几行的完整记录，而非加载整个文件。
    """
```

**MVP vs FAISS 对 data_file 的调用对比**：

```
MVP（全量缓存在 Segment 中）：
  启动时: read_data_file(path) → Segment          ← 一次加载，常驻内存
  搜索时: 直接访问 Segment 的 NumPy 数组            ← 不读磁盘

FAISS（未来，按需读取）：
  启动时: read_data_file(path) → Segment + build FAISS index
  搜索时: Segment 提供 pk/seq（bitmap）
          FAISS index 提供向量搜索
          read_rows(path, top_k_indices) → 回查完整记录  ← 只读 top-k 行
```

---

## 17. 模块结构更新

### 17.1 新增和变更的模块

```
storage/
  segment.py        ← 新增：Parquet 文件的内存表示 (Segment 类)
  data_file.py      ← 扩展：新增 read_columns(), read_rows()
  memtable.py       ← 扩展：新增 get_pk_seq_arrays(), get_vectors(), search()

search/
  bitmap.py         ← 变更：build_valid_mask() → build_segment_masks()
  distance.py       ← 不变
  executor.py       ← 变更：输入从裸数组 → segments + memtable
  merge.py          ← 新增：Hit 数据类 + merge_results()
```

### 17.2 更新后的完整结构

```
milvus_lite/
├── storage/
│   ├── wal.py           # WAL 读写
│   ├── memtable.py      # 内存缓冲（含搜索支持）
│   ├── segment.py       # ★ Parquet 文件的内存缓存（含搜索能力）
│   ├── data_file.py     # Parquet 磁盘 IO（含列投影/行选择）
│   ├── delta_log.py     # 删除记录
│   └── manifest.py      # 全局状态
│
├── search/
│   ├── bitmap.py        # 全局去重 → per-segment mask
│   ├── distance.py      # 距离计算
│   ├── executor.py      # 分段搜索编排
│   └── merge.py         # ★ Hit + 多路合并
│
├── engine/
│   ├── collection.py    # 核心引擎（管理 Segment 列表）
│   ├── flush.py         # Flush 管线（含 Segment 创建回调）
│   ├── recovery.py      # 崩溃恢复
│   └── compaction.py    # Compaction（含 Segment 替换回调）
│
├── schema/              # 不变
├── db.py                # 不变
├── constants.py         # 不变
└── exceptions.py        # 不变
```

### 17.3 更新后的依赖图

```
Level 0:  constants.py, exceptions.py
Level 1:  schema/*
Level 2:  storage/wal, storage/memtable, storage/data_file,
          storage/delta_log, storage/manifest
Level 3:  storage/segment                         ← 依赖 data_file + schema
Level 4:  search/bitmap, search/distance          ← 依赖 segment + delta_log
Level 5:  search/merge                            ← 依赖 Hit 定义
Level 6:  search/executor                         ← 依赖 L3-L5
Level 7:  engine/flush, recovery, compaction       ← 依赖 L2-L6
Level 8:  engine/collection                        ← 依赖 L2-L7
Level 9:  db.py                                    ← 依赖 L8
```

---

## 18. 全部设计决策汇总

### WAL 决策（§1-§12）

| 决策 | 选择 | 理由 |
|------|------|------|
| 文件格式 | Arrow IPC Streaming | 向量无写放大，零拷贝解析，Schema 内建 |
| 文件结构 | 双文件（data + delta） | Schema 不同，延迟初始化，与 MemTable 对称 |
| Writer 初始化 | 延迟（首次写入时创建） | 避免空文件 |
| fsync | MVP 不 fsync | 嵌入式场景，进程崩溃由 OS cache 保护 |
| 截断处理 | 尽力恢复（读到哪算哪） | 最大化数据恢复 |
| Recovery 后旧 WAL | 保留，等 flush 清理 | 二次崩溃安全 |
| WAL 编号 | 单调递增，从 Manifest 恢复 | 不重复，可追溯 |
| close_and_delete | 幂等 | 崩溃重试安全 |
| 清理范围 | flush 时清理所有 <= frozen 编号的 WAL | 统一处理残留 + 冻结 |

### Segment 与搜索决策（§13-§17）

| 决策 | 选择 | 理由 |
|------|------|------|
| Upsert 实现 | 纯 insert，_seq 去重 | 单文件写入，无原子性风险 |
| 数据缓存 | Segment 常驻内存 | 避免每次搜索读磁盘 |
| 缓存失效 | 跟随 Manifest（无 LRU） | 文件不可变，增删明确 |
| NumPy 预提取 | 加载时一次性转换 | 搜索热路径零转换开销 |
| Flush 建 Segment | 复用内存 Table，不重读磁盘 | 减少一次磁盘 IO |
| 搜索模型 | 全局去重 → 分段搜索 → 合并 | 去重必须全局；搜索可分段并行 + 适配 FAISS |
| MemTable 搜索 | 暴力搜索（Growing Segment） | 数据量小，无需索引 |
| 合并策略 | MVP 拼接排序 | 段少够快；未来可换 k-way merge |

### 去重与演进决策（§19-§21）

| 决策 | 选择 | 理由 |
|------|------|------|
| delta_log.is_deleted 比较 | `delete_seq > data_seq`（严格大于） | _seq 单调递增，语义精确 |
| 全局去重数据来源 | Segment 内存中预提取的 pk/seq 数组 | 零磁盘 IO |
| Milvus 式优化时机 | MVP 不引入，Phase 2 渐进引入 | 先保证正确，再优化性能 |
| Upsert 原子性（未来） | 方案 A 单 WAL 文件为最终形态 | RecordBatch 级原子，根本解决 |
| _seq 去重的定位 | 正确性安全网，永远保留 | 即使引入 bitset 优化，仍作为兜底 |
| 去重时机（MVP） | 前置去重（搜索前） | 暴力搜索要求精确 top-k，不可丢结果 |
| 去重时机（FAISS） | 可切换后置去重 | ANN 本身近似，over-fetch 补偿足够 |
| PK 唯一性范围 | Collection 级（跨 Partition） | _seq 全局去重天然实现，比 Milvus 更强保证 |

---

# Part III: 全局去重详解与演进

---

## 19. 全局去重详细逻辑

### 19.1 去重规则

```
对于每一个 PK：
  1. 在所有 Sealed Segment + MemTable 中找出该 PK 的所有出现
  2. 保留 _seq 最大的那个（最新版本），淘汰其余
  3. 对保留的版本检查 delta_log：
     如果 delete_seq > data_seq → 已删除 → 过滤掉
```

### 19.2 delta_log.is_deleted 判定逻辑

```python
class DeltaLog:
    _deleted_map: Dict[Any, int]   # pk → max_delete_seq

    def is_deleted(self, pk, data_seq: int) -> bool:
        """判断一条数据记录是否已被删除。

        规则：存在 delete_seq 且 delete_seq > data_seq
              即删除操作发生在该条数据写入之后。
        """
        if pk not in self._deleted_map:
            return False
        return self._deleted_map[pk] > data_seq
```

**为什么是 `>` 而不是 `>=`？**

_seq 是单调递增的，insert 和 delete 不会拿到同一个 _seq：
- insert：每条记录分配独立 _seq（批量 N 条 = N 个不同 _seq）
- delete：批量共享一个 _seq，但与 insert 的 _seq 不会重复（来自同一个计数器）

所以 `>` 和 `>=` 实际等价，但 `>` 语义更精确：
"删除操作的 _seq 严格大于数据记录的 _seq" = "删除发生在写入之后"。

### 19.3 逐场景推演

**场景 1：正常 upsert（跨 Segment 去重）**

```
Segment A (早期 flush):  doc_1  _seq=100  vec=[0.1, ...]
Segment B (后来 flush):  doc_1  _seq=500  vec=[0.9, ...]

best["doc_1"] = max(100, 500) = 500 → Segment B 的版本胜出
delta_log.is_deleted("doc_1", 500) → "doc_1" 不在 deleted_map → False

seg_A_mask: doc_1 → False   ← 旧版本，跳过
seg_B_mask: doc_1 → True    ← 最新版本，参与搜索
```

**场景 2：Segment + MemTable 去重**

```
Segment A:  doc_1  _seq=100
MemTable:   doc_1  _seq=800   ← 最近又 upsert 了

best["doc_1"] = max(100, 800) = 800 → MemTable 胜出

seg_A_mask: doc_1 → False
mem_mask:   doc_1 → True
```

**场景 3：先删除，再插入（delete + re-insert）**

```
时间线：
  T1: insert doc_1 → _seq=100 → flush 到 Segment A
  T2: delete doc_1 → _seq=300 → delta_log: {doc_1: 300}
  T3: insert doc_1 → _seq=500 → flush 到 Segment B

去重：best["doc_1"] = max(100, 500) = 500 → Segment B

删除检查：
  delta_log._deleted_map = {"doc_1": 300}
  is_deleted("doc_1", data_seq=500)
  → 300 > 500?  → False → 没有被删除 ✅

结果：doc_1 可见，使用 Segment B 的版本
      （删除发生在重新插入之前，不影响新版本）
```

**场景 4：先 upsert，再删除**

```
时间线：
  T1: insert doc_1 → _seq=100 → Segment A
  T2: insert doc_1 → _seq=500 → Segment B (upsert)
  T3: delete doc_1 → _seq=700 → delta_log: {doc_1: 700}

去重：best["doc_1"] = max(100, 500) = 500 → Segment B

删除检查：
  is_deleted("doc_1", data_seq=500)
  → 700 > 500?  → True → 已删除

结果：doc_1 不可见（Segment A 的旧版本在去重阶段就被淘汰了，
      Segment B 的新版本在删除检查阶段被过滤了）
```

**场景 5：MemTable 中的 delete 覆盖 Segment 数据**

```
Segment A:  doc_1  _seq=100  (磁盘上的旧数据)
MemTable delete_buf:  delete doc_1, _seq=200

delta_log._deleted_map（含 MemTable delete_buf 和磁盘 delta 文件）:
  {"doc_1": 200}

去重：best["doc_1"] = 100 → Segment A（唯一 insert 版本）

删除检查：
  is_deleted("doc_1", 100) → 200 > 100 → True → 已删除

结果：doc_1 不可见 ✅
```

**场景 6：多次删除，保留最大 delete_seq**

```
时间线：
  T1: insert doc_1 → _seq=100
  T2: delete doc_1 → _seq=200
  T3: insert doc_1 → _seq=300   ← re-insert
  T4: delete doc_1 → _seq=400   ← 再次删除

delta_log._deleted_map = {"doc_1": 400}  ← 保留 max(200, 400) = 400

去重：best["doc_1"] = 300
删除检查：is_deleted("doc_1", 300) → 400 > 300 → True
结果：doc_1 不可见 ✅
```

### 19.4 一个重要性质：Segment 内 PK 唯一

单个 Segment 内部，每个 PK 只出现一次，原因：
- **MemTable flush**：insert_buf 是 dict（按 PK 去重），输出的 Table 每个 PK 只一行
- **Compaction 合并**：同 PK 保留 max_seq，输出唯一

因此全局去重只处理**跨 Segment 重复**，不需要处理段内重复。
`best` dict 的大小 = 唯一 PK 数（≤ 总行数），不会额外膨胀。

### 19.5 算法复杂度

```
Phase 1（收集 max_seq）:
  遍历所有 Segment 的 pks + seqs 数组 → O(N)  N = 总行数
  dict 查找/更新 → O(1) per entry

Phase 2（生成 mask）:
  遍历 best dict → O(U)  U = 唯一 PK 数
  delta_log.is_deleted() → O(1) per PK (dict 查找)

总计: O(N) 时间，O(U) 空间
```

### 19.6 全局信息的来源

搜索时不需要额外查询机制——**数据已全部在内存中**：

```
Collection._segments = {
    "_default": [seg_A, seg_B, seg_C],     ← Segment 对象常驻内存
}
Collection.memtable                         ← 内存中
Collection.delta_log._deleted_map           ← 内存中

每个 Segment 在加载时预提取了 NumPy 数组：
  seg_A.pks  = np.array(["doc_1", "doc_2", ...])
  seg_A.seqs = np.array([100, 101, ...])

全局去重 = 遍历这些内存数组 + 查 delta_log 内存 dict
         全程零磁盘 IO
```

---

## 20. Milvus 的去重方案对比

### 20.1 Milvus 的做法：写入时显式删除

Milvus 把去重代价从搜索时转移到写入时：

```
Milvus upsert("doc_1", new_vec):
  → Step 1: delete("doc_1")        ← 先显式删除所有旧版本
  → Step 2: insert("doc_1", ...)   ← 再插入新版本
  → 搜索时：每个 Segment 只看自己的 bitset，不需要跨 Segment 去重
```

### 20.2 关键机制：Bloom Filter

Milvus 每个 Sealed Segment 维护一个 Bloom Filter，记录该 Segment 含哪些 PK：

```
Segment A: bloom_filter_A = {doc_1, doc_2, doc_3 的指纹}
Segment B: bloom_filter_B = {doc_4, doc_5 的指纹}
Segment C: bloom_filter_C = {doc_1, doc_6 的指纹}
```

当 delete("doc_1") 到达时：

```
检查每个 Segment 的 Bloom Filter：
  bloom_filter_A.might_contain("doc_1") → True   → 标记 A 中 doc_1 为删除
  bloom_filter_B.might_contain("doc_1") → False  → 跳过（doc_1 肯定不在 B）
  bloom_filter_C.might_contain("doc_1") → True   → 标记 C 中 doc_1 为删除

复杂度：O(Segment 数量)，每个 Bloom Filter 查询 O(1)
```

Bloom Filter 特性：
- 说"不在" → **一定不在**（无假阴性）
- 说"可能在" → **可能误报**（假阳性，但无害——多记一条无效删除而已）

### 20.3 Per-Segment Bitset

每个 Segment 有自己的 bitset，搜索时只看本 Segment 的 bitset：

```
Segment A (1000 行):
  bitset = [1,1,0,1,1,0,1,...]     ← 0 = 已删除/过期，1 = 有效

搜索 Segment A 时：
  index.search(query, bitset=bitset)   ← FAISS 直接跳过 bitset=0 的行
  不需要知道其他 Segment 有什么 ← 与我们的方案的根本区别
```

### 20.4 两种方案全面对比

```
                        我们的设计                    Milvus
                        (隐式去重)                   (显式删除)
                        ─────────                    ─────────

upsert 实现             只写 wal_data               delete(旧) + insert(新)
写入复杂度              O(1)                        O(S) S=Segment 数（Bloom Filter 查询）
写入额外结构            无                           每个 Segment 维护 Bloom Filter
搜索时去重              全局扫描 O(N)                不需要（旧版本已被显式删除）
                        N=总行数                     O(1) per row（只看本段 bitset）
Compaction             清理旧版本 + 已删除            清理已删除
正确性保证              _seq 比较                     Bloom Filter + 显式 delete log
```

核心 trade-off：

```
我们：写入简单 O(1)，搜索代价 O(N)
      → 适合写多读少、数据量小的嵌入式场景

Milvus：写入代价 O(S)，搜索代价 O(1) per row
        → 适合读多写少、数据量大的生产环境
```

### 20.5 Milvus 搜索流程 vs 我们的搜索流程

```
Milvus search:                              我们的 search:
  │                                           │
  ├─ for each segment:                        ├─ 全局扫描所有 seg.pks + seg.seqs
  │    ├─ 读本 segment 的 bitset              │   构建 best = {pk → max_seq}
  │    │   (已包含删除+去重信息，                │   O(N)
  │    │    不需要跨 segment 查)               │
  │    ├─ index.search(query, bitset)         ├─ 生成 per-segment masks
  │    └─ 返回 local top-k                    │
  │                                           ├─ for each segment:
  ├─ merge local top-k → global top-k        │    seg.search(query, mask, top_k)
  │                                           │
  └─ done                                    ├─ merge → global top-k
                                              └─ done

  搜索时零全局协调                              搜索时需要全局扫描去重
```

---

## 21. 未来演进：Upsert 原子性问题与解决方案

### 21.1 问题：显式删除引入双文件写入

一旦采用 Milvus 风格的 upsert = delete + insert，一次 upsert 就需要同时写 wal_data 和 wal_delta：

```
upsert("doc_1", new_vec)
  │
  ├─ wal_delta.write_delete(doc_1)     ← 删除旧版本
  │                               ← ⚡ 崩溃
  └─ wal_data.write_insert(doc_1)      ← 插入新版本
```

**崩溃在两步之间：delete 写了但 insert 没写 → doc_1 被删除，新版本丢失 → 数据丢了。**

反过来先 insert 后 delete：

```
  ├─ wal_data.write_insert(doc_1, _seq=500)    ← 先写 insert
  │                                       ← ⚡ 崩溃
  └─ wal_delta.write_delete(doc_1)              ← 再写 delete

崩溃：insert 在，delete 不在
→ 旧版本 (_seq=100) 没被显式删除
→ 新版本 (_seq=500) 存在
→ 同一个 PK 出现两次（如果只靠 per-segment bitset，不做 _seq 去重）
```

**不管什么顺序，双文件写入都无法保证原子性。**

### 21.2 方案 A：合并为单 WAL 文件（根本解决）

把 wal_data 和 wal_delta 合并为一个 WAL 文件，用 `_op` 列区分操作类型。

**合并后的 WAL schema**：

```
_op:        utf8          "INSERT" | "DELETE"
_seq:       uint64
_partition: utf8
{pk_field}: ...
{vec_field}: list<f32>    (DELETE 行填 null)
{其他字段}: ...            (DELETE 行填 null)
$meta:      utf8          (DELETE 行填 null)
```

**Upsert 写入一个 RecordBatch**，包含两行：

```
一次 write_batch() 调用，两行在同一个 RecordBatch 中：

  Row 0: _op="DELETE", _seq=499, pk="doc_1", vec=null,       ...
  Row 1: _op="INSERT", _seq=500, pk="doc_1", vec=[0.9,...],  ...
```

**原子性保证**：

```
Arrow IPC write_batch() 要么完整写入一个 RecordBatch，要么截断丢弃。
两行在同一个 RecordBatch 中 → 同生共死：
  写入成功 → delete + insert 都在
  写入中崩溃 → recovery 丢弃不完整的 batch → 都不在
  ✅ RecordBatch 级原子
```

Recovery 时按 `_op` 列分流：

```python
for batch in wal_batches:
    for row in range(batch.num_rows):
        op = batch.column("_op")[row].as_py()
        if op == "INSERT":
            memtable.put(seq, partition, **fields)
        elif op == "DELETE":
            memtable.delete(pk, seq, partition)
            # + 通过 Bloom Filter 更新相关 Segment 的 bitset
```

### 21.3 方案 B：保持双文件 + 写入顺序 + _seq 兜底（渐进方案）

保持双文件结构不变，利用 _seq 作为最终正确性保证：

```
写入顺序：先 insert，后 delete

upsert("doc_1", new_vec):
  ├─ Step A: wal_data.write_insert(doc_1, _seq=500)    ← 先写 insert
  └─ Step B: wal_delta.write_delete(doc_1, _seq=499)   ← 再写 delete

崩溃在 A 和 B 之间：
  insert 在，delete 不在
  → 旧版本 (_seq=100) 没被显式删除
  → 新版本 (_seq=500) 存在
  → 但 _seq 去重仍然生效：同 PK 保留 max_seq → 500 > 100 → 新版本胜出 ✅
```

**本质**：Bloom Filter + per-segment bitset 是**性能优化层**，_seq 去重是**正确性保证层**。两层配合：

```
正常情况（99.9%）：
  bitset 快速过滤（O(1) per row） → 不需要全局扫描

异常情况（崩溃导致 bitset 不完整）：
  _seq 去重兜底 → 正确性不受影响
  Compaction 时修复 bitset → 最终恢复到正常状态
```

### 21.4 方案 C：Recovery 补偿（方案 B 的增强）

在方案 B 基础上，Recovery 时主动检测并补全不完整的 upsert：

```python
def _fix_incomplete_upserts(wal_data_batches, wal_delta_batches, segments):
    """检测 insert 有但对应 delete 缺失的情况，补发 delete。"""

    # 收集 WAL 中 insert 的 PK
    inserted_pks = set()
    for batch in wal_data_batches:
        for row in range(batch.num_rows):
            inserted_pks.add(batch.column(pk_field)[row].as_py())

    # 收集 WAL 中 delete 的 PK
    deleted_pks = set()
    for batch in wal_delta_batches:
        for row in range(batch.num_rows):
            deleted_pks.add(batch.column(pk_field)[row].as_py())

    # insert 了但没 delete 的 PK → 可能是崩溃导致的不完整 upsert
    missing_deletes = inserted_pks - deleted_pks

    for pk in missing_deletes:
        for seg in segments:
            if seg.bloom_filter.might_contain(pk):
                seg.mark_deleted(pk)    # 补全 per-segment bitset
```

### 21.5 三种方案对比

| | 方案 A：单 WAL 文件 | 方案 B：双文件 + _seq 兜底 | 方案 C：Recovery 补偿 |
|--|-------------------|------------------------|-------------------|
| 原子性 | RecordBatch 级原子 | 不原子，靠 _seq 兜底 | 不原子，靠 Recovery 修复 |
| WAL 改动 | 大（合并为单文件，schema 加 _op + nullable） | 无 | 无 |
| 搜索正确性 | 完美 | _seq 兜底保证（偶尔需全局去重） | Recovery 后完美 |
| 复杂度 | schema 改动 | 搜索需保留 _seq 去重路径 | Recovery 加检测逻辑 |
| 性能 | DELETE 行的 null 字段浪费少量空间 | 异常时搜索退化 | 无额外搜索开销 |

### 21.6 推荐演进路径

```
MVP（当前设计）
  upsert = 纯 insert，_seq 隐式去重
  双 WAL 文件，没有原子性问题（每个 API 只写单文件）
  搜索时全局扫描去重，O(N)
  ✅ 简单正确

Phase 2（加 Bloom Filter + bitset 优化搜索性能）
  采用方案 B：upsert = 先 insert 后 delete，_seq 兜底
  正常情况 bitset 快速过滤，异常情况 _seq 兜底
  可选方案 C 在 Recovery 时补偿
  ✅ 渐进优化，不破坏已有正确性

Phase 3（追求完美原子性）
  采用方案 A：合并为单 WAL 文件 + _op 列
  RecordBatch 级原子，彻底消除不一致窗口
  可以安全降低 _seq 全局去重的权重（仍保留作为防御性检查）
  ✅ 最终形态
```

### 21.7 _seq 去重的定位

**_seq 去重是系统的安全网，贯穿所有演进阶段，不应被移除。**

```
即使 Phase 3 实现了完美的单 WAL 原子性 + Bloom Filter + per-segment bitset：
  → 仍保留 _seq 去重作为防御性检查
  → 防止 Bloom Filter 误报 + 代码 bug + 未预见的边界情况
  → 代价极低（已经有 _seq 列，比较操作 O(1)）
  → 收益极高（最后一道正确性防线）
```

---

## 22. 前置去重 vs 后置去重

### 22.1 后置去重的思路

不在搜索前做全局去重，而是搜索后再过滤：

```
前置去重（当前方案）：
  全局扫描 PK+_seq → 生成 mask → 只搜有效行 → 合并

后置去重（替代方案）：
  每个 Segment 独立搜索（含过期数据）→ 合并 → 去重 → 取 top-k
```

后置去重更简单——每个 Segment 完全独立，不需要搜索前的全局扫描。

### 22.2 问题：过期数据挤占 local top-k 名额

```
场景：top_k = 3

Segment A (旧):
  doc_1  _seq=100  vec=[0.1,...]  dist=0.05  ← 过期版本，离 query 很近
  doc_2                           dist=0.30
  doc_7                           dist=0.31
  doc_4                           dist=0.32  ← 第 4 名
  ...

Segment B (新):
  doc_1  _seq=500  vec=[0.9,...]  dist=0.80  ← 最新版本，离 query 远
  doc_3                           dist=0.40
```

**前置去重（正确结果）**：

```
去重：doc_1 → Segment B 胜 (_seq=500)
Segment A 有效行搜索：doc_2(0.30), doc_7(0.31), doc_4(0.32) → local top-3
Segment B 有效行搜索：doc_1(0.80), doc_3(0.40) → local top-2

合并 top-3: [doc_2(0.30), doc_7(0.31), doc_4(0.32)]  ✅
```

**后置去重（丢结果）**：

```
Segment A 搜索全部（含过期 doc_1）：
  local top-3: [doc_1(0.05), doc_2(0.30), doc_7(0.31)]
                  ↑ 过期数据占了名额，doc_4(0.32) 排第 4 被截断

Segment B：
  local top-3: [doc_3(0.40), doc_1(0.80)]

合并: [doc_1(A,0.05), doc_2(0.30), doc_7(0.31), doc_3(0.40), doc_1(B,0.80)]
去重: doc_1 保留 B 版本(0.80)，丢弃 A 版本(0.05)
结果: [doc_2(0.30), doc_7(0.31), doc_3(0.40)]

正确答案: [doc_2(0.30), doc_7(0.31), doc_4(0.32)]
                                       ↑ doc_4 丢了！
```

**根因**：过期的 doc_1（dist=0.05）在 Segment A 的 local top-3 中挤掉了 doc_4（dist=0.32）。
去重后 doc_1 被丢弃，但 doc_4 已经在 local top-k 截断时消失了，无法找回。

### 22.3 缓解方案：Over-fetch

每个 Segment 取 `top_k × factor` 而不是 `top_k`，给去重留出裕量：

```
factor = 2：Segment A 取 top-6 而不是 top-3
  → [doc_1(0.05), doc_2(0.30), doc_7(0.31), doc_4(0.32), ...]
  → doc_4 保住了 ✅

合并 → 去重 → 取 top-3
  → [doc_2(0.30), doc_7(0.31), doc_4(0.32)] ✅
```

**但 factor 多大才够？**

```
取决于"有多少 PK 跨 Segment 重复"：
  刚 flush，几乎无重复         → factor=1.1 够了
  大量 upsert，很多重复        → 需要 factor=3, 5, 甚至更大
  极端情况：所有 PK 都 upsert 过 → 需要 factor≥2
  无法提前知道 → 只能猜，猜错就丢结果
```

### 22.4 两种方案对比

| | 前置去重 | 后置去重 + over-fetch |
|--|--------|---------------------|
| 正确性 | 精确（不丢结果） | 可能丢结果（factor 不够大时） |
| 实现复杂度 | 需要全局扫描 PK+_seq | 每段独立，更简单 |
| 距离计算量 | 只算有效行 | 过期行也要算（浪费） |
| 适用场景 | 暴力搜索（要求精确） | ANN 近似搜索（本身就不精确） |
| 分布式友好度 | 差（需全局协调） | 好（每段独立） |

### 22.5 结论：取决于搜索精度要求

```
暴力搜索（MVP）→ 必须前置去重
  用户期望精确的 top-k，丢结果不可接受
  数据全在内存中，O(N) 全局扫描代价可接受

FAISS ANN（未来）→ 后置去重可接受
  ANN 本身就是近似的，丢一个边界结果影响不大
  每段有独立 FAISS 索引，前置去重收益更小
    （FAISS 搜索复杂度不随数据量线性增长，过滤几条过期数据省不了多少）
  over-fetch factor=2 在大多数场景够用
  分布式搜索时更自然（segment 可在不同节点上）
```

### 22.6 与演进路径的关系

```
MVP:     前置去重（精确） + 暴力搜索
Phase 2: 前置去重（精确） + Bloom Filter 优化写入端
Phase 3: 可选切换到后置去重 + FAISS ANN
         此时 _seq 去重仍作为安全网：
         即使后置去重丢了边界结果（精度问题），
         _seq 机制至少保证不会返回过期数据的错误内容
```

---

## 23. Milvus 跨 Partition 重复 PK 调研

### 23.1 Milvus 的行为：不做任何 PK 唯一性检查

Milvus 官方文档明确声明：

> "Milvus does not support primary key de-duplication for now. Therefore, there can be duplicate primary keys in the same collection."

同一个 PK 写入两个不同 Partition，**两条记录都会被保存**，不报错、不覆盖。

### 23.2 读取时的行为（不一致）

| 操作 | 行为 | 问题 |
|------|------|------|
| query(pk=X) | 返回**最早插入的那条**（post-reduce 去重，保留第一条找到的） | 不一定是最新版本 |
| search(ANN) | **可能返回多条同 PK 结果**，取向量最相似的那条 | 语义不明确 |
| count(\*) | **包含所有重复**，PK 插入 2 次则 count=2 | 计数不准 |
| Limit=10 | 先取 10 条再去重 → 实际返回可能 **< 10 条** | 分页不可靠 |

### 23.3 Upsert 也不跨 Partition

Milvus 的 `upsert()` 是 **Partition 级别的** delete + insert：

```
Partition A 有 PK=1
调用 upsert(PK=1, partition=B)

→ Partition A 的记录 不会被删除
→ 结果：Partition A 和 B 各有一条 PK=1
```

要跨 Partition 清理，必须手动 `delete(pk=1, partition_name=None)`（None 表示全 Partition 扫描）。

### 23.4 Compaction 也不跨 Partition

Compaction 在 Partition 内部的 Segment 间合并，**不处理跨 Partition 的重复 PK**。

### 23.5 Milvus 的官方态度

- Issue #36199 标记为 **"resolution/by-design"** — 承认这是已知行为，不视为 bug
- **Global PK Dedup** 列入 Roadmap，目标 v2.6 / v3.0（尚未实现）
- 当前官方建议：用 `auto_id=True` 避免问题，或应用层自行去重

相关 Issue 汇总：

| Issue / PR | 标题 | 要点 |
|------------|------|------|
| Issue #36199 | Duplicate primary key values leads to inconsistent query results | 标记 by-design；query 返回最早版本，search 返回最相似版本 |
| Issue #28615 | Enhance constraints for inserting duplicate primary key data | Limit 参数因去重后变少 |
| Issue #31552 | Support primary key dedup and vector dedup when insert | 特性请求，分配到 milestone 3.0 |
| Issue #37389 | Does Milvus actually perform deduplication? | 仅 upsert 去重，insert 不去重 |
| Issue #33353 | Insert data with duplicate primary key | 默默接受重复 PK |
| Discussion #18202 | The Scope of Primary Key | PK 范围是 Collection 级，但唯一性不保证 |
| Discussion #18201 | What happens if I insert data with the same id several times? | query 返回第一个；search 可能返回多个同 PK |
| PR #10967 | Remove primary key duplicated query result on proxy | Proxy 层 post-reduce 去重实现 |

### 23.6 对 MilvusLite 的设计启示

**PK 唯一性范围有三种选择**：

| 方案 | PK 唯一性范围 | 优点 | 缺点 |
|------|-------------|------|------|
| A: Collection 级唯一 | 跨所有 Partition | 语义清晰，search/query 结果一致 | insert 时需要全局检查（或靠 _seq 去重兜底） |
| B: Partition 级唯一 | 仅 Partition 内 | 实现简单，Partition 间完全隔离 | 同 Milvus 问题：跨 Partition 查询出现重复 |
| C: 不保证唯一（同 Milvus） | 无 | 最简单 | 结果不可预测，count 不准 |

**我们选择方案 A，且已经天然实现了**：

```
MilvusLite 的 _seq 全局去重机制天然提供 Collection 级 PK 唯一性：

1. insert 时不需要额外检查
   → 直接写入，不查旧 Partition

2. search 时 build_segment_masks 全局扫描所有 Partition 的 Segment
   → 遍历所有 seg.pks + seg.seqs
   → 只保留 max_seq → 同 PK 跨 Partition 自动取最新版本
   → 过期版本自动被过滤，不管在哪个 Partition

3. delete(partition_name=None) 已设计为全 Partition 扫描
   → 跨 Partition 删除也是原生支持的

4. get(pks, partition_names=None) 同理
   → 全局 _seq 去重，返回最新版本
```

**这是比 Milvus 更强的语义保证，且不需要额外开销**：

```
                        Milvus                     MilvusLite
                        ────────                   ──────────
PK 唯一性范围            不保证                      Collection 级（跨 Partition）
upsert 跨 Partition     不处理（只在目标 Partition    _seq 去重自动处理
                        内 delete+insert）
search 跨 Partition     可能返回重复 PK              全局去重，每个 PK 只返回最新版本
                        query/search 行为不一致
count 准确性            不准（含重复）                准确（去重后计数）
额外实现开销            需要 Global PK Dedup          无（_seq 天然支持）
                        (Roadmap v3.0)
```

**但也有 trade-off**：

```
代价：搜索时全局扫描 O(N) 做去重
     → MVP 可接受（全量暴力搜索本身就 O(N)，去重扫描相比之下微不足道）
     → Phase 2 引入 Bloom Filter + bitset 后，可把去重从搜索热路径移到写入端
```
