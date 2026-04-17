# Storage 子系统设计

## 1. 概述

Storage 层是 LiteVecDB 的持久化基础，实现了 LSM-Tree 风格的写入流水线：

```
用户写入 → WAL (Arrow IPC) → MemTable (内存) → Flush → Parquet (磁盘)
```

所有持久化状态由 Manifest 作为 single source of truth 管理。

### 设计原则

| 原则 | 体现 |
|------|------|
| 文件不可变 | Parquet 文件一旦写入永不修改，只能整体删除 |
| `_seq` 全序 | 所有覆盖/丢弃/去重判断基于 `_seq`，不依赖调用顺序 |
| Insert/Delete 分离 | 数据流和删除流使用不同的文件类型和 schema |
| 原子 Manifest | tmp + rename 保证持久化原子性 |
| 崩溃安全 | WAL 保证写入持久性，recovery 保证一致性恢复 |

### 代码位置

| 文件 | 职责 |
|------|------|
| `storage/wal.py` | Write-Ahead Log（Arrow IPC Streaming，双文件） |
| `storage/memtable.py` | 内存写缓冲（RecordBatch 列表 + 双索引） |
| `storage/data_file.py` | Data Parquet 文件读写 |
| `storage/delta_file.py` | Delta Parquet 文件读写（tombstone） |
| `storage/delta_index.py` | 内存删除水位图（pk → max_delete_seq） |
| `storage/segment.py` | 不可变 Parquet 缓存 + 向量索引绑定 |
| `storage/manifest.py` | 原子状态快照（JSON + .prev 备份） |
| `engine/flush.py` | Flush 管线（7 步同步） |
| `engine/recovery.py` | 崩溃恢复管线（5 步） |

---

## 2. 磁盘布局

```
data_dir/
├── manifest.json              # 当前 Manifest
├── manifest.json.prev         # 上一版本备份
├── schema.json                # Collection schema（不可变）
├── LOCK                       # fcntl.flock 进程锁
├── wal/
│   ├── wal_data_000001.arrow  # WAL 数据流（Arrow IPC）
│   └── wal_delta_000001.arrow # WAL 删除流（Arrow IPC）
└── partitions/
    ├── _default/
    │   ├── data/
    │   │   ├── data_000001_000050.parquet
    │   │   └── data_000051_000100.parquet
    │   ├── delta/
    │   │   └── delta_000030_000030.parquet
    │   └── indexes/
    │       └── data_000001_000050.vec.hnsw.idx
    └── p1/
        ├── data/
        ├── delta/
        └── indexes/
```

### 文件命名规则

| 文件类型 | 模板 | 示例 |
|----------|------|------|
| Data Parquet | `data_{seq_min:06d}_{seq_max:06d}.parquet` | `data_000001_000050.parquet` |
| Delta Parquet | `delta_{seq_min:06d}_{seq_max:06d}.parquet` | `delta_000030_000030.parquet` |
| WAL Data | `wal_data_{number:06d}.arrow` | `wal_data_000001.arrow` |
| WAL Delta | `wal_delta_{number:06d}.arrow` | `wal_delta_000001.arrow` |
| Vector Index | `{data_stem}.{field}.{type}.idx` | `data_000001_000050.vec.hnsw.idx` |

---

## 3. 四种 Arrow/Parquet Schema

Insert 和 Delete 是独立数据流，WAL 和磁盘使用不同 schema（共 4 种）：

| Schema | 用途 | 列 |
|--------|------|-----|
| `wal_data` | WAL 数据流 | `_seq`, `_partition`, pk, 用户字段..., `[$meta]` |
| `wal_delta` | WAL 删除流 | `_seq`, `_partition`, pk |
| `data` | Data Parquet | `_seq`, pk, 用户字段..., `[$meta]` |
| `delta` | Delta Parquet | `_seq`, pk |

**关键区别**：WAL schema 包含 `_partition` 列（因为 WAL 是 Collection 级共享，需标记每行属于哪个 partition）；Parquet schema 不含 `_partition`（因为 partition 已编码在目录路径中）。

`[$meta]` 是动态字段的 JSON 聚合列，用于存储 schema 中未显式定义的扩展字段。

---

## 4. WAL（Write-Ahead Log）

### 4.1 设计

WAL 使用 Arrow IPC Streaming 格式，每个 WAL 轮次由一对文件组成：

```
wal_data_{N}.arrow   ← insert 流（自描述 schema + RecordBatch 序列）
wal_delta_{N}.arrow  ← delete 流（自描述 schema + RecordBatch 序列）
```

双文件设计的原因：insert 和 delete 的 schema 不同，Arrow IPC 流要求整个文件使用同一 schema。

### 4.2 同步模式

```python
SYNC_MODES = ("none", "close", "batch")
```

| 模式 | fsync 时机 | 适用场景 |
|------|-----------|----------|
| `"none"` | 从不 | 测试/基准测试 |
| `"close"` | 关闭前一次性 fsync | **默认**，覆盖容器 OOM 重启场景 |
| `"batch"` | 每次 write 后 fsync | 最强持久性，最慢 |

### 4.3 生命周期

```
创建: Collection 初始化 / flush 后创建新 WAL
写入: write_insert / write_delete（lazy 初始化 writer）
关闭: close_and_delete（flush Step 6 后删除）
```

**Lazy 初始化**：writer 在首次写入时才创建文件，避免空 WAL 文件占磁盘。

**Exception-safe 关闭**：`close_and_delete` 收集所有异常，尝试关闭每个 writer/sink 并删除文件，最后重抛第一个异常。`_closed` 标记保证幂等。

### 4.4 截断容忍

Recovery 读取 WAL 时使用截断容忍策略：

```python
def _read_wal_file(path: str) -> List[pa.RecordBatch]:
    # 文件不存在 → []
    # 文件完整   → 所有 batch
    # 文件截断   → 截断点之前的 batch（静默丢弃不完整尾部）
    # 严重损坏   → []
```

这保证了进程在写入中间崩溃时，已完成的 batch 仍可恢复。

---

## 5. MemTable（内存写缓冲）

### 5.1 数据结构

```python
class MemTable:
    _insert_batches: List[pa.RecordBatch]   # append-only，物理存储
    _pk_index: Dict[pk, (batch_idx, row_idx, seq)]  # 活跃行索引
    _delete_index: Dict[pk, (max_delete_seq, partition)]  # tombstone 索引
```

**append-only 设计**：写入时仅追加 RecordBatch，不复制/修改 Arrow 缓冲区。旧行通过索引失效（shadow）而非物理删除。

### 5.2 Seq-aware 写入

MemTable 的 `apply_insert` 和 `apply_delete` 是**顺序无关**的——任意交错调用都能得到正确的最终状态（架构不变量 §1-2）。

**apply_insert 逻辑**：
```
对每一行 (pk, seq):
  if delete_index[pk].seq >= seq → 跳过（更新的删除阻止旧插入）
  if pk_index[pk].seq >= seq     → 跳过（更新的插入覆盖旧插入）
  否则 → 更新 pk_index，清除 delete_index 中该 pk 的条目
```

**apply_delete 逻辑**：
```
对每一行 (pk, seq):
  if pk_index[pk].seq >= seq → 跳过（更新的插入阻止旧删除）
  否则 → 更新 delete_index（取 max seq），从 pk_index 中驱逐该 pk
```

### 5.3 读取路径

| 方法 | 用途 | 返回 |
|------|------|------|
| `get(pk)` | 点查 | `Optional[dict]` |
| `get_active_records(partitions)` | 全量活跃记录 | `List[dict]` |
| `to_search_arrays(vector_field, partitions)` | 搜索用 numpy 数组 | `(pks, seqs, vectors, row_refs)` |
| `to_arrow_table(partitions)` | 标量过滤用 Arrow Table | `pa.Table` |
| `materialize_row(batch_idx, row_idx)` | 延迟物化单行 | `dict` |

**延迟物化**：search 路径只提取 pk/seq/vector，不物化完整行。只有 top-k 结果的行才通过 `materialize_row` 获取全部字段。

### 5.4 Flush

```python
def flush(known_partitions) -> Dict[str, Tuple[Optional[Table], Optional[Table]]]:
    # 1. 遍历 pk_index，按 _partition 分组活跃行
    # 2. 构建 per-partition data Table（data schema，无 _partition 列）
    # 3. 遍历 delete_index，按目标 partition 分组
    # 4. 跨 partition 删除（ALL_PARTITIONS）复制到所有 known_partitions
    # 5. 构建 per-partition delta Table
    # 6. 合并返回 {partition: (data_table, delta_table)}
```

MemTable 本身不被清除——调用方（Collection）丢弃冻结副本。

### 5.5 大小度量

```python
def size() -> int:
    return len(pk_index) + len(delete_index)  # 活跃行 + tombstone
```

当 `size() >= MEMTABLE_SIZE_LIMIT`（默认 10,000）时触发 flush。这是**逻辑大小**，不是物理行数（`_insert_batches` 中包含被 shadow 的旧行）。

---

## 6. Data / Delta 文件

### 6.1 Data Parquet

存储用户数据行，使用 `data` schema（`_seq` + pk + 用户字段 + `[$meta]`）。

```python
write_data_file(table, partition_dir, seq_min, seq_max) → rel_path
read_data_file(path) → pa.Table
```

**不可变**：写入后永不修改。由 compaction 合并后产生新文件，旧文件在 manifest 更新后删除。

**路径是相对的**：返回相对于 `partition_dir` 的路径（如 `data/data_000001_000050.parquet`），使 `data_dir` 可重定位。

### 6.2 Delta Parquet

存储删除 tombstone，使用 `delta` schema（`_seq` + pk）。

```python
write_delta_file(table, partition_dir, seq_min, seq_max) → rel_path
read_delta_file(path) → pa.Table
```

内容极简——只有主键和序列号，没有完整记录。

### 6.3 Seq 范围解析

```python
parse_seq_range(filename) → (seq_min, seq_max)
# "data_000001_000050.parquet" → (1, 50)
# "delta_000030_000030.parquet" → (30, 30)
```

文件名编码 seq 范围，供 compaction 分桶和 tombstone GC 使用，无需打开文件即可获取元数据。

---

## 7. DeltaIndex（删除水位图）

### 7.1 语义

```
is_deleted(pk, data_seq) ⟺ _map.get(pk, -1) > data_seq
```

严格大于：如果 delete_seq == data_seq，数据行**不被视为已删除**（same-seq 不可能发生因为 seq 全局唯一，但语义上明确）。

### 7.2 生命周期

```
启动:   DeltaIndex.rebuild_from(delta_files)    ← recovery Step 5
运行时: delta_index.add_table(delta_table)       ← flush Step 4
GC:    delta_index.gc_below(min_active_seq)     ← compaction 后
```

### 7.3 并发隔离

```python
def frozen_copy() -> DeltaIndex:
    copy = DeltaIndex(self._pk_name)
    copy._map = dict(self._map)  # O(N) 浅拷贝
    return copy
```

读路径（search/query/get/num_entities）在请求开始时调用 `frozen_copy()`，获取快照后独立使用。后台 compaction 的 `gc_below` 修改 live DeltaIndex 不影响正在进行的读操作。

### 7.4 Tombstone GC

```python
def gc_below(min_active_data_seq) -> int:
    # 删除所有 delete_seq < min_active_data_seq 的条目
    # 正确性：没有任何 data 文件包含 seq_min <= delete_seq 的行
    # → 这些 tombstone 不可能被任何数据行需要
```

---

## 8. Segment（不可变 Parquet 缓存）

### 8.1 设计

每个 Segment 对应一个 data Parquet 文件，加载后缓存：

```python
class Segment:
    file_path: str                    # 源 Parquet 绝对路径
    partition: str                    # partition 名
    pks: List[Any]                    # 主键列表
    seqs: np.ndarray[uint64]          # 序列号数组
    vectors: np.ndarray[float32]      # 向量矩阵 (N, dim)
    vector_null_mask: Optional[np.ndarray]  # null 向量掩码
    table: pa.Table                   # 原始表（用于字段提取）
    pk_to_row: Dict[pk, row_idx]      # O(1) 点查索引
    indexes: Dict[str, VectorIndex]   # per-field 向量索引
```

**加载时预计算**：numpy 数组在 `Segment.load()` 时一次性提取，摊销到后续所有搜索。

**null 向量处理**：null 向量用零向量填充，`vector_null_mask` 标记哪些行有效（True=有效）。搜索时通过 bitmap pipeline 过滤 null 行。

### 8.2 索引绑定（Phase 9）

Segment 与 VectorIndex 1:1 绑定：

```python
def build_or_load_index(spec, index_dir):
    # 如果 .idx 文件已存在 → 加载
    # 否则 → 从 vectors 构建 + 持久化 .idx
    # 幂等：已有索引则跳过

def attach_index(index, field_name):
    self.indexes[field_name] = index

def release_index(field_name=None):
    # field_name=None → 释放所有索引
```

**生命周期对齐**（架构不变量 §11）：compaction 删除旧 segment 时，对应的 .idx 文件由 `_cleanup_orphan_index_files` 清理。

---

## 9. Manifest（原子状态快照）

### 9.1 持久化状态

```python
_version: int                                      # 每次 save() +1
_current_seq: int                                  # 已 flush 的最大 _seq
_schema_version: int                               # schema 版本
_active_wal_number: Optional[int]                  # 当前 WAL 编号
_partitions: Dict[str, Dict[str, List[str]]]      # {partition: {data_files, delta_files}}
_index_specs: Dict[str, IndexSpec]                 # 索引规格（Phase 9.3+）
```

### 9.2 原子写入协议

```
save():
  1. 构建 payload（版本号 +1，但不立即更新内存状态）
  2. 写入 manifest.json.tmp + fsync
  3. 复制 manifest.json → manifest.json.prev（best-effort）
  4. os.rename(manifest.json.tmp, manifest.json)  ← commit point
  5. 更新内存 _version（仅在 rename 成功后）
  6. fsync 目录（best-effort）
```

**崩溃安全**：
- rename 前失败 → `.tmp` 被清理，内存版本不变
- rename 成功 → 一个版本被原子提交
- `.prev` 复制失败 → 非致命，不阻断 rename

### 9.3 加载回退链

```
load():
  1. 尝试 manifest.json
  2. 失败 → 尝试 manifest.json.prev（warning）
  3. 都失败 → ManifestCorruptedError
  4. 都不存在 → 返回全新 Manifest
```

### 9.4 格式演进

`MANIFEST_FORMAT_VERSION = 2`（Phase 9.3 新增 `index_spec` 字段）。v1 manifest 加载正常——`index_spec` 默认 None，下次 `save()` 透明升级到 v2。

---

## 10. Flush 管线

### 10.1 整体流程

```
Collection._trigger_flush():
  Step 1: 冻结当前 MemTable + WAL，创建新的一对
  └── execute_flush(frozen_memtable, frozen_wal, ...):
      Step 2: frozen_memtable.flush() → {partition: (data, delta)}
      Step 3: 写 data + delta Parquet 文件
      Step 4: 将 delta 表折入 delta_index（内存提交）
      Step 5: Manifest 原子更新
      Step 6: WAL 关闭 + 删除 + 清理旧 WAL
      Step 7: 触发后台 compaction（异步）
```

### 10.2 Step 4 在 Step 5 之前的原因

delta_index 在 manifest.save() 之前更新，原因：

1. **读一致性**：如果反过来（先 save 后更新 delta_index），并发 reader 可能看到新版 manifest 但拿到旧的 `frozen_copy()`，导致返回应被删除的行
2. **崩溃安全**：WAL 在 Step 6 才删除。如果 crash 发生在 Step 4-5 之间，recovery 会重放 WAL 重建 delta_index
3. **窗口短暂**：Step 4 和 Step 5 之间只有内存操作，窗口极小

### 10.3 空 MemTable 快速路径

```python
if not flushed:
    manifest.active_wal_number = new_wal_number
    manifest.save()
    frozen_wal.close_and_delete()
    return
```

即使 MemTable 为空，也必须更新 `active_wal_number` 并清理冻结 WAL，否则 WAL 会成为孤儿。

---

## 11. Recovery 管线

### 11.1 整体流程

```
execute_recovery(data_dir, schema, manifest):
  前提: Manifest 已加载（含 .prev 回退）

  Step 2: 重放 WAL
    ├── WAL.find_wal_files() → 发现磁盘上所有 WAL 编号
    ├── replay_wal_operations() → 按 _seq 排序的 Operation 流
    └── memtable.apply_insert / apply_delete（顺序无关）

  Step 3: 验证 Manifest 引用的文件存在
    └── 缺失文件仅 warning，不 crash（允许手动恢复）

  Step 4: 清理孤儿文件
    ├── data/ 目录中不在 manifest 中的 Parquet → 删除
    ├── delta/ 目录中不在 manifest 中的 Parquet → 删除
    └── indexes/ 目录中源 data 文件不在 manifest 中的 .idx → 删除

  Step 5: 从 delta Parquet 文件重建 DeltaIndex
    └── DeltaIndex.rebuild_from(pk_name, {partition: [abs_paths]})

  返回: (memtable, delta_index, next_wal_number)
```

### 11.2 顺序无关性

WAL 重放的正确性**不依赖操作顺序**（架构不变量 §2）。`replay_wal_operations` 按 `_seq` 排序仅是为了 "last yielded op.seq is the new max seq" 的整洁性。MemTable 的 seq-aware 逻辑保证任意顺序都得到相同最终状态。

### 11.3 next_wal_number 推导

```python
candidates = []
if found_wal_numbers:
    candidates.append(max(found_wal_numbers) + 1)
if manifest.active_wal_number is not None:
    candidates.append(manifest.active_wal_number)
next_wal_number = max(candidates) if candidates else 1
```

取最大值以避免编号冲突。

---

## 12. 崩溃安全总结

### 写入路径

```
用户 insert/delete
  → WAL.write_insert/write_delete      sync_mode 控制 fsync
  → MemTable.apply_insert/apply_delete  纯内存
```

崩溃时：WAL 文件已持久化的 batch 可恢复，MemTable 丢失但 WAL 重放补回。

### Flush 路径

| 崩溃点 | 影响 | 恢复 |
|--------|------|------|
| Step 3（写 Parquet）| orphan 文件 | recovery Step 4 清理 |
| Step 5（manifest save）| 原子 rename | 旧或新 manifest |
| Step 5 后 Step 6 前 | manifest 已更新，WAL 残留 | WAL 重放幂等（seq 去重）|
| Step 6（WAL 删除）| 部分 WAL 残留 | `_cleanup_old_wals` 清理 |

### Compaction 路径

| 崩溃点 | 影响 | 恢复 |
|--------|------|------|
| 写新文件后 manifest 前 | orphan 新文件 | recovery 清理 |
| manifest 更新后删旧文件前 | orphan 旧文件 | recovery 清理 |
| tombstone GC（内存）| delta_index 状态丢失 | 从 delta 文件重建 |
| delta file GC | manifest 已保存 | orphan delta 文件被清理 |

**核心原则**：所有路径上，**manifest 是 commit point**。持久化变更先写 manifest，再执行物理文件操作。物理操作失败只产生 orphan，由 recovery 清理。

---

## 13. 数据流全景

```
                          ┌──────────────────────────┐
                          │     用户 API 层           │
                          │  insert / delete / search │
                          └────────┬─────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────────┐
              │   WAL    │  │ MemTable │  │   Segment    │
              │(持久化)  │  │ (内存)   │  │   Cache      │
              └────┬─────┘  └────┬─────┘  │ (不可变缓存) │
                   │             │        └──────┬───────┘
                   │        flush│               │ 读取
                   │             ▼               │
                   │    ┌────────────────┐       │
                   │    │  Flush 管线    │       │
                   │    │ (7 步同步)     │       │
                   │    └───┬────┬───────┘       │
                   │        │    │               │
                   │        ▼    ▼               │
                   │   Data    Delta             │
                   │  Parquet  Parquet            │
                   │    │        │               │
                   │    │        ▼               │
                   │    │   DeltaIndex ◀─────────┤
                   │    │   (内存删除图)          │
                   │    │                        │
                   │    ▼                        │
                   │  ┌────────────┐             │
                   │  │ Manifest   │─────────────┘
                   │  │ (JSON)     │  manifest 驱动 segment 加载
                   │  └────────────┘
                   │
                   │  崩溃恢复
                   ▼
              ┌──────────────┐
              │  Recovery    │
              │  (5 步)      │
              │  WAL 重放    │
              │  孤儿清理    │
              │  DeltaIndex  │
              │    重建      │
              └──────────────┘
```

### 写入数据流

```
insert(records)
  → Collection._apply(InsertOp)
    → WAL.write_insert(batch)         # 持久化
    → MemTable.apply_insert(batch)    # 内存
    → if memtable.size >= limit:
        → _trigger_flush()
          → execute_flush()           # 同步 Step 2-7
          → _schedule_bg_maintenance()  # 异步 compaction
```

### 搜索数据流

```
search(query_vectors)
  → segments = _segments_snapshot()     # 快照 segment cache
  → delta_snap = delta_index.frozen_copy()  # 快照 tombstone
  → for each segment:
      → bitmap = build_bitmap(segment, delta_snap, expr)
      → top_k = index.search(query, k, bitmap) or brute_force
  → merge memtable results
  → global top_k
```
