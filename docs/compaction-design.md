# Compaction 子系统设计

## 1. 概述

Compaction 是 LiteVecDB LSM-Tree 架构的核心后台维护机制。随着 flush 不断产生新的 data Parquet 文件，文件数量膨胀会导致：

- **读放大**：search/query 需扫描更多 segment
- **空间放大**：已被覆盖（upsert）或删除的行仍占磁盘
- **tombstone 累积**：delta_index 内存持续增长

Compaction 通过合并小文件、去重、过滤已删除行来控制这三个维度的膨胀。

### 设计目标

| 目标 | 实现方式 |
|------|----------|
| 控制文件数量 | Size-Tiered 分桶 + 同桶合并 |
| 消除重复行 | pk 去重（保留 max `_seq`） |
| 回收删除空间 | tombstone 过滤 + delta file GC |
| 不阻塞用户写入 | 后台单线程 executor + `_maintenance_lock` |
| 崩溃安全 | manifest commit-point 设计 + orphan 清理 |
| 索引生命周期对齐 | compaction 后自动重建 segment 索引（Phase 9） |

### 代码位置

| 文件 | 职责 |
|------|------|
| `litevecdb/engine/compaction.py` | `CompactionManager` — 分桶、选择、合并、GC |
| `litevecdb/engine/collection.py` | `_bg_compact_and_index` — 后台调度、锁协调 |
| `litevecdb/storage/delta_index.py` | `gc_below` — in-memory tombstone 回收 |
| `litevecdb/constants.py` | 可调参数 |

---

## 2. 触发条件

Compaction 由 flush 后的 `_schedule_bg_maintenance()` 异步触发，对每个 partition 独立判断。满足以下**任一**条件即触发：

1. **同桶文件数 ≥ `COMPACTION_MIN_FILES_PER_BUCKET`（默认 4）**
   同一大小区间内积累了足够多的小文件，合并收益高。

2. **总文件数 > `MAX_DATA_FILES`（默认 32）**
   强制合并，防止 segment 扫描数过多。此时跨桶贪心选择最小文件优先。

短路优化：如果 partition 总文件数 < `COMPACTION_MIN_FILES_PER_BUCKET`，直接跳过（连分桶都不做）。

---

## 3. Size-Tiered 分桶策略

### 桶边界

```python
COMPACTION_BUCKET_BOUNDARIES = [1_000_000, 10_000_000, 100_000_000]  # bytes
```

产生 4 个桶：

| 桶编号 | 文件大小范围 | 典型内容 |
|--------|-------------|----------|
| 0 | < 1 MB | 小 flush 文件 |
| 1 | 1 MB – 10 MB | 中等 segment |
| 2 | 10 MB – 100 MB | 大 segment |
| 3 | ≥ 100 MB | 超大 segment（通常跳过） |

### 分桶流程

```
_bucket_files(partition_dir, files)
  ├── 遍历 manifest 中的 data files
  ├── os.path.getsize → 获取文件字节大小
  └── _bucket_index(size) → 分配到对应桶
```

### 目标选择

```
_select_target(buckets, total_files, partition_dir)
  ├── 优先: 找到第一个 len(bucket) >= MIN_FILES_PER_BUCKET 的桶
  │         └── _cap_by_row_limit: 取前缀，使合计行数 ≤ 2×MAX_SEGMENT_ROWS
  │              └── 若取完后仍 >= MIN_FILES_PER_BUCKET → 选中
  ├── 次选: total_files > MAX_DATA_FILES（强制模式）
  │         └── 所有桶的文件合并后 _cap_by_row_limit，≥ 2 个文件即选中
  └── 否则: 返回 None，不触发
```

**行数上限**：`_cap_by_row_limit` 读取 Parquet 元数据（不加载数据），累加行数，限制在 `2 × MAX_SEGMENT_ROWS`（200K 行）以内。2x 预算考虑了去重 + 删除过滤后的缩减——如果合并后存活行数仍超过 `MAX_SEGMENT_ROWS`，输出阶段会自动拆分。

---

## 4. 核心合并流程

`_compact_files` 执行 6 步：

```
                    输入文件列表
                         │
                 ┌───────┴───────┐
                 │ 1. 读取文件   │  read_data_file × N
                 └───────┬───────┘
                         │  pa.concat_tables
                 ┌───────┴───────┐
                 │ 2. PK 去重    │  _dedup_max_seq
                 └───────┬───────┘
                         │
                 ┌───────┴───────┐
                 │ 3. 删除过滤   │  _filter_deleted (via delta_index)
                 └───────┬───────┘
                         │
                 ┌───────┴───────┐
                 │ 4. 写新文件   │  write_data_file (可能拆分)
                 └───────┬───────┘
                         │
                 ┌───────┴───────┐
                 │ 5. Manifest   │  remove old + add new + save()
                 │    原子更新   │  ← commit point
                 └───────┬───────┘
                         │
                 ┌───────┴───────┐
                 │ 6. 删除旧文件 │  os.remove × N (best-effort)
                 └───────────────┘
```

### 4.1 PK 去重 — `_dedup_max_seq`

同一 pk 可能因 upsert 出现在多个输入文件中。去重保留 `_seq` 最大的行。

**算法**：纯 Arrow C++ 层操作，避免 Python 逐行循环。

```
1. sort by (pk ASC, _seq DESC)
2. 相邻行比较: mask[i] = (pk[i] != pk[i-1])，mask[0] = True
3. filter(mask) → 每个 pk 只保留第一行（即 _seq 最大的）
```

```python
sort_idx = pc.sort_indices(table, sort_keys=[
    (pk_name, "ascending"), ("_seq", "descending"),
])
sorted_t = table.take(sort_idx)
pk_col = sorted_t.column(pk_name)
n = pk_col.length()
changed = pc.not_equal(pk_col.slice(0, n - 1), pk_col.slice(1))
if isinstance(changed, pa.ChunkedArray):
    changed = changed.combine_chunks()
mask = pa.concat_arrays([pa.array([True]), changed])
return sorted_t.filter(mask)
```

> **注意**：`pa.concat_tables` 产生 ChunkedArray，`pc.not_equal` 在 ChunkedArray 上返回 ChunkedArray，而 `pa.concat_arrays` 只接受 Array。因此需要 `combine_chunks()`。

### 4.2 删除过滤 — `_filter_deleted`

对去重后的每一行，检查 `delta_index.is_deleted(pk, data_seq)`：

```
is_deleted(pk, data_seq) ⟺ _map.get(pk, -1) > data_seq
```

若 tombstone 的 `delete_seq` **严格大于** 数据行的 `_seq`，则该行被过滤。等于时不过滤（same-seq 不是删除）。

> 当前实现使用 Python 逐行循环（因 delta_index 是 dict，无法直接用 Arrow compute）。对于典型 compaction 规模（≤200K 行），耗时在 100ms 量级，可接受。

### 4.3 输出拆分

如果过滤后存活行数超过 `MAX_SEGMENT_ROWS`（100K），按 `_seq` 升序排列后切分为多个 segment：

```python
chunk_size = MAX_SEGMENT_ROWS
chunks = [filtered.slice(i, min(chunk_size, n - i))
          for i in range(0, n, chunk_size)]
```

每个 chunk 独立写入一个 data Parquet 文件，确保：
- 每个 segment 内 `_seq` 范围连续
- 文件名 `data_{seq_min}_{seq_max}.parquet` 通过 `_pick_unique_seq_range` 保证唯一

### 4.4 文件名冲突处理 — `_pick_unique_seq_range`

合并后的文件 seq 范围可能与某个输入文件重叠（例如输入 `[1,10]` + `[3,5]`，合并范围 `[1,10]` 与输入冲突）。

解决方案：保持 `seq_min` 不变，递增 `seq_max` 直到文件名不冲突：

```python
for _ in range(_MAX_SEQ_BUMP_ATTEMPTS):  # 上限 10,000 次
    filename = f"data_{seq_min:06d}_{candidate_max:06d}.parquet"
    if not os.path.exists(filename):
        return seq_min, candidate_max
    candidate_max += 1
raise RuntimeError(...)
```

`seq_max` 是文件名中的**上界**（允许大于实际最大 `_seq`），因此 bump 不影响正确性。

---

## 5. Tombstone GC

Compaction 完成后立即执行 tombstone GC，分两部分：

### 5.1 In-memory GC — `delta_index.gc_below`

**架构不变量 §3**（正确性证明见 `modules.md` §9.16）：

> 一个 tombstone `(pk, delete_seq)` 可以安全丢弃 ⟺ 不存在任何 `seq_min ≤ delete_seq` 的活跃 data 文件。

保守实现：计算所有 partition 所有 data 文件中最小的 `seq_min`（`_global_min_active_data_seq`），删除所有 `delete_seq < global_min` 的 tombstone。

```
_global_min_active_data_seq(manifest)
  ├── 遍历所有 partition 的所有 data files
  ├── parse_seq_range(filename) → 提取 seq_min
  ├── 取全局最小值
  └── 无 data files → sys.maxsize（可清空整个 delta_index）
```

**注意**：使用严格小于（`<`），不是小于等于。`delete_seq == global_min` 的 tombstone 必须保留——可能仍有 `seq_min == delete_seq` 的 data 文件包含该 pk。

### 5.2 On-disk GC — `_gc_delta_files`

Delta Parquet 文件在 `seq_max < global_min` 时完全过时——文件内所有 tombstone 都已被 in-memory GC 清除，文件本身不再需要。

**关键：crash-safe 顺序**

```
1. 收集所有过时的 delta files（per-partition）
2. manifest.remove_delta_files + manifest.save()   ← 先持久化
3. os.remove                                        ← 后删文件
```

如果反过来（先删文件后保存 manifest），crash 后 manifest 仍引用已删除的文件，recovery 的 `DeltaIndex.rebuild_from` 会失败。

---

## 6. 崩溃安全分析

| 崩溃时机 | 磁盘状态 | 恢复行为 |
|----------|----------|----------|
| Step 4 写新文件途中 | 新文件已写（或部分），manifest 未变 | recovery `_cleanup_orphan_files` 清理不在 manifest 中的文件 |
| Step 5 manifest.save() | 原子 rename：要么旧 manifest 要么新 manifest | 旧 manifest → 新文件成 orphan 被清理；新 manifest → 正常 |
| Step 6 删旧文件途中 | manifest 已更新，部分旧文件残留 | orphan 旧文件不在 manifest 中，recovery 清理 |
| Tombstone GC 途中 | delta_index 是纯内存结构 | recovery 从 delta files 重建，不受影响 |
| Delta file GC 途中 | manifest 已保存（先于文件删除） | orphan delta files 被 recovery 清理 |

**核心原则**：manifest 是 commit point。所有持久化变更先反映到 manifest，再执行物理文件操作。物理文件操作失败只产生 orphan，由 recovery 清理。

---

## 7. 并发模型

### 后台执行

```
Collection._trigger_flush()
  ├── execute_flush()          ← 同步，持 _maintenance_lock
  └── _schedule_bg_maintenance()
        └── bg_executor.submit(_bg_compact_and_index)
              ├── Phase A: compaction      ← 持 _maintenance_lock
              │   ├── maybe_compact() × N partitions
              │   ├── _refresh_segment_cache()
              │   └── _cleanup_orphan_index_files()
              └── Phase B: index build     ← 不持锁
                  └── _ensure_loaded_segments_indexed()
```

- **`_bg_executor`**：`ThreadPoolExecutor(max_workers=1)`，单线程串行化所有后台任务
- **`_maintenance_lock`**：`threading.RLock`，保护 manifest + segment_cache 的并发访问

### 读写隔离

| 操作 | 锁行为 | 隔离机制 |
|------|--------|----------|
| 用户 insert/delete | 不直接持锁 | WAL + MemTable（独立于 compaction） |
| 用户 search/query/get | 不持锁 | `_segments_snapshot()` + `delta_index.frozen_copy()` |
| flush | 持 `_maintenance_lock` | 阻塞期间后台 compaction 等待 |
| compaction | 持 `_maintenance_lock` | 阻塞期间 flush 等待 |
| index build | 不持锁 | 操作不可变 Segment，写独立 .idx 文件 |

**关键设计**：读路径在请求开始时获取快照（`_segments_snapshot` + `frozen_copy`），之后完全不依赖锁。后台 compaction 可以自由修改 manifest 和 segment_cache，不影响正在进行的读操作。

---

## 8. Index 生命周期集成（Phase 9）

Compaction 与 vector index 1:1 绑定的 segment 架构深度集成：

```
compaction 删除旧 segment
  → _refresh_segment_cache 驱逐旧 Segment 对象
    → _cleanup_orphan_index_files 删除旧 .idx 文件

compaction 创建新 segment
  → _refresh_segment_cache 加载新 Segment
    → _ensure_loaded_segments_indexed 为新 Segment 构建索引
```

Index build 在锁外执行（Phase B），避免 HNSW 构建（可能数分钟）阻塞用户写入。

---

## 9. 可调参数

| 常量 | 默认值 | 含义 |
|------|--------|------|
| `COMPACTION_MIN_FILES_PER_BUCKET` | 4 | 同桶触发阈值 |
| `MAX_DATA_FILES` | 32 | 单 partition 最大文件数（超过则强制合并） |
| `COMPACTION_BUCKET_BOUNDARIES` | [1M, 10M, 100M] | 文件大小分桶边界（字节） |
| `MAX_SEGMENT_ROWS` | 100,000 | 单 segment 最大行数（影响输出拆分和索引构建耗时） |
| `_MAX_SEQ_BUMP_ATTEMPTS` | 10,000 | 文件名冲突重试上限 |

---

## 10. 数据流全景

```
用户写入
   │
   ▼
MemTable ──flush──▶ data_*.parquet (N 个小文件)
   │                      │
   │                      ▼
   │               ┌─────────────┐
   │               │ Compaction  │
   │               │ (后台线程)  │
   │               └──────┬──────┘
   │                      │
   │         ┌────────────┼────────────┐
   │         ▼            ▼            ▼
   │    读取 N 个     去重+过滤     写 1~M 个
   │    输入文件     (Arrow C++)    输出文件
   │                                   │
   │                      ┌────────────┘
   │                      ▼
   │               Manifest 原子更新
   │               (remove old + add new)
   │                      │
   │                      ▼
   │               删除旧文件 + Tombstone GC
   │                      │
   │                      ▼
   │               Index 重建 (Phase 9)
   │
   ▼
delta_*.parquet ◀──flush──┤
   │                      │
   ▼                      ▼
DeltaIndex ◀── gc_below ──┤
(in-memory)    (回收过时    │
                tombstone)  │
                            ▼
                     _gc_delta_files
                     (删除过时 delta 文件)
```
