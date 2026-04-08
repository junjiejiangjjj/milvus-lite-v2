# LiteVecDB 开发路线图

本文档是从当前状态（仅 `wal.py` 落地）走到 MVP 的纵向切片开发计划。每个 Phase 都是 end-to-end 可跑的状态，配套验证手段贯穿始终。

## 核心原则

1. **设计先冻结**：modules.md 中识别的 P1 设计漏洞必须先在文档层面确定，再写代码。
2. **纵向切片 > 横向分层**：每个里程碑必须 end-to-end 跑通，不堆叠"先把存储层写完"。
3. **每 Phase 一个 demo**：`examples/m{N}_demo.py` 是活文档 + 冒烟测试。
4. **每 Phase 一个 git tag**：出问题可二分。
5. **崩溃注入测试是 LSM 的灵魂**：从 Phase 3 起常态化运行。

---

## Phase 0 — 设计冻结（不写代码）

把前期讨论结论落进 `modules.md` / `wal-design.md`。

| 决议项 | 影响模块 |
|---|---|
| MemTable cross-clear 必须 seq-aware（避免 recovery 顺序敏感 bug） | MemTable |
| Tombstone GC 规则：`delete_seq < min_active_data_seq` 时可丢 | DeltaIndex + Compaction |
| Concurrency：MVP **同步 flush**（异步列入 future） | Collection |
| MemTable 内部表示：append-only RecordBatch list + pk_index + delete_index | MemTable |
| Manifest 保留 `.prev` 备份 | Manifest |
| Operation 抽象：`InsertOp` / `DeleteOp` 统一编排层 | engine/operation.py |
| DeltaLog 拆为 `delta_file.py`（IO）+ `delta_index.py`（内存索引） | 存储层 |
| WAL 加 `sync_mode="close"` 默认 fsync | WAL |
| LOCK 文件防多进程 | db.py |
| Schema 不可变（MVP 不支持 alter） | 顶层不变量 |

**完成标志**：modules.md diff 有意义，无歧义，无遗留 "再讨论一下"。

---

## Phase 1 — WAL 修补 + 基础工具

| 任务 | 文件 | 验证 |
|---|---|---|
| `_read_wal_file` 用 `with` + 收紧 except | `storage/wal.py` | 单测：截断文件不泄漏 fd |
| `close_and_delete` 异常安全（每个 writer 各自 try/finally） | `storage/wal.py` | 单测：mock close 抛异常，验证 `_closed=True` 且第二个 writer 也被尝试关闭 |
| 删 `recover()` 死参数 schema | `storage/wal.py` | 编译/测试通过 |
| `sync_mode="close"` + close 时 fsync | `storage/wal.py` | 单测：fsync mock 被调用一次 |
| `schema/validation.py` | new | 单测覆盖各 DataType |
| `schema/persistence.py` | new | dump → load 往返 |
| `exceptions.py` 补齐 | — | 后续模块自然消费 |

**完成标志**：`pytest tests/storage/test_wal.py tests/schema/` 全绿。

---

## Phase 2 — 第一个纵向切片：内存里走通 insert/get

**目标**：能 `insert()` 一条记录、`get()` 读回来，仅碰 WAL（不 flush）。

**新增模块**：

```
storage/manifest.py      # 最小子集：load/save/add_partition/has_partition
engine/operation.py      # InsertOp / DeleteOp / Operation 类型
storage/memtable.py      # apply_insert / apply_delete / get / size
                         # 内部：RecordBatch list + pk_index + delete_index
engine/collection.py     # insert + get + _seq + WAL/MemTable orchestration
                         # 不实现：flush, search, delete, compaction
```

**Demo**：`examples/m2_demo.py` — 5 行 insert + get。

**验证矩阵**：

| 类型 | 内容 |
|---|---|
| 单测 `test_operation.py` | InsertOp / DeleteOp 构造、属性 |
| 单测 `test_manifest.py` | load 不存在文件 → 初始状态；save→load 往返；`.prev` 生效；version 单调 +1 |
| 单测 `test_memtable.py` | upsert 覆盖；delete + put 顺序敏感的反例（验证 seq-aware）；size 准确 |
| **关键单测** | 乱序 apply：先 seq=7 insert，再 seq=5 insert，再 seq=6 delete → get(X) 必须返回 seq=7 数据 |
| 集成测试 | M2 demo 跑通；duplicate pk 走 upsert |

**完成标志**：M2 demo 通过；`kill -9` 后重启 get 不到（符合预期，下个 phase 解决）。

---

## Phase 3 — 持久化：flush + recovery

**目标**：进程崩溃后数据不丢。LSM 核心价值证明点。

**新增模块**：

```
storage/data_file.py     # write_data_file / read_data_file / parse_seq_range
storage/delta_file.py    # write_delta_file / read_delta_file
storage/delta_index.py   # add_batch / is_deleted / rebuild_from
storage/manifest.py      # 补全 add/remove_data_file, current_seq
engine/flush.py          # 同步 7 步管线
engine/recovery.py       # 5 步恢复
```

**Collection 升级**：触发同步 flush；`__init__` 调 recovery；仍不实现 delete。

**Demo**：`examples/m3_demo.py` — write 进程 `os._exit(0)`，read 进程恢复后 get 能读到。

**验证矩阵**：

| 类型 | 内容 |
|---|---|
| 单测 `test_data_file.py` | 读写往返；parse_seq_range 边界 |
| 单测 `test_delta_index.py` | rebuild_from 多文件；is_deleted 边界；GC 规则 |
| 单测 `test_flush.py` | 7 步逐步骤 manifest + 磁盘状态符合预期 |
| 单测 `test_recovery.py` | wal-design.md §7.1 五个场景 A-E 各一 case |
| **崩溃注入** `test_crash_recovery.py` | 在 flush 每步之间 `os._exit(0)`，验证 recovery 后正确 |
| **Property 测试** | hypothesis 生成随机 insert 序列 + flush + crash + recover，state 一致 |

**关键崩溃注入模板**：

```python
@pytest.mark.parametrize("crash_after_step", range(1, 8))
def test_crash_during_flush(tmp_path, crash_after_step):
    # monkeypatch 让 flush 在第 N 步后抛 SystemExit
    col = Collection("t", str(tmp_path), schema)
    col.insert([...])  # 灌满 memtable
    with pytest.raises(SystemExit):
        col.insert([trigger])

    col2 = Collection("t", str(tmp_path), schema)  # recovery
    # 不变量：所有"已成功 insert"的都能 get 到
    # 不变量：孤儿文件被 recovery 清理
```

**完成标志**：M3 demo 通过；7 个崩溃注入 case 全绿；property test 跑 1000 次随机 case 无挂。

---

## Phase 4 — Search

**目标**：暴力 KNN search 能跑（MemTable + Disk 都覆盖）。

**新增模块**：

```
search/distance.py       # cosine / l2 / ip
search/bitmap.py         # build_valid_mask
search/assembler.py      # segment cache + memtable → numpy 拼接
search/executor.py       # execute_search
storage/segment.py       # data Parquet 内存缓存（pks/seqs/vectors numpy）
```

**Collection 升级**：`search()` 实现；维护 `_segment_cache`；flush 后注册新 segment。

**Demo**：`examples/m4_demo.py` — 1000 条向量 + top-10 query。

**验证矩阵**：

| 类型 | 内容 |
|---|---|
| 单测 `test_distance.py` | 三种距离手算对比 |
| 单测 `test_bitmap.py` | 同 pk 多 seq 去重；空输入；全删 |
| 单测 `test_segment.py` | 加载 parquet → numpy 形态正确 |
| 单测 `test_executor.py` | top-k 正确性；nq > 1 |
| 集成 `test_search_e2e.py` | 1000 条随机向量，search 结果与 numpy 暴力对比 |
| **flush 边界** | 一半 MemTable 一半 Parquet，结果正确 |

**完成标志**：M4 demo top-10 与 numpy 暴力计算一致。

---

## Phase 5 — Delete + Delta + 完整 recovery

**目标**：delete 走通；delete 的 recovery 路径走通。

**新增/升级**：

```
engine/collection.py     # 加 delete()
engine/flush.py          # flush 也写 delta parquet
engine/recovery.py       # 重放 wal_delta + 重建 delta_index
search/bitmap.py         # 启用 is_deleted 过滤
```

**Demo**：`examples/m5_demo.py` — insert → delete → search 不返回 → 重启后仍不返回。

**验证矩阵**：

| 类型 | 内容 |
|---|---|
| 单测 `test_collection_delete.py` | 单/多 pk delete；不存在 pk；跨 partition (`partition_name=None`) |
| 单测 `test_delta_index.py` | 重建后 is_deleted 正确 |
| **关键集成** `test_insert_delete_insert.py` | insert(X) → delete(X) → insert(X) → search 返回新 X（漏洞 1 修复的端到端验证）|
| **崩溃注入** | delete 后 flush 中途崩溃，recovery 后 X 仍删除 |
| **Property** | 随机 insert/delete 混合 + crash + recover state 一致 |

**完成标志**：M5 demo 通过；50 种随机交错顺序的 insert/delete 都正确。

---

## Phase 6 — Compaction + Tombstone GC

**目标**：长跑下文件数和 deleted_map 不无限增长。

**新增**：

```
engine/compaction.py     # CompactionManager, maybe_compact, size-tiered
                         # GC: 调用 delta_index.gc_below(min_active_seq)
```

**Collection 升级**：flush 末尾调 `compaction_mgr.maybe_compact()`。

**Demo**：`examples/m6_demo.py` — 100 万 insert + 周期 delete，验证文件数与内存有界。

**验证矩阵**：

| 类型 | 内容 |
|---|---|
| 单测 `test_compaction.py` | 分桶；触发条件；合并去重正确性 |
| 单测 `test_tombstone_gc.py` | 构造 delete_seq < min_data_seq 场景，GC 后 entry 消失且查询仍正确 |
| 回归 | 触发 compaction 后所有之前测试通过 |
| 崩溃注入 | compaction 写新文件时崩溃 → 旧文件还在 → recovery 后查询正确 |
| **长跑** `@pytest.mark.slow` | M6 demo 跑完后断言：data 文件 ≤ 32；delete_index 大小 ≤ 5%×总插入量 |

**完成标志**：长跑测试通过；100 万 insert 后文件数和内存都有上界。

---

## Phase 7 — DB 层 + 收尾

**目标**：多 collection 生命周期；跨 collection 隔离；干净对外 API。

**新增/升级**：

```
db.py                    # LiteVecDB.create_collection / drop_collection / list_collections
                         # LOCK 文件 (fcntl.flock)
__init__.py              # 公开 API
```

**Demo**：`examples/m7_demo.py` — 创建多 collection，分别写入，drop 一个，list 验证。

**验证矩阵**：

| 类型 | 内容 |
|---|---|
| 单测 `test_db.py` | create / drop / list；同名 create 报错；drop 后磁盘清理 |
| 多进程隔离 | 第二进程 open 同一 data_dir 被 LOCK 拒绝 |
| 冒烟 `test_smoke_e2e.py` | 跑完 M2-M7 所有 demo（CI 入口） |

**完成标志**：能写一段 README quickstart 跑通。

---

## 验证体系

| 层次 | 工具 | 触发 | 价值 |
|---|---|---|---|
| L1 单测 | pytest | 每次 commit | 模块级正确性 |
| L2 集成 | pytest | 每次 commit | 模块组合正确 |
| L3 demo 脚本 | python `examples/m*.py` | 里程碑结束 | "真的能用"信心 |
| L4 Property | hypothesis | 每次 commit | 随机输入下不变量成立 |
| L5 崩溃注入 | pytest + monkeypatch + `os._exit` | 每次 commit | LSM 灵魂——崩溃安全 |
| L6 长跑 | `@pytest.mark.slow` | nightly / 手动 | 资源有界 |

**关键**：L4 + L5 从 Phase 3 起常态化运行，每次改 flush/recovery 都过一遍。这是防止"沉默回归"的唯一办法。

---

## 风险与缓冲

| 风险 | 缓解 |
|---|---|
| Phase 3 flush+recovery 比想象难 | 第一版宁可同步+简陋；不过早优化 |
| Phase 4 segment cache 内存爆炸 | M4 先全量 cache（Segment 永不失效），LRU 列入 P3 优化 |
| Phase 6 compaction 与 search 并发 | MVP 同步路径不会有并发，搁置 |
| Schema 变更需求半路冒出 | Phase 0 写死"MVP 不支持 alter"，新需求开新文档讨论 |

---

## 使用说明

1. **Phase 0 必须先做**——后续所有 phase 的前提
2. **Phase 1-7 严格按序**，phase 内部任务可乱序
3. **每 phase 完成 = git tag**（`m1-fixes`, `m2-write`, …）
4. **每 phase 一个 `examples/m{N}_demo.py`** 长期保留
5. **PR description 引用本文档对应章节**，留决策轨迹
