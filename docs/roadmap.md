# MilvusLite 开发路线图

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
db.py                    # MilvusLite.create_collection / drop_collection / list_collections
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

## Phase 8 — 标量过滤（Scalar Filter Expression）

**目标**：让 `Collection.search` / `get` / `query` 接受 Milvus-style 过滤表达式
（如 `"age > 18 and category == 'tech'"`），打通"向量召回 + 标量过滤"的混合查询。

**架构**：三阶段编译 + 双 backend dispatcher。

```
parse_expr(s)            → Expr (raw AST, schema 无关)
compile_expr(expr, schema) → CompiledExpr (类型检查 + backend 选择)
evaluate(compiled, table) → pa.BooleanArray
```

**新增模块**（详见 modules.md §9.19-9.28）：

```
milvus_lite/search/filter/
├── __init__.py        # parse_expr / compile_expr / evaluate
├── exceptions.py      # FilterError 系列
├── tokens.py          # Tokenizer
├── ast.py             # 11 个 frozen AST 节点
├── parser.py          # Pratt parser (借鉴 Milvus Plan.g4)
├── semantic.py        # compile_expr + 类型推断
└── eval/
    ├── __init__.py    # backend dispatcher
    ├── arrow_backend.py  # pyarrow.compute (主)
    └── python_backend.py # row-wise (兜底 + 差分基准)
```

**Collection 升级**：
- `search(query_vectors, ..., expr=None)`
- `get(pks, ..., expr=None)`
- **新方法** `query(expr, output_fields=None, partition_names=None, limit=None) → List[dict]`
  （纯标量查询，无 query vector）

### 子阶段拆分

| 子阶段 | grammar 增量 | Backend | Status |
|---|---|---|---|
| **F1** | Tier 1：`==/!=/<.../in/and/or/not` + 字面量 + 字段引用 + 括号 | 仅 arrow_backend；python_backend 仅做差分测试基准 | ✅ done |
| **F2a** | + `like` + 算术 (`+ - * / %`) + `is null` | 仍 arrow_backend | ✅ done |
| **F2b** | + `$meta["key"]` 动态字段 | 引入 python_backend dispatch | ✅ done |
| **F2c** | filter LRU cache + `query()` 公开方法 | 与 backend 无关 | ✅ done |
| **F3+** | 性能优化：per-batch JSON 预处理 → arrow path；hybrid 取代 python 作 $meta 默认 dispatch | 引入 hybrid_backend；python_backend 仍作 fallback 与差分基准 | ✅ done |
| **F3** | + `json_contains` / `array_contains` / UDF / 严格 Milvus 兼容 | 扩展 python_backend；可选 ANTLR parser swap | — |

### Phase F1 任务清单

| # | 任务 | 文件 |
|---|---|---|
| F1.1 | exceptions.py + 渲染逻辑 | `search/filter/exceptions.py` |
| F1.2 | tokens.py (TokenKind + Token + tokenize) | `search/filter/tokens.py` + `tests/.../test_tokens.py` |
| F1.3 | ast.py (11 个 frozen dataclass) | `search/filter/ast.py` |
| F1.4 | parser.py (Pratt parser) | `search/filter/parser.py` + `test_parser.py` |
| F1.5 | semantic.py (compile_expr + 类型推断) | `search/filter/semantic.py` + `test_semantic.py` |
| F1.6 | eval/arrow_backend.py | + `test_arrow_backend.py` |
| F1.7 | eval/python_backend.py | + `test_python_backend.py` |
| F1.8 | eval/__init__.py (dispatcher) + 差分测试 | + `test_e2e.py` |
| F1.9 | filter/__init__.py 公开 API | — |
| F1.10 | bitmap.py 加 filter_mask 参数 | + 测试更新 |
| F1.11 | assembler.py 调用 evaluator + 返回 mask | + `test_assembler_filter.py` |
| F1.12 | executor.py 接 filter_mask | + 测试更新 |
| F1.13 | Collection.search/get 加 expr 参数 | + `test_collection_filter.py` 部分 |
| F1.14 | Collection.query 新方法 | + 完整集成测试 |
| F1.15 | __init__.py 公开 query / FilterError | + smoke 补充 |
| F1.16 | examples/m8_demo.py | — |
| F1.17 | 跑全量 pytest | — |

**M8 demo**：`examples/m8_demo.py` — 100 条记录 + 含 `age + category + score` 字段
+ search/get/query 三种用法 + 各种 expr 表达式。

### 验证策略：差分测试

`test_e2e.py` 里每个 case **同时跑两个 backend**，断言结果相等：

```python
@pytest.mark.parametrize("expr_str", [
    "age > 18",
    "category == 'tech'",
    "age in [10, 20, 30]",
    "age >= 18 and category == 'tech' or score > 0.5",
    "not (status == 'draft')",
    # ... 50+ cases
])
def test_arrow_python_equivalence(expr_str, sample_table, sample_schema):
    expr = parse_expr(expr_str)
    compiled = compile_expr(expr, sample_schema)

    arrow_result = evaluate_arrow(compiled, sample_table)

    py_compiled = CompiledExpr(ast=compiled.ast, fields=compiled.fields, backend="python")
    py_result = evaluate_python(py_compiled, sample_table)

    assert arrow_result.equals(py_result)
```

差分测试是 F1 的安全网：写两份实现互相校验，任何一边的 bug 都会被另一边暴露。
NULL 三值逻辑、类型 promotion、边界值这些容易写错的地方靠对称性 catch。

### 不在 Phase F1 范围内（明确推迟）

- ❌ `like` 算子 → F2a
- ❌ 算术 (`+, -, *, /, %`) → F2a
- ❌ NULL 算子 (`is null` / `is not null`) → F2a
- ❌ `$meta` 动态字段 → F2b
- ❌ JSON / array 函数 → F3
- ❌ UDF → F3
- ❌ Expression cache → F2c
- ❌ ANTLR-based parser → F3+ 可选切换
- ❌ DuckDB 后端 → F3+ opt-in extra

### 完成标志

- F1 done：`col.search([[...]], expr="age > 18 and category in ['tech', 'news']")` 跑通；
  差分测试 50+ case 全绿；m8 demo 通过。
- F2 done：`col.search(expr="title like 'AI%' and $meta['priority'] > 5")` 跑通。
- F3+ done：hybrid_backend 取代 python_backend 作 $meta 默认 dispatch；差分测试 hybrid vs python 一致。
- F3 done：跑通 pymilvus 表达式测试套件子集。

---

## Phase 9 — 向量索引（FAISS HNSW + segment-level）

**目标**：把 `Collection.search` 的检索路径从 NumPy 暴力扫描升级为 FAISS HNSW，让 100K+ 向量的搜索 latency 下降 1-2 个数量级。

**架构决策**（详见 `plan/index-design.md`）：
- **索引绑定 segment-level**：每个 data parquet 文件对应一个 .idx 文件，1:1 绑定，与 LSM immutable 架构天然对齐
- **索引库选 FAISS-cpu**：`IDSelectorBitmap` 与 Phase 8 bitmap pipeline 同构；索引家族对齐 Milvus
- **BruteForceIndex 长期保留**：差分测试基准 + 无 faiss 时的 fallback
- **load/release 状态机在 Phase 9.3 引入**：与 Milvus 行为对齐

**新增模块**：

```
milvus_lite/index/
├── __init__.py
├── protocol.py            # VectorIndex ABC
├── spec.py                # IndexSpec dataclass
├── brute_force.py         # BruteForceIndex
├── faiss_hnsw.py          # FaissHnswIndex (Phase 9.5 引入)
└── factory.py             # build_index_from_spec / load_index
```

### 子阶段拆分

| 子阶段 | 内容 | 改动文件 | 工作量 |
|---|---|---|---|
| **9.1** | 补齐 pymilvus quickstart 前置 API：`Collection.create_partition / drop_partition / list_partitions / num_entities / describe` + `search(output_fields=...)` + `MilvusLite.get_collection_stats` | `engine/collection.py`, `db.py`, `search/executor.py` | S |
| **9.2** | `VectorIndex` protocol + `BruteForceIndex` + `Segment.index` + 新 `execute_search_with_index` 路径 | new `index/`, `storage/segment.py`, `search/executor.py` | M |
| **9.3** | `IndexSpec` + Manifest v2 升级 + `Collection.create_index / drop_index / load / release` + `_load_state` 状态机 + `CollectionNotLoadedError` | `engine/collection.py`, `storage/manifest.py`, `exceptions.py` | M |
| **9.4** | Index 文件持久化（`indexes/<stem>.<type>.idx`）+ flush / compaction / recovery 钩子 + 孤儿清理 | `engine/flush.py`, `engine/compaction.py`, `engine/recovery.py`, `storage/segment.py` | M |
| **9.5** | `FaissHnswIndex` + factory 路由 + `[faiss]` extras + metric 对齐 + IDSelectorBitmap + 差分测试 | new `index/faiss_hnsw.py`, `pyproject.toml`, `tests/index/test_index_differential.py` | L |
| **9.6** | `examples/m9_demo.py` + 长跑测试 + 文档 backfill | new `examples/m9_demo.py` | S |

### Phase 9 任务清单（关键文件细化）

| # | 任务 | 文件 |
|---|---|---|
| 9.1.1 | `Collection.create_partition / drop_partition / list_partitions / has_partition` | `engine/collection.py` + `tests/engine/test_collection_partition.py` |
| 9.1.2 | `Collection.num_entities` + `Collection.describe()` | `engine/collection.py` + `tests/engine/test_collection_describe.py` |
| 9.1.3 | `Collection.search(output_fields=...)` 参数支持 | `engine/collection.py`, `search/executor.py` + 测试更新 |
| 9.1.4 | `MilvusLite.get_collection_stats(name)` | `db.py` + `tests/test_db.py` |
| 9.2.1 | `VectorIndex` protocol + `BruteForceIndex` 实现 | new `index/protocol.py`, `index/brute_force.py` + `tests/index/test_brute_force_index.py` |
| 9.2.2 | `Segment.index / attach_index / release_index / build_or_load_index` | `storage/segment.py` + `tests/storage/test_segment_index.py` |
| 9.2.3 | `execute_search_with_index` 新路径 + Collection.search 切换 | `search/executor.py` (新增 `executor_indexed.py` 或扩展) + `tests/search/test_executor_with_index.py` |
| 9.3.1 | `IndexSpec` dataclass + Manifest v2 schema | new `index/spec.py`, `storage/manifest.py` + `tests/storage/test_manifest_v2_compat.py` |
| 9.3.2 | `Collection.create_index / drop_index / has_index / get_index_info` | `engine/collection.py` + `tests/engine/test_collection_create_index.py` |
| 9.3.3 | `_load_state` 状态机 + `Collection.load / release` + `CollectionNotLoadedError` | `engine/collection.py`, `exceptions.py` + `tests/engine/test_collection_load_release.py` |
| 9.4.1 | Index 文件命名约定 + `Segment.build_or_load_index` 持久化路径 | `storage/segment.py` + `tests/index/test_index_persistence.py` |
| 9.4.2 | flush 末尾 build index 钩子（loaded 态时） | `engine/flush.py` + `tests/engine/test_flush_with_index.py` |
| 9.4.3 | compaction 后旧 .idx 清理 + 新 .idx 构建 | `engine/compaction.py` + `tests/engine/test_compaction_with_index.py` |
| 9.4.4 | recovery 启动时 `_cleanup_orphan_index_files` | `engine/recovery.py` + `tests/engine/test_recovery_orphan_idx.py` |
| 9.5.1 | `FaissHnswIndex` build / search / save / load + metric 对齐 | new `index/faiss_hnsw.py` + `tests/index/test_faiss_hnsw.py` |
| 9.5.2 | `IDSelectorBitmap` 接入（packbits 顺序、selector params） | `index/faiss_hnsw.py` + `tests/index/test_faiss_id_selector.py` |
| 9.5.3 | factory 路由 + `[faiss]` extras + try-import 降级 | `index/factory.py`, `pyproject.toml` |
| 9.5.4 | 差分测试 recall@10 ≥ 0.95 + distance parity | `tests/index/test_index_differential.py` |
| 9.5.5 | benchmark 脚本（QPS 对比） | `examples/m9_benchmark.py`（可选） |
| 9.6.1 | `examples/m9_demo.py` | new |
| 9.6.2 | 长跑 `@pytest.mark.slow` 100K + 周期 compaction | `tests/test_smoke_index_longrun.py` |
| 9.6.3 | 跑全量 pytest | — |

### 验证策略：差分测试 + recall

```python
# tests/index/test_index_differential.py
@pytest.mark.parametrize("dim,n,metric", [
    (4, 100, "COSINE"), (32, 10000, "L2"), (128, 10000, "IP"),
])
def test_faiss_hnsw_recall_vs_brute_force(dim, n, metric):
    vectors = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(20, dim).astype(np.float32)
    brute = BruteForceIndex.build(vectors, metric, {})
    faiss_idx = FaissHnswIndex.build(vectors, metric, {"M": 16, "efConstruction": 200})
    brute_ids, _ = brute.search(queries, 10)
    faiss_ids, _ = faiss_idx.search(queries, 10, params={"ef": 64})
    for i in range(20):
        recall = len(set(brute_ids[i]) & set(faiss_ids[i])) / 10
        assert recall >= 0.95
```

### 不在 Phase 9 范围

- IVF / IVF-PQ / OPQ 等量化索引 → Phase 9.5+ 之后
- GPU 索引 → Future
- 异步 index build → Future
- 多向量字段 → Future
- Sparse / Binary vector 索引 → Future

### 完成标志

- `col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16}})` persist 到 manifest
- `col.load()` 后 `col.search` 走 FAISS 路径，100K 向量 search QPS ≥ brute-force × 50
- `col.release()` 后 search 抛 `CollectionNotLoadedError`
- 重启 → load 秒级完成（直接 read .idx 文件）
- compaction 后 .idx 文件 1:1 同步
- recall@10 ≥ 0.95 全绿
- m9 demo 通过

---

## Phase 10 — gRPC 适配层（pymilvus 兼容）

**目标**：在 engine 之上构造 gRPC 服务层，让 pymilvus 客户端无需修改代码即可连接 MilvusLite。"本地版 Milvus" 协议兼容性的最终一公里。

**前置依赖**：Phase 9 必须先完成（CreateIndex / LoadCollection / Search RPC 直接映射 Phase 9 的 API）。

**架构决策**（详见 `plan/grpc-adapter-design.md`）：
- **proto 来源**：直接拷 milvus 官方 proto，但只实现 quickstart 子集，其他返回 `UNIMPLEMENTED`（方案 C — 兼顾 pymilvus 兼容 + 实现工作量可控）
- **错误码对齐 Milvus 2.3 numeric code**
- **依赖 grpcio 是 optional extra `[grpc]`**

**新增模块**：

```
milvus_lite/adapter/grpc/
├── __init__.py
├── server.py                  # run_server(data_dir, host, port)
├── servicer.py                # MilvusServicer
├── cli.py                     # python -m milvus_lite.adapter.grpc 入口
├── errors.py                  # MilvusLiteError → grpc Status
├── translators/
│   ├── schema.py
│   ├── records.py             # FieldData ↔ list[dict] 列行转置
│   ├── search.py
│   ├── result.py
│   ├── expr.py
│   └── index.py
└── proto/                     # 生成的 stub（commit 到 repo）
    ├── milvus_pb2.py
    ├── milvus_pb2_grpc.py
    ├── schema_pb2.py
    └── common_pb2.py
```

### 子阶段拆分

| 子阶段 | 内容 | 工作量 |
|---|---|---|
| **10.1** | proto 拉取 + stub 生成 + 空 servicer + `run_server` + CLI | M |
| **10.2** | Collection 生命周期 RPC (create / drop / has / describe / list) + `translators/schema.py` | M |
| **10.3** | insert/get/delete/query RPC + `translators/records.py`（FieldData ↔ records 双向转置） | L |
| **10.4** | search + create_index + load + release RPC + `translators/{search,result,expr,index}.py` | L |
| **10.5** | Partition RPC + flush + stats + `examples/m10_demo.py` + pymilvus quickstart 冒烟测试 | M |
| **10.6** | 错误码映射 + UNIMPLEMENTED 友好消息 | S |

### Phase 10 任务清单

| # | 任务 | 文件 |
|---|---|---|
| 10.1.1 | 拉 milvus proto → `proto/`，记录 source commit | `adapter/grpc/proto/README.md` |
| 10.1.2 | 用 grpcio-tools 生成 _pb2 / _pb2_grpc，commit | `adapter/grpc/proto/*_pb2.py` |
| 10.1.3 | `MilvusServicer` 空框架（继承 + 全部 UNIMPLEMENTED） | `adapter/grpc/servicer.py` |
| 10.1.4 | `run_server` + `cli.py` + `__main__.py` + `[grpc]` extras | `adapter/grpc/server.py`, `cli.py`, `pyproject.toml` |
| 10.1.5 | server startup 测试（pymilvus.connect 通） | `tests/adapter/test_grpc_server_startup.py` |
| 10.2.1 | `translators/schema.py`：Milvus FieldSchema ↔ MilvusLite 双向 | + `tests/adapter/test_grpc_translators_schema.py` |
| 10.2.2 | `CreateCollection / DropCollection / HasCollection / DescribeCollection / ShowCollections` | `servicer.py` + `tests/adapter/test_grpc_collection_lifecycle.py` |
| 10.3.1 | `translators/records.py`：FieldData ↔ records，覆盖所有支持类型 | + `tests/adapter/test_grpc_translators_records.py` |
| 10.3.2 | `Insert / Upsert / Delete / Query / Get` RPC | `servicer.py` + `tests/adapter/test_grpc_crud.py` |
| 10.4.1 | `translators/expr.py`：filter 透传 + 不支持函数检测 | + `tests/adapter/test_grpc_translators_expr.py` |
| 10.4.2 | `translators/index.py`：IndexParams ↔ IndexSpec | + `tests/adapter/test_grpc_translators_index.py` |
| 10.4.3 | `translators/search.py + result.py`：SearchRequest 解析 + SearchResults 生成 | + `tests/adapter/test_grpc_translators_search.py` |
| 10.4.4 | `Search / CreateIndex / DropIndex / DescribeIndex / LoadCollection / ReleaseCollection / GetLoadState` | `servicer.py` + `tests/adapter/test_grpc_search.py`, `test_grpc_index.py` |
| 10.5.1 | `CreatePartition / DropPartition / ShowPartitions / HasPartition / Flush / GetCollectionStatistics / ListDatabases` | `servicer.py` + `tests/adapter/test_grpc_partition.py` |
| 10.5.2 | `examples/m10_demo.py` | new |
| 10.5.3 | pymilvus quickstart L3 冒烟 | `tests/adapter/test_grpc_quickstart.py` |
| 10.6.1 | `errors.py`：MilvusLiteError → ErrorCode 映射 | + `tests/adapter/test_grpc_error_mapping.py` |
| 10.6.2 | servicer 异常 wrapping 中间件 | `servicer.py` |
| 10.6.3 | UNIMPLEMENTED stub 友好消息（rename / hybrid_search / aliases / RBAC stub …） | `servicer.py` |

### 验证策略：pymilvus quickstart 冒烟

```python
# tests/adapter/test_grpc_quickstart.py
def test_pymilvus_quickstart(grpc_server):
    client = MilvusClient(uri=f"http://localhost:{grpc_server.port}")
    client.create_collection("demo", dimension=4)
    client.insert("demo", data=[{"id": i, "vector": [float(i)]*4} for i in range(100)])
    client.flush("demo")
    client.create_index("demo", index_params={
        "field_name": "vector", "index_type": "HNSW",
        "metric_type": "COSINE", "params": {"M": 16},
    })
    client.load_collection("demo")
    res = client.search("demo", data=[[0.1, 0.2, 0.3, 0.4]], limit=10)
    assert len(res[0]) == 10
    client.query("demo", filter="id >= 50", limit=20)
    client.delete("demo", ids=[1, 2, 3])
    client.release_collection("demo")
    client.drop_collection("demo")
```

### 不在 Phase 10 范围

- TLS / mTLS 加密 → Future
- Token / Username-Password 认证 → Future
- RBAC / 多租户 → Future（嵌入式不必）
- Backup / Restore RPC → Future
- Bulk insert / Import → Future
- Replica / Resource Group → Future
- Aliases → Future
- Hybrid search（多向量） → Future
- Search iterator / pagination → Future
- Database 多实例 → Future（永远 default）
- Binary vector 类型 → Future

### 完成标志

- `python -m milvus_lite.adapter.grpc --data-dir ./data --port 19530` 起 server
- pymilvus quickstart 全流程跑通
- recall parity 测试：grpc search 与 engine 直接 search 的 top-k 完全一致
- 所有不支持 RPC 返回 `UNIMPLEMENTED` + 友好消息（不 silent fail）
- m10 demo 通过

---

## Phase 11 — 全文检索（Full Text Search）

**目标**：支持 Milvus 兼容的 BM25 全文检索 + text_match 过滤，pymilvus 用户可直接使用 `Function(type=BM25)` 进行文本搜索。

**前置依赖**：Phase 10 完成。gRPC 适配层提供 FunctionSchema / SparseFloatArray 的 proto 基础设施。

**深度设计文档**：`plan/fts-design.md`

### 核心架构

```
Insert: text → Analyzer 分词 → BM25 Function → {term_hash: TF} → 稀疏向量列
Load:   稀疏向量 → 构建倒排索引 + segment 内统计量 (docCount, avgdl, df)
Search: query text → 分词 → 倒排索引查找 → BM25 评分 → top-k
```

### 任务分解

| 编号 | 任务 | 交付物 |
|---|---|---|
| 11.1 | Schema 扩展 ✅ | DataType.SPARSE_FLOAT_VECTOR, Function/FunctionType, FieldSchema 新属性, 校验, persistence |
| 11.2 | Analyzer 分词子系统 | `analyzer/` 包：StandardAnalyzer（正则）+ JiebaAnalyzer（可选）+ factory + hash |
| 11.3 | 稀疏向量存储 | sparse_to_bytes/bytes_to_sparse 编解码；WAL/Parquet 稀疏向量列支持 |
| 11.4 | BM25 Function 引擎 | insert 时自动分词 → 生成 TF 稀疏向量；engine search 支持 anns_field |
| 11.5 | 稀疏倒排索引 + BM25 搜索 | SparseInvertedIndex：build/search/save/load；集成到 segment 状态机 |
| 11.6 | text_match 过滤器 | filter 子系统新增 text_match 函数；三个后端实现 |
| 11.7 | gRPC 适配层扩展 | FunctionSchema 翻译；SparseFloatArray 编解码；BM25 搜索请求处理 |
| 11.8 | 集成测试 | pymilvus 端到端 BM25 搜索 + text_match + 混合场景测试 |

### 关键设计决策

- **Per-segment 倒排索引**：与 Phase 9 的 VectorIndex 1:1 绑定一致，segment 不可变 → 索引不可变
- **BM25 查询时计算**：insert 存 TF，search 时基于段内统计量实时计算 IDF + BM25 score
- **term ID = hash**：MurmurHash3 映射 term → uint32，无需全局词表
- **距离约定**：distance = -bm25_score（取负，与 VectorIndex 协议一致）
- **anns_field 参数**：search API 支持指定搜索向量字段，打破单向量限制

### 不在 Phase 11 范围

- Multi-Analyzer（多语言动态选择）→ Future
- phrase_match（短语匹配 + slop）→ Future
- LexicalHighlighter（结果高亮）→ Future
- TextEmbedding Function → Future
### 完成标志

- pymilvus BM25 全文检索端到端通过
- text_match 过滤器与向量搜索组合使用
- Flush / Compaction 后 BM25 索引正确重建
- Load / Release 状态机覆盖 BM25 索引
- 所有已有测试不回归

---

## Phase 12 — Hybrid Search（多路向量融合检索）

**目标**：支持 pymilvus `hybrid_search()` API，允许同时执行多路 ANN 搜索（如 dense COSINE + BM25 sparse）并通过 Reranker 合并结果。

**前置依赖**：Phase 11 完成。BM25 搜索 + 密集向量搜索 + `anns_field` 路由已就绪。

### 核心架构

```
pymilvus.hybrid_search(reqs=[dense_req, bm25_req], ranker=WeightedRanker(0.6, 0.4))
  ↓
HybridSearchRequest proto (多个 SearchRequest + rank_params)
  ↓
servicer.HybridSearch:
  1. 解析每个子 SearchRequest → (query_vectors, anns_field, metric, filter, limit)
  2. 对每路分别调用 Collection.search() → List[List[dict]]
  3. 应用 Reranker 合并结果 → 统一 top-k
  4. 构建 SearchResults 响应
```

### 任务分解

| 编号 | 任务 | 交付物 |
|---|---|---|
| 12.1 | Reranker 实现 | `adapter/grpc/reranker.py`：WeightedRanker（加权分数归一化合并）+ RRFRanker（Reciprocal Rank Fusion） |
| 12.2 | HybridSearch RPC 实现 | `servicer.py`：解析 HybridSearchRequest → 多路 search → rerank → SearchResults |
| 12.3 | 集成测试 | pymilvus `hybrid_search()` 端到端：dense+BM25、多 dense、filter、output_fields |

### 关键设计决策

**Reranker 策略**：

| 策略 | 公式 | 说明 |
|---|---|---|
| **WeightedRanker** | `final_score = Σ(weight_i × normalize(score_i))` | 各路分数归一化到 [0,1] 后加权求和 |
| **RRFRanker** | `final_score = Σ 1/(k + rank_i)` | 基于排名融合，k 默认 60，不依赖分数量纲 |

**分数归一化**（WeightedRanker 需要）：
- 各路搜索的距离量纲不同（dense COSINE ∈ [0,2]，BM25 score ∈ (-∞,0]）
- 归一化方式：per-query min-max → [0,1]，然后 `1 - normalized`（统一为越大越好）

**HybridSearchRequest 解析**：
- `requests`：repeated SearchRequest，每个子请求有独立的 placeholder_group、anns_field、search_params、dsl（filter）
- `rank_params`：KeyValuePair list，包含 strategy（"weighted"/"rrf"）、params（weights/k）、limit、offset
- `output_fields`：全局输出字段，所有子搜索共享

**结果合并**：
- 各路搜索返回 `List[List[dict]]`（nq × top_k_per_route）
- Reranker 按 pk 去重 + 合并分数 → 全局 top-k
- 同一 pk 出现在多路结果中时，合并策略由 Reranker 决定

### 不在 Phase 12 范围

- FunctionScore reranker（外部函数重排序）→ Future
- group_by 分组重排序 → Future
- 异步并行执行多路搜索 → Future（MVP 串行足够）

### 完成标志

- pymilvus `hybrid_search(reqs=[dense, bm25], ranker=WeightedRanker(...))` 端到端通过
- pymilvus `hybrid_search(reqs=[...], ranker=RRFRanker(...))` 端到端通过
- 各子搜索可带独立 filter 表达式
- 结果按 reranker 分数正确排序
- 所有已有测试不回归

---

## Phase 13 — Group By Search（搜索结果分组去重）

**目标**：支持 pymilvus `search(group_by_field=...)` 和 `hybrid_search(group_by_field=...)`，按标量字段分组返回搜索结果。

**前置依赖**：Phase 12 完成。

### 核心机制

```
Search → top-N 候选 → 按 group_by_field 分组 → 每组 top group_size → 返回前 limit 个组
```

### 任务分解

| 编号 | 任务 | 交付物 |
|---|---|---|
| 13.1 | Engine 层 group_by 后处理 | Collection.search() 新增 group_by_field/group_size/strict_group_size 参数 |
| 13.2 | gRPC 适配 | search_params 解析 group_by 参数；SearchResultData 添加 group_by_field_value |
| 13.3 | 测试 | pymilvus 端到端分组搜索 + hybrid group_by |

### 完成标志

- pymilvus `search(group_by_field="category", group_size=3)` 端到端通过
- strict_group_size=True/False 行为正确
- group_by 与 hybrid_search 组合工作
- 支持 INT64/VARCHAR/BOOL 分组字段

---

## Phase 14 — Range Search（距离范围过滤搜索）

**目标**：支持 pymilvus `search(search_params={"params": {"radius": ..., "range_filter": ...}})` 距离范围过滤。

### 参数语义

- `radius`：距离下界（exclusive），`range_filter`：距离上界（inclusive）
- 结果范围：`radius < distance <= range_filter`
- 两个参数均可选；同时存在时要求 `radius < range_filter`

### 任务分解

| 编号 | 任务 | 交付物 |
|---|---|---|
| 14.1 | Engine 层 range 过滤 | search() 新增 radius/range_filter，搜索后按距离过滤 |
| 14.2 | gRPC 适配 + 测试 | search_params 解析 + pymilvus 端到端测试 |

### 完成标志

- pymilvus `search(params={"radius": ..., "range_filter": ...})` 端到端通过
- L2/COSINE/IP/BM25 各 metric 下范围过滤正确
- 只有 radius 或只有 range_filter 的情况正确处理

---

## Phase 15 — Auto ID（自增主键）

**已完成。** FieldSchema.auto_id=True，INT64 主键自动递增生成。

---

## Phase 16 — Iterator（query_iterator / search_iterator）

**已完成。** query(expr=None) 返回全部记录，支持 pymilvus 客户端侧的 pk 游标分页和距离范围分页。

---

## Phase 17 — Offset 分页

**已完成。** search(offset=N) 和 query(offset=N) 跳过前 N 条结果。

---

## Phase 18 — 多向量独立建索引

**已完成。** Manifest/Segment/Collection 从单 IndexSpec 重构为 Dict[str, IndexSpec]，每个向量字段独立建索引。

---

## 性能优化（已完成）

1. **批量 .to_pylist()** — BM25 搜索 segment 数据批量转换替代逐行 .as_py()
2. **延迟物化** — assembler/executor/memtable 全链路仅 top-k winner 物化记录
3. **BM25 segment 级索引缓存** — 不可变 segment 的倒排索引一次构建永久复用

---

## CRUD 对齐修复（已完成）

- delete(filter=...) 不再要求 load_collection
- query(output_fields=["count(*)"]) 计数聚合
- get(ids, output_fields=[...]) 字段过滤
- search(round_decimal=N) 距离四舍五入
- output_fields=["*"] 通配符展开
- JSON 字段 field["key"] 路径过滤语法
- JSON dict 值 Arrow 序列化
- Nullable FLOAT_VECTOR 端到端支持

---

## CI/CD + 打包（已完成）

- GitHub Actions：Python 3.10-3.13 × ubuntu + macos 矩阵测试
- PyPI 打包：Apache-2.0 license，完整元数据
- 1529 测试，0 skip

---

## 待做（TODO）

### 代码 TODO

| 位置 | 描述 |
|------|------|
| `engine/collection.py:552` | BM25 per-segment IDF 应改为全局统计量（跨 segment 汇总 doc_count/avgdl/df） |

### 未实现的 RPC（UNIMPLEMENTED stubs）

| RPC | 原因 |
|-----|------|
| CreateAlias / DropAlias | 别名未在 MVP 范围 |
| AlterCollection | schema 不可变 |
| LoadPartitions / ReleasePartitions | 仅支持 collection 级 load/release |

### 与 Milvus 功能对比（2026-04-17 更新）

#### 已对齐

| 功能 | 说明 |
|------|------|
| Collection CRUD | create/drop/has/describe/list/rename |
| Insert / Upsert / Delete | upsert 语义 + partial update（字段合并） |
| Get / Query / Search | 含 scalar filter 表达式 |
| Vector Search (dense) | COSINE / L2 / IP |
| HNSW / HNSW_SQ Index | FAISS HNSW + scalar quantization |
| IVF_FLAT / IVF_SQ8 Index | FAISS IVF 系列 |
| AUTOINDEX | 自动选择索引类型 |
| BruteForce Index | 小 segment fallback + 差分基准 |
| SPARSE_INVERTED_INDEX | BM25 倒排索引 |
| Load / Release 状态机 | 对齐 Milvus 行为 |
| Partition 管理 | create/drop/list/has |
| Partition Key | 自动哈希分桶路由 |
| Array 字段类型 | DataType.ARRAY + array_contains/all/any + array_length + 下标访问 |
| Scalar Filter | `==, !=, <, >, in, and, or, not, like, is null, $meta["key"]` |
| gRPC 协议兼容 | pymilvus 直连 |
| BM25 全文检索 | Function(type=BM25) + text_match |
| TEXT_EMBEDDING Function | 自动文本转向量 |
| RERANK Function | Cohere + Decay (gauss/exp/linear) |
| Hybrid Search | WeightedRanker + RRFRanker |
| Group By Search | group_by_field + group_size |
| Range Search | radius + range_filter |
| Auto ID | INT64 自增 |
| Nullable fields + Default values | 含 is null / is not null |
| Dynamic fields | enable_dynamic_field + $meta["key"] |
| Iterator | 客户端侧 pk 游标 / 距离范围分页 |
| Offset 分页 | search/query(offset=N) |
| 多向量独立建索引 | 每个向量字段独立 IndexSpec |
| Compaction | Size-Tiered + tombstone GC |
| count(\*) 聚合 | query(output_fields=["count(*)"]) |
| output_fields=["\*"] | 通配符展开 |
| JSON 字段 + 路径过滤 | field["key"] 语法 |
| Sparse Vector (BM25) | SPARSE_FLOAT_VECTOR |
| round_decimal | search 距离四舍五入 |
| Analyzer | Standard + Jieba 中英文分词 |
| Crash Recovery | WAL + 崩溃注入测试 |
| delete(filter=...) 无需 load | 已修复 |

#### 功能缺口

##### P0 — 基本增删改查直接相关，用户高频使用

| 功能 | 对应 Milvus API | 说明 | 工作量 |
|------|-----------------|------|--------|
| Alias 管理 | create/drop/alter/describe/list_aliases | 集合别名，版本切换常用 | 小 |
| Truncate Collection | truncate_collection | 清空集合数据，保留 schema | 小 |
| list_indexes | list_indexes | 列出集合所有索引 | 小 |
| get_partition_stats | get_partition_stats | 分区级统计（num_entities） | 小 |
| Search Iterator | search_iterator | 大结果集分页遍历（服务端游标） | 中 |
| Query Iterator | query_iterator | 查询结果分页遍历（区别于 offset/limit） | 中 |

##### P1 — 常用进阶功能，影响用户迁移体验

| 功能 | 对应 Milvus API | 说明 | 工作量 |
|------|-----------------|------|--------|
| LoadPartitions / ReleasePartitions | load_partitions / release_partitions | 分区级加载释放 | 中 |
| Schema 变更 | add_collection_field / alter_collection_field | 动态加字段、改字段属性 | 大 |
| Collection 属性 | alter/drop_collection_properties | 集合级配置（TTL 等） | 中 |
| Index 属性 | alter/drop_index_properties | 修改索引参数 | 小 |
| Collection Functions | add/alter/drop_collection_function | 动态管理 BM25 等函数 | 中 |
| query order_by | query(... order_by=...) | 标量排序（Milvus 2.5+） | 中 |
| Scalar Index | INVERTED / BITMAP index on scalar fields | 大数据量下标量过滤性能 | 大 |
| FLOAT16 / BFLOAT16 向量 | FLOAT16_VECTOR / BFLOAT16_VECTOR | 省内存的向量格式 | 中 |

##### P2 — 进阶功能，特定场景需要

| 功能 | 对应 Milvus API | 说明 | 工作量 |
|------|-----------------|------|--------|
| Database 管理 | create/drop/list/use_database | 多数据库隔离 | 中 |
| DiskANN Index | DiskANN index type | 磁盘索引，大数据量场景 | 大 |
| IVF_PQ / OPQ / SCANN | 量化索引系列 | 量化压缩，内存效率 | 大 |
| Binary Vector | BIN_FLAT / BIN_IVF_FLAT | 二值向量索引 | 中 |
| Bulk Insert | import / get_import_state | 批量文件导入 | 大 |
| phrase_match | text_match 增强 | 有序短语匹配 + slop | 中 |
| run_analyzer | run_analyzer | 分析器调试接口 | 小 |
| SPARSE_WAND | 稀疏向量加速 | 稀疏向量加速检索 | 中 |
| JSON Path Index | JSON 字段索引 | JSON 字段索引加速 | 中 |
| Boost Ranker | 加权搜索排名 | 多字段权重搜索 | 中 |
| Text Highlighter | 搜索结果高亮 | FTS 结果片段高亮 | 中 |
| Snapshot | create/restore_snapshot | 集合快照备份恢复 | 大 |
| RESTful API | HTTP 接口 | REST 替代 gRPC | 大 |

##### P3 — 低优先级

| 功能 | 说明 |
|------|------|
| Geometry 类型 | WKT 格式 + 空间查询 |
| Struct/Array 嵌套 | 结构化数组字段 |
| MinHash 向量 | MinHash 向量类型 |
| TimestampTZ | 时区感知时间戳 |
| Clustering Key | 聚簇压缩 |
| Warmup | 集合预热 |
| MMap | 内存映射存储 |
| Nullable vector Parquet 持久化 | 当前 null 向量存为零向量，重启后 null 语义丢失 |

##### 明确不做（嵌入式场景不需要）

| 功能 | 理由 |
|------|------|
| 用户/角色/权限 (RBAC) | 嵌入式单用户 |
| GPU Index | 嵌入式场景无需 |
| TLS / mTLS | 本地嵌入式 |
| 多副本 / Resource Group | 单进程架构 |
| 流式 CDC | 无分布式消费需求 |
| Row-Level Security | 企业安全功能 |
| Privilege Groups | 嵌入式无需权限分组 |
| Metrics / GetComponentStates | 分布式监控 |
| Consistency Levels | 单进程同步架构天然 Strong Consistency，无需多级别 |

**覆盖率**：以 Milvus pymilvus 测试套件（55 个测试文件）衡量，MilvusLite 覆盖约 80% 核心功能。P0 补齐后预计可达 ~85%，P0+P1 补齐后预计可达 ~90%。剩余缺口集中在高级索引类型、扩展数据类型和企业级运维能力。

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
| Phase 9 FAISS macOS arm64 wheel 安装失败 | optional extra `[faiss]` + BruteForceIndex fallback；CI 跑双 matrix |
| Phase 9 FAISS metric 符号对齐写错 | 差分测试 distance value parity 是必经关卡 |
| Phase 10 milvus proto 跨版本漂移 | proto/README.md 记录 source commit；future 升级时 git diff |
| Phase 10 pymilvus 客户端版本兼容 | 测试矩阵覆盖至少 pymilvus 2.3.x / 2.4.x / 2.5.x |

---

## 使用说明

1. **Phase 0 必须先做**——后续所有 phase 的前提
2. **Phase 1-10 严格按序**（Phase 8 含 F1/F2/F3 子阶段，Phase 9 含 9.1-9.6，Phase 10 含 10.1-10.6），phase 内部任务可乱序
3. **每 phase 完成 = git tag**（`m1-fixes`, `m2-write`, …）
4. **每 phase 一个 `examples/m{N}_demo.py`** 长期保留
5. **PR description 引用本文档对应章节**，留决策轨迹
6. **Phase 9 / Phase 10 的深度设计在独立文档**：`plan/index-design.md` 和 `plan/grpc-adapter-design.md`
