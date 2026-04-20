# Search Iterator 设计文档

## 1. 背景与目标

pymilvus 提供两种 SearchIterator 实现：

- **V1（客户端侧）**：基于距离范围的自适应分页。客户端通过 `radius` + `range_filter` 参数反复调用 Search RPC，逐步扩大搜索范围，用 PK 排除去重。
- **V2（服务端侧）**：基于 token 的分页。服务端每次执行 `top_k=batch_size` 的搜索，用 `last_bound` 距离阈值跳过已返回的结果，实现逐批推进。

### 当前状态

MilvusLite 已支持 V1 所依赖的全部底层能力：
- ✅ Range Search（`radius` + `range_filter`）— Phase 14
- ✅ Offset 分页 — Phase 17
- ✅ PK `not in [...]` 过滤表达式 — Phase 8
- ✅ test_iterator.py 中的 `search_iterator` 测试已通过

**结论：V1 已经可以工作。** pymilvus 的 SearchIterator V1 是纯客户端实现，只要 Search RPC 正确支持 `radius`、`range_filter`、`offset` 和 filter 表达式，V1 就能运行。

### 设计目标

实现 **V2 服务端迭代器**，原因：
1. V2 是 pymilvus 的默认路径（优先尝试 V2，失败才 fallback V1）
2. V2 避免了 V1 的多次自适应重试，延迟更低
3. V2 避免了客户端构造 PK 排除表达式的开销（大批量时表达式可能很长）
4. V1 在 `distance` 相同的大量记录场景下可能死循环（PK 排除列表爆炸）

---

## 2. Milvus 实现分析

### 2.1 Milvus V2 实际架构

Milvus 的 V2 **不是**全量搜索 + 缓存，而是**逐批搜索 + last_bound 距离过滤**：

1. 每次请求都执行真正的搜索，`topK = batch_size`
2. C++ segcore 层的 `CachedSearchIterator` 缓存的是**向量索引的迭代器状态**（HNSW 图遍历位置），不是搜索结果
3. 用 `last_bound`（上一批最后一条的距离值）过滤掉 `distance ≤ last_bound` 的结果
4. Token 不对应服务端状态存储，真正的"游标"是 `last_bound` 值

```
第1次: Search(batch_size=100, last_bound=None)
  → 返回最近的 100 条 + last_bound=0.85

第2次: Search(batch_size=100, last_bound=0.85)
  → 跳过 distance ≤ 0.85，返回下一批 + last_bound=1.23

第3次: Search(batch_size=100, last_bound=1.23)
  → 跳过 distance ≤ 1.23，返回下一批...

结果为空 → 客户端终止
```

### 2.2 为什么 Milvus 不用全量缓存

全量缓存意味着 `top_k` 要设成很大的值（如 16384），对 HNSW 等 ANN 索引来说等价于退化为暴力搜索，完全浪费了索引加速的意义。逐批搜索每次只取 `batch_size` 条，索引的 early termination 优势得以保留。

### 2.3 pymilvus V2 协议

**握手流程**：

```
Client                                    Server
  |                                          |
  |-- Search(batch_size=1, iter_v2=true) --> |   # probe 调用
  |<--- SearchResults + {token, last_bound} -|   # 返回 token
  |                                          |
  |-- Search(token=T, last_bound=B) -------> |   # 后续调用
  |<--- SearchResults + {token, last_bound} -|
  |                                          |
  |       ... 重复直到结果为空 ...               |
```

**关键参数**（嵌入在 search_params KV 中）：

| Key | 类型 | 说明 |
|-----|------|------|
| `iterator` | bool | 标记为迭代器模式 |
| `search_iter_v2` | bool | 使用 V2 协议 |
| `search_iter_batch_size` | int | 每批返回数量 |
| `search_iter_id` | str | 迭代器 token（首次请求无此参数） |
| `search_iter_last_bound` | float | 上一批最后一条的距离值 |
| `guarantee_timestamp` | int | MVCC 时间戳（确保一致性快照） |

**SearchResultData 扩展字段**：

```protobuf
message SearchIteratorV2Results {
    string token = 1;       // 迭代器唯一标识
    float last_bound = 2;   // 本批最后一条的距离值
}

message SearchResultData {
    ...
    optional SearchIteratorV2Results search_iterator_v2_results = 11;
}
```

**pymilvus 客户端行为**：

1. **Probe**：首次调用 `batch_size=1`，获取 token + last_bound
2. **Iterate**：后续调用携带 `token` + `last_bound`，`limit = batch_size`
3. **Terminate**：返回空结果时停止
4. **Timestamp**：首次响应中获取 `session_ts`，后续请求携带 `guarantee_timestamp` 确保快照一致性

**约束**：
- 只支持 nq=1（单向量查询）
- 不能与 groupBy / offset / order_by 组合

---

## 3. MilvusLite 设计方案

### 3.1 核心思路

对齐 Milvus：**逐批搜索 + last_bound 距离过滤**。

每次迭代请求都执行一次 `Collection.search(top_k=batch_size)`，通过 `last_bound` 参数过滤掉已返回的结果（`distance ≤ last_bound`），实现逐批推进。

```
pymilvus SearchIteratorV2
    │
    ▼
gRPC Search RPC（带 iterator 参数）
    │
    ▼
MilvusServicer.Search()
    │  检测 search_iter_v2=true
    ▼
Collection.search(top_k=batch_size, last_bound=B)
    │  搜索时过滤 distance ≤ last_bound 的结果
    ▼
返回 SearchResults + SearchIteratorV2Results{token, last_bound}
```

### 3.2 核心设计决策

#### 决策 1：逐批搜索 + last_bound 过滤（对齐 Milvus）

**方案**：每次迭代请求执行 `search(top_k=batch_size)`，引擎层在距离计算后过滤 `distance ≤ last_bound` 的候选。

**理由**：
- 保留 HNSW 索引的 early termination 优势，不退化为暴搜
- 每批只计算 batch_size 条结果的距离，内存开销小
- 无需管理服务端缓存和生命周期
- 与 Milvus 行为一致

#### 决策 2：last_bound 过滤在 executor 层实现

过滤时机在搜索结果排序后、返回前——将 `distance ≤ last_bound` 的结果剔除。

**距离约定**（内部统一 "smaller = more similar"）：
- `COSINE`：`1 - similarity`，范围 [0, 2]，越小越相似
- `L2`：欧氏距离，越小越相似
- `IP`：`-dot_product`（内部取反），越小越相似

过滤规则统一为：**跳过 `distance ≤ last_bound` 的结果**（因为内部距离已统一为"小 = 相似"，last_bound 之前的结果距离更小，应跳过）。

注意：返回给 pymilvus 的 `last_bound` 是**外部距离**（IP 距离已取反回正值），与内部距离符号不同。需要在 adapter 层做转换。

#### 决策 3：Token 是无状态标识

与 Milvus 一致，token 不对应服务端状态。首次请求生成 `uuid4` 返回给客户端，后续请求原样回传。真正的游标是 `last_bound` 值。

服务端无状态 = 无需管理 TTL / 缓存清理 / 内存泄漏。

#### 决策 4：基于 `_seq` 的快照隔离

**问题**：迭代器的每次 `.next()` 调用都是独立的 `Collection.search()`。如果两次调用之间有 insert/delete，会导致结果不一致：

```
iterator.next()  → 返回 batch 1 (distance 0.1~0.5)
    ↓
collection.insert(new_record)  # 新记录 distance=0.3
    ↓
iterator.next()  → 返回 batch 2 (distance > 0.5)
                   新记录 distance=0.3 ≤ last_bound，被过滤 → 永远丢失
```

**方案**：利用 `_seq`（全局单调递增序列号）实现快照隔离，与 Milvus 的 `session_ts` 机制对齐。

本质上这就是 **MVCC（Multi-Version Concurrency Control）**：
- `_seq` = 版本号（每条记录自带，单调递增）
- `snapshot_seq` = 读快照的版本上界（等价于经典 MVCC 的 `read_ts`）
- `seq_mask = (seqs <= snapshot_seq)` = 版本可见性判定

MilvusLite 的 `_seq` 机制天然具备 MVCC 能力，此前只用于 dedup 和 tombstone 判定，这里扩展为快照读。

基于这个底层能力，Consistency Levels 理论上可以实现：
- **Strong**：`snapshot_seq = current_seq`（当前默认行为，单进程同步天然满足）
- **Session**：`snapshot_seq = 该 session 最后一次写入的 _seq`
- **Eventually**：`snapshot_seq = 上一次 flush 时的 _seq`（只读持久化数据）

但对单进程嵌入式架构，Strong 是天然的，其他级别没有实际收益，因此 **Consistency Levels 不作为用户功能暴露**。`snapshot_seq` 仅在 Search Iterator 内部使用。

**流程**：

1. **首次迭代请求**：捕获当前 `Collection._seq` 作为 `snapshot_seq`
2. **返回 `session_ts`**：将 `snapshot_seq` 通过 SearchResults 的 `session_ts` 字段返回
3. **pymilvus 自动回传**：后续请求携带 `guarantee_timestamp = snapshot_seq`
4. **后续搜索**：bitmap pipeline 中增加 `seq_mask = (seqs <= snapshot_seq)`，合并到 valid_mask

**实现位置**：`search/bitmap.py` 的 `build_valid_mask()` 中，在现有的 dedup + tombstone + scalar filter 之后，追加 seq 过滤：

```python
if snapshot_seq is not None:
    seq_mask = (all_seqs <= snapshot_seq)
    valid_mask &= seq_mask
```

这样快照之后插入的记录（`_seq > snapshot_seq`）和删除操作（`delete_seq > snapshot_seq`）都不可见，保证迭代过程中数据视图一致。

**优势**：
- 零额外存储——`_seq` 已存在于每条记录中，MVCC 是"免费"的
- 零服务端状态——`snapshot_seq` 由客户端通过 `guarantee_timestamp` 回传
- 复用 pymilvus 已有的 `session_ts` / `guarantee_timestamp` 协议字段
- 对非迭代器请求无影响（`snapshot_seq=None` 时跳过过滤）

#### 决策 5：同距离记录处理

当多条记录距离完全相同时，`last_bound` 可能导致部分同距离记录被跳过。处理策略：

- 使用**严格小于** `distance < last_bound` 可能导致重复
- 使用**小于等于** `distance ≤ last_bound` 可能导致遗漏

Milvus 使用 `distance > last_bound`（严格大于，等价于跳过 ≤），接受同距离遗漏。我们对齐此行为。理由：
- 向量搜索本身是近似的（ANN），少量同距离遗漏可接受
- 比重复好——重复会让客户端逻辑复杂化

---

## 4. 实现细节

### 4.1 引擎层改动：Collection.search 增加迭代器参数

```python
def search(
    self,
    query_vectors,
    top_k=10,
    metric_type="COSINE",
    ...,
    last_bound=None,       # 新增：迭代器距离阈值
    snapshot_seq=None,      # 新增：快照序列号（迭代器一致性）
) -> List[List[dict]]:
```

**last_bound 过滤**——在搜索结果排序后、截取 top_k 前：

```python
# 在 executor 层，top-k 选出后：
if last_bound is not None:
    hits = [h for h in hits if h["distance"] > last_bound]
```

**snapshot_seq 过滤**——在 bitmap pipeline 中：

```python
# 在 search/bitmap.py 的 build_valid_mask() 中：
if snapshot_seq is not None:
    seq_mask = (all_seqs <= snapshot_seq)
    valid_mask &= seq_mask
```

### 4.2 距离转换

pymilvus 传入的 `last_bound` 是**外部距离**（返回给用户的值）。引擎内部使用统一约定（smaller = more similar）。需要在 adapter 层转换：

| metric | 外部距离 | 内部距离 | 转换 |
|--------|---------|---------|------|
| COSINE | `1 - sim` | `1 - sim` | 无需转换 |
| L2 | `euclidean` | `euclidean` | 无需转换 |
| IP | `dot_product`（正值） | `-dot_product`（负值） | `internal = -external` |

返回 `last_bound` 时做反向转换。

### 4.3 过滤实现位置

在 `search/executor_indexed.py` 的全局合并阶段，top-k 结果已按距离排序。追加一步过滤：

```python
def _apply_last_bound(hits: List[dict], last_bound: float) -> List[dict]:
    """过滤掉 distance ≤ last_bound 的结果（已在内部距离空间）。"""
    return [h for h in hits if h["distance"] > last_bound]
```

注意：这个过滤发生在 top-k 选择**之后**。因此实际流程是：
1. 正常搜索 `top_k = batch_size + 余量`（over-fetch 以补偿被过滤的部分）
2. 过滤 `distance ≤ last_bound`
3. 截取前 `batch_size` 条

over-fetch 的余量：Milvus C++ 层的做法是让索引迭代器自然跳过 ≤ last_bound 的结果（HNSW 图遍历中直接跳过）。MilvusLite 的 FAISS 不支持这种定制，所以用 over-fetch + post-filter 替代。over-fetch 倍率设为 2x（即 `top_k = batch_size * 2`），如果过滤后不足 batch_size，不重试——返回实际数量，客户端自行判断是否继续。

### 4.4 Servicer 改动

```python
def Search(self, request, context):
    # ... 现有解析逻辑 ...
    parsed = parse_search_request(request, ...)

    # ── 新增：检测 V2 迭代器模式 ──
    if parsed.get("search_iter_v2"):
        return self._handle_search_iterator_v2(request, col, parsed)

    # ── 原有搜索逻辑 ──
    results = col.search(...)
    ...

def _handle_search_iterator_v2(self, request, col, parsed):
    batch_size = parsed.get("search_iter_batch_size", 1000)
    last_bound_external = parsed.get("search_iter_last_bound")  # 外部距离
    token = parsed.get("search_iter_id")
    guarantee_ts = parsed.get("guarantee_timestamp", 0)

    # 首次请求：生成 token，捕获快照 _seq
    is_first = (token is None)
    if is_first:
        import uuid
        token = str(uuid.uuid4())
        snapshot_seq = col._seq          # 捕获当前序列号作为快照
    else:
        snapshot_seq = guarantee_ts if guarantee_ts > 0 else None

    # 转换 last_bound 到内部距离空间
    metric = parsed["metric_type"]
    last_bound_internal = None
    if last_bound_external is not None:
        if metric == "IP":
            last_bound_internal = -last_bound_external
        else:
            last_bound_internal = last_bound_external

    # 执行搜索（over-fetch 2x 补偿 last_bound 过滤）
    results = col.search(
        query_vectors=parsed["query_vectors"],
        top_k=batch_size * 2,
        metric_type=metric,
        partition_names=parsed["partition_names"],
        expr=parsed["expr"],
        output_fields=parsed["output_fields"],
        anns_field=parsed.get("anns_field"),
        last_bound=last_bound_internal,
        snapshot_seq=snapshot_seq,        # 快照隔离
    )

    # 截取 batch_size
    for i, hits in enumerate(results):
        results[i] = hits[:batch_size]

    # 计算本批的 last_bound（最后一条的外部距离）
    new_last_bound = 0.0
    if results and results[0]:
        last_hit_distance = results[0][-1]["distance"]
        # 内部距离 → 外部距离
        if metric == "IP":
            new_last_bound = -last_hit_distance
        else:
            new_last_bound = last_hit_distance

    # 构建响应
    result_data = build_search_result_data(...)

    # 设置 V2 迭代器信息
    result_data.search_iterator_v2_results.token = token
    result_data.search_iterator_v2_results.last_bound = new_last_bound

    # 首次请求：返回 session_ts，pymilvus 会自动回传为 guarantee_timestamp
    session_ts = snapshot_seq if is_first else 0

    return milvus_pb2.SearchResults(
        status=common_pb2.Status(**success_status_kwargs()),
        results=result_data,
        session_ts=session_ts,
    )
```

### 4.5 parse_search_request 扩展

在现有的 `parse_search_request` 返回值中新增：

```python
# V2 iterator 参数
"search_iter_v2": bool(raw_params.get("search_iter_v2", False)),
"search_iter_batch_size": int(raw_params.get("search_iter_batch_size", 1000)),
"search_iter_id": raw_params.get("search_iter_id"),        # str or None
"search_iter_last_bound": raw_params.get("search_iter_last_bound"),  # float or None
"iterator": bool(raw_params.get("iterator", False)),
```

---

## 5. 模块拆分

| 文件 | 变更 | 说明 |
|------|------|------|
| `search/bitmap.py` | 修改 | build_valid_mask() 增加 `snapshot_seq` 过滤 |
| `search/executor_indexed.py` | 修改 | top-k 后增加 last_bound 过滤，透传 snapshot_seq |
| `engine/collection.py` | 修改 | search() 增加 `last_bound` + `snapshot_seq` 参数 |
| `adapter/grpc/servicer.py` | 修改 | Search() 增加 V2 分支，快照 _seq 捕获 |
| `adapter/grpc/translators/search.py` | 修改 | 提取 V2 iterator 参数 + guarantee_timestamp |
| `adapter/grpc/translators/result.py` | 修改 | 设置 search_iterator_v2_results 字段 |

无需新增文件。服务端无状态（snapshot_seq 由客户端通过 guarantee_timestamp 回传）。

---

## 6. 边界条件

| 场景 | 处理 |
|------|------|
| probe 请求（batch_size=1） | 正常搜索 top_k=2，过滤后返回 1 条 |
| 首次请求（无 last_bound） | 不过滤，正常返回 top batch_size；捕获 _seq 作为快照 |
| 结果为空 | 返回空结果 + token + last_bound=0，pymilvus 终止 |
| over-fetch 后仍不足 batch_size | 返回实际数量，不重试 |
| 同距离大量记录 | 部分遗漏，与 Milvus 行为一致 |
| nq > 1 | 拒绝，返回错误（V2 仅支持单向量） |
| 与 groupBy / offset 组合 | 拒绝，返回错误 |
| 迭代过程中 insert | 新记录 `_seq > snapshot_seq`，被 seq_mask 过滤，不可见 |
| 迭代过程中 delete | 删除操作 `_seq > snapshot_seq`，被忽略，已返回记录不受影响 |

---

## 7. 测试计划

| 测试 | 内容 |
|------|------|
| **单测：last_bound 过滤** | 验证 distance ≤ last_bound 的结果被正确过滤 |
| **单测：snapshot_seq 过滤** | 验证 _seq > snapshot_seq 的记录不可见 |
| **单测：距离转换** | IP metric 的内外距离转换正确性 |
| **集成：pymilvus search_iterator** | 完整迭代流程，验证无重复、结果按距离递增 |
| **集成：不同 batch_size** | 1, 10, 100, 大于总数 |
| **集成：带 filter** | 标量过滤 + 迭代器 |
| **集成：不同 metric** | COSINE / L2 / IP |
| **集成：limit 参数** | pymilvus limit < 总数 |
| **集成：迭代中插入** | 迭代过程中 insert 新记录，验证不影响迭代结果 |
| **集成：稀疏场景** | 大部分结果被 filter 过滤，验证 over-fetch 表现 |
| **回归：普通 search 不受影响** | 非迭代器请求走原路径 |

---

## 8. Query Iterator

Query Iterator 已在 Phase 16 通过 pymilvus 客户端侧实现工作正常（基于 offset 分页调用 Query RPC）。暂不需要服务端优化。

---

## 9. 实现步骤

1. **Step 1**：`search/bitmap.py` — build_valid_mask() 增加 `snapshot_seq` 过滤（含单测）
2. **Step 2**：`search/executor_indexed.py` — 增加 `last_bound` 过滤逻辑，透传 snapshot_seq（含单测）
3. **Step 3**：`engine/collection.py` — search() 增加 `last_bound` + `snapshot_seq` 参数
4. **Step 4**：`adapter/grpc/translators/search.py` — 提取 V2 参数 + guarantee_timestamp
5. **Step 5**：`adapter/grpc/servicer.py` — Search() 增加 V2 分支，快照捕获，距离转换
6. **Step 6**：`adapter/grpc/translators/result.py` — 设置 SearchIteratorV2Results + session_ts
7. **Step 7**：集成测试（pymilvus search_iterator + 快照隔离 + 多场景）
