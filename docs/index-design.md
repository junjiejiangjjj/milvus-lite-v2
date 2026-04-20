# 深入设计：向量索引子系统（Phase 9）

## 1. 概述

MilvusLite Phase 9 引入向量索引（vector index），把 `Collection.search` 的检索路径从 NumPy 暴力扫描升级为 ANN（Approximate Nearest Neighbor）检索。**默认实现是 FAISS HNSW**，同时保留 BruteForceIndex 作为差分基准 + 无依赖兜底。

**为什么现在做**：
- Phase 8 标量过滤系统建立了 `bitmap pipeline + filter_mask` 抽象，FAISS 的 `IDSelectorBitmap` 与之天然同构 — Phase 9 一次性把这条管线接通
- 项目定位是"本地版 Milvus"，pymilvus 用户的核心诉求是"快速 top-k"，没有 ANN 索引的"本地 Milvus"对早期用户是错误的第一印象
- 存储层（MVP.md §10）从一开始就为索引接入预留了不变量：**数据文件不可变 + 不含删除标记**，使索引可以"一次构建、永不修改"

**为什么选 FAISS 而不是 hnswlib / USearch**：
- `IDSelectorBitmap` 与 Phase 8 的 `valid_mask` 天然对齐，pre-filter（不是后验）可以原生支持
- 索引家族对齐 Milvus（HNSW / IVF_FLAT / IVF_SQ8 / IVF_PQ 都是 FAISS 名字），用户从 pymilvus 迁移时 `index_params` 不用改
- 后续扩展性最好（量化索引、混合索引、GPU 路径）
- 风险：macOS arm64 wheel 历史上有坑（faiss-cpu 1.7.4+ 已稳定），通过"FAISS 是 optional extra + BruteForce fallback"的方式隔离

---

## 2. 架构决策

### 2.1 索引绑定层级：Segment-level（决定）

**决定：每个 data Parquet 文件对应一个独立的 VectorIndex 文件，1:1 绑定。**

候选方案对比：

| 维度 | Segment-level（选定） | Collection-level |
|---|---|---|
| 与 LSM 不可变架构匹配度 | 完美 — segment 不可变 → index 也不可变 → 永远不需要"删除某个向量" | 差 — 全局图必须"增量更新 + 删除"，而 FAISS HNSW 不支持真删除 |
| 增量更新成本 | 极低 — 新 segment flush 后只需建一份新 index | 中 — 每次 flush 要 add_items，HNSW 支持但要锁 + 可能 resize |
| 与 compaction 协同 | 自然 — 旧 segment 删 → 旧 index 文件删；新 segment 写 → 新 index 构建。1:1 对应 | 痛苦 — 删 N 个 segment 后要从全局图里删 N 批 pk，HNSW 只能 mark-deleted，空间不回收，recall 漂移；最终被迫周期性"全量重建全局 index" |
| recall | 多 segment 时略低于全局图（"per-segment top-k 后合并"），实际损失 < 5% | 理论最优 |
| 内存占用 | 略高（每段图重复一些辅助数据） | 略省 |
| 与 brute-force fallback 共存 | 自然 — 小 segment 不建 index，search 时直接 brute-force；其他段走 index | 不自然 — memtable 永远旁路全局图 |
| 与 Milvus 实际架构一致性 | ✅ Milvus 本身就是 per-segment 建索引 | ✗ |

**决定理由**：
1. LSM-Tree 的 immutable segment 不变量是项目的根本架构红利；segment-level index 让 Phase 9 不引入任何新的可变状态
2. FAISS HNSW 不支持真删除，全局 index 方案会被这个约束反制；segment-level 完美回避
3. "本地版 Milvus" 的定位倾向抄 Milvus 本身的架构决策

### 2.2 索引库选型：FAISS-cpu（决定）

| 维度 | hnswlib | **FAISS-cpu**（选定） | USearch |
|---|---|---|---|
| 依赖体积 | ~2 MB | ~30-80 MB（macOS arm64 wheel 已成熟） | ~3 MB |
| 索引类型 | HNSW only | HNSW / IVF_FLAT / IVF_SQ8 / IVF_PQ / OPQ / Flat / ... | HNSW only |
| Pre-filter（IDSelector） | 不支持 callback | **`IDSelectorBitmap` 原生支持，与 bitmap pipeline 同构** | 支持但不如 FAISS 成熟 |
| Metric | cosine / l2 / ip | 齐全 | 齐全 |
| 维护活跃度 | 低（作者 2023 后较少响应） | 高（Meta 官方） | 高（unum-cloud） |
| 与 Milvus 的语义对齐 | 部分 | **完全 — Milvus index_params 直接复用** | 部分 |
| Phase 9 MVP 选 | ❌ | ✅ | ❌ |

**风险与缓解**：
- **macOS arm64 wheel**：截至 2024 末已稳定，但安装失败要降级 — `try: import faiss` 失败时自动 fallback 到 BruteForceIndex
- **HNSW 不支持真删除**：架构上由 segment-level + immutable 规避
- **小 segment 上 FAISS 比 brute-force 慢**：对 < `INDEX_BUILD_THRESHOLD`（默认 10000 行）的 segment 不建索引，search 时走 brute-force

### 2.3 BruteForceIndex 的双重定位（决定）

`BruteForceIndex` 不是临时的占位实现，而是一个**长期保留**的一等公民：

1. **零依赖兜底**：用户不装 faiss-cpu 时仍能用 MilvusLite（性能受限但功能完整）
2. **差分测试基准**：`tests/index/test_index_differential.py` 用 BruteForceIndex 作为 groundtruth，验证 FaissHnswIndex 的 recall@10 ≥ 0.95
3. **小 segment 实际选择**：低于阈值的 segment 实际就用它

这个决策的设计代价：必须保证 `VectorIndex` protocol 足够通用，能同时容纳 brute-force 和 ANN 两种范式的实现。protocol 设计要先满足 brute-force（最简单），ANN 实现"撑大"接口。

### 2.4 load / release 状态机（决定在 Phase 9.3 引入）

**决定：在 Phase 9.3 引入显式的 `_load_state` 状态机**，而不是等到 Phase 10 gRPC 适配层。

状态：

```
                  ┌──────────┐
                  │ released │  ◄── 初始 / Collection 刚 open / 显式 release()
                  └─────┬────┘
                        │ load()
                        ▼
                  ┌──────────┐
                  │ loading  │  ◄── 正在构建/加载 index 文件
                  └─────┬────┘
                        │ 全部 segment 就绪
                        ▼
                  ┌──────────┐
                  │  loaded  │  ◄── search 可用
                  └──────────┘
```

**行为**：
- `Collection.search` / `query` / `get` 在非 `loaded` 态抛 `CollectionNotLoadedError`
- `Collection.insert` / `delete` 不需要 loaded 态（写入路径不依赖索引）
- 重启后默认 `released`，必须显式 `load()`（与 Milvus 行为对齐）
- 无 IndexSpec 的 Collection 也允许 `load()` —— `load_state` 仍然进入 `loaded`，但 segment 不构建任何索引，search 走 brute-force（与 Milvus "无 index 的 collection 也可以 load + search" 行为对齐）

**为什么不延后到 Phase 10**：
- pymilvus 用户已经习惯 `load_collection / release_collection`，Phase 10 必须有东西可映射
- 状态机本身只有几十行代码，但语义早期定下来对 Phase 10 的 servicer 写法有决定性影响
- Phase 9.4 的 index 持久化和 load 机制有强耦合，分两阶段做反而麻烦

---

## 3. VectorIndex 抽象

### 3.1 protocol 定义

```python
# milvus_lite/index/protocol.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

class VectorIndex(ABC):
    """Abstract interface for any per-segment vector index implementation.

    Implementations: BruteForceIndex (NumPy), FaissHnswIndex (FAISS HNSW),
    future: FaissIvfFlatIndex, FaissIvfPqIndex, ...

    Lifetime: build → save → load → search → close. After close, all
    methods raise. Indexes are immutable — there is no add/remove after
    build. Compaction creates a new index for the merged segment instead.
    """

    metric: str    # "COSINE" | "L2" | "IP"
    num_vectors: int
    dim: int

    @classmethod
    @abstractmethod
    def build(
        cls,
        vectors: np.ndarray,        # (N, dim) float32
        metric: str,                # "COSINE" | "L2" | "IP"
        params: dict,               # implementation-specific
    ) -> "VectorIndex":
        """Construct a fresh index from a set of vectors. The local id
        of each vector is its row index 0..N-1; mapping back to pk is
        the Segment's responsibility, not the index's."""

    @abstractmethod
    def search(
        self,
        queries: np.ndarray,            # (nq, dim) float32
        top_k: int,
        valid_mask: Optional[np.ndarray] = None,  # (num_vectors,) bool
        params: Optional[dict] = None,  # impl-specific (efSearch, nprobe, ...)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (local_ids, distances), each shape (nq, top_k).

        valid_mask is the bitmap pipeline output AFTER dedup + tombstone
        + scalar filter. The index uses it to skip excluded rows DURING
        search (not after — this is the whole point of using FAISS
        IDSelectorBitmap instead of post-filtering).

        distances are returned in the canonical "smaller is more similar"
        convention regardless of metric:
            - L2:    raw L2 distance (NOT squared L2 like FAISS internal)
            - IP:    -dot(q, v)  (negated so smaller = more similar)
            - COSINE: 1 - dot(q_norm, v_norm)
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk. Format is implementation-specific."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, metric: str, dim: int) -> "VectorIndex":
        """Reload a previously saved index from disk."""

    @property
    @abstractmethod
    def index_type(self) -> str:
        """A string tag like 'BRUTE_FORCE' / 'HNSW' / 'IVF_FLAT'."""
```

**关键设计点**：

1. **local_id 只在 segment 内有效**：Index 不知道也不关心 pk 是什么。Segment 通过自己的 `pks` 数组把 local_id 翻译回 pk。这保证 index 实现完全 schema-agnostic、pk-type-agnostic。
2. **distance 归一化在 index 内部完成**：FAISS L2 返回 squared L2、IP 返回越大越相似，与我们的 `compute_distances` 约定不符。归一化在 `FaissHnswIndex.search` 内部做，确保上层看到的距离语义与 brute-force 一致 —— 这是差分测试能跑的前提。
3. **valid_mask 是 search 参数，不是 build 参数**：因为 mask 取决于运行时的 delta_index 和 filter_mask。FAISS 通过 `IDSelectorBitmap` 在 search 路径上吃下 mask，brute-force 通过 `vectors[mask]` 取子集后再算距离。
4. **没有 add / remove 接口**：immutable 是契约的一部分。任何"修改 index"的需求都通过"丢弃旧 segment + 建新 segment + 建新 index"完成。

### 3.2 IndexSpec

```python
# milvus_lite/index/spec.py

from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass(frozen=True)
class IndexSpec:
    """Persisted on the Collection (via Manifest) and used by every
    Segment to decide what kind of index to build.

    Mirrors Milvus's IndexParams structure for direct pymilvus mapping.
    """
    field_name: str          # which vector field this index covers
    index_type: str          # "BRUTE_FORCE" | "HNSW" | "IVF_FLAT" | ...
    metric_type: str         # "COSINE" | "L2" | "IP"
    build_params: Dict       # impl-specific: {"M": 16, "efConstruction": 200}
    search_params: Dict = field(default_factory=dict)  # impl-specific defaults: {"ef": 64}

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "IndexSpec": ...
```

**为什么 frozen**：和 `CompiledExpr` / `FieldSchema` 一致 — Manifest 持久化 + 跨 segment 共享 + hash 安全。

**为什么 build_params 是 dict 而不是 typed**：不同 index_type 的参数差异巨大（HNSW 有 M / efConstruction，IVF 有 nlist），typed 会导致 `Union[HnswParams, IvfParams, ...]` 的类型膨胀。dict + impl 内部校验是更轻的选择，与 Milvus proto 的 `KeyValuePair` 表达直接对齐。

---

## 4. 与现有代码的接入点

### 4.1 Segment 改动

```python
# milvus_lite/storage/segment.py

class Segment:
    __slots__ = (
        ..., "index",   # 新增
    )

    def __init__(self, ...):
        ...
        self.index: Optional[VectorIndex] = None

    def attach_index(self, index: "VectorIndex") -> None:
        """Attach a built or loaded index. Idempotent — replaces any
        existing index. Used by build_or_load_index."""
        self.index = index

    def release_index(self) -> None:
        """Drop the index reference. Memory freed when GC collects."""
        self.index = None

    def build_or_load_index(
        self,
        spec: "IndexSpec",
        index_dir: str,
    ) -> None:
        """Try to load index from disk; build + save if not found.

        Called by:
            - Collection.load() for every existing segment
            - flush.execute_flush() for newly created segments (if
              Collection is in 'loaded' state)
            - compaction.run_compaction() for the merged segment
        """
        path = self._index_file_path(index_dir, spec.index_type)
        if os.path.exists(path):
            self.index = build_index_from_factory(...).load(path, spec.metric_type, self.vector_dim)
            return
        # Build from scratch
        self.index = factory.build_index_from_spec(spec, self.vectors)
        self.index.save(path)

    def _index_file_path(self, index_dir: str, index_type: str) -> str:
        """Convention: indexes/<data_filename_stem>.<index_type>.idx"""
        stem = os.path.splitext(os.path.basename(self.file_path))[0]
        return os.path.join(index_dir, f"{stem}.{index_type.lower()}.idx")
```

**关键不变量**：`Segment.file_path` 和 `Segment.index 的存盘路径`是 1:1 对应的。任何 segment 删除都必须同步删除对应的 index 文件，反之亦然。

### 4.2 Collection 改动

```python
# milvus_lite/engine/collection.py

class Collection:
    def __init__(self, ...):
        ...
        self._index_spec: Optional[IndexSpec] = self._manifest.index_spec
        self._load_state: Literal["released", "loading", "loaded"] = "released"

    # ── new public API ───────────────────────────────────────────

    def create_index(self, field_name: str, index_params: dict) -> None:
        """Persist an IndexSpec on the manifest. Does NOT actually build
        any index — that happens at load() time. Mirrors Milvus behavior.

        Raises:
            IndexAlreadyExistsError: if create_index already called
            FilterFieldError: if field_name is not a vector field
        """
        if self._index_spec is not None:
            raise IndexAlreadyExistsError(...)
        spec = IndexSpec(
            field_name=field_name,
            index_type=index_params["index_type"],
            metric_type=index_params["metric_type"],
            build_params=index_params.get("params", {}),
            search_params=index_params.get("search_params", {}),
        )
        self._index_spec = spec
        self._manifest.set_index_spec(spec)
        self._manifest.save()

    def drop_index(self, field_name: str) -> None:
        """Remove the IndexSpec and all on-disk .idx files."""
        if self._index_spec is None:
            return
        # Release in-memory indexes
        for seg in self._segment_cache.values():
            seg.release_index()
        # Delete .idx files
        for seg in self._segment_cache.values():
            path = seg._index_file_path(self._index_dir(seg.partition), self._index_spec.index_type)
            if os.path.exists(path):
                os.remove(path)
        self._index_spec = None
        self._manifest.set_index_spec(None)
        self._manifest.save()
        self._load_state = "released"

    def load(self) -> None:
        """Build or load all segment indexes. Required before search."""
        if self._load_state == "loaded":
            return
        self._load_state = "loading"
        try:
            if self._index_spec is not None:
                for seg in self._segment_cache.values():
                    if seg.index is None:
                        index_dir = self._index_dir(seg.partition)
                        os.makedirs(index_dir, exist_ok=True)
                        seg.build_or_load_index(self._index_spec, index_dir)
            self._load_state = "loaded"
        except Exception:
            self._load_state = "released"
            raise

    def release(self) -> None:
        """Drop all in-memory indexes. Subsequent search() raises."""
        for seg in self._segment_cache.values():
            seg.release_index()
        self._load_state = "released"

    def has_index(self) -> bool:
        return self._index_spec is not None

    def get_index_info(self) -> Optional[dict]:
        return self._index_spec.to_dict() if self._index_spec else None

    # ── search guard ─────────────────────────────────────────────

    def search(self, ...):
        if self._load_state != "loaded":
            raise CollectionNotLoadedError(self.name)
        ...
```

### 4.3 search executor 改动

新增 `execute_search_with_index` 函数，**与原 `execute_search` 并存**。Collection.search 根据 `_load_state` 选择路径：

- 始终走新路径 `execute_search_with_index`
- 新路径内部按 segment 分别召回，每个 segment 看自己有没有 index：有就用 index，没有就走 brute-force

```python
# milvus_lite/search/executor_indexed.py  (or extend executor.py)

def execute_search_with_index(
    query_vectors: np.ndarray,
    segments: List["Segment"],
    memtable: "MemTable",
    delta_index: "DeltaIndex",
    top_k: int,
    metric_type: str,
    pk_field: str,
    vector_field: str,
    compiled_filter: Optional["CompiledExpr"] = None,
    partition_names: Optional[List[str]] = None,
) -> List[List[dict]]:
    """Per-segment recall + global merge.

    Algorithm:
        1. For each segment (filtered by partition):
            a. Build per-segment valid_mask via bitmap (dedup intra-segment +
               tombstone + scalar filter mask for THIS segment's rows)
            b. If segment.index is not None: index.search(q, top_k, valid_mask)
               Else: brute-force on segment.vectors[valid_indices]
            c. Translate local_ids back to pks via segment.pks
            d. Collect (pk, distance, segment, row_idx) tuples
        2. Process memtable similarly (always brute-force; no index there)
        3. Global dedup by pk (keep max-seq) — different segments may
           have the same pk due to upsert
        4. Global top-k by distance
        5. Materialize result dicts via segment.row_to_dict / memtable record
    """
    ...
```

**关键复杂度点**：
- **per-segment top-k 后的全局合并**：每个 segment 召回 `top_k` 个，最后从 `N_segments * top_k + memtable_topk` 个候选里再取全局 top-k。理论上"per-segment 取 k"会丢失部分召回（如果某个 segment 实际有 k+1 个该 query 的近邻），实践上 N_segments 较小时影响很小。要不要"per-segment 取 2k"是 Phase 9.5 的可调参数。
- **跨 segment dedup**：upsert 场景下同一个 pk 可能在多个 segment 出现，要按 max-seq 去重。这一步在原 `execute_search` 里由 `bitmap.build_valid_mask` 处理，新路径要在合并阶段重做。
- **valid_mask 的 per-segment 切分**：原来的 `filter_mask` 是全局拼接的；新路径下要按 segment 切回去（assembler 可以提供"分 segment 的 filter mask 列表"而不是合并版本）。

### 4.4 flush / compaction 钩子

```python
# milvus_lite/engine/flush.py — Step 8 (new)

def execute_flush(collection):
    ... # Steps 1-7 unchanged

    # Step 8: build indexes for newly created segments if collection is loaded
    if collection._load_state == "loaded" and collection._index_spec is not None:
        for new_segment in newly_added_segments:
            index_dir = collection._index_dir(new_segment.partition)
            os.makedirs(index_dir, exist_ok=True)
            new_segment.build_or_load_index(collection._index_spec, index_dir)
```

```python
# milvus_lite/engine/compaction.py — at end of run_compaction

def run_compaction(collection, ...):
    ...
    # After: new merged segment created, old segments dropped
    # Delete old .idx files for the dropped segments
    for old_seg in dropped_segments:
        old_idx_path = old_seg._index_file_path(...)
        if os.path.exists(old_idx_path):
            os.remove(old_idx_path)
    # Build new index for merged segment if loaded
    if collection._load_state == "loaded" and collection._index_spec is not None:
        index_dir = collection._index_dir(merged_segment.partition)
        merged_segment.build_or_load_index(collection._index_spec, index_dir)
```

### 4.5 recovery 改动

```python
# milvus_lite/engine/recovery.py

def recover(collection):
    ... # Steps 1-5 unchanged

    # Step 6 (new): default load_state is "released"
    # Even if the manifest has an index_spec, segments are NOT loaded
    # automatically — caller must explicitly call collection.load().
    # This matches Milvus behavior and avoids surprise startup latency.
    collection._load_state = "released"

    # Orphan cleanup also covers .idx files
    _cleanup_orphan_index_files(collection)
```

### 4.6 Manifest schema bump

```python
# milvus_lite/storage/manifest.py

# Bump format_version from 1 to 2 to add index_spec support.
# Backward-compat: old manifests without index_spec field load with
# index_spec=None.

@dataclass
class ManifestState:
    ...
    index_spec: Optional[IndexSpec] = None  # NEW
    format_version: int = 2                  # bumped from 1

def to_dict(self) -> dict:
    return {
        ...,
        "index_spec": self.index_spec.to_dict() if self.index_spec else None,
        "format_version": self.format_version,
    }

@classmethod
def from_dict(cls, d: dict) -> "ManifestState":
    fv = d.get("format_version", 1)
    spec_dict = d.get("index_spec")
    spec = IndexSpec.from_dict(spec_dict) if spec_dict else None
    return cls(..., index_spec=spec, format_version=2)
```

**兼容策略**：旧 v1 manifest 加载时 `index_spec` 字段缺失 → 默认 None，下一次 save 时升级到 v2。无需迁移工具。

---

## 5. 目录布局

```
data_dir/
└── collections/
    └── <collection_name>/
        ├── schema.json
        ├── manifest.json          # 含 index_spec
        ├── manifest.json.prev
        ├── wal/
        │   ├── data_*.arrow
        │   └── delta_*.arrow
        └── partitions/
            └── <partition_name>/
                ├── data/
                │   ├── data_000001_000500.parquet
                │   └── data_000501_001000.parquet
                ├── delta/
                │   └── delta_000501_000503.parquet
                └── indexes/                                  ← Phase 9 新增
                    ├── data_000001_000500.brute_force.idx
                    └── data_000501_001000.hnsw.idx
```

**命名约定**：`<data_filename_stem>.<index_type_lowercase>.idx`

**严格不变量**：
1. 一个 segment 的 .idx 文件名由 segment 文件名 + 当前 IndexSpec 的 index_type 唯一确定
2. compaction 删 segment 时同时删 .idx；写新 segment 时同时写新 .idx
3. recovery 启动时扫 indexes/ 目录，孤儿 .idx（对应的 segment 文件已不存在）一律删除

---

## 6. FAISS 接入坑点

### 6.1 Metric 符号对齐（最大坑）

| Metric | FAISS 内部约定 | MilvusLite 上层约定 | 转换 |
|---|---|---|---|
| L2 | squared L2（越小越相似） | raw L2（越小越相似） | `dist = sqrt(faiss_dist)` |
| IP | dot product（越大越相似） | -dot（越小越相似） | `dist = -faiss_dist` |
| COSINE | 等价于"对 vector 做 L2 normalize 后的 IP" | `1 - cosine_sim` | `vectors_norm = normalize(vectors); query_norm = normalize(query); dist = 1 - faiss_ip(query_norm, vectors_norm)` |

**实现位置**：`FaissHnswIndex.search` 的 distance 后处理，确保返回给 executor 的 distance 与 brute-force 完全一致。

**测试方法**：差分测试 — 同一份数据 build 两个 index（brute / faiss），对 100 个随机 query 验证 distance 误差 < 1e-3（对召回命中的 pk）。

### 6.2 IDSelectorBitmap 的字节对齐

```python
# faiss.IDSelectorBitmap 接受的是按 bit packing 的 uint8 数组
# numpy bool array 不能直接传，需要 packbits

import faiss
import numpy as np

mask_bool = np.array([True, False, True, ...], dtype=bool)
mask_packed = np.packbits(mask_bool, bitorder='little')  # uint8 array
selector = faiss.IDSelectorBitmap(num_vectors, faiss.swig_ptr(mask_packed))
params = faiss.SearchParametersHNSW(sel=selector)
D, I = index.search(queries, top_k, params=params)
```

注意 `bitorder='little'` 必须显式指定（FAISS 期望 LSB-first packing）。这个细节调试要靠单测覆盖。

### 6.3 FAISS HNSW 不需要 train，但 IVF 要

```python
# HNSW: just add
index = faiss.IndexHNSWFlat(dim, M)
index.add(vectors)

# IVF: train first, then add
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(vectors[:training_subset])
index.add(vectors)
```

Phase 9 MVP 只支持 HNSW，IVF 类索引留到 Phase 9.5+ 之后的扩展。

### 6.4 持久化

```python
# Save
faiss.write_index(self._index, path)

# Load
loaded = faiss.read_index(path)
```

注意 FAISS 的 write_index/read_index 是 C++ 序列化格式，不是 numpy。一份 index 文件 = 一个 FAISS object。

### 6.5 macOS arm64 wheel

最新 `faiss-cpu>=1.7.4` 在 PyPI 上有 macOS arm64 wheel。但仍然推荐：

```toml
# pyproject.toml
[project.optional-dependencies]
faiss = ["faiss-cpu>=1.7.4"]
```

并在 `milvus_lite/index/factory.py` 用 try-import 模式：

```python
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

def build_index_from_spec(spec: IndexSpec, vectors: np.ndarray) -> VectorIndex:
    if spec.index_type in ("HNSW", "IVF_FLAT", ...):
        if not _FAISS_AVAILABLE:
            raise IndexBackendUnavailableError(
                f"index_type={spec.index_type} requires faiss-cpu; "
                "install with `pip install milvus_lite[faiss]`"
            )
        ...
    elif spec.index_type == "BRUTE_FORCE":
        return BruteForceIndex.build(vectors, spec.metric_type, spec.build_params)
```

---

## 7. 生命周期时序图

### 7.1 正常写入 + 搜索（Collection 已 loaded）

```
client                Collection           flush.py        Segment      VectorIndex
  │                       │                    │              │              │
  ├─ insert(records) ─────►                    │              │              │
  │                       ├─ memtable.append ──┤              │              │
  │                       │  (memtable 满)     │              │              │
  │                       ├─ flush() ──────────►              │              │
  │                       │                    ├─ write data parquet ────────►
  │                       │                    ├─ load Segment ──────────────►
  │                       │                    ├─ if loaded:                 │
  │                       │                    │   build_or_load_index ──────►
  │                       │                    │                             ├─ FaissHnsw.build
  │                       │                    │                             ├─ save .idx
  │                       │                    ◄──────────── attach ─────────┤
  │                       ◄────────────────────┤              │              │
  │                                                                          │
  ├─ search([q], top_k) ──►                                                  │
  │                       ├─ assemble per-segment masks ──────►              │
  │                       │                                  │              │
  │                       ├─ for each segment:                              │
  │                       │   if has index: ──────────────────────────────►
  │                       │                                                ├─ search(q, mask)
  │                       │   else: brute force                            │
  │                       ├─ merge top-k from all segments + memtable      │
  ◄───────────────────────┤                                                │
```

### 7.2 重启 + 显式 load

```
client                Collection           recovery        Manifest      Segment    VectorIndex
  │                       │                    │              │             │            │
  ├─ open(data_dir) ──────►                    │              │             │            │
  │                       ├─ recover() ────────►              │             │            │
  │                       │                    ├─ load manifest (incl. index_spec)       │
  │                       │                    ├─ replay WAL → memtable                  │
  │                       │                    ├─ load Segments (no index)               │
  │                       │                    ├─ load_state = released                  │
  │                       ◄────────────────────┤              │             │            │
  │                                                                                       │
  ├─ search(...) ─────────►                                                              │
  │                       ├─ raise CollectionNotLoadedError                              │
  ◄───────────────────────┤                                                              │
  │                                                                                       │
  ├─ load() ──────────────►                                                              │
  │                       ├─ load_state = loading                                        │
  │                       ├─ for each segment:                                           │
  │                       │   build_or_load_index ────────────────────────►              │
  │                       │                                                ├─ load .idx (fast) │
  │                       │                                                │   OR build (slow)│
  │                       ├─ load_state = loaded                                         │
  ◄───────────────────────┤                                                              │
```

### 7.3 compaction 时序

```
flush.py            compaction.py         Segment(old)    Segment(new)   VectorIndex
  │                       │                    │               │              │
  │                       ├─ pick small files ─►               │              │
  │                       ├─ merge & dedup ───────────────────►               │
  │                       ├─ write new parquet                                │
  │                       ├─ load new Segment ────────────────►               │
  │                       ├─ remove old segments ──────────────►              │
  │                       │   (also rm old .idx files)         │              │
  │                       ├─ if loaded:                                       │
  │                       │   build_or_load_index ─────────────────────────────►
  │                       │                                                   ├─ build + save
  │                       ├─ manifest.swap                                    │
```

---

## 8. recall 验证策略

### 8.1 差分测试结构

`tests/index/test_index_differential.py`：

```python
@pytest.mark.parametrize("dim", [4, 32, 128])
@pytest.mark.parametrize("n", [100, 10_000])
@pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
def test_faiss_hnsw_recall_vs_brute_force(dim, n, metric):
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(20, dim).astype(np.float32)

    brute = BruteForceIndex.build(vectors, metric, {})
    faiss_idx = FaissHnswIndex.build(vectors, metric, {"M": 16, "efConstruction": 200})

    brute_ids, brute_dists = brute.search(queries, top_k=10)
    faiss_ids, faiss_dists = faiss_idx.search(queries, top_k=10, params={"ef": 64})

    # 1) recall@10 ≥ 0.95
    for i in range(20):
        overlap = len(set(brute_ids[i]) & set(faiss_ids[i]))
        assert overlap / 10 >= 0.95, \
            f"recall@10 = {overlap/10} for query {i}, dim={dim}, metric={metric}"

    # 2) distance value parity for hits — within 1e-3 relative error
    for i in range(20):
        faiss_id_to_dist = dict(zip(faiss_ids[i], faiss_dists[i]))
        brute_id_to_dist = dict(zip(brute_ids[i], brute_dists[i]))
        for pid, fdist in faiss_id_to_dist.items():
            if pid in brute_id_to_dist:
                bdist = brute_id_to_dist[pid]
                assert abs(fdist - bdist) < 1e-3 + 1e-3 * abs(bdist), \
                    f"distance mismatch for pk={pid}: faiss={fdist} brute={bdist}"
```

**为什么这个 setup**：
- BruteForce 是数学上的 groundtruth，FAISS 是被测对象 — 与 Phase 8 的差分测试结构对称
- recall@10 ≥ 0.95 是 HNSW 在合理参数下的常规水平
- distance value parity 验证 metric 符号对齐没出错（一旦 metric 转换写错，第二个断言一定挂）
- 三个 metric × 三个 dim × 两个规模 = 18 个 case，跑得快但覆盖广

### 8.2 端到端 search 路径的差分

`tests/engine/test_search_index_vs_brute.py`：

```python
def test_collection_search_index_path_matches_brute_force(tmp_path):
    """The full Collection.search going through index path returns the
    same top-k as if we forced brute-force everywhere."""
    db = MilvusLite(str(tmp_path))
    col = db.create_collection("test", schema=...)
    col.insert([...])  # 1000 records
    col.flush()

    # Path 1: brute-force
    col.create_index("vec", {"index_type": "BRUTE_FORCE", "metric_type": "COSINE"})
    col.load()
    brute_results = col.search(query_vectors=[[...]], top_k=10)

    # Path 2: HNSW
    col.release()
    col.drop_index("vec")
    col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16}})
    col.load()
    hnsw_results = col.search(query_vectors=[[...]], top_k=10)

    # Top-1 must match (HNSW recall@1 is essentially 1.0 for n=1000)
    assert brute_results[0][0]["id"] == hnsw_results[0][0]["id"]
    # Top-10 set overlap ≥ 9/10
    brute_ids = {r["id"] for r in brute_results[0]}
    hnsw_ids = {r["id"] for r in hnsw_results[0]}
    assert len(brute_ids & hnsw_ids) >= 9
```

---

## 9. Phase 9 子阶段拆分

| 子阶段 | 内容 | 完成标志 | 工作量 |
|---|---|---|---|
| **9.1** | 补齐 pymilvus quickstart 前置 API：`Collection.create_partition / drop_partition / list_partitions / num_entities / describe` + `search(output_fields=...)` + `MilvusLite.get_collection_stats` | 6 个新方法 + 测试齐全；不引入任何 index 概念 | S |
| **9.2** | `VectorIndex` protocol + `BruteForceIndex` + 接入 `Segment.index` + 新 `execute_search_with_index` 路径（仍走 brute-force 实现） | 全部老搜索测试在新路径下通过；差分测试 brute-force-via-index ≡ 老 execute_search | M |
| **9.3** | `IndexSpec` + `Manifest` v2 升级 + `Collection.create_index / drop_index / load / release / has_index / get_index_info` + `_load_state` 状态机 + `CollectionNotLoadedError` | `col.create_index → col.load → col.search → col.release → col.search raise` 全链路通；manifest v1→v2 兼容测试通 | M |
| **9.4** | Index 文件持久化（`indexes/<stem>.<type>.idx`）+ flush / compaction / recovery 钩子 + 孤儿 .idx 清理 | Collection 重启 → load → search 等价；compaction 后无孤儿 .idx；崩溃注入测试通 | M |
| **9.5** | `FaissHnswIndex` + factory 路由 + `[faiss]` extras + metric 对齐 + `IDSelectorBitmap` 接入 + 差分测试 + benchmark | 100K 向量 search QPS 比 brute-force 高 ≥50x；recall@10 ≥ 0.95；macOS arm64 + Linux 双平台 CI 通 | L |
| **9.6** | `examples/m9_demo.py` + 长跑测试 + Phase 9 文档 backfill | m9 demo 通；`@pytest.mark.slow` 100K 测试通；`plan/index-design.md` 与最终代码对齐 | S |

合计：2S + 3M + 1L

---

## 10. 关键风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| FAISS metric 符号对齐写错 | 高 | 高（搜索结果错） | 差分测试是必经关卡，metric 错一定挂 |
| FAISS macOS arm64 wheel 装不上 | 中 | 中 | optional extra + BruteForce fallback；CI 跑双 matrix |
| `IDSelectorBitmap` packbits 顺序错 | 中 | 高 | 单测覆盖各种 mask pattern |
| segment-level top-k 合并丢召回 | 低 | 低 | per-segment 取 `2*top_k` 候选；可调参数 |
| flush 末尾同步 build index 让写入变慢 | 高 | 中（UX） | 接受 MVP 行为；future 把 index build 移出 flush 同步路径 |
| compaction 后 .idx 文件孤儿 | 中 | 低 | recovery 启动时 cleanup_orphan_index_files |
| Manifest v1 → v2 兼容失败 | 低 | 高 | 明确测试覆盖：旧 manifest 文件 → 新代码读取 → 升级保存 |
| 多线程 load 期间 search 调用 | 低 | 中 | `_load_state == "loading"` 时 search 抛错；load 不并发 |

---

## 11. 不在 Phase 9 范围

| 功能 | 推迟到 |
|---|---|
| IVF / IVF-PQ / OPQ 等量化索引 | Phase 9.5+ 之后扩展 |
| GPU 加速（faiss-gpu） | Future |
| 向量 quantization（int8, fp16, bf16, binary） | Future |
| 异步 index build（不阻塞 flush） | Future |
| 索引 warmup / 预读 | Future |
| 多向量字段（一个 Collection 多个 vector 列） | Future（Milvus 也是后期才加） |
| Index 参数自动调优 | Future |
| Sparse vector index | Future |

Phase 9 MVP 的目标是"让 search 不再是 brute-force"，复杂索引家族留给后续。

---

## 12. 完成标志

- `col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})` 可以正常 persist 到 manifest
- `col.load()` 后 `col.search([[...]], top_k=10)` 走 FAISS 路径，性能比 brute-force 高一个数量级
- `col.release()` 后 search 抛 `CollectionNotLoadedError`
- 重启进程 → `col.load()` 秒级完成（直接 load .idx 文件，不重建）
- compaction 后旧 .idx 自动清理，新 .idx 自动构建
- 差分测试 recall@10 ≥ 0.95 全绿
- `examples/m9_demo.py` 跑通
- 跑 `pytest` 全部老测试 + 新测试全绿
