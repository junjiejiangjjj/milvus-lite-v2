# 深入设计：FuncChain 函数链（Function Chain）

## 1. 概述

LiteVecDB 当前的函数执行逻辑分散在 Collection 类中：4 组 per-type 列表（`_bm25_functions`、`_embedding_functions`、`_rerank_functions`、`_decay_functions`）+ 4 个 apply 方法，每新增一种函数类型都要在初始化、insert、search 三处添加 if/elif 分支。

**FuncChain 的目标**：参考 Milvus `internal/util/function/chain/` 的设计，用统一的 **Operator 管道** 替代 per-type 分支，让多个 Function 可以 **串行组合** 执行。

**为什么现在做**：
- Milvus 已经用 FuncChain 统一了 reranking 管线（Merge → Map → Sort → Limit → Select），并预留了 `StageIngestion` 用于将 BM25/Embedding 也纳入 chain 体系
- LiteVecDB 已经积累了 4 种函数类型（BM25、TEXT_EMBEDDING、RERANK、DECAY），分散逻辑的维护成本越来越高
- 后续要支持更复杂的串联场景（如 text → Embedding A → dense_vec_a → DimReduce → dense_vec_b），需要通用的链式执行框架

**核心定位**：与 Milvus chain 保持概念对齐（FunctionExpr / Operator / FuncChain / Stage），但用 Python 原生数据结构（`List[dict]`）替代 Arrow DataFrame，适配嵌入式场景。

---

## 2. 与 Milvus chain 的对齐关系

| Milvus (Go) | LiteVecDB (Python) | 差异说明 |
|---|---|---|
| `types.FunctionExpr` interface | `FunctionExpr` ABC | 相同语义：无状态列计算 |
| `DataFrame` (Arrow Chunked) | `DataFrame` (List[List[dict]]) | Python GC 替代 Arrow Allocator；chunk = per-query 结果 |
| `Operator` interface | `Operator` ABC | 相同语义：`execute(ctx, df) → df` |
| `MapOp` + `BaseOp` | `MapOp` | 列映射 + FunctionExpr 调用 |
| `MergeOp` (5 strategies) | `MergeOp` (RRF/Weighted/Max/Sum/Avg) | 多路搜索结果合并 |
| `SortOp` (per-chunk sort) | `SortOp` (per-chunk sort) | 按列排序 |
| `LimitOp` (per-chunk) | `LimitOp` (per-chunk) | offset + limit |
| `SelectOp` | `SelectOp` | 列投影 |
| `GroupByOp` | `GroupByOp` | 分组搜索 |
| `FilterOp` | `FilterOp` | 布尔表达式过滤行 |
| `FuncChain` | `FuncChain` | 有序管道 + stage 校验 + fluent API |
| `rerank_builder.go` | `builder.py` | 从 schema.functions 构建 chain |
| `types.FuncContext` | `FuncContext` | 执行上下文（stage） |
| `types.FunctionFactory` + Registry | `create_function_expr()` factory | 嵌入式不需要动态注册，直接 factory |
| `StageIngestion / StageL2Rerank / ...` | `STAGE_INGESTION / STAGE_RERANK` | 简化为两个 stage（嵌入式无分布式多级 rerank） |

---

## 3. 架构决策

### 3.1 数据容器：Python DataFrame（决定）

**决定：用 `List[List[dict]]` 作为 DataFrame 内部表示，每个内层 list 是一个 chunk（对应一个 query 的结果集）。**

候选方案对比：

| 维度 | Python List[List[dict]]（选定） | PyArrow Table | 自定义列式结构 |
|---|---|---|---|
| 实现复杂度 | 零依赖，Python 原生 | 需要 Arrow 类型派发 | 需大量样板代码 |
| 与现有代码对接 | insert records 本身就是 List[dict] | 需要 dict↔Arrow 转换 | 需要 dict↔自定义 转换 |
| per-chunk 语义 | 二维 list 天然支持 | 需要 ChunkedArray | 需自建 chunk 机制 |
| 性能 | 足够（嵌入式规模） | 大数据量更优 | 居中 |
| Operator 实现 | 标准 Python list 操作 | 需 Arrow compute | 视结构而定 |

**理由**：
- LiteVecDB 是嵌入式单进程，数据规模有限，Python 原生结构足够
- 现有 insert/search 接口已经是 `List[dict]`，无需转换层
- Operator 实现更直观，调试更容易

**chunk 语义**：
- Ingestion 阶段：单 chunk — `chunks = [records]`
- Rerank 阶段：nq 个 chunk — `chunks[i]` = 第 i 个 query 的搜索结果

### 3.2 Stage 设计：两阶段（决定）

**决定：只定义两个 stage — `ingestion` 和 `rerank`。**

Milvus 有 6 个 stage（Ingestion、L2_Rerank、L1_Rerank、L0_Rerank、PreProcess、PostProcess），因为分布式系统中 reranking 可以在 Proxy / QueryNode / Segment 三层执行。LiteVecDB 是单进程嵌入式，所有 reranking 在同一处完成，无需多级 rerank stage。

```python
STAGE_INGESTION = "ingestion"   # insert/upsert 时执行
STAGE_RERANK    = "rerank"      # search 后处理时执行
```

### 3.3 Operator 集合：6 种（决定）

**决定：实现 Map、Merge、Sort、Limit、Select、GroupBy 六种 Operator。**

这是 Milvus `rerank_builder.go` 实际用到的完整 Operator 集合。以 Decay reranker chain 为例：

```
Merge(strategy) → Map(DecayExpr) → Map(ScoreCombineExpr) → Sort($score, DESC) → Limit(limit, offset) → [Map(RoundDecimal)] → Select($id, $score)
```

只有 Map 无法表达 Merge（多路合并）、Sort（排序）、Limit（分页）、Select（投影）这些搜索后处理操作。四种 reranker（RRF / Weighted / Decay / Model）都共用 Sort → Limit → Select 尾部，仅 Merge 策略和中间 Map 步骤不同。

FilterOp 暂不在 MVP 范围（当前 scalar filter 在搜索侧独立处理），但接口预留。

### 3.4 FunctionExpr 与 Operator 的分层（决定）

**决定：FunctionExpr 只负责纯列计算（input columns → output columns），列映射（从 DataFrame 读哪些列、写到哪些列）由 Operator 负责。**

这与 Milvus 的设计一致：

```
MapOp（Operator 层）  ─┬─ 负责：从 DataFrame 读 input_cols，写 output_cols
                       └─ 调用：FunctionExpr.execute(inputs) → outputs

FunctionExpr（计算层） ─── 负责：纯计算逻辑（如 BM25 分词、decay 衰减）
                          不知道 DataFrame 的存在
```

好处：同一个 FunctionExpr 可以用不同的列映射复用（如 DecayExpr 作用于不同字段）。

---

## 4. 核心接口设计

### 4.1 FunctionExpr — 无状态列计算单元

```python
# function/types.py

from abc import ABC, abstractmethod
from typing import List, FrozenSet

STAGE_INGESTION = "ingestion"
STAGE_RERANK    = "rerank"

# DataFrame 内部使用的虚拟列名
ID_FIELD    = "$id"
SCORE_FIELD = "$score"


class FunctionExpr(ABC):
    """无状态的列级计算单元。

    只负责 input columns → output columns 的纯计算。
    不感知 DataFrame 结构、列名映射由 MapOp 处理。

    对应 Milvus: internal/util/function/chain/types.FunctionExpr
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """函数名（如 "bm25", "decay", "score_combine"）。"""

    @property
    @abstractmethod
    def supported_stages(self) -> FrozenSet[str]:
        """此函数支持在哪些 stage 执行。"""

    @abstractmethod
    def execute(self, inputs: List[list]) -> List[list]:
        """执行计算。

        Args:
            inputs: 输入列列表。inputs[i] 是一列值的 list，
                    长度 = chunk 内记录数。
        Returns:
            输出列列表。长度由函数定义（通常 = 输出字段数）。
        """

    def is_runnable(self, stage: str) -> bool:
        """检查此函数是否支持指定 stage。"""
        return stage in self.supported_stages
```

### 4.2 DataFrame — 轻量数据容器

```python
# function/dataframe.py

from typing import List, Optional


class DataFrame:
    """轻量级列式数据容器。

    内部存储为 List[List[dict]]，每个内层 list 是一个 chunk。

    - Ingestion 阶段：单 chunk，chunks = [records]
    - Rerank 阶段：nq 个 chunk，chunks[i] = 第 i 个 query 的搜索结果

    对应 Milvus: internal/util/function/chain/dataframe.go
    """

    __slots__ = ("_chunks",)

    def __init__(self, chunks: List[List[dict]]):
        self._chunks = chunks

    # ── 工厂方法 ──

    @classmethod
    def from_records(cls, records: List[dict]) -> "DataFrame":
        """从 insert records 创建（单 chunk）。"""
        return cls([records])

    @classmethod
    def from_search_results(cls, results: List[List[dict]]) -> "DataFrame":
        """从 search 返回值创建（per-query chunks）。"""
        return cls(results)

    # ── 导出 ──

    def to_records(self) -> List[dict]:
        """导出为扁平 records（仅限单 chunk）。"""
        assert len(self._chunks) == 1, "to_records() requires single chunk"
        return self._chunks[0]

    def to_search_results(self) -> List[List[dict]]:
        """导出为 per-query 搜索结果。"""
        return self._chunks

    # ── 访问器 ──

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def chunk(self, idx: int) -> List[dict]:
        return self._chunks[idx]

    def column(self, name: str, chunk_idx: int) -> list:
        """读取指定 chunk 中某列的所有值。"""
        return [r.get(name) for r in self._chunks[chunk_idx]]

    def set_column(self, name: str, chunk_idx: int, values: list) -> None:
        """将一列值写回指定 chunk（就地修改）。"""
        chunk = self._chunks[chunk_idx]
        for r, v in zip(chunk, values):
            r[name] = v

    def column_names(self, chunk_idx: int = 0) -> List[str]:
        """获取列名集合（取自第一条记录的 keys）。"""
        chunk = self._chunks[chunk_idx]
        return list(chunk[0].keys()) if chunk else []
```

### 4.3 FuncContext — 执行上下文

```python
# function/types.py (续)

class FuncContext:
    """函数链执行上下文。

    对应 Milvus: internal/util/function/chain/types.FuncContext
    """

    __slots__ = ("_stage",)

    def __init__(self, stage: str):
        self._stage = stage

    @property
    def stage(self) -> str:
        return self._stage
```

### 4.4 Operator — 算子基类

```python
# function/operator.py

from abc import ABC, abstractmethod
from typing import List


class Operator(ABC):
    """算子基类。

    Operator 工作在 DataFrame 上，接收输入 DataFrame、返回输出 DataFrame。
    每个 Operator 声明自己读取（inputs）和产出（outputs）的列名。

    对应 Milvus: internal/util/function/chain/chain.go Operator interface
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """算子名称（如 "Map", "Sort", "Merge"）。"""

    @abstractmethod
    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        """执行算子。

        Args:
            ctx: 执行上下文
            df:  输入 DataFrame
        Returns:
            输出 DataFrame（可能是就地修改后的同一对象，也可能是新对象）
        """
```

### 4.5 FuncChain — 有序管道

```python
# function/chain.py

from typing import List, Optional


class FuncChain:
    """有序的 Operator 管道。

    - Fluent API：chain.merge(...).map(...).sort(...).limit(...)
    - Execute 时按顺序执行所有 Operator
    - 如果第一个 Operator 是 MergeOp，支持多路输入

    对应 Milvus: internal/util/function/chain/chain.go FuncChain
    """

    def __init__(self, name: str, stage: str):
        self._name = name
        self._stage = stage
        self._operators: List[Operator] = []

    @property
    def stage(self) -> str:
        return self._stage

    # ── Fluent API ──

    def add(self, op: Operator) -> "FuncChain":
        """添加一个 Operator 到管道末尾。"""
        self._operators.append(op)
        return self

    def map(self, expr: FunctionExpr,
            input_cols: List[str], output_cols: List[str]) -> "FuncChain":
        """添加 MapOp。"""
        if not expr.is_runnable(self._stage):
            raise ValueError(
                f"FunctionExpr '{expr.name}' does not support "
                f"stage '{self._stage}'"
            )
        return self.add(MapOp(expr, input_cols, output_cols))

    def merge(self, strategy: str, **kwargs) -> "FuncChain":
        """添加 MergeOp（必须是 chain 的第一个 Operator）。"""
        return self.add(MergeOp(strategy, **kwargs))

    def sort(self, column: str, desc: bool = True) -> "FuncChain":
        """添加 SortOp。"""
        return self.add(SortOp(column, desc))

    def limit(self, limit: int, offset: int = 0) -> "FuncChain":
        """添加 LimitOp。"""
        return self.add(LimitOp(limit, offset))

    def select(self, *columns: str) -> "FuncChain":
        """添加 SelectOp。"""
        return self.add(SelectOp(list(columns)))

    def group_by(self, field: str, group_size: int,
                 limit: int, offset: int = 0,
                 scorer: str = "max") -> "FuncChain":
        """添加 GroupByOp。"""
        return self.add(GroupByOp(field, group_size, limit, offset, scorer))

    # ── 执行 ──

    def execute(self, *inputs: DataFrame) -> DataFrame:
        """执行整条 chain。

        如果第一个 Operator 是 MergeOp，接受多路输入；
        否则只接受单路输入。
        """
        ctx = FuncContext(self._stage)
        start_idx = 0

        if self._operators and isinstance(self._operators[0], MergeOp):
            result = self._operators[0].execute_multi(ctx, list(inputs))
            start_idx = 1
        else:
            if len(inputs) != 1:
                raise ValueError(
                    f"Chain expects 1 input but got {len(inputs)} "
                    f"(first operator is not MergeOp)"
                )
            result = inputs[0]

        for op in self._operators[start_idx:]:
            result = op.execute(ctx, result)

        return result

    # ── 调试 ──

    def __repr__(self) -> str:
        ops = " → ".join(op.name for op in self._operators)
        return f"FuncChain({self._name}, stage={self._stage}): {ops}"
```

---

## 5. Operator 详细设计

### 5.1 MapOp — 列变换

**职责**：从 DataFrame 读取 input_cols，调用 FunctionExpr.execute()，将结果写入 output_cols。

**对应 Milvus**：`operator_map.go`

```python
class MapOp(Operator):
    """对每个 chunk 独立执行 FunctionExpr 列变换。

    input_cols: 从 DataFrame 读取的列名
    output_cols: 将 FunctionExpr 输出写回 DataFrame 的列名

    output_cols 可以与 input_cols 重叠（如 ScoreCombine 将
    $score 和 _decay_score 合并后写回 $score）。
    """

    name = "Map"

    def __init__(self, expr: FunctionExpr,
                 input_cols: List[str], output_cols: List[str]):
        self._expr = expr
        self._input_cols = input_cols
        self._output_cols = output_cols

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            # 1. 读取输入列
            inputs = [df.column(col, chunk_idx) for col in self._input_cols]
            # 2. 执行函数
            outputs = self._expr.execute(inputs)
            # 3. 写回输出列
            for col_name, col_data in zip(self._output_cols, outputs):
                df.set_column(col_name, chunk_idx, col_data)
        return df
```

### 5.2 MergeOp — 多路合并

**职责**：将多路搜索结果合并为一路。这是 hybrid search 的核心 —— 多个 ANN 子搜索返回独立结果，MergeOp 按策略合并并去重。

**对应 Milvus**：`operator_merge.go`

**5 种合并策略**：

| 策略 | 公式 | 说明 |
|---|---|---|
| `rrf` | `score = Σ 1/(k + rank_i)` | 基于排名融合，不依赖分数量纲 |
| `weighted` | `score = Σ weight_i × normalize(score_i)` | 加权求和，需分数归一化 |
| `max` | `score = max(score_i)` | 取最高分（如 decay 前置的合并） |
| `sum` | `score = Σ score_i` | 分数求和 |
| `avg` | `score = mean(score_i)` | 分数平均 |

```python
class MergeOp(Operator):
    """合并多路搜索结果。

    - 必须是 chain 的第一个 Operator
    - execute_multi() 接收多路 DataFrame，按 pk 去重合并
    - 对齐 Milvus 的 MergeStrategy
    """

    name = "Merge"

    def __init__(self, strategy: str, **kwargs):
        self._strategy = strategy      # "rrf" | "weighted" | "max" | "sum" | "avg"
        self._weights = kwargs.get("weights", [])
        self._rrf_k = kwargs.get("rrf_k", 60.0)
        self._metric_types = kwargs.get("metric_types", [])
        self._normalize = kwargs.get("normalize", False)

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        raise RuntimeError("MergeOp requires execute_multi()")

    def execute_multi(self, ctx: FuncContext,
                      inputs: List[DataFrame]) -> DataFrame:
        """合并多路 DataFrame。

        每路 DataFrame 的 chunk 数应相同（= nq）。
        同一 pk 出现在多路中时，按 strategy 合并分数。
        """
        if not inputs:
            raise ValueError("MergeOp requires at least one input")
        if len(inputs) == 1:
            return inputs[0]

        nq = inputs[0].num_chunks
        merged_chunks = []

        for q in range(nq):
            # 收集所有路的 (pk, score, hit) + 该路的 rank
            pk_map = {}  # pk → {hit, scores: [(路idx, score, rank)]}
            for path_idx, inp in enumerate(inputs):
                chunk = inp.chunk(q)
                for rank, hit in enumerate(chunk):
                    pk = hit.get(ID_FIELD)
                    if pk not in pk_map:
                        pk_map[pk] = {"hit": dict(hit), "entries": []}
                    pk_map[pk]["entries"].append(
                        (path_idx, hit.get(SCORE_FIELD, 0.0), rank)
                    )

            # 按 strategy 计算最终 score
            results = []
            for pk, info in pk_map.items():
                score = self._compute_score(info["entries"], len(inputs))
                merged_hit = info["hit"]
                merged_hit[SCORE_FIELD] = score
                results.append(merged_hit)

            merged_chunks.append(results)

        return DataFrame(merged_chunks)

    def _compute_score(self, entries, num_paths):
        """按策略计算合并分数。"""
        if self._strategy == "rrf":
            return sum(1.0 / (self._rrf_k + rank) for _, _, rank in entries)
        elif self._strategy == "weighted":
            total = 0.0
            for path_idx, score, _ in entries:
                w = self._weights[path_idx] if path_idx < len(self._weights) else 1.0
                total += w * score
            return total
        elif self._strategy == "max":
            return max(score for _, score, _ in entries)
        elif self._strategy == "sum":
            return sum(score for _, score, _ in entries)
        elif self._strategy == "avg":
            scores = [score for _, score, _ in entries]
            return sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown merge strategy: {self._strategy}")
```

### 5.3 SortOp — 排序

**职责**：对每个 chunk 内的记录按指定列排序。

**对应 Milvus**：`operator_sort.go`

```python
class SortOp(Operator):
    """per-chunk 排序。

    每个 chunk（query）独立排序。
    支持 tie-break by $id ASC（与 Milvus 对齐）。
    """

    name = "Sort"

    def __init__(self, column: str, desc: bool = True,
                 tie_break_col: str = ID_FIELD):
        self._column = column
        self._desc = desc
        self._tie_break_col = tie_break_col

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            chunk.sort(
                key=lambda r: (
                    self._sort_key(r.get(self._column)),
                    r.get(self._tie_break_col, 0)
                ),
                reverse=self._desc,
            )
        return df

    @staticmethod
    def _sort_key(val):
        """None 排到最后。"""
        if val is None:
            return (1, 0)  # (is_none=1, val) — 保证 None 排末尾
        return (0, val)
```

### 5.4 LimitOp — 分页截取

**职责**：对每个 chunk 应用 offset + limit。

**对应 Milvus**：`operator_limit.go`

```python
class LimitOp(Operator):
    """per-chunk offset + limit。"""

    name = "Limit"

    def __init__(self, limit: int, offset: int = 0):
        self._limit = limit
        self._offset = offset

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            start = min(self._offset, len(chunk))
            end = min(start + self._limit, len(chunk))
            new_chunks.append(chunk[start:end])
        return DataFrame(new_chunks)
```

### 5.5 SelectOp — 列投影

**职责**：只保留指定的列，移除其余字段。

**对应 Milvus**：`operator_select.go`

```python
class SelectOp(Operator):
    """保留指定列，移除其余字段。"""

    name = "Select"

    def __init__(self, columns: List[str]):
        self._columns = set(columns)

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)
            new_chunks.append([
                {k: v for k, v in r.items() if k in self._columns}
                for r in chunk
            ])
        return DataFrame(new_chunks)
```

### 5.6 GroupByOp — 分组搜索

**职责**：按字段分组，每组保留 top-N，返回前 limit 个组。

**对应 Milvus**：`operator_group_by.go`

```python
class GroupByOp(Operator):
    """per-chunk 分组搜索。

    1. 按 group_by_field 分组
    2. 每组内按 $score DESC 排序，保留 top group_size
    3. 用 scorer 计算每组的 group_score（max/sum/avg）
    4. 组间按 group_score DESC 排序
    5. 跳过 offset 组，取 limit 组
    6. 添加 $group_score 列
    """

    name = "GroupBy"

    def __init__(self, field: str, group_size: int,
                 limit: int, offset: int = 0,
                 scorer: str = "max"):
        self._field = field
        self._group_size = group_size
        self._limit = limit
        self._offset = offset
        self._scorer = scorer  # "max" | "sum" | "avg"

    def execute(self, ctx: FuncContext, df: DataFrame) -> DataFrame:
        new_chunks = []
        for chunk_idx in range(df.num_chunks):
            chunk = df.chunk(chunk_idx)

            # 1. 分组
            groups = {}  # field_val → [hits]
            for hit in chunk:
                key = hit.get(self._field)
                groups.setdefault(key, []).append(hit)

            # 2. 每组内排序 + 截取 top group_size
            scored_groups = []
            for key, hits in groups.items():
                hits.sort(key=lambda r: r.get(SCORE_FIELD, 0), reverse=True)
                top_hits = hits[:self._group_size]
                group_score = self._compute_group_score(top_hits)
                scored_groups.append((group_score, key, top_hits))

            # 3. 组间排序 + offset + limit
            scored_groups.sort(key=lambda g: g[0], reverse=True)
            selected = scored_groups[self._offset:self._offset + self._limit]

            # 4. 展平 + 添加 $group_score
            result = []
            for group_score, key, hits in selected:
                for hit in hits:
                    hit["$group_score"] = group_score
                    result.append(hit)

            new_chunks.append(result)
        return DataFrame(new_chunks)

    def _compute_group_score(self, hits):
        scores = [h.get(SCORE_FIELD, 0) for h in hits]
        if not scores:
            return 0.0
        if self._scorer == "max":
            return max(scores)
        elif self._scorer == "sum":
            return sum(scores)
        elif self._scorer == "avg":
            return sum(scores) / len(scores)
        return max(scores)
```

---

## 6. FunctionExpr 实现

### 6.1 BM25Expr — 文本 → 稀疏向量（Ingestion）

```python
class BM25Expr(FunctionExpr):
    """text → analyze → compute_tf → sparse vector dict"""

    name = "bm25"
    supported_stages = frozenset({STAGE_INGESTION})

    def __init__(self, analyzer):
        self._analyzer = analyzer

    def execute(self, inputs: List[list]) -> List[list]:
        from milvus_lite.analyzer.sparse import compute_tf
        texts = inputs[0]
        sparse_vecs = []
        for text in texts:
            if text is None or not isinstance(text, str):
                sparse_vecs.append({})
            else:
                term_ids = self._analyzer.analyze(text)
                sparse_vecs.append(compute_tf(term_ids))
        return [sparse_vecs]
```

### 6.2 EmbeddingExpr — 文本 → 密集向量（Ingestion）

```python
class EmbeddingExpr(FunctionExpr):
    """text → embedding provider → dense vector"""

    name = "text_embedding"
    supported_stages = frozenset({STAGE_INGESTION})

    def __init__(self, provider):
        self._provider = provider

    def execute(self, inputs: List[list]) -> List[list]:
        texts = inputs[0]
        # 批量处理非空文本
        indices = []
        batch = []
        for i, text in enumerate(texts):
            if text is not None and isinstance(text, str) and text:
                indices.append(i)
                batch.append(text)

        vectors = [None] * len(texts)
        if batch:
            embeddings = self._provider.embed_documents(batch)
            for i, emb in zip(indices, embeddings):
                vectors[i] = emb

        # 空值填零向量
        zero_vec = [0.0] * self._provider.dimension
        for i in range(len(vectors)):
            if vectors[i] is None:
                vectors[i] = zero_vec

        return [vectors]
```

### 6.3 DecayExpr — 数值 → 衰减因子（Rerank）

**对应 Milvus**：`expr/decay_expr.go`

```python
class DecayExpr(FunctionExpr):
    """numeric column → decay factor [0, 1]

    三种衰减函数（与 Milvus 一致）：
    - gauss:  exp(-0.5 * ((max(0, |val-origin|-offset)) / scale)^2)
    - exp:    exp(ln(decay) * max(0, |val-origin|-offset) / scale)
    - linear: max(0, (scale - max(0, |val-origin|-offset)) / scale)
    """

    name = "decay"
    supported_stages = frozenset({STAGE_RERANK})

    def __init__(self, function: str, origin: float, scale: float,
                 offset: float = 0.0, decay: float = 0.5):
        self._function = function  # "gauss" | "exp" | "linear"
        self._origin = origin
        self._scale = scale
        self._offset = offset
        self._decay = decay

    def execute(self, inputs: List[list]) -> List[list]:
        import math
        values = inputs[0]
        factors = []
        for val in values:
            if val is None:
                factors.append(0.0)
                continue
            dist = max(0.0, abs(float(val) - self._origin) - self._offset)
            if self._function == "gauss":
                factor = math.exp(-0.5 * (dist / self._scale) ** 2)
            elif self._function == "exp":
                factor = math.exp(
                    math.log(self._decay) * dist / self._scale
                )
            elif self._function == "linear":
                factor = max(0.0, (self._scale - dist) / self._scale)
            else:
                factor = 0.0
            factors.append(factor)
        return [factors]
```

### 6.4 ScoreCombineExpr — 分数合并（Rerank）

**对应 Milvus**：`expr/score_combine_expr.go`

```python
class ScoreCombineExpr(FunctionExpr):
    """($score, factor) → $score * factor

    将多列分数合并为一个最终分数。
    mode="multiply" 是 decay reranker 的默认行为。
    """

    name = "score_combine"
    supported_stages = frozenset({STAGE_RERANK})

    def __init__(self, mode: str = "multiply"):
        self._mode = mode  # "multiply" | "sum" | "max" | "min" | "avg"

    def execute(self, inputs: List[list]) -> List[list]:
        n = len(inputs[0])
        results = []
        for row_idx in range(n):
            vals = [col[row_idx] for col in inputs]
            if None in vals:
                results.append(0.0)
                continue
            if self._mode == "multiply":
                r = 1.0
                for v in vals:
                    r *= v
                results.append(r)
            elif self._mode == "sum":
                results.append(sum(vals))
            elif self._mode == "max":
                results.append(max(vals))
            elif self._mode == "min":
                results.append(min(vals))
            elif self._mode == "avg":
                results.append(sum(vals) / len(vals))
            else:
                results.append(0.0)
        return [results]
```

### 6.5 RoundDecimalExpr — 距离四舍五入（Rerank）

**对应 Milvus**：`expr/round_decimal_expr.go`

```python
class RoundDecimalExpr(FunctionExpr):
    """$score → round($score, decimal)"""

    name = "round_decimal"
    supported_stages = frozenset({STAGE_INGESTION, STAGE_RERANK})

    def __init__(self, decimal: int):
        self._decimal = decimal

    def execute(self, inputs: List[list]) -> List[list]:
        scores = inputs[0]
        return [[round(s, self._decimal) if s is not None else None
                 for s in scores]]
```

### 6.6 RerankModelExpr — 语义重排序（Rerank）

```python
class RerankModelExpr(FunctionExpr):
    """document_text column → relevance_score column

    调用外部 rerank 模型（如 Cohere rerank）重新评分。
    需要在创建时绑定 query texts。
    """

    name = "rerank_model"
    supported_stages = frozenset({STAGE_RERANK})

    def __init__(self, provider, query_texts: List[str]):
        self._provider = provider
        self._query_texts = query_texts

    def execute(self, inputs: List[list]) -> List[list]:
        # inputs[0] = document texts for this chunk
        doc_texts = inputs[0]
        # 简化实现：假设每次 execute 处理一个 chunk
        # query_text 需要从 FuncContext 或外部传入
        # 这里通过 _query_texts[chunk_idx] 获取
        # 实际实现时需要通过 FuncContext 传递 chunk_idx
        rerank_results = self._provider.rerank(
            self._query_texts[0], doc_texts, top_n=len(doc_texts)
        )
        scores = [0.0] * len(doc_texts)
        for r in rerank_results:
            scores[r.index] = r.relevance_score
        return [scores]
```

> **注意**：RerankModelExpr 需要知道当前 chunk 对应哪个 query text。两种方式：(1) FuncContext 携带 chunk_idx → query_texts 映射；(2) 创建 chain 时为每个 nq 创建独立 expr。实现时选择方式 (1)，给 FuncContext 增加 chunk_idx 字段。

---

## 7. Chain Builder — 从 schema.functions 构建 chain

### 7.1 Ingestion Chain Builder

```python
# function/builder.py

def build_ingestion_chain(schema, field_by_name) -> Optional[FuncChain]:
    """从 schema.functions 构建 ingestion chain。

    遍历 schema 中所有 function，将支持 ingestion stage
    的 function 按声明顺序添加到 chain。

    Returns:
        FuncChain 或 None（无 ingestion function 时）
    """
    if not schema.functions:
        return None

    chain = FuncChain("ingestion", STAGE_INGESTION)
    has_steps = False

    for func in schema.functions:
        if func.function_type == FunctionType.BM25:
            in_name = func.input_field_names[0]
            out_name = func.output_field_names[0]
            in_field = field_by_name[in_name]
            analyzer = create_analyzer(in_field.analyzer_params)
            chain.map(BM25Expr(analyzer), [in_name], [out_name])
            has_steps = True

        elif func.function_type == FunctionType.TEXT_EMBEDDING:
            in_name = func.input_field_names[0]
            out_name = func.output_field_names[0]
            provider = create_embedding_provider(func.params)
            chain.map(EmbeddingExpr(provider), [in_name], [out_name])
            has_steps = True

        # 未来扩展：其他 ingestion-stage function

    return chain if has_steps else None
```

### 7.2 Rerank Chain Builder

**直接参照 Milvus `rerank_builder.go` 的 4 种 chain 模式**：

```python
def build_rerank_chain(
    schema,
    search_params: dict,      # {limit, offset, round_decimal, group_by_field, group_size}
    search_metrics: List[str], # 各路搜索的 metric type
) -> Optional[FuncChain]:
    """从 schema.functions 中的 RERANK/DECAY function 构建 rerank chain。

    4 种 chain 模式（与 Milvus rerank_builder.go 对齐）：

    RRF:      Merge(RRF) → Sort → Limit → [RoundDecimal] → Select
    Weighted: Merge(Weighted) → Sort → Limit → [RoundDecimal] → Select
    Decay:    Merge(strategy) → Map(Decay) → Map(ScoreCombine) → Sort → Limit → [RoundDecimal] → Select
    Model:    Merge(Max) → Map(RerankModel) → Sort → Limit → [RoundDecimal] → Select
    """
    rerank_func = _find_rerank_function(schema)
    if rerank_func is None:
        return None

    chain = FuncChain("rerank", STAGE_RERANK)
    reranker_type = _get_reranker_type(rerank_func)

    # ── 头部：Merge ──
    if reranker_type == "rrf":
        rrf_k = rerank_func.params.get("k", 60.0)
        chain.merge("rrf", rrf_k=rrf_k)

    elif reranker_type == "weighted":
        weights = rerank_func.params.get("weights", [])
        normalize = rerank_func.params.get("norm_score", False)
        chain.merge("weighted", weights=weights,
                    metric_types=search_metrics, normalize=normalize)

    elif reranker_type == "decay":
        score_mode = rerank_func.params.get("score_mode", "max")
        chain.merge(score_mode, metric_types=search_metrics)
        # Map(DecayExpr)
        in_name = rerank_func.input_field_names[0]
        decay_expr = DecayExpr(
            function=rerank_func.params["function"],
            origin=rerank_func.params["origin"],
            scale=rerank_func.params["scale"],
            offset=rerank_func.params.get("offset", 0.0),
            decay=rerank_func.params.get("decay", 0.5),
        )
        chain.map(decay_expr, [in_name], ["_decay_score"])
        # Map(ScoreCombineExpr)
        chain.map(ScoreCombineExpr("multiply"),
                  [SCORE_FIELD, "_decay_score"], [SCORE_FIELD])

    elif reranker_type == "model":
        chain.merge("max")
        in_name = rerank_func.input_field_names[0]
        provider = create_rerank_provider(rerank_func.params)
        model_expr = RerankModelExpr(provider, query_texts=[])  # query_texts 在 execute 时注入
        chain.map(model_expr, [in_name], [SCORE_FIELD])

    # ── 尾部：Sort / GroupBy → [RoundDecimal] → Select ──
    group_by_field = search_params.get("group_by_field")
    limit = search_params.get("limit", 10)
    offset = search_params.get("offset", 0)
    round_decimal = search_params.get("round_decimal", -1)

    if group_by_field:
        group_size = search_params.get("group_size", 1)
        chain.group_by(group_by_field, group_size, limit, offset)
    else:
        chain.sort(SCORE_FIELD, desc=True)
        chain.limit(limit, offset)

    if round_decimal >= 0:
        chain.map(RoundDecimalExpr(round_decimal),
                  [SCORE_FIELD], [SCORE_FIELD])

    select_cols = [ID_FIELD, SCORE_FIELD]
    if group_by_field:
        select_cols.extend([group_by_field, "$group_score"])
    chain.select(*select_cols)

    return chain
```

---

## 8. Collection 重构

### 8.1 初始化：4 组列表 → 2 条 chain

**Before**:
```python
# Collection.__init__ 中约 40 行 if/elif 分支
self._bm25_functions: List[Tuple[str, str, Any]] = []
self._embedding_functions: List[Tuple[str, str, Any]] = []
self._rerank_functions: List[Tuple[str, Any]] = []
self._decay_functions: List[Tuple[str, Any]] = []
for func in schema.functions:
    if func.function_type == FunctionType.BM25: ...
    elif func.function_type == FunctionType.TEXT_EMBEDDING: ...
    elif func.function_type == FunctionType.RERANK:
        if reranker_type == "decay": ...
        else: ...
```

**After**:
```python
# Collection.__init__
from milvus_lite.function.builder import build_ingestion_chain

field_by_name = {f.name: f for f in schema.fields}
self._ingestion_chain = build_ingestion_chain(schema, field_by_name)
# rerank chain 在 search 时按需构建（因为依赖 search_params）
```

### 8.2 Insert：4 行 apply → 1 行 execute

**Before**:
```python
# insert() 中
if self._bm25_functions:
    self._apply_bm25_functions(records)
if self._embedding_functions:
    self._apply_embedding_functions(records)
```

**After**:
```python
# insert() 中
if self._ingestion_chain:
    df = DataFrame.from_records(records)
    self._ingestion_chain.execute(df)
    # records 已就地修改，无需额外操作
```

### 8.3 Search 后处理：分散的 rerank/decay → rerank chain

**Before**:
```python
# search() 中约 30 行
if self._query_texts is not None:
    raw_results = self._apply_rerank(raw_results, self._query_texts, ...)
    scores_replaced = True
if self._decay_functions:
    raw_results = self._apply_decay(raw_results, metric_type, scores_replaced)
    scores_replaced = True
# + group_by 后处理
# + offset 处理
```

**After**:
```python
# search() 中 — rerank chain（Hybrid Search 场景）
if rerank_chain:
    # 多路搜索结果 → DataFrame
    dfs = [DataFrame.from_search_results(r) for r in per_path_results]
    merged = rerank_chain.execute(*dfs)
    raw_results = merged.to_search_results()
    # Sort + Limit + GroupBy + RoundDecimal + Select 全在 chain 内完成
```

### 8.4 可删除的方法

chain 重构后，以下 Collection 方法可以删除：

| 方法 | 替代 |
|---|---|
| `_apply_bm25_functions()` | `BM25Expr` + `MapOp` |
| `_apply_embedding_functions()` | `EmbeddingExpr` + `MapOp` |
| `_apply_rerank()` | `RerankModelExpr` + `MapOp` |
| `_apply_decay()` | `DecayExpr` + `ScoreCombineExpr` + `MapOp` |

---

## 9. 数据流图

### 9.1 Ingestion Chain 数据流

```
用户 records: [{"text": "hello world", "id": 1}, ...]
    │
    ▼
DataFrame.from_records(records)
    │ chunks = [[{"text": "hello world", "id": 1}, ...]]
    │
    ├─ MapOp(BM25Expr, ["text"] → ["sparse_vec"])
    │   │  extract: texts = ["hello world", ...]
    │   │  compute: BM25Expr.execute([texts]) → [sparse_vecs]
    │   │  write:   records[i]["sparse_vec"] = sparse_vecs[i]
    │   ▼
    ├─ MapOp(EmbeddingExpr, ["text"] → ["dense_vec"])
    │   │  extract: texts = ["hello world", ...]
    │   │  compute: EmbeddingExpr.execute([texts]) → [vectors]
    │   │  write:   records[i]["dense_vec"] = vectors[i]
    │   ▼
    ├─ [未来: MapOp(DimReduceExpr, ["dense_vec"] → ["reduced_vec"])]
    │   │  串联：消费上一步的 dense_vec 输出
    │   ▼
    │
df.to_records() → 修改后的 records，继续走 validate → WAL → MemTable
```

### 9.2 Rerank Chain 数据流（Decay 示例）

```
多路搜索结果:
  path_0: [[{$id: 1, $score: 0.9, ts: 100}, {$id: 2, $score: 0.8, ts: 200}], ...]  (dense)
  path_1: [[{$id: 2, $score: -3.5, ts: 200}, {$id: 3, $score: -4.1, ts: 50}], ...]  (BM25)
    │
    ▼
MergeOp(strategy="max", metric_types=["COSINE", "BM25"])
    │  per-query: pk 去重 + 按 max 策略取最高 $score
    │  result: [[{$id: 1, $score: 0.9, ts: 100}, {$id: 2, $score: 0.8, ts: 200}, {$id: 3, ...}], ...]
    ▼
MapOp(DecayExpr(gauss, origin=now, scale=86400), ["ts"] → ["_decay_score"])
    │  per-row: _decay_score = gauss(|ts - now|)
    │  result: 每条 hit 新增 _decay_score 字段
    ▼
MapOp(ScoreCombineExpr("multiply"), ["$score", "_decay_score"] → ["$score"])
    │  per-row: $score = $score * _decay_score
    │  result: $score 已更新
    ▼
SortOp("$score", desc=True)
    │  per-chunk: 按 $score 降序排列
    ▼
LimitOp(limit=10, offset=0)
    │  per-chunk: 取前 10 条
    ▼
MapOp(RoundDecimalExpr(4), ["$score"] → ["$score"])     [可选]
    │  per-row: $score = round($score, 4)
    ▼
SelectOp("$id", "$score")
    │  per-row: 只保留 $id 和 $score
    ▼
DataFrame.to_search_results() → 最终返回值
```

---

## 10. 模块结构

### 新增 `function/` 包

```
milvus_lite/function/
├── __init__.py           # 公开 API: FuncChain, build_ingestion_chain, build_rerank_chain
├── types.py              # FunctionExpr ABC, FuncContext, Stage 常量, 列名常量
├── dataframe.py          # DataFrame
├── chain.py              # FuncChain
├── operator.py           # Operator ABC
├── ops/
│   ├── __init__.py
│   ├── map_op.py         # MapOp
│   ├── merge_op.py       # MergeOp
│   ├── sort_op.py        # SortOp
│   ├── limit_op.py       # LimitOp
│   ├── select_op.py      # SelectOp
│   └── group_by_op.py    # GroupByOp
├── expr/
│   ├── __init__.py
│   ├── bm25_expr.py      # BM25Expr
│   ├── embedding_expr.py # EmbeddingExpr
│   ├── decay_expr.py     # DecayExpr
│   ├── score_combine.py  # ScoreCombineExpr
│   ├── round_decimal.py  # RoundDecimalExpr
│   └── rerank_model.py   # RerankModelExpr
└── builder.py            # build_ingestion_chain, build_rerank_chain
```

### 修改的现有模块

| 模块 | 变更 |
|---|---|
| `engine/collection.py` | 删除 4 组 function 列表 + 4 个 apply 方法；改用 `_ingestion_chain` |
| `engine/collection.py` | search 后处理逻辑重构为 rerank chain 调用 |

---

## 11. 子阶段拆分

| 子阶段 | 内容 | 交付物 |
|---|---|---|
| **FC-1** | 基础框架 | `types.py`（FunctionExpr, FuncContext, constants）、`dataframe.py`（DataFrame）、`operator.py`（Operator ABC）、`chain.py`（FuncChain）、单测 |
| **FC-2** | MapOp + Ingestion Exprs | `ops/map_op.py`、`expr/bm25_expr.py`、`expr/embedding_expr.py`、`builder.py`（build_ingestion_chain）、单测 |
| **FC-3** | Collection insert 重构 | 删除 `_bm25_functions` / `_embedding_functions` / `_apply_*` 方法，改用 ingestion chain；**回归测试全绿** |
| **FC-4** | Rerank Operators | `ops/merge_op.py`、`ops/sort_op.py`、`ops/limit_op.py`、`ops/select_op.py`、`ops/group_by_op.py`、单测 |
| **FC-5** | Rerank Exprs | `expr/decay_expr.py`、`expr/score_combine.py`、`expr/round_decimal.py`、`expr/rerank_model.py`、单测 |
| **FC-6** | Rerank Chain Builder | `builder.py`（build_rerank_chain）、4 种 chain 模式的集成测试 |
| **FC-7** | Collection search 重构 | 删除 `_apply_rerank` / `_apply_decay` / 内联 group_by / offset 逻辑，改用 rerank chain；**回归测试全绿** |
| **FC-8** | gRPC Hybrid Search 对接 | `servicer.py` HybridSearch RPC 使用 rerank chain 替代内联 reranker；pymilvus 端到端测试 |

### 验证策略

| 子阶段 | 测试要点 |
|---|---|
| FC-1 | DataFrame 创建/导出/列读写；FuncChain 空 chain / 单 op / 多 op 执行 |
| FC-2 | BM25Expr 分词+TF 正确；EmbeddingExpr 批量 + 空值处理；MapOp 列映射正确 |
| FC-3 | **所有已有 insert 相关测试不回归**（BM25 insert、embedding insert、mixed insert） |
| FC-4 | MergeOp 5 种策略的去重+合并分数正确；SortOp 正序/逆序/None 处理；LimitOp offset 边界；SelectOp 列过滤；GroupByOp 分组+scorer |
| FC-5 | DecayExpr 三种衰减函数与手算对比；ScoreCombineExpr multiply/sum/max；RoundDecimalExpr 精度 |
| FC-6 | 4 种 rerank chain（RRF/Weighted/Decay/Model）端到端输入输出验证 |
| FC-7 | **所有已有 search/rerank 测试不回归**；chain 路径与原始路径结果一致 |
| FC-8 | pymilvus hybrid_search 端到端测试 |

### 完成标志

- `milvus_lite/function/` 包完整实现
- Collection 中无 per-type function 分支代码
- 4 种 rerank chain（RRF / Weighted / Decay / Model）可由 builder 自动构建
- 新增 function 类型只需：新增 FunctionExpr 子类 + 注册到 builder
- **1529+ 测试全绿，0 回归**
