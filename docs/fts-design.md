# 深入设计：全文检索子系统（Phase 11）

## 1. 概述

MilvusLite Phase 11 引入全文检索（Full Text Search, FTS），让用户能够通过自然语言文本进行语义相关性搜索。**核心实现是 BM25 评分 + 稀疏倒排索引**，与 Milvus 的 Full Text Search API 完全兼容。

**为什么现在做**：
- 全文检索是 Milvus 2.5 的核心新能力，"本地版 Milvus"必须跟进
- BM25 搜索与密集向量检索互补 — 密集向量擅长语义理解，BM25 擅长精确关键词匹配
- Phase 9/10 建立了索引状态机 + gRPC 适配层，Phase 11 在此基础上扩展，不需要新的架构范式

**核心定位**：与 Milvus 的 FTS API 兼容，pymilvus 用户可以直接使用 `Function(type=BM25)` + `text_match` + 稀疏向量搜索，无需代码修改。

---

## 2. 架构决策

### 2.1 全文检索管线：Function 驱动（决定）

**决定：采用 Milvus 的 Function 机制 — 用户在 schema 中声明 BM25 Function，引擎在 insert 时自动分词 + 生成稀疏向量。**

数据流：

```
Insert:
  用户 record {"text": "machine learning"}
  → Analyzer 分词 → ["machine", "learning"]
  → term hash → {2847: 1, 9134: 1}  (term_hash → TF)
  → 存储为 SPARSE_FLOAT_VECTOR 列

Search:
  query "learning algorithm"
  → 分词 → ["learning", "algorithm"]
  → 查倒排索引 → 对每个 term 的 posting list 累加 BM25 score
  → top-k
```

**为什么不用 dense embedding**：
- BM25 是确定性算法，不依赖外部模型，零推理成本
- 精确关键词匹配场景下 BM25 >> 密集向量
- 与 Milvus API 一致（FunctionType.BM25）

### 2.2 分词策略：内置轻量级 Analyzer（决定）

**决定：内置 StandardAnalyzer（正则分词）作为默认实现，jieba 作为可选中文依赖。**

候选方案对比：

| 维度 | 内置轻量级（选定） | 依赖 NLTK/spaCy | 依赖外部服务 |
|---|---|---|---|
| 部署简单度 | 零额外依赖 | 需装 NLTK 数据包 / spaCy 模型 | 需启动分词服务 |
| 中文支持 | jieba（optional extra） | spaCy 中文模型 | 灵活但复杂 |
| 与 Milvus 对齐 | Milvus 内置 standard/jieba/icu | 不对齐 | 不对齐 |
| 性能 | 足够（正则非瓶颈） | 较慢（模型加载） | 网络开销 |

**实现**：
- `StandardAnalyzer`：`re.findall(r'\w+', text.lower())` — 按非单词字符分割 + 转小写
- `JiebaAnalyzer`：`jieba.cut(text)` — 可选 exact/search 模式
- 支持 stop words filter

### 2.3 稀疏向量存储：packed binary 格式（决定）

**决定：SPARSE_FLOAT_VECTOR 在 Arrow/Parquet 中存储为 `pa.binary()` 列，每行是 packed bytes（交错 uint32 index + float32 value），与 Milvus SparseFloatArray.contents 格式一致。**

候选对比：

| 格式 | 优点 | 缺点 |
|---|---|---|
| **packed binary（选定）** | 紧凑；与 proto 格式一致，零拷贝转换 | 人类不可读 |
| JSON string | 可读 | 3x 体积，序列化/反序列化开销 |
| Arrow Map<int,float> | 原生类型 | Parquet map 支持不稳定 |

**编码格式**：
```
每行 = N 组 (uint32_le index, float32_le value)，按 index 升序排列
空稀疏向量 = 空 bytes b""
```

### 2.4 倒排索引：Per-segment（决定）

**决定：每个 segment 拥有独立的倒排索引（SparseInvertedIndex），与 Phase 9 的 VectorIndex 1:1 绑定策略一致。**

| 维度 | Per-segment（选定） | 全局 |
|---|---|---|
| 与 LSM 不可变架构匹配度 | 完美 — segment 不可变 → 索引不可变 | 差 — 需维护可变全局索引 |
| IDF 准确性 | 段内 IDF（可能偏差，Elasticsearch 也这样做） | 精确 |
| 增量更新成本 | 零 — 新 segment 建新索引 | 每次 flush 要更新全局 |
| 与 compaction 协同 | 自然 — 合并 segment 重建索引 | 复杂 |
| 实现复杂度 | 低 | 高 |

**IDF 准确性补偿**：Elasticsearch 在生产中也使用 per-segment IDF，对于大部分场景足够准确。小 segment 的 IDF 偏差可通过 compaction 合并来缓解。

### 2.5 BM25 评分策略：查询时计算（决定）

**决定：insert 时只存储 term frequency (TF)，BM25 完整评分在查询时基于 segment 内统计量实时计算。**

**理由**：
- IDF 随文档增删变化，预计算 BM25 score 会因 IDF 漂移而失准
- 查询时计算的开销集中在倒排索引查找（O(postings_per_term)），不是瓶颈
- 每个 segment 在 build 时预计算好 doc_count、avgdl、df_map，查询时只做一次除法和乘法

### 2.6 term ID 映射：Hash（决定）

**决定：使用确定性 hash 函数将 term string 映射为 uint32 ID，不维护全局词表文件。**

- 优点：无状态，segment 间无需共享词表，简化存储
- 冲突风险：32 位 hash 空间 ~43 亿，实际 vocabulary < 100 万时冲突概率极低（< 0.01%）
- 选用 `mmh3`（MurmurHash3）或内置 `hash` + 取模

### 2.7 多向量字段支持：anns_field 参数（决定）

**决定：search API 新增 `anns_field` 参数指定搜索的向量字段，打破"只有一个向量字段"的 MVP 限制。**

- 当 schema 含有 FLOAT_VECTOR + SPARSE_FLOAT_VECTOR 时，用户通过 `anns_field` 选择搜索目标
- 默认值：第一个 FLOAT_VECTOR 字段（保持向后兼容）
- gRPC 层从 SearchRequest 中提取 anns_field 并传递给 engine

---

## 3. 模块结构

### 3.1 新增模块

```
milvus_lite/
├── analyzer/                      # Phase 11.2: 分词子系统
│   ├── __init__.py
│   ├── protocol.py                # Analyzer ABC
│   ├── standard.py                # StandardAnalyzer (正则)
│   ├── jieba_analyzer.py          # JiebaAnalyzer (可选)
│   ├── factory.py                 # create_analyzer(params) → Analyzer
│   └── hash.py                    # term_to_id(term) → uint32
│
├── index/
│   └── sparse_inverted.py         # Phase 11.5: SparseInvertedIndex
│
└── adapter/grpc/translators/
    └── sparse.py                  # Phase 11.7: 稀疏向量编解码
```

### 3.2 修改模块

| 模块 | 改动 |
|---|---|
| `schema/types.py` | ✅ (11.1 已完成) DataType.SPARSE_FLOAT_VECTOR, Function, FunctionType, FieldSchema 新属性 |
| `schema/validation.py` | ✅ (11.1 已完成) 向量字段约束放宽, BM25 函数校验, 稀疏向量验证 |
| `schema/arrow_builder.py` | ✅ (11.1 已完成) SPARSE_FLOAT_VECTOR → pa.binary() |
| `schema/persistence.py` | ✅ (11.1 已完成) schema.json 支持 functions 和新属性 |
| `engine/collection.py` | insert 自动生成 function output; search 支持 anns_field; 稀疏向量序列化 |
| `storage/segment.py` | attach SparseInvertedIndex |
| `search/filter/` | text_match / phrase_match 函数 |
| `adapter/grpc/servicer.py` | FunctionSchema 处理; anns_field 传递 |
| `adapter/grpc/translators/` | schema/records/search 层扩展 |

---

## 4. BM25 算法详解

### 4.1 评分公式

```
score(D, Q) = Σ_{qi ∈ Q} IDF(qi) · f(qi, D) · (k1 + 1) / (f(qi, D) + k1 · (1 - b + b · |D| / avgdl))
```

其中：
- `f(qi, D)` = term qi 在文档 D 中的词频 (TF)
- `|D|` = 文档 D 的总 token 数
- `avgdl` = segment 内所有文档的平均 token 数
- `k1` = 饱和参数（默认 1.5，控制 TF 增长速度）
- `b` = 长度归一化参数（默认 0.75，控制长文档惩罚）
- `IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)`
  - `N` = segment 内总文档数
  - `df(qi)` = segment 内包含 term qi 的文档数

### 4.2 存储时数据

| 存储项 | 位置 | 格式 |
|---|---|---|
| term hash → TF（per-row） | 稀疏向量列（Parquet binary） | packed uint32+float32 |
| doc_length（per-row） | 倒排索引元数据 | 从稀疏向量的 value 之和推导 |
| doc_count, avgdl, df_map | SparseInvertedIndex 内存状态 | build 时计算, save 时持久化 |
| posting_lists | SparseInvertedIndex | term_hash → [(local_id, tf), ...] |

### 4.3 搜索时流程

```python
def search_bm25(query_text, index, top_k):
    terms = analyzer.analyze(query_text)  # → List[int] (term hashes)
    scores = {}  # doc_id → accumulated score

    for term_hash in set(terms):
        posting = index.posting_lists.get(term_hash, [])
        df = len(posting)
        idf = math.log((index.doc_count - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in posting:
            dl = index.doc_lengths[doc_id]
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / index.avgdl))
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

    return top_k_by_score(scores, top_k)
```

---

## 5. Analyzer 子系统

### 5.1 Analyzer ABC

```python
class Analyzer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """将文本分割为 token 列表。"""

    def analyze(self, text: str) -> List[int]:
        """tokenize + hash → term_id 列表。"""
        return [term_to_id(t) for t in self.tokenize(text)]
```

### 5.2 StandardAnalyzer

```python
class StandardAnalyzer(Analyzer):
    def __init__(self, stop_words: Optional[Set[str]] = None):
        self._stop_words = stop_words or set()
        self._pattern = re.compile(r'\w+')

    def tokenize(self, text: str) -> List[str]:
        tokens = self._pattern.findall(text.lower())
        if self._stop_words:
            tokens = [t for t in tokens if t not in self._stop_words]
        return tokens
```

### 5.3 JiebaAnalyzer

```python
class JiebaAnalyzer(Analyzer):
    def __init__(self, mode: str = "search", stop_words=None, user_dict=None):
        import jieba  # optional dependency
        self._mode = mode
        # ...

    def tokenize(self, text: str) -> List[str]:
        if self._mode == "search":
            return list(jieba.cut_for_search(text))
        return list(jieba.cut(text))
```

### 5.4 Factory

```python
def create_analyzer(params: Optional[dict]) -> Analyzer:
    if params is None:
        return StandardAnalyzer()
    tokenizer = params.get("tokenizer", "standard")
    if tokenizer == "standard":
        return StandardAnalyzer(stop_words=_parse_stop_words(params))
    if tokenizer == "jieba" or (isinstance(tokenizer, dict) and tokenizer.get("type") == "jieba"):
        return JiebaAnalyzer(...)
    raise SchemaValidationError(f"unknown tokenizer: {tokenizer}")
```

---

## 6. 稀疏向量编解码

### 6.1 Python dict ↔ packed bytes

```python
import struct

def sparse_to_bytes(sv: dict[int, float]) -> bytes:
    """dict[int, float] → packed bytes (sorted by index)."""
    if not sv:
        return b""
    pairs = sorted(sv.items())
    return struct.pack(f"<{len(pairs) * 2}I",
                       *[x for idx, val in pairs
                         for x in (idx, struct.unpack('I', struct.pack('f', val))[0])])

def bytes_to_sparse(b: bytes) -> dict[int, float]:
    """packed bytes → dict[int, float]."""
    if not b:
        return {}
    n = len(b) // 8  # 每对 8 bytes (uint32 + float32)
    result = {}
    for i in range(n):
        idx = struct.unpack_from('<I', b, i * 8)[0]
        val = struct.unpack_from('<f', b, i * 8 + 4)[0]
        result[idx] = val
    return result
```

### 6.2 与 Milvus SparseFloatArray 的映射

| Milvus Proto | MilvusLite |
|---|---|
| `SparseFloatArray.contents[i]` (bytes) | `sparse_to_bytes(row_dict)` |
| `SparseFloatArray.dim` | `max(all_indices) + 1` across all rows |

---

## 7. SparseInvertedIndex

### 7.1 类签名

```python
class SparseInvertedIndex(VectorIndex):
    """Per-segment 倒排索引，实现 VectorIndex 协议。"""

    def build(self, vectors: list[dict], valid_mask: np.ndarray) -> None:
        """从稀疏向量列表构建倒排索引。

        vectors: list of dict[int, float] (term_hash → tf)
        valid_mask: boolean array, True = 该行有效

        构建后的状态:
        - posting_lists: dict[int, list[tuple[int, float]]]  # term → [(local_id, tf)]
        - doc_count: int
        - doc_lengths: np.ndarray  # per-doc token count
        - avgdl: float
        - df_map: dict[int, int]  # term → document frequency
        """

    def search(self, query_vectors, top_k, valid_mask=None):
        """query_vectors: list of dict[int, float] (query term hashes → weight)

        Returns: (ids, distances) — distances 是 BM25 score 的负值
        （与 VectorIndex 协议一致：smaller = more similar）
        """

    def save(self, f) -> None: ...
    def load(self, f) -> None: ...
```

### 7.2 BM25 距离约定

VectorIndex 协议要求"smaller distance = more similar"。BM25 score 越高越相关，所以：
- **distance = -bm25_score**（取负）
- 搜索结果按 distance 升序 = 按 BM25 score 降序
- 与 Milvus 行为一致（pymilvus 从 SearchResults 拿到的 score 取负回来）

### 7.3 持久化

存储为 `.sidx` 文件（JSON 或 pickle），包含：
- posting_lists（序列化为 {term_hash: [[local_id, tf], ...]}）
- doc_count, avgdl, df_map
- doc_lengths
- bm25_k1, bm25_b 参数

---

## 8. text_match 过滤器

### 8.1 语法

```
text_match(field_name, 'token1 token2 token3')
```

- 多个 token 之间是 **OR** 逻辑：匹配包含任一 token 的文档
- token 会通过该字段的 Analyzer 进行分词处理
- 需要字段设置 `enable_match=True` + `enable_analyzer=True`

### 8.2 实现方式

在 filter 子系统中新增 `TextMatchNode` AST 节点：

```python
class TextMatchNode(ASTNode):
    field_name: str
    query_text: str  # 原始查询文本
```

评估时：
1. 用字段对应的 Analyzer 对 query_text 分词 → query_tokens
2. 对每行的文本字段值分词 → doc_tokens
3. 返回 `bool(set(query_tokens) & set(doc_tokens))`

---

## 9. gRPC 适配层扩展

### 9.1 Schema 翻译

| Milvus Proto | MilvusLite |
|---|---|
| `FieldSchema.data_type = 104` (SparseFloatVector) | `DataType.SPARSE_FLOAT_VECTOR` |
| `FieldSchema.type_params["enable_analyzer"]` | `FieldSchema.enable_analyzer` |
| `FieldSchema.type_params["analyzer_params"]` | `FieldSchema.analyzer_params` (JSON decoded) |
| `FieldSchema.type_params["enable_match"]` | `FieldSchema.enable_match` |
| `CollectionSchema.functions` | `CollectionSchema.functions` |
| `FunctionSchema.type = BM25 (1)` | `FunctionType.BM25` |
| `FieldSchema.is_function_output = True` | `FieldSchema.is_function_output` |

### 9.2 FieldData 编解码

**SparseFloatArray 解码**（insert path）：
```python
sfa = fd.vectors.sparse_float_vector
for content_bytes in sfa.contents:
    sv = bytes_to_sparse(content_bytes)  # → dict[int, float]
    column.append(sv)
```

**SparseFloatArray 编码**（response path）：
```python
sfa = schema_pb2.SparseFloatArray()
max_dim = 0
for sv in column:
    sfa.contents.append(sparse_to_bytes(sv))
    if sv:
        max_dim = max(max_dim, max(sv.keys()) + 1)
sfa.dim = max_dim
fd.vectors.sparse_float_vector.CopyFrom(sfa)
```

### 9.3 Search 请求处理

BM25 搜索时，pymilvus 发送文本查询的方式：
- `PlaceholderGroup` 中 `PlaceholderValue.type = 104`（SPARSE_FLOAT_VECTOR）
- 但实际数据是文本 — pymilvus 内部先分词生成稀疏向量再编码
- MilvusLite 需要支持两种搜索入口：
  1. 客户端已分词的稀疏向量搜索
  2. 文本直接搜索（engine 内部分词）

### 9.4 Index 参数

| Milvus index_type | MilvusLite 处理 |
|---|---|
| `SPARSE_INVERTED_INDEX` | 映射到 SparseInvertedIndex |
| `metric_type = BM25` | 使用 BM25 评分 |
| `bm25_k1`, `bm25_b` | 传递给 SparseInvertedIndex |

---

## 10. 分阶段实现计划

### Phase 11.1 — Schema 扩展 ✅

已完成。新增 SPARSE_FLOAT_VECTOR DataType, Function/FunctionType, FieldSchema 新属性, schema 校验, Arrow 类型映射, persistence。30 个新测试。

### Phase 11.2 — Analyzer 分词子系统

新建 `milvus_lite/analyzer/` 包：
- `protocol.py` — Analyzer ABC
- `standard.py` — StandardAnalyzer
- `jieba_analyzer.py` — JiebaAnalyzer（optional）
- `factory.py` — create_analyzer
- `hash.py` — term_to_id

验证：单测覆盖分词结果、hash 确定性、stop words。

### Phase 11.3 — 稀疏向量存储

- `sparse_to_bytes` / `bytes_to_sparse` 编解码函数
- engine/collection.py 的 `_build_wal_data_batch` 支持稀疏向量列序列化
- validate_record 在 insert 时跳过 function output 字段

验证：稀疏向量 round-trip（dict → bytes → dict），WAL 写入/读取。

### Phase 11.4 — BM25 Function 引擎

- engine/collection.py insert 路径：检测 BM25 function → 分词 → 生成 TF 稀疏向量 → 注入 record
- 文本字段 + 稀疏向量字段同时写入 WAL / Parquet

验证：insert + flush → 读 Parquet 验证稀疏向量列存在且正确。

### Phase 11.5 — 稀疏倒排索引 + BM25 搜索

- `index/sparse_inverted.py` — SparseInvertedIndex 实现
- 集成到 segment attach_index / load 状态机
- search 路径支持 anns_field 选择 + BM25 距离

验证：
- 单独的 BM25 搜索准确性测试
- 插入 + 搜索端到端测试
- BM25 score 正确性验证（手动计算对比）

### Phase 11.6 — text_match 过滤器

- 扩展 filter 子系统支持 `text_match(field, 'tokens')` 函数
- 新增 TextMatchNode AST 节点
- 三个后端实现

验证：text_match 单独使用 + 与向量搜索组合使用。

### Phase 11.7 — gRPC 适配层扩展

- FunctionSchema ↔ Function 翻译
- SparseFloatArray FieldData 编解码
- 搜索请求处理（稀疏向量 / 文本查询）
- CreateIndex 支持 SPARSE_INVERTED_INDEX / BM25

验证：pymilvus 端到端全文检索流程。

### Phase 11.8 — 集成测试

从 Milvus 测试套件提取关键场景：
1. 基本 BM25 搜索（英文）
2. text_match 过滤
3. BM25 + FLOAT_VECTOR 混合 schema
4. Flush/Compaction 后索引重建
5. Load/Release 状态机与 BM25 索引

---

## 11. 依赖变更

```toml
[project.optional-dependencies]
chinese = ["jieba>=0.42"]
```

jieba 作为可选依赖，不影响基础安装。StandardAnalyzer 零额外依赖。

---

## 12. 不在 Phase 11 范围

- Multi-Analyzer（多语言动态选择）→ Future
- phrase_match（短语匹配 + slop 控制）→ Future
- LexicalHighlighter（搜索结果高亮）→ Future
- TextEmbedding Function（调用外部 embedding 模型）→ Future
- ICU tokenizer → Future
- BM25 全局 IDF 统计（跨 segment 合并 IDF）→ Future
- **Hybrid Search RPC（多路 ANN 重排序）→ Phase 12**（见 roadmap.md）

---

## 13. 风险与缓冲

| 风险 | 缓解 |
|---|---|
| Hash 冲突导致 BM25 评分不准 | 32 位空间足够；未来可升级 64 位或加冲突检测 |
| Per-segment IDF 偏差 | 与 Elasticsearch 相同策略，compaction 合并段后缓解 |
| jieba 安装问题 | optional extra，不影响英文场景 |
| 稀疏向量 Parquet 体积 | binary 列已是最紧凑格式；compaction 时自然合并 |
| search API 新增 anns_field 破坏向后兼容 | 默认值 = 第一个 FLOAT_VECTOR 字段，不改行为 |
