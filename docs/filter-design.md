# 深入设计：标量过滤表达式系统（Scalar Filter）

## 1. 概述

MilvusLite Phase 8 引入 Milvus-style 标量过滤表达式系统，让 `Collection.search` /
`get` / `query` 接受字符串形式的谓词表达式（如 `"age > 18 and category == 'tech'"`），
打通"向量召回 + 标量过滤"的混合查询。

**为什么自己写**：pymilvus 只是把表达式字符串透传给 Milvus 服务端，所有 lex/parse/eval
都在服务端完成。MilvusLite 是嵌入式的，没有"服务端"，必须自己实现完整的 lexer + parser
+ type checker + evaluator。

**为什么"Milvus-inspired"而非 binary 兼容**：
- Milvus grammar 跨版本会变（2.3 vs 2.4 表达式有差异）
- 完整 grammar 含 JSON 路径、array 操作、UDF 等冷门特性
- "Milvus-like" + 文档化我们支持的子集，足以让用户从 pymilvus 迁移
- F3+ 才考虑追严格兼容（届时可能切换到 ANTLR-generated parser）

---

## 2. 三阶段编译流水线

```
┌──────────────────────────────────────────────────────────────┐
│  source string                                               │
│      "age > 18 and category in ['tech', 'news']"             │
│                              │                               │
│                              │  parse_expr(s)                │
│                              ▼                               │
│  Expr (raw AST)                                              │
│      And(operands=(CmpOp("==", ...), InOp(...)))             │
│                              │                               │
│                              │  compile_expr(expr, schema)   │
│                              ▼                               │
│  CompiledExpr  ─── 字段绑定 + 类型检查 + backend 选择          │
│                              │                               │
│                              │  evaluate(compiled, table)    │
│                              ▼                               │
│  pa.BooleanArray (length == table.num_rows)                  │
└──────────────────────────────────────────────────────────────┘
```

**为什么三段**：
- **parse 与 schema 无关**：相同表达式可缓存（F2c 优化）
- **compile 与具体数据无关**：一次绑定、多次执行
- **evaluate 是热路径**：只走 backend dispatch，零解析开销

---

## 3. Grammar 子集（Tier 1，对齐 Milvus Plan.g4）

### 3.1 操作符 + 优先级

| Prec | Operator | Associativity | Notes |
|---|---|---|---|
| 1 | `or`, `OR`, `\|\|` | left | |
| 2 | `and`, `AND`, `&&` | left | |
| 3 | `not`, `NOT`, `!` | right (前缀) | |
| 4 | `==`, `!=`, `<`, `<=`, `>`, `>=` | left | 链式比较 parse 接受、semantic 拒绝 |
| 4 | `in [...]`, `not in [...]` | non-assoc | RHS 必须是字面量数组 |
| 5 | `-` (unary) | right | 仅 `Unary(SUB, expr)` |
| 6 | literal / ident / `(...)` | — | |

### 3.2 字面量

| 类型 | 语法 | 例 |
|---|---|---|
| 整数 | 十进制 | `42`, `0`, `-7`（负号是 unary）|
| 浮点 | 十进制 + 科学计数 | `3.14`, `1e3`, `1.5e-2`, `-0.5` |
| 字符串 | 双/单引号 + C 风格 escape | `"hello"`, `'world'`, `"a\"b"` |
| 布尔 | 三种形式（与 Milvus 一致） | `true`/`True`/`TRUE`, `false`/`False`/`FALSE` |
| 数组 | `[lit, lit, ...]` 含 trailing comma | `[1, 2, 3]`, `["a", "b",]` |

**注意**：Milvus 接受 `True`/`true`/`TRUE` 三种但**不接受** `tRuE`。F1 与 Milvus 一致：
非这 6 种形式的 mixed-case 一律 lex 阶段拒绝并给 did-you-mean 提示。

### 3.3 字符串 escape

| Escape | 含义 |
|---|---|
| `\"` | 双引号 |
| `\'` | 单引号 |
| `\\` | 反斜杠 |
| `\n` | 换行 |
| `\r` | CR |
| `\t` | tab |

F1 暂不支持 `\xHH`、`\uXXXX`、`\OOO` 八进制等罕见 escape（Milvus 支持，留给 F2）。

### 3.4 标识符规则

- `[a-zA-Z_][a-zA-Z_0-9]*`
- **大小写敏感**（与 Milvus 一致）
- 关键字大小写**不**敏感（`and == AND == &&`，但不接受 `And`）
- **保留前缀** `_seq` / `_partition`：parse 阶段允许，semantic 阶段拒绝
- **`$meta`**：F1 完全拒绝（推迟到 F2b）

### 3.5 空白与注释

- 空白：` `, `\t`, `\r`, `\n` 全部跳过
- **没有注释**（与 Milvus 一致）

### 3.6 完整 BNF（F1 实现的子集）

```
expr            : or_expr ;
or_expr         : and_expr (OR and_expr)* ;
and_expr        : not_expr (AND not_expr)* ;
not_expr        : NOT not_expr | term ;
term            : cmp_term | in_term | unary | primary ;
cmp_term        : (unary | primary) CMP_OP (unary | primary) ;
in_term         : Identifier (NOT)? IN array_literal ;
unary           : SUB primary ;        // -7, -age
primary         : literal
                | Identifier
                | '(' expr ')'
                ;
literal         : INT | FLOAT | STRING | BOOL ;
array_literal   : '[' (literal (',' literal)* (',')?)? ']' ;

CMP_OP          : '==' | '!=' | '<' | '<=' | '>' | '>=' ;

// Lexer
INT             : [1-9][0-9]* | '0' ;
FLOAT           : [0-9]+ '.' [0-9]+ ([eE][+-]?[0-9]+)?
                | [0-9]+ [eE][+-]?[0-9]+ ;
STRING          : '"' DoubleStrChar* '"' | "'" SingleStrChar* "'" ;
BOOL            : 'true' | 'True' | 'TRUE' | 'false' | 'False' | 'FALSE' ;

AND             : 'and' | 'AND' | '&&' ;
OR              : 'or'  | 'OR'  | '||' ;
NOT             : 'not' | 'NOT' | '!' ;
IN              : 'in'  | 'IN' ;
SUB             : '-' ;

Identifier      : [a-zA-Z_][a-zA-Z_0-9]* ;
Whitespace      : [ \t\r\n]+ -> skip ;
```

---

## 4. AST 节点

11 个 frozen dataclass，全部值语义、可哈希、自动 `__eq__`。详见 `modules.md §9.21`。

```
Literal:    IntLit, FloatLit, StringLit, BoolLit
List:       ListLit
Reference:  FieldRef
Operations: CmpOp, InOp, And, Or, Not
```

**关键设计**：
- 用 `tuple` 不用 `list`（frozen 友好）
- 没有共同 base class — 用 `Union` + `isinstance` dispatch（与 Operation 抽象一致）
- 没有方法 — 行为在 backend 里
- 每节点带 `pos` 用于错误信息溯源

---

## 5. 编译期：semantic.py

### 5.1 编译步骤

```
1. Walk AST → collect all FieldRef
2. For each FieldRef:
   - lookup in schema.fields
   - if not found → FilterFieldError with did-you-mean
   - if reserved (_seq / _partition / $meta) → FilterFieldError
   - if FLOAT_VECTOR → FilterTypeError
3. Walk AST again → infer + check types
4. Choose backend:
   - F1: 永远 "arrow"
   - F2b: 含 $meta 引用 → "python"
   - F3: 含 UDF → "python"
5. Wrap in CompiledExpr
```

### 5.2 类型推断 + 兼容性

```
int  ≈ int     ✓
int  ≈ float   ✓ (晋升)
str  ≈ str     ✓
bool ≈ bool    ✓
其他           ✗
```

链式比较 `a == b == c` 在 parse 阶段**不**拒绝（与 Milvus 一致），semantic 阶段
报类型错误："left side is bool (result of `a == b`), right side is int — comparison
between bool and int not supported"。

### 5.3 错误信息要求

错误信息是 parser 的脸面。F1 必须做到：

```
>>> col.search([[...]], expr="age >> 18")
FilterParseError: unexpected token '>' at column 5
  age >> 18
      ^
expected: expression after '>'
```

```
>>> col.search([[...]], expr="ag > 18")
FilterFieldError: unknown field 'ag' at column 1
  ag > 18
  ^^
available fields: [id, age, category, score]
did you mean 'age'?
```

```
>>> col.search([[...]], expr="age > 'eighteen'")
FilterTypeError: type mismatch at column 7
  age > 'eighteen'
        ^^^^^^^^^
left side is int (field 'age'), right side is string
```

实现要点：
- 所有异常继承 `MilvusLiteError`，user 可以一把 catch
- 异常带 `source: str` + `pos: int`，`__str__` 自动渲染 caret
- "did you mean" 用 `difflib.get_close_matches`（标准库）
- 每个错误指出**字段名**和**类型**，不只是"type mismatch"

---

## 6. Backend 设计

### 6.1 三 backend 决策

| Backend | 用途 | 速度（100K 行）|
|---|---|---|
| `arrow_backend` | 纯 schema 字段表达式（F1+F2a 全部） | ~5ms |
| `hybrid_backend` | 含 `$meta` 动态字段表达式（F3+） | ~50–100ms |
| `python_backend` | 差分测试基准 + hybrid fallback + 未来 UDF | ~500ms |

**Backend 在 compile 时静态决定**——不在 evaluate 热路径上 dispatch。
- 纯 schema 字段 → `arrow`
- 含 `$meta` → `hybrid`（per-batch JSON 预处理后委托 arrow_backend）
- `python` 不会被 dispatcher 自动选中，仅作为：
  1. test_e2e 差分测试的基准
  2. hybrid_backend 在遇到异构 JSON 类型 / 不兼容 arrow kernel 时的运行时 fallback
  3. 未来 F3 UDF / 真正动态语义的最终落点

### 6.2 arrow_backend 实现策略

**AST → pyarrow.compute 调用树**。pyarrow.compute 是向量化 C++ 实现，所有比较 /
布尔 / IN 都有现成 kernel。

| AST | pyarrow operation |
|---|---|
| 字面量 | `pa.scalar(value)` |
| FieldRef | `table.column(name)` |
| CmpOp | `pc.equal / less / ...` |
| InOp | `pc.is_in(col, value_set=values)` + 可选 `pc.invert` |
| And | `functools.reduce(pc.and_kleene, masks)` |
| Or | `functools.reduce(pc.or_kleene, masks)` |
| Not | `pc.invert` |

**关键细节**：
- 用 `and_kleene` / `or_kleene` 而不是 `and_` / `or_`：pyarrow 推荐对 nullable
  数据用 Kleene 三值逻辑
- 字面量用 `pa.scalar`，compute kernel 接受 array vs scalar 自动 broadcast
- 顶层结果调 `pc.fill_null(False)`：null 表示"无信息"，filter 语义下当 false

### 6.3 python_backend 实现策略

Row-wise 解释器：把 pa.Table 转成 list of dicts，对每行调 Python eval。

```python
def evaluate_python(compiled, data) -> pa.BooleanArray:
    rows = data.to_pylist()
    out = [False] * len(rows)
    for i, row in enumerate(rows):
        result = _eval_row(compiled.ast, row)
        out[i] = bool(result) if result is not None else False
    return pa.array(out, type=pa.bool_())
```

NULL 三值逻辑：用 Kleene 实现 AND/OR/NOT，最终 None → False。

**性能**：100K 行 ~500ms。慢但通用。F3+ 阶段不再被 dispatcher 自动选中，仅作为差分基准 + hybrid fallback。

### 6.3a hybrid_backend 实现策略 (F3+)

`$meta["key"]` 在 F2b 最初实现是 `python_backend` 直接 row-wise 解释 — 每行付出
"AST walk + JSON parse" 的双重开销，100K 行 ~500ms。F3+ 引入 hybrid_backend：

**思路**：把 JSON 解析与列物化提到 per-batch 一次，让比较/算术/布尔仍走 arrow 向量化。

**步骤**：
1. `collect_meta_keys(ast)` 扫一遍 AST 收集所有 `$meta["key"]` 的 key 集合
2. `_augment_table(data, keys)`：
   - 一次 `to_pylist()` 把 `$meta` 列拉出来
   - 每行 `json.loads` 一次（容错 None / dict / 坏 JSON）
   - 对每个 key 用 `pa.array([d.get(key) for d in parsed])` 物化成 Arrow 列
   - append 到原 table，列名约定 `__meta__<key>`（双下划线前缀避免冲突）
3. `_rewrite_meta_access(ast, keys)`：把 AST 里所有 `MetaAccess(key)` 节点替换为
   `FieldRef("__meta__<key>")`，得到一份新 AST
4. 用 `dataclasses.replace` 临时把 backend 改成 `"arrow"`，调 `evaluate_arrow`

**性能**：100K 行 ~50–100ms（瓶颈从 row-wise Python 转为 JSON 解析），约 5–10×。

**Fallback**：整段 try/except 包裹 augment + arrow eval。任意失败（异构类型 / 全 null
列没有匹配 kernel / arrow 不支持的型变换）→ 落回 `python_backend` 跑这一次 evaluate。
fallback 是 per-evaluate 而不是 per-row，开销可控。

**正确性**：差分测试 (`test_meta_hybrid_vs_python_parity`) 跑遍 14 个 `$meta` 表达式，
逐行比对 hybrid 与 python 输出。语义源头是 `python_backend`，hybrid 偏离即测试失败。

### 6.4 差分测试

`test_e2e.py` 里每个 case **同时跑两个 backend**，断言结果相等：

```python
@pytest.mark.parametrize("expr_str", [...50+ cases...])
def test_arrow_python_equivalence(expr_str, sample_table, sample_schema):
    expr = parse_expr(expr_str)
    compiled = compile_expr(expr, sample_schema)

    arrow_result = evaluate_arrow(compiled, sample_table)

    py_compiled = CompiledExpr(
        ast=compiled.ast, fields=compiled.fields, backend="python",
    )
    py_result = evaluate_python(py_compiled, sample_table)

    assert arrow_result.equals(py_result), \
        f"backend mismatch on '{expr_str}'"
```

**为什么差分测试是关键**：
1. 写两份实现，互相校验 — 任何一边的 bug 都被另一边暴露
2. NULL 三值逻辑、类型 promotion、边界值 — 这些容易写错的地方靠对称性 catch
3. F2b 引入 `$meta` 后，差分测试自然扩展到验证"backend selection 是否选对"

---

## 7. 与现有 search pipeline 的集成

### 7.1 数据流

```
        ┌──────────────────────────────────┐
        │  Collection.search(expr=...)     │
        └────────────┬─────────────────────┘
                     │ parse + compile
                     ▼
        ┌──────────────────────────────────┐
        │  assemble_candidates(            │
        │    filter_compiled=...           │
        │  )                               │
        │                                  │
        │  per source (segment / memtable):│
        │    bool_arr = evaluate(          │
        │      compiled, source_table      │
        │    )                             │
        │    chunks.append(bool_arr)       │
        │                                  │
        │  filter_mask = concat(chunks)    │
        └────────────┬─────────────────────┘
                     │ returns + filter_mask
                     ▼
        ┌──────────────────────────────────┐
        │  execute_search(                 │
        │    filter_mask=...               │
        │  )                               │
        │                                  │
        │  build_valid_mask(               │
        │    dedup + tombstone             │
        │      + filter_mask               │
        │  )                               │
        │                                  │
        │  for each query: distance + topk │
        └──────────────────────────────────┘
```

### 7.2 bitmap.py 改动

```python
def build_valid_mask(
    all_pks, all_seqs, delta_index,
    filter_mask: Optional[np.ndarray] = None,  # NEW
) -> np.ndarray:
    mask = ...  # existing dedup + tombstone
    if filter_mask is not None:
        mask = mask & filter_mask
    return mask
```

### 7.3 assembler.py 改动

```python
def assemble_candidates(
    segments, memtable, vector_field,
    partition_names=None,
    filter_compiled=None,  # NEW
):
    ...
    filter_chunks = []
    for segment in scoped_segments:
        if filter_compiled:
            bool_arr = filter.evaluate(filter_compiled, segment.table)
            filter_chunks.append(bool_arr.to_numpy(zero_copy_only=False))
        ...
    if filter_compiled and mt_pks:
        mt_table = _memtable_to_table(memtable, partition_names)
        bool_arr = filter.evaluate(filter_compiled, mt_table)
        filter_chunks.append(bool_arr.to_numpy(zero_copy_only=False))

    filter_mask = np.concatenate(filter_chunks) if filter_chunks else None
    return all_pks, all_seqs, all_vectors, all_records, filter_mask
```

**为什么 filter 在 assembler 而不是 bitmap**：
- 数据已经是 pa.Table 形式（segment 持有原始 Table）
- pyarrow.compute 需要 columnar 输入，bitmap 阶段已经 numpy 化了
- 让 bitmap.py 保持纯 numpy（与 distance / executor 一致）

`assembler` 是 search 子系统中**唯一同时知道 storage 类型 (Segment, MemTable) 和
filter 子系统**的模块。

---

## 8. Collection API 升级

```python
class Collection:
    def search(
        self,
        query_vectors: List[list],
        top_k: int = 10,
        metric_type: str = "COSINE",
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,        # NEW
    ) -> List[List[dict]]: ...

    def get(
        self,
        pks: List[Any],
        partition_names: Optional[List[str]] = None,
        expr: Optional[str] = None,        # NEW
    ) -> List[dict]: ...

    def query(                              # NEW METHOD
        self,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Pure scalar query — no vector. Returns all matching rows."""

    def _compile_filter(self, expr_str: str) -> CompiledExpr:
        from milvus_lite.search.filter import parse_expr, compile_expr
        return compile_expr(parse_expr(expr_str), self._schema)
```

`query()` 是新方法 — 纯标量查询，相当于 `search(query=None, expr=...)` 但不需要
query vector，也不计算距离，直接返回所有匹配行（可选 `limit`）。

---

## 9. Phase 8 子阶段拆分

| Phase | 目标 grammar | Backend | Status |
|---|---|---|---|
| **F1** | Tier 1：比较 + 布尔 + IN + 字面量 + 字段引用 + 括号 | 仅 arrow_backend；python_backend 仅做差分测试 | ✅ done |
| **F2a** | + `like` + 算术 (`+ - * / %`) + `is null` | 仍 arrow_backend | ✅ done |
| **F2b** | + `$meta["key"]` 动态字段 | 引入 python_backend dispatch | ✅ done |
| **F2c** | filter LRU cache + `query()` 接入 | 与 backend 无关 | ✅ done |
| **F3+** | 性能优化：per-batch JSON 预处理 → arrow_backend；hybrid 取代 python 作 $meta 默认 dispatch | 引入 hybrid_backend | ✅ done |
| **F3** | + `json_contains` / `array_contains` / UDF / 严格 Milvus 兼容 | 扩展 python_backend；可选 ANTLR parser swap | — |

---

## 10. 关于 ANTLR

Milvus 使用 ANTLR4 + Plan.g4 生成 C++ parser。我们 F1 选**手写 Pratt parser** 而
非引入 ANTLR Python target，原因：

1. **F1 grammar 小**（10 个算子），手写 ~300 行 Python，调试友好
2. **零依赖**（不引入 antlr4-python3-runtime + 1500 行生成代码）
3. **错误信息可控**（手写 caret + did-you-mean 比 ANTLR override BaseErrorListener 简单）
4. **AST 是稳定接口** — 未来 F3 切换到 ANTLR 后端时，type checker / evaluator 都不动

但**借鉴 Milvus Plan.g4 的语法设计**：操作符优先级表、关键字大小写、字面量语法、
AST 节点形态都对齐 Milvus（方便未来真要做 binary 兼容时切换 parser 实现）。

参考：[milvus-io/milvus Plan.g4](https://github.com/milvus-io/milvus/blob/master/internal/parser/planparserv2/Plan.g4)

---

## 11. 不在 Phase F1 范围

| 特性 | 推迟到 |
|---|---|
| `like` 算子 | F2a |
| 算术 (`+, -, *, /, %`) | F2a |
| `is null` / `is not null` | F2a |
| `$meta` 动态字段 | F2b |
| JSON / array 函数 | F3 |
| UDF | F3 |
| Expression cache | F2c |
| ANTLR-based parser | F3+ |
| DuckDB 后端 | F3+ |

F1 grammar 之外的算子，lex/parse 阶段会 reject 并给 "Phase F2/F3 will support" 提示，
而不是 silent error。

---

## 12. 完成标志

- **F1 done**：
  - `col.search([[...]], expr="age > 18 and category in ['tech', 'news']")` 跑通
  - 差分测试 50+ case 全绿
  - `examples/m8_demo.py` 通过
  - `Collection.search` / `get` / `query` 三个方法都接受 expr
  - 错误信息含 caret + did-you-mean

- **F2 done**：
  - `col.search(expr="title like 'AI%' and $meta['priority'] > 5")` 跑通
  - filter LRU cache + `query()` 接入

- **F3+ done**：
  - hybrid_backend 取代 python_backend 作 `$meta` 默认 dispatch
  - 差分测试 hybrid vs python 在所有 `$meta` 表达式上一致
  - 异构 JSON 类型 / 全 null 列等异常自动 fallback 到 python_backend

- **F3 done**：
  - 跑通 pymilvus 表达式测试套件子集
  - 可选 ANTLR backend
