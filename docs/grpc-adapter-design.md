# 深入设计：gRPC 适配层（Phase 10）

## 1. 概述

MilvusLite Phase 10 在内部 engine 之上构造一个 gRPC 服务层，让 **pymilvus 客户端无需修改代码即可连接 MilvusLite**。这是项目作为"本地版 Milvus"在协议兼容性上的最终一公里。

**核心定位**：协议翻译，不增加任何 engine 能力。能跑的 RPC 完全等于 engine 已实现的方法集。这一层的工作量集中在：
1. 协议到 engine API 的字段映射 + 数据结构转换（最大头）
2. Milvus FieldData 列式 → engine records 行式 list[dict] 的转置
3. 错误码翻译
4. 实现下的 RPC stub + 友好的 UNIMPLEMENTED 消息

**前置依赖**：Phase 9 必须先完成 — Phase 10 的 `CreateIndex / LoadCollection / Search` RPC 直接映射到 Phase 9 的 `Collection.create_index / load / search`，两阶段顺序倒过来会让 Phase 10 的 servicer 失去稳定的下层 API 可对接。

---

## 2. 服务定义来源决策

### 2.1 三个候选方案

| 方案 | 优点 | 缺点 |
|---|---|---|
| **A. 直接拷 milvus 官方 proto** | pymilvus 开箱即连；行为一致性最强；future Milvus 版本升级时跟着拉新 proto | proto 量大（~100 个 RPC，几百 KB），实现 stub 工作量大；某些 RPC 的 server-side 行为依赖 Milvus 内部状态机 |
| **B. 手写最小 proto 子集** | proto 简单，实现量小 | pymilvus 不认识，失去最大卖点 |
| **C. 拷 milvus proto，但只实现 quickstart 子集，其他返回 UNIMPLEMENTED** | 兼顾 A 的 pymilvus 兼容性 + B 的可控工作量 | 用户调到未实现 RPC 时需要友好错误消息 |

### 2.2 决定：方案 C

**理由**：
1. **pymilvus 必须能连**是项目的核心目标，方案 B 一票否决
2. 方案 A 实现所有 RPC 是不必要的浪费 — Milvus 的 backup / RBAC / replica / resource group 等大量 RPC 对本地嵌入式数据库没有意义
3. 方案 C 等于"protocol surface = Milvus，functional surface = MilvusLite"，pymilvus 能 connect + 大部分 quickstart 调用能跑通，少数 RPC 返回明确的"MilvusLite 不支持 X"

**前例参考**：Zilliz 官方的 `milvus-lite` 项目（SQLite 后端）走的就是这条路。我们本质上是"用 LSM Parquet + FAISS 替换 milvus-lite 的 SQLite"。

### 2.3 Proto 文件来源

从 [milvus-io/milvus-proto](https://github.com/milvus-io/milvus-proto) 拉以下文件：
- `proto/milvus.proto` — 主 RPC 定义
- `proto/schema.proto` — schema / FieldData 类型
- `proto/common.proto` — 公共类型（Status, KeyValuePair, MsgBase 等）

放到 `milvus_lite/adapter/grpc/proto/`，用 `grpcio-tools` 生成 `_pb2.py` / `_pb2_grpc.py`，**生成结果 commit 到 repo**（不 runtime 生成，避免 build 时依赖 grpcio-tools）。

`proto/README.md` 记录"从哪个 milvus-proto commit 生成"，便于追溯和升级。

---

## 3. 模块结构

```
milvus_lite/
└── adapter/
    └── grpc/
        ├── __init__.py
        ├── server.py                   # run_server(data_dir, host, port)
        ├── servicer.py                 # MilvusServicer — 所有 RPC 实现
        ├── translators/
        │   ├── __init__.py
        │   ├── schema.py               # Milvus FieldSchema ↔ milvus_lite FieldSchema
        │   ├── records.py              # FieldData (列式) ↔ list[dict] (行式)
        │   ├── search.py               # SearchRequest 解析
        │   ├── result.py               # engine 结果 → SearchResults proto
        │   ├── expr.py                 # Milvus filter expr ↔ MilvusLite filter
        │   └── index.py                # IndexParams ↔ IndexSpec
        ├── proto/                      # 生成的 stub
        │   ├── __init__.py
        │   ├── milvus_pb2.py
        │   ├── milvus_pb2_grpc.py
        │   ├── schema_pb2.py
        │   ├── common_pb2.py
        │   └── README.md               # source commit reference
        ├── errors.py                   # MilvusLiteError → grpc Status mapping
        └── cli.py                      # python -m milvus_lite.adapter.grpc 入口
```

**依赖**：
```toml
# pyproject.toml
[project.optional-dependencies]
grpc = ["grpcio>=1.50", "protobuf>=4.21"]
```

`grpcio-tools` 只在 dev / build 阶段需要（生成 stub），不进 runtime 依赖。

---

## 4. pymilvus → engine API 完整映射表

### 4.1 Collection 生命周期

| pymilvus | Milvus RPC | engine API | 说明 |
|---|---|---|---|
| `MilvusClient(uri="...")` | (TCP connect) | `MilvusLite(data_dir)` 在 server 启动时持有 | server 模式只服务一个 data_dir |
| `create_collection(name, dim, ...)` 快速模式 | `CreateCollection` | `db.create_collection(name, schema)` | translator 把 quickstart 参数生成默认 schema (id INT64 + vector FLOAT_VECTOR) |
| `create_collection(name, schema)` 完整 schema | `CreateCollection` | 同上 | translator 解析 `CollectionSchema` proto，逐字段转 `FieldSchema` |
| `drop_collection(name)` | `DropCollection` | `db.drop_collection(name)` | 直接映射 |
| `has_collection(name)` | `HasCollection` | `db.has_collection(name)` | bool 包成 `BoolResponse` |
| `describe_collection(name)` | `DescribeCollection` | `db.get_collection(name).describe()` + schema 序列化 | translator 把 MilvusLite schema 转回 Milvus proto schema |
| `list_collections()` | `ShowCollections` | `db.list_collections()` | 直接映射 |
| `get_collection_stats(name)` | `GetCollectionStatistics` | `col.num_entities` | 包成 `KeyValuePair[("row_count", str(n))]` |
| `rename_collection(old, new)` | `RenameCollection` | `db.rename_collection(old, new)` | ✅ 直接映射 |
| `alter_collection_properties` | `AlterCollection` | ❌ schema 不可变 | UNIMPLEMENTED |

### 4.2 Partition

| pymilvus | Milvus RPC | engine API | 说明 |
|---|---|---|---|
| `create_partition(collection, partition)` | `CreatePartition` | `col.create_partition(name)` | Phase 9.1 补的 API |
| `drop_partition(collection, partition)` | `DropPartition` | `col.drop_partition(name)` | Phase 9.1 补的 API |
| `has_partition(collection, partition)` | `HasPartition` | `partition in col.list_partitions()` | bool 包装 |
| `list_partitions(collection)` | `ShowPartitions` | `col.list_partitions()` | 直接映射 |
| `get_partition_stats` | `GetPartitionStatistics` | `col.partition_num_entities(name)` | Phase 9.1 可选补 |
| `load_partitions` / `release_partitions` | `LoadPartitions` / `ReleasePartitions` | ❌ engine load/release 是 Collection 级别 | UNIMPLEMENTED 或映射到 collection load/release |

### 4.3 数据 CRUD

| pymilvus | Milvus RPC | engine API | 说明 |
|---|---|---|---|
| `insert(collection, data, partition_name=None)` | `Insert` | `col.insert(records, partition_name)` | **最复杂转换** — InsertRequest.fields_data 是 FieldData 列式结构，需要转置为 records list |
| `upsert(collection, data, partition_name=None)` | `Upsert` | `col.insert(records, partition_name)` | engine insert 已是 upsert 语义；两个 RPC 共享同一个 servicer 方法 |
| `delete(collection, ids=, partition_name=None)` | `Delete` | `col.delete(pks, partition_name)` | DeleteRequest 的 expr 字段为 `id in [...]` 形式时走 pk 路径 |
| `delete(collection, filter=, partition_name=None)` | `Delete` | `col.query(filter) → 提取 pk → col.delete(pks)` | 表达式删除：先 query 找 pk 再 delete |
| `get(collection, ids=, partition_names=, output_fields=)` | `Query`(`id in [...]` expr) | `col.get(pks, partition_names, expr)` | pymilvus get 实际走 Query RPC |
| `query(collection, filter, output_fields, limit, partition_names)` | `Query` | `col.query(expr, output_fields, partition_names, limit)` | 直接映射 |

### 4.4 搜索

| pymilvus | Milvus RPC | engine API | 说明 |
|---|---|---|---|
| `search(collection, data, anns_field, limit, filter, output_fields, search_params, partition_names)` | `Search` | `col.search(query_vectors, top_k, metric_type, partition_names, expr, output_fields)` | translator 解析 SearchParams 提取 metric / topk / search_params；返回值结构转换最复杂 |
| `hybrid_search(collection, reqs, ...)` | `HybridSearch` | 多路 `col.search()` + `reranker.rerank()` | ✅ Phase 12：解析每个子 SearchRequest 独立搜索，通过 WeightedRanker / RRFRanker 融合结果 |
| `search_iterator(...)` | (client-side wrapper) | engine 加 offset 支持 | 可选；MVP UNIMPLEMENTED |

### 4.5 索引

| pymilvus | Milvus RPC | engine API | 说明 |
|---|---|---|---|
| `create_index(collection, index_params)` | `CreateIndex` | `col.create_index(field, params)` | translator 把 IndexParams 的 KeyValuePair list 转 IndexSpec |
| `drop_index(collection, field_name, index_name)` | `DropIndex` | `col.drop_index(field)` | index_name 忽略（engine 一个 field 只支持一个 index） |
| `describe_index(collection, field_name)` | `DescribeIndex` | `col.get_index_info()` | IndexSpec → IndexDescription proto |
| `list_indexes(collection)` | `ListIndexedField`（Milvus 内部 RPC，pymilvus 客户端有封装） | `col.list_indexes()` | engine 加一个 helper 方法 |
| `get_index_state` / `get_index_build_progress` | `GetIndexState` / `GetIndexBuildProgress` | engine 是同步 build，永远返回 `Finished` / `100%` | trivial 实现 |

### 4.6 Load / Release

| pymilvus | Milvus RPC | engine API | 说明 |
|---|---|---|---|
| `load_collection(collection, replica_number=1)` | `LoadCollection` | `col.load()` | replica_number 忽略 |
| `release_collection(collection)` | `ReleaseCollection` | `col.release()` | 直接映射 |
| `get_load_state(collection)` | `GetLoadState` | `col._load_state` | 枚举映射：released → NotLoad，loading → Loading，loaded → Loaded |
| `get_loading_progress(collection)` | `GetLoadingProgress` | engine load 是同步，永远 100% | trivial |

### 4.7 其他

| pymilvus | Milvus RPC | engine API | 说明 |
|---|---|---|---|
| `flush(collection)` | `Flush` | `col.flush()` | 直接映射 |
| `compact(collection)` | `ManualCompaction` | `col.compact()` | engine 加一个手动 compact 触发方法 |
| `list_databases()` | `ListDatabases` | stub 返回 `["default"]` | MilvusLite 没有 database 多实例 |
| `using_database(name)` | `UseDatabase` | 仅接受 "default"，其他报错 | trivial |
| Aliases (`create_alias` 等) | `CreateAlias` 等 | ❌ engine 不支持 | UNIMPLEMENTED |
| User / Role / Privilege | `CreateCredential` 等 | ❌ 嵌入式无 RBAC | 一律返回 OK 空结果（pymilvus 不会因此 crash） |
| Backup / Restore | 各种 | ❌ | UNIMPLEMENTED |
| Resource Group | `CreateResourceGroup` 等 | ❌ | UNIMPLEMENTED |
| Replica | `GetReplicas` 等 | ❌ 单进程 | UNIMPLEMENTED |
| QueryNode / DataNode 等内部 RPC | 各种 | — | 这些是 Milvus 内部组件间通信，pymilvus 不调，不实现 |

### 4.8 未支持 RPC 的处理策略

**三档策略**：

1. **明确 UNIMPLEMENTED**（推荐默认）：返回 `grpc.StatusCode.UNIMPLEMENTED` + 友好消息
   - 应用于：bulk_insert 等"功能缺失"的 RPC
   - 错误消息格式：`"MilvusLite does not support X. Reason: <one-line reason>. See https://...”`

2. **静默成功**（少数情况）：返回 `Success` + 空结果
   - 应用于：RBAC 系列（嵌入式没有用户概念，pymilvus 调用时不应该 crash）
   - 应用于：多 database（永远是 "default"）
   - **决策原则**：只有当"假装成功"对用户体验完全无害时才用，一旦有误导性立即改回 UNIMPLEMENTED

3. **忽略可选参数**（向前兼容）：知道的字段处理，未知字段忽略
   - 应用于：SearchRequest 的 consistency_level、travel_timestamp 等
   - **不算"假装支持"**，是合理的 forward compatibility

---

## 5. 关键转换：FieldData ↔ records

### 5.1 Milvus FieldData 结构

Milvus InsertRequest 是按字段的列式结构：

```protobuf
message InsertRequest {
  string collection_name = 1;
  string partition_name = 2;
  repeated FieldData fields_data = 3;
  repeated uint32 hash_keys = 4;
  uint32 num_rows = 5;
}

message FieldData {
  schema.DataType type = 1;
  string field_name = 2;
  oneof field {
    schema.ScalarField scalars = 3;
    schema.VectorField vectors = 4;
  }
  int64 field_id = 5;
}

message ScalarField {
  oneof data {
    BoolArray bool_data = 1;
    IntArray int_data = 2;
    LongArray long_data = 3;
    FloatArray float_data = 4;
    DoubleArray double_data = 5;
    StringArray string_data = 6;
    BytesArray bytes_data = 7;
    ArrayArray array_data = 8;
    JSONArray json_data = 9;
  }
}

message VectorField {
  int64 dim = 1;
  oneof data {
    FloatArray float_vector = 2;
    bytes binary_vector = 3;
    bytes float16_vector = 4;
    bytes bfloat16_vector = 5;
    SparseFloatArray sparse_float_vector = 6;
  }
}
```

**关键差异点**：
- Milvus 是列式 — 每个字段一个 array，所有字段长度相同 = `num_rows`
- MilvusLite engine 是行式 — `List[Dict[field_name, value]]`
- Vector 字段在 Milvus 里是 flat float array（长度 = `num_rows * dim`），需要按 dim 切片
- 类型多 — 每个 oneof 分支对应一组转换逻辑

### 5.2 转置算法

```python
# milvus_lite/adapter/grpc/translators/records.py

def fields_data_to_records(
    fields_data: List["FieldData"],
    num_rows: int,
) -> List[Dict[str, Any]]:
    """Transpose Milvus columnar fields_data into engine row-wise records.

    Args:
        fields_data: list of FieldData proto messages
        num_rows: declared row count from InsertRequest.num_rows

    Returns:
        records: list of dicts, length num_rows. Each dict has all field
                 names from fields_data.

    Raises:
        ValueError: if any FieldData length mismatches num_rows
        UnsupportedFieldTypeError: if a FieldData uses a type MilvusLite
                                    doesn't support (e.g. SparseFloat)
    """
    records: List[Dict[str, Any]] = [{} for _ in range(num_rows)]

    for fd in fields_data:
        column = _extract_column(fd, num_rows)
        for i in range(num_rows):
            records[i][fd.field_name] = column[i]

    return records


def _extract_column(fd: "FieldData", num_rows: int) -> List[Any]:
    """Pull a single FieldData out as a length-num_rows Python list."""
    if fd.HasField("scalars"):
        scalars = fd.scalars
        if scalars.HasField("long_data"):
            data = list(scalars.long_data.data)
        elif scalars.HasField("int_data"):
            data = list(scalars.int_data.data)
        elif scalars.HasField("float_data"):
            data = list(scalars.float_data.data)
        elif scalars.HasField("double_data"):
            data = list(scalars.double_data.data)
        elif scalars.HasField("string_data"):
            data = list(scalars.string_data.data)
        elif scalars.HasField("bool_data"):
            data = list(scalars.bool_data.data)
        elif scalars.HasField("json_data"):
            # JSON values are stored as bytes in proto; decode + parse
            data = [json.loads(b.decode("utf-8")) for b in scalars.json_data.data]
        else:
            raise UnsupportedFieldTypeError(
                f"unsupported scalar field type for {fd.field_name}"
            )
    elif fd.HasField("vectors"):
        vectors = fd.vectors
        dim = vectors.dim
        if vectors.HasField("float_vector"):
            flat = vectors.float_vector.data
            data = [list(flat[i*dim:(i+1)*dim]) for i in range(num_rows)]
        elif vectors.HasField("binary_vector"):
            raise UnsupportedFieldTypeError("binary vectors not supported in MVP")
        elif vectors.HasField("sparse_float_vector"):
            raise UnsupportedFieldTypeError("sparse vectors not supported in MVP")
        else:
            raise UnsupportedFieldTypeError(
                f"unsupported vector field type for {fd.field_name}"
            )
    else:
        raise ValueError(f"FieldData {fd.field_name} has no scalars or vectors")

    if len(data) != num_rows:
        raise ValueError(
            f"FieldData {fd.field_name} has {len(data)} rows, expected {num_rows}"
        )
    return data
```

### 5.3 反向转换：records → FieldData

仅在 `Query` / `Get` / `Search` RPC 的返回值中需要 — 把 engine 返回的 list[dict] 转回 Milvus FieldData 列式结构。算法对称：

```python
def records_to_fields_data(
    records: List[Dict[str, Any]],
    schema: "CollectionSchema",
    output_fields: Optional[List[str]] = None,
) -> List["FieldData"]:
    """Build columnar FieldData list from row-wise records, based on
    the collection schema (which knows each field's type).

    output_fields: optional whitelist; only these fields are emitted.
    """
    if not records:
        return []

    field_names = output_fields or [f.name for f in schema.fields]
    fields_data = []

    for fname in field_names:
        fschema = schema.get_field(fname)
        column = [r.get(fname) for r in records]
        fd = _build_field_data(fname, fschema, column)
        fields_data.append(fd)

    return fields_data
```

### 5.4 测试覆盖

`tests/adapter/test_grpc_translators_records.py`：
- 每种类型一个 round-trip 测试：build FieldData → fields_data_to_records → records_to_fields_data → 应等价
- num_rows mismatch → ValueError
- 不支持的类型 → UnsupportedFieldTypeError
- vector dim 切片正确性
- 空 fields_data → 空 records list
- 部分字段 nullable → None 值的处理

---

## 6. Filter 表达式翻译

### 6.1 大部分情况：透传

MilvusLite Phase 8 的 filter grammar 就是抄的 Milvus，**绝大多数表达式直接透传**：

```
"age > 18 and category in ['tech', 'news']"   # 完全兼容，原样传给 col.search(expr=...)
```

### 6.2 需要 rewrite 的少数场景

| Milvus 写法 | MilvusLite 是否支持 | 处理策略 |
|---|---|---|
| `field == value` 等比较 | ✅ | 透传 |
| `field in [...]` | ✅ | 透传 |
| `field like "pattern"` | ✅ (Phase F2a) | 透传 |
| `$meta["key"] == value` | ✅ (Phase F2b) | 透传 |
| 算术 + - * / | ✅ (Phase F2a) | 透传 |
| `is null` / `is not null` | ✅ (Phase F2a) | 透传 |
| `json_contains(json_field, value)` | ✅ | 透传（parser 原生支持） |
| `array_contains(array_field, value)` | ✅ | 透传（parser 原生支持 array_contains / array_contains_all / array_contains_any） |
| `text_match(text_field, query)` | ✅ (Phase 11) | 透传（engine 内置 BM25 全文索引 + analyzer） |
| `phrase_match` | ❌ | UNIMPLEMENTED |

### 6.3 实现方式

**当前状态**：Phase 11 之后，MilvusLite parser 原生支持 `text_match`、`json_contains`、`array_contains` 等函数，表达式全部透传给 engine parser 处理，无需适配层做 rewrite 或拦截。仅 `phrase_match` 仍不支持（parser 遇到未知函数会抛 `FilterParseError`）。独立的 `translators/expr.py` 文件未创建 — 表达式直接透传。

---

## 7. Servicer 实现骨架

```python
# milvus_lite/adapter/grpc/servicer.py

import grpc
from .proto import milvus_pb2, milvus_pb2_grpc, common_pb2
from .errors import to_grpc_status
from .translators.records import fields_data_to_records, records_to_fields_data
from .translators.schema import milvus_to_milvus_lite_schema, milvus_lite_to_milvus_schema
from .translators.expr import translate_filter_expr
from milvus_lite import MilvusLite
from milvus_lite.exceptions import MilvusLiteError

class MilvusServicer(milvus_pb2_grpc.MilvusServiceServicer):
    def __init__(self, db: MilvusLite):
        self._db = db

    # ── Collection lifecycle ─────────────────────────────────────

    def CreateCollection(self, request, context):
        try:
            schema = milvus_to_milvus_lite_schema(request.schema)
            self._db.create_collection(request.collection_name, schema)
            return common_pb2.Status(code=0, reason="")
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def DropCollection(self, request, context):
        try:
            self._db.drop_collection(request.collection_name)
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def HasCollection(self, request, context):
        try:
            exists = self._db.has_collection(request.collection_name)
            return milvus_pb2.BoolResponse(
                status=common_pb2.Status(code=0),
                value=exists,
            )
        except MilvusLiteError as e:
            return milvus_pb2.BoolResponse(status=common_pb2.Status(**to_grpc_status(e)))

    # ── Data CRUD ────────────────────────────────────────────────

    def Insert(self, request, context):
        try:
            records = fields_data_to_records(request.fields_data, request.num_rows)
            col = self._db.get_collection(request.collection_name)
            partition = request.partition_name or "_default"
            inserted_pks = col.insert(records, partition_name=partition)
            return milvus_pb2.MutationResult(
                status=common_pb2.Status(code=0),
                IDs=_pks_to_ids_proto(inserted_pks, schema=col.schema),
                insert_cnt=len(inserted_pks),
                succ_index=list(range(len(inserted_pks))),
            )
        except MilvusLiteError as e:
            return milvus_pb2.MutationResult(status=common_pb2.Status(**to_grpc_status(e)))

    def Search(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            queries = _decode_search_query(request)
            top_k, metric_type, _ = _parse_search_params(request.search_params)
            expr = translate_filter_expr(request.dsl) if request.dsl else None
            results = col.search(
                query_vectors=queries,
                top_k=top_k,
                metric_type=metric_type,
                partition_names=list(request.partition_names) or None,
                expr=expr,
                output_fields=list(request.output_fields) or None,
            )
            return _build_search_results(results, col.schema, request.output_fields)
        except MilvusLiteError as e:
            return milvus_pb2.SearchResults(status=common_pb2.Status(**to_grpc_status(e)))

    # ── Index lifecycle ──────────────────────────────────────────

    def CreateIndex(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            params = _kv_pairs_to_dict(request.extra_params)
            index_params = {
                "index_type": params.get("index_type", "HNSW"),
                "metric_type": params.get("metric_type", "COSINE"),
                "params": json.loads(params.get("params", "{}")) if params.get("params") else {},
            }
            col.create_index(request.field_name, index_params)
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def LoadCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.load()
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    def ReleaseCollection(self, request, context):
        try:
            col = self._db.get_collection(request.collection_name)
            col.release()
            return common_pb2.Status(code=0)
        except MilvusLiteError as e:
            return common_pb2.Status(**to_grpc_status(e))

    # ── Catch-all UNIMPLEMENTED ──────────────────────────────────

    def _unimplemented(self, context, rpc_name: str, reason: str = ""):
        msg = f"MilvusLite does not support {rpc_name}"
        if reason:
            msg += f": {reason}"
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(msg)
        return common_pb2.Status(code=2, reason=msg)

    def CreateAlias(self, request, context):
        return self._unimplemented(context, "CreateAlias", "aliases are not in MVP scope")

    def HybridSearch(self, request, context):
        # ✅ Phase 12: 多路搜索 + WeightedRanker/RRFRanker 融合
        # 解析每个 sub-SearchRequest → col.search() → reranker.rerank()
        # 实现见 servicer.py + reranker.py
        ...

    # ... (其他 UNIMPLEMENTED stub)
```

---

## 8. 错误码映射

```python
# milvus_lite/adapter/grpc/errors.py

from milvus_lite.exceptions import (
    MilvusLiteError,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    PartitionNotFoundError,
    SchemaError,
    FilterParseError,
    FilterTypeError,
    FilterFieldError,
    CollectionNotLoadedError,
    IndexAlreadyExistsError,
    IndexNotFoundError,
    IndexBackendUnavailableError,
)

# Milvus 标准 ErrorCode
# 0  Success
# 1  UnexpectedError
# 4  CollectionNotExists
# 6  IllegalArgument
# 11 IndexNotExist
# 26 IndexBuildFailed
# 101 CollectionNotLoaded
# 1100 (newer) BadRequest
# ... (full list in milvus common.proto)

_EXCEPTION_TO_CODE = {
    CollectionNotFoundError:       (4,   "CollectionNotExists"),
    CollectionAlreadyExistsError:  (1,   "CollectionAlreadyExists"),
    PartitionNotFoundError:        (200, "PartitionNotExists"),
    SchemaError:                   (6,   "IllegalArgument"),
    FilterParseError:              (6,   "IllegalArgument"),
    FilterTypeError:               (6,   "IllegalArgument"),
    FilterFieldError:              (6,   "IllegalArgument"),
    CollectionNotLoadedError:      (101, "CollectionNotLoaded"),
    IndexAlreadyExistsError:       (35,  "IndexAlreadyExists"),
    IndexNotFoundError:            (11,  "IndexNotExist"),
    IndexBackendUnavailableError:  (26,  "IndexBuildFailed"),
}

def to_grpc_status(exc: MilvusLiteError) -> dict:
    code, _ = _EXCEPTION_TO_CODE.get(type(exc), (1, "UnexpectedError"))
    return {
        "code": code,
        "reason": str(exc),
    }
```

**注意**：Milvus 的 ErrorCode 在 2.3 → 2.4 之间做过较大调整（从 numeric code 转向 string-based 错误）。Phase 10 MVP 对齐 2.3 风格的 numeric code，pymilvus 客户端两个版本都能识别。

---

## 9. server.py + CLI

```python
# milvus_lite/adapter/grpc/server.py

import grpc
from concurrent import futures
from .servicer import MilvusServicer
from .proto import milvus_pb2_grpc
from milvus_lite import MilvusLite

def run_server(
    data_dir: str,
    host: str = "0.0.0.0",
    port: int = 19530,
    max_workers: int = 10,
):
    db = MilvusLite(data_dir)
    servicer = MilvusServicer(db)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    milvus_pb2_grpc.add_MilvusServiceServicer_to_server(servicer, server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    print(f"MilvusLite gRPC server listening on {addr} (data_dir={data_dir})")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop(grace=5)
        db.close()
```

```python
# milvus_lite/adapter/grpc/cli.py / __main__.py

import argparse
from .server import run_server

def main():
    parser = argparse.ArgumentParser(prog="python -m milvus_lite.adapter.grpc")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19530)
    args = parser.parse_args()
    run_server(args.data_dir, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
```

启动方式：
```bash
python -m milvus_lite.adapter.grpc --data-dir ./data --port 19530
```

---

## 10. Phase 10 子阶段拆分

| 子阶段 | 内容 | 完成标志 | 工作量 |
|---|---|---|---|
| **10.1** | proto 拉取 + stub 生成 + 空 servicer + `run_server` + CLI | `python -m milvus_lite.adapter.grpc --data-dir /tmp/x --port 19530` 起 server，pymilvus.connect() 不报错；所有 RPC 返回 UNIMPLEMENTED | M |
| **10.2** | Collection 生命周期 RPC + `translators/schema.py`（最小类型集：INT64 / VARCHAR / FLOAT_VECTOR / BOOL / FLOAT / DOUBLE） | pymilvus 跑 `create / list / has / describe / drop` 全通；非支持类型报 UnsupportedFieldTypeError | M |
| **10.3** | insert/get/delete/query RPC + `translators/records.py` 双向转置 + 单测覆盖每种支持类型 | pymilvus 灌入 100 条数据 → query 出来等价；delete by id 通；delete by filter 通 | L |
| **10.4** | search + create_index + load + release RPC + `translators/search.py` + `translators/result.py` + `translators/expr.py` + `translators/index.py` | pymilvus quickstart 全流程跑通：create_collection → insert → create_index(HNSW) → load → search(filter) → release → drop | L |
| **10.5** | Partition RPC + flush + stats RPC + `examples/m10_demo.py` + `tests/adapter/test_grpc_quickstart.py` 作为 L3 冒烟 | m10 demo 通；冒烟测试在 CI 跑通 | M |
| **10.6** | 错误码映射 + 异常 wrapping 中间件 + UNIMPLEMENTED 友好消息 | 每种 MilvusLiteError 都有对应的 grpc status code 测试 | S |

合计：1S + 3M + 2L

---

## 11. 验证策略

### 11.1 单元测试

| 测试文件 | 覆盖 |
|---|---|
| `tests/adapter/test_grpc_server_startup.py` | server 启动、shutdown、port binding |
| `tests/adapter/test_grpc_translators_schema.py` | Milvus FieldSchema ↔ MilvusLite FieldSchema 双向 |
| `tests/adapter/test_grpc_translators_records.py` | FieldData ↔ records 每种类型 round-trip |
| `tests/adapter/test_grpc_translators_expr.py` | 透传 + UNIMPLEMENTED 函数检测 |
| `tests/adapter/test_grpc_translators_index.py` | IndexParams ↔ IndexSpec |
| `tests/adapter/test_grpc_collection_lifecycle.py` | create / list / has / describe / drop |
| `tests/adapter/test_grpc_crud.py` | insert / upsert / delete / query / get |
| `tests/adapter/test_grpc_search.py` | search 全参数（filter / top_k / output_fields / partition_names） |
| `tests/adapter/test_grpc_index.py` | create_index / load / release / drop_index |
| `tests/adapter/test_grpc_error_mapping.py` | 每种异常 → grpc status code |

### 11.2 集成测试 — pymilvus 冒烟

`tests/adapter/test_grpc_quickstart.py`：

```python
import pytest
from pymilvus import MilvusClient

@pytest.fixture
def grpc_server(tmp_path):
    """Start MilvusLite gRPC server in a thread, yield port, stop after test."""
    from milvus_lite.adapter.grpc.server import run_server_in_thread
    port = _find_free_port()
    server, db = run_server_in_thread(str(tmp_path), port=port)
    yield port
    server.stop(grace=2)
    db.close()


def test_pymilvus_quickstart(grpc_server):
    client = MilvusClient(uri=f"http://localhost:{grpc_server}")

    # 1. Create
    client.create_collection("demo", dimension=4)
    assert client.has_collection("demo")
    assert "demo" in client.list_collections()

    # 2. Insert
    data = [{"id": i, "vector": [float(i)]*4} for i in range(100)]
    res = client.insert("demo", data=data)
    assert res["insert_count"] == 100

    # 3. Flush + Index
    client.flush("demo")
    client.create_index(
        "demo",
        index_params={
            "field_name": "vector",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )

    # 4. Load
    client.load_collection("demo")

    # 5. Search
    results = client.search(
        "demo",
        data=[[0.1, 0.2, 0.3, 0.4]],
        limit=10,
    )
    assert len(results[0]) == 10

    # 6. Query
    rows = client.query("demo", filter="id >= 50", limit=20)
    assert len(rows) == 20

    # 7. Delete
    client.delete("demo", ids=[1, 2, 3])

    # 8. Release + Drop
    client.release_collection("demo")
    client.drop_collection("demo")
    assert not client.has_collection("demo")
```

这是 Phase 10 的**完成标志测试** — 必须绿。

### 11.3 recall 一致性测试

`tests/adapter/test_grpc_search_parity.py`：

```python
def test_grpc_search_returns_same_topk_as_engine_directly(grpc_server, tmp_path):
    """Search via gRPC and via engine directly should return the same top-k."""
    db = MilvusLite(str(tmp_path))
    col = db.create_collection("test", schema=...)
    col.insert([...])
    col.create_index("vec", {"index_type": "HNSW", ...})
    col.load()

    # Direct engine
    direct_results = col.search([[...]], top_k=10)

    # Via gRPC
    client = MilvusClient(uri=f"http://localhost:{grpc_server}")
    grpc_results = client.search("test", data=[[...]], limit=10)

    # IDs should match exactly (HNSW recall@1 = 1.0 for small datasets)
    direct_ids = [r["id"] for r in direct_results[0]]
    grpc_ids = [r["id"] for r in grpc_results[0]]
    assert direct_ids == grpc_ids
```

---

## 12. 依赖与构建

```toml
# pyproject.toml additions

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "grpcio-tools>=1.50",   # for regenerating proto stubs
    "pymilvus>=2.4",         # for grpc integration tests
]
faiss = ["faiss-cpu>=1.7.4"]
grpc = ["grpcio>=1.50", "protobuf>=4.21"]
all = ["milvus_lite[faiss,grpc]"]

[project.scripts]
milvus_lite-grpc = "milvus_lite.adapter.grpc.cli:main"
```

`pip install -e ".[dev,faiss,grpc]"` 安装完整开发环境。

---

## 13. 不在 Phase 10 范围

| 功能 | 推迟到 |
|---|---|
| TLS / mTLS 加密 | Future |
| Token / Username-Password 认证 | Future |
| RBAC / 多租户 | Future（嵌入式不必） |
| Backup / Restore RPC | Future |
| Bulk insert / Import | Future |
| Replica / Resource Group | Future（单进程不需要） |
| Aliases | Future |
| Hybrid search（多向量） | Future |
| Search iterator / pagination | Future（offset 参数加在 engine 即可） |
| Database 概念（多 db 实例） | Future |
| Async stream RPC（如 grpc client streaming） | Future |
| Sparse / Binary vector 类型 | Future |

---

## 14. 完成标志

- `python -m milvus_lite.adapter.grpc --data-dir ./data --port 19530` 能起 server
- pymilvus quickstart（第 11.2 节的脚本）从 `connect → create → insert → create_index → load → search → query → delete → release → drop` 一遍跑通
- recall parity 测试通过：grpc search 和直接 engine search 的 top-k 完全一致
- 错误码映射测试覆盖所有 MilvusLiteError 子类
- `examples/m10_demo.py` 是 README quickstart 的 1:1 对应
- 不支持的 RPC 返回 `UNIMPLEMENTED` + 友好消息（不 silent fail，不假装成功）
- 跑 `pytest tests/adapter/` 全绿
