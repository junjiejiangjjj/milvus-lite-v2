# 元数据设计

## 1. 概述

LiteVecDB 的元数据由两个 JSON 文件组成，分别管理**静态定义**和**运行时状态**：

| 文件 | 内容 | 生命周期 | 更新频率 |
|------|------|----------|----------|
| `schema.json` | Collection 的 schema 定义 | 创建时写入，之后不可变 | 一次 |
| `manifest.json` | 运行时状态（文件列表、计数器、索引配置） | 每次 flush / compaction 更新 | 频繁 |

两者都使用原子写入协议（write tmp → fsync → rename），确保崩溃安全。

### 磁盘位置

```
data_dir/
├── schema.json                # Collection schema（不可变）
├── manifest.json              # 当前运行时状态
├── manifest.json.prev         # 上一版本备份（单步回退）
└── ...
```

---

## 2. schema.json — Collection Schema

### 2.1 格式

```json
{
  "collection_name": "my_collection",
  "schema_format_version": 2,
  "version": 1,
  "enable_dynamic_field": true,
  "fields": [ ... ],
  "functions": [ ... ]
}
```

### 2.2 顶层字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `collection_name` | string | Collection 名称 |
| `schema_format_version` | int | schema 文件格式版本（当前 2） |
| `version` | int | schema 逻辑版本号 |
| `enable_dynamic_field` | bool | 是否启用动态字段（存入 `$meta` JSON 列） |
| `fields` | array | 字段定义列表 |
| `functions` | array | 函数定义列表（v2+，可选，BM25/TEXT_EMBEDDING/RERANK） |

### 2.3 Field 定义

每个 field 对象的完整结构：

```json
{
  "name": "vec",
  "dtype": "float_vector",
  "is_primary": false,
  "auto_id": false,
  "dim": 128,
  "max_length": null,
  "element_type": null,
  "max_capacity": null,
  "nullable": false,
  "default_value": null,
  "enable_analyzer": false,
  "analyzer_params": null,
  "enable_match": false,
  "is_function_output": false,
  "is_partition_key": false
}
```

| 字段 | 类型 | 含义 | 适用 dtype |
|------|------|------|------------|
| `name` | string | 字段名 | 所有 |
| `dtype` | string | 数据类型枚举值 | 所有 |
| `is_primary` | bool | 是否为主键 | int64, varchar |
| `auto_id` | bool | 是否自动生成主键值 | int64, varchar |
| `dim` | int\|null | 向量维度 | float_vector |
| `max_length` | int\|null | 最大字符长度 | varchar |
| `element_type` | string\|null | 数组元素类型枚举值 | array |
| `max_capacity` | int\|null | 数组最大容量 | array |
| `nullable` | bool | 是否允许 null | 所有（主键除外） |
| `default_value` | any\|null | 默认值 | 标量类型 |
| `enable_analyzer` | bool | 是否启用分词器 | varchar（FTS） |
| `analyzer_params` | dict\|null | 分词器配置 | varchar（FTS） |
| `enable_match` | bool | 是否支持 text_match 过滤 | varchar（FTS） |
| `is_function_output` | bool | 是否为 Function 的输出字段 | sparse_float_vector |
| `is_partition_key` | bool | 是否为 partition key | int64, varchar |

> FTS 相关属性（`enable_analyzer`、`analyzer_params`、`enable_match`、`is_function_output`、`is_partition_key`）仅在值为非默认时才写入文件，保持 v1 向后兼容。

### 2.4 支持的 DataType

| dtype 值 | 含义 | PyArrow 类型 |
|----------|------|-------------|
| `"bool"` | 布尔 | `pa.bool_()` |
| `"int8"` | 8 位整数 | `pa.int8()` |
| `"int16"` | 16 位整数 | `pa.int16()` |
| `"int32"` | 32 位整数 | `pa.int32()` |
| `"int64"` | 64 位整数 | `pa.int64()` |
| `"float"` | 单精度浮点 | `pa.float32()` |
| `"double"` | 双精度浮点 | `pa.float64()` |
| `"varchar"` | 变长字符串 | `pa.string()` |
| `"json"` | JSON 对象 | `pa.string()`（JSON 编码） |
| `"array"` | 变长数组 | 运行时根据 `element_type` 确定 |
| `"float_vector"` | 稠密浮点向量 | `pa.list_(pa.float32(), dim)` |
| `"sparse_float_vector"` | 稀疏浮点向量 | `pa.binary()`（packed uint32+float32 pairs） |

### 2.5 Function 定义

```json
{
  "name": "bm25_fn",
  "function_type": 1,
  "input_field_names": ["title"],
  "output_field_names": ["sparse_vec"],
  "params": {"analyzer_params": {"type": "standard"}}
}
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `name` | string | 函数名称 |
| `function_type` | int | 函数类型：1=BM25, 2=TEXT_EMBEDDING, 3=RERANK |
| `input_field_names` | string[] | 输入字段名列表 |
| `output_field_names` | string[] | 输出字段名列表 |
| `params` | dict | 函数参数 |

### 2.6 完整示例

```json
{
  "collection_name": "articles",
  "schema_format_version": 2,
  "version": 1,
  "enable_dynamic_field": true,
  "fields": [
    {
      "name": "id",
      "dtype": "int64",
      "is_primary": true,
      "auto_id": true,
      "dim": null,
      "max_length": null,
      "element_type": null,
      "max_capacity": null,
      "nullable": false,
      "default_value": null
    },
    {
      "name": "title",
      "dtype": "varchar",
      "is_primary": false,
      "auto_id": false,
      "dim": null,
      "max_length": 512,
      "element_type": null,
      "max_capacity": null,
      "nullable": false,
      "default_value": null,
      "enable_analyzer": true,
      "analyzer_params": {"type": "standard"},
      "enable_match": true
    },
    {
      "name": "embedding",
      "dtype": "float_vector",
      "is_primary": false,
      "auto_id": false,
      "dim": 768,
      "max_length": null,
      "element_type": null,
      "max_capacity": null,
      "nullable": false,
      "default_value": null
    },
    {
      "name": "tags",
      "dtype": "array",
      "is_primary": false,
      "auto_id": false,
      "dim": null,
      "max_length": null,
      "element_type": "varchar",
      "max_capacity": 50,
      "nullable": true,
      "default_value": null
    },
    {
      "name": "sparse_embedding",
      "dtype": "sparse_float_vector",
      "is_primary": false,
      "auto_id": false,
      "dim": null,
      "max_length": null,
      "element_type": null,
      "max_capacity": null,
      "nullable": false,
      "default_value": null,
      "is_function_output": true
    }
  ],
  "functions": [
    {
      "name": "title_bm25",
      "function_type": 1,
      "input_field_names": ["title"],
      "output_field_names": ["sparse_embedding"],
      "params": {}
    }
  ]
}
```

### 2.7 不可变性

schema.json 在 Collection 创建时写入，之后**永不修改**。这是一个设计决策：

- 避免 schema migration 的复杂性
- 保证所有 Parquet 文件使用同一 schema（否则旧文件需要 schema evolution）
- 与 Milvus MVP 行为一致

---

## 3. manifest.json — 运行时状态

### 3.1 格式

```json
{
  "manifest_format_version": 2,
  "version": 42,
  "current_seq": 15000,
  "schema_version": 1,
  "active_wal_number": 3,
  "partitions": {
    "_default": {
      "data_files": ["data/data_000100_000500.parquet"],
      "delta_files": ["delta/delta_000300_000300.parquet"]
    }
  },
  "index_specs": {
    "embedding": {
      "field_name": "embedding",
      "index_type": "HNSW",
      "metric_type": "COSINE",
      "build_params": {"M": 16, "efConstruction": 200},
      "search_params": {"ef": 64}
    }
  }
}
```

### 3.2 顶层字段

| 字段 | 类型 | 含义 | 更新时机 |
|------|------|------|----------|
| `manifest_format_version` | int | manifest 格式版本（当前 2） | 固定 |
| `version` | int | 单调递增版本号，每次 `save()` +1 | 每次 save |
| `current_seq` | int | 已 flush 到磁盘的最大 `_seq` | flush Step 5 |
| `schema_version` | int | 对应 schema.json 的 version | 创建时 |
| `active_wal_number` | int\|null | 当前正在写入的 WAL 编号 | flush Step 5 |
| `partitions` | dict | 每个 partition 的文件列表 | flush / compaction |
| `index_specs` | dict | 每个向量字段的索引配置 | create_index / drop_index |

### 3.3 partitions 结构

```json
{
  "_default": {
    "data_files": [
      "data/data_000001_000050.parquet",
      "data/data_000051_000100.parquet"
    ],
    "delta_files": [
      "delta/delta_000030_000030.parquet"
    ]
  },
  "user_partition": {
    "data_files": [],
    "delta_files": []
  }
}
```

- `_default` partition 始终存在，不可删除
- 文件路径是**相对路径**（相对于 `partitions/{partition_name}/`），使 `data_dir` 可重定位
- data_files 和 delta_files 是有序列表，新文件 append 到末尾
- compaction 会 remove 旧文件 + add 新文件

### 3.4 index_specs 结构

```json
{
  "embedding": {
    "field_name": "embedding",
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "build_params": {"M": 16, "efConstruction": 200},
    "search_params": {"ef": 64}
  },
  "sparse_vec": {
    "field_name": "sparse_vec",
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "BM25",
    "build_params": {},
    "search_params": {}
  }
}
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `field_name` | string | 向量字段名（与外层 key 一致） |
| `index_type` | string | 索引类型：BRUTE_FORCE / HNSW / IVF_FLAT / SPARSE_INVERTED_INDEX / AUTOINDEX |
| `metric_type` | string | 距离类型：COSINE / L2 / IP / BM25 |
| `build_params` | dict | 构建参数（如 HNSW 的 M、efConstruction） |
| `search_params` | dict | 搜索参数（如 HNSW 的 ef），RPC 可按请求覆盖 |

**格式演进**：v1 manifest 使用单数 `index_spec`（单索引），v2 改为复数 `index_specs`（多索引 dict）。加载时自动兼容：读到 `index_spec` 会迁移到 `index_specs`，下次 `save()` 透明升级。

### 3.5 各字段的语义与用途

#### version

单调递增。每次 `save()` 加 1。用于变更检测和调试。版本号仅在 rename 成功后才在内存中更新——如果 save 失败，版本号不变，不会出现跳号。

#### current_seq

已持久化到 Parquet 文件的最大 `_seq` 值。重启时 Collection 从此值恢复 `_seq` 计数器，保证新分配的 seq 不与已持久化的冲突。

```
重启恢复: Collection._seq = manifest.current_seq
         → 后续 _alloc_seq() 从 current_seq + 1 开始
```

#### active_wal_number

当前正在写入的 WAL 文件编号。Recovery 用它定位需要重放的 WAL：

```
recovery: 扫描 wal/ 目录 → 找到所有 WAL 文件 → 重放
          next_wal_number = max(found, active_wal_number) + 1
```

#### partitions

Manifest 是文件列表的 **single source of truth**：
- 在 manifest 中但不在磁盘上 → recovery 警告（数据可能丢失）
- 在磁盘上但不在 manifest 中 → recovery 清理（orphan 文件）

---

## 4. 原子写入协议

两个元数据文件都使用相同的原子写入模式：

### 4.1 schema.json 写入

```
1. json.dump → schema.json.tmp
2. fsync(tmp)
3. os.rename(tmp, schema.json)
```

只写一次，无需 .prev 备份。

### 4.2 manifest.json 写入

```
1. 构建 payload（version + 1，但不更新内存状态）
2. json.dump → manifest.json.tmp + fsync
3. shutil.copy2(manifest.json → manifest.json.prev)  ← best-effort
4. os.rename(manifest.json.tmp → manifest.json)       ← commit point
5. 更新内存 _version（仅 rename 成功后）
6. fsync(data_dir)                                     ← best-effort
```

### 4.3 崩溃安全

| 崩溃时机 | 磁盘状态 | 恢复 |
|----------|----------|------|
| Step 2（写 tmp）| tmp 不完整或不存在，manifest.json 不变 | tmp 被清理，使用旧 manifest |
| Step 3（备份 prev）| tmp 已 fsync，prev 可能损坏 | 非致命，继续 rename |
| Step 4（rename）| 原子操作，要么旧要么新 | 内核保证 |
| Step 6（fsync dir）| rename 已完成，dir fsync 失败 | best-effort，极端情况下重启可回退到 .prev |

### 4.4 加载回退链

```
load(data_dir):
  1. 尝试 manifest.json → 成功则返回
  2. manifest.json 损坏 → 尝试 manifest.json.prev（warning）
  3. 两者都损坏 → ManifestCorruptedError
  4. 两者都不存在 → 返回全新 Manifest
```

---

## 5. 元数据与其他组件的关系

```
                schema.json                    manifest.json
                (不可变)                         (频繁更新)
                    │                               │
        ┌───────────┼───────────┐       ┌───────────┼───────────┐
        ▼           ▼           ▼       ▼           ▼           ▼
   FieldSchema  DataType    Function  文件列表   current_seq  index_specs
        │        mapping       │     (data/delta)     │           │
        ▼           ▼         ▼       ▼           ▼           ▼
   Arrow Schema  PyArrow   BM25/   Segment     WAL编号     VectorIndex
   (4种变体)     类型映射   FTS     Cache       恢复起点     构建配置
```

### schema.json 的消费者

| 组件 | 用途 |
|------|------|
| `arrow_builder.py` | 构建 4 种 Arrow Schema（wal_data/wal_delta/data/delta） |
| `MemTable` | 写入验证、flush 时构建 per-partition 表 |
| `Collection` | 创建时持久化，重启时加载验证 |
| `gRPC adapter` | Milvus schema ↔ LiteVecDB schema 双向翻译 |

### manifest.json 的消费者

| 组件 | 用途 |
|------|------|
| `Collection` | 启动时恢复 `_seq`、加载 Segment Cache |
| `Recovery` | 定位 WAL、验证文件完整性、清理 orphan |
| `Flush` | 注册新写入的 data/delta 文件 |
| `Compaction` | 读取/更新文件列表、tombstone GC 阈值计算 |
| `Index lifecycle` | 持久化 IndexSpec、加载时恢复索引配置 |
