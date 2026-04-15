# LiteVecDB vs Milvus Feature Comparison

## Data Types

| Data Type | LiteVecDB | Milvus |
|---|:---:|:---:|
| Bool | Y | Y |
| Int8 / Int16 / Int32 / Int64 | Y | Y |
| Float / Double | Y | Y |
| VarChar | Y | Y |
| JSON | Y | Y |
| Array | Y | Y |
| FLOAT_VECTOR | Y | Y |
| SPARSE_FLOAT_VECTOR | Y | Y |
| BinaryVector | - | Y |
| Float16Vector | - | Y |
| BFloat16Vector | - | Y |

## Index Types

| Index Type | LiteVecDB | Milvus |
|---|:---:|:---:|
| FLAT / BRUTE_FORCE | Y | Y |
| HNSW | Y (FAISS) | Y |
| IVF_FLAT | Y (FAISS) | Y |
| IVF_SQ8 | - | Y |
| IVF_PQ | - | Y |
| SCANN | - | Y |
| DISKANN | - | Y |
| AUTOINDEX | Y | Y |
| SPARSE_INVERTED_INDEX | Y | Y |
| SPARSE_WAND | - | Y |
| BIN_FLAT / BIN_IVF_FLAT | - | Y |
| GPU_IVF_FLAT / GPU_IVF_PQ / GPU_CAGRA | - | Y |
| Scalar index (INVERTED / BITMAP / Trie) | - | Y |

## Metric Types

| Metric | LiteVecDB | Milvus |
|---|:---:|:---:|
| COSINE | Y | Y |
| L2 | Y | Y |
| IP (Inner Product) | Y | Y |
| BM25 | Y | Y |
| JACCARD / HAMMING | - | Y |

## Search Features

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| Top-K ANN search | Y | Y |
| Dense vector search | Y | Y |
| Sparse vector search (BM25) | Y | Y |
| Hybrid search (multi-vector + reranker) | Y | Y |
| WeightedRanker | Y | Y |
| RRFRanker | Y | Y |
| Scalar filter expressions | Y | Y |
| Range search (radius / range_filter) | Y | Y |
| Group-by search | Y | Y |
| Offset / pagination | Y | Y |
| Search iterator | Y (v1 fallback) | Y (v1 + v2) |
| Query iterator | Y | Y |
| Multi-query (nq > 1) | Y | Y |
| Text match (tokenized keyword filter) | Y | Y |

## Filter Expressions

| Operator | LiteVecDB | Milvus |
|---|:---:|:---:|
| Comparison (==, !=, <, <=, >, >=) | Y | Y |
| Logical (and, or, not) | Y | Y |
| IN / NOT IN | Y | Y |
| LIKE (% and _ wildcards) | Y | Y |
| IS NULL / IS NOT NULL | Y | Y |
| Arithmetic (+, -, *, /) | Y | Y |
| $meta["key"] (dynamic field access) | Y | Y |
| Bare dynamic field name (auto-rewrite) | Y | Y |
| JSON field path access | Y | Y |
| text_match() | Y | Y |
| array_contains / array_contains_all / array_contains_any | Y | Y |
| array_length() | Y | Y |
| Array index access (field[0]) | Y | Y |

## Schema & Field Features

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| Primary key (INT64 / VARCHAR) | Y | Y |
| Auto ID | Y | Y |
| Nullable fields | Y | Y |
| Default values | Y | Y |
| Dynamic fields (enable_dynamic_field) | Y | Y |
| Schema functions (BM25) | Y | Y |
| Schema functions (TEXT_EMBEDDING) | Y | Y |
| Schema functions (RERANK) | Y | Y |
| Partition key | Y (auto-bucket routing) | Y |
| Clustering key | - | Y |
| Field-level analyzer config | Y | Y |

## Collection Management

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| Create / Drop / Has / Describe / List | Y | Y |
| Rename collection | Y | Y |
| Collection aliases | - | Y |
| Collection TTL | - | Y |
| Alter collection | - | Y |

## Data Operations

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| Insert | Y | Y |
| Upsert (full replace) | Y | Y |
| Upsert (partial update) | Y | - |
| Delete (by PK) | Y | Y |
| Delete (by filter expression) | Y | Y |
| Get (point read by PK) | Y | Y |
| Query (scalar filter) | Y | Y |
| Flush | Y | Y |
| Bulk insert (file import) | - | Y |

## Index & Load Lifecycle

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| Create / Drop / Describe index | Y | Y |
| Load / Release collection | Y | Y |
| Load state query | Y | Y |
| Load progress query | Y | Y |
| Partition-level load/release | - | Y |
| Multiple replicas | - | Y |

## Partition Support

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| Create / Drop / Has / List partitions | Y | Y |
| Per-partition insert / search / query | Y | Y |
| Partition-level load/release | - | Y |
| Partition key (auto-routing) | - | Y |

## Text Search & Analysis

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| BM25 full text search | Y | Y |
| Standard analyzer (regex tokenizer) | Y | Y |
| Jieba analyzer (Chinese) | Y | Y |
| Text embedding (auto text-to-vector) | Y | Y |
| Semantic reranking (Cohere) | Y | - |
| Decay reranking (gauss/exp/linear) | Y | - |

## Enterprise / Operations

| Feature | LiteVecDB | Milvus |
|---|:---:|:---:|
| Multi-database | - | Y |
| RBAC (users, roles, privileges) | - | Y |
| CDC (cross-cluster replication) | - | Y |
| Backup / Restore | - | Y |
| Resource groups | - | Y |
| Consistency levels | - | Y (Strong/Bounded/Session/Eventually) |
| MMap (memory-mapped storage) | - | Y |
| GPU acceleration | - | Y |
| RESTful API | - | Y |
| Web GUI (Attu) | - | Y |
| Clustering compaction | - | Y |

## Architecture

| Aspect | LiteVecDB | Milvus |
|---|---|---|
| Language | Pure Python | C++/Go |
| Deployment | Embedded / single-process | Distributed (standalone or cluster) |
| Storage | WAL (Arrow IPC) + Parquet | MinIO/S3 + RocksMQ/Kafka/Pulsar |
| Metadata | JSON manifest | etcd |
| Process model | Single writer per data_dir | Multi-node, multi-replica |
| Platform | Any (macOS/Linux/Windows) | Linux (macOS for dev) |
| Install size | ~50MB (pip) | ~200MB+ (binary / Docker) |
| Max recommended scale | < 1M vectors | Billions |

## Summary

**LiteVecDB 的优势：**
- 纯 Python，跨平台（Windows/macOS/Linux），安装简单
- 支持 partial update upsert（Milvus 不支持）
- 内置语义 reranking（Cohere）和 decay reranking
- 代码可读、可调试、可扩展
- 启动快（0.07s vs 0.57s）

**Milvus 的优势：**
- 分布式架构，支持十亿级向量
- 更多索引类型（DISKANN, IVF_PQ, IVF_SQ8, SCANN, GPU 系列）
- 更多向量类型（BinaryVector, Float16Vector, BFloat16Vector）
- 企业级功能（RBAC, CDC, 备份恢复, 多副本, 一致性级别）
- GPU 加速
- Scalar index（INVERTED, BITMAP, Trie）
- RESTful API + Web GUI

**定位：** LiteVecDB 面向原型开发、测试、小规模应用（<1M 向量），作为 Milvus 的轻量替代。功能覆盖了 pymilvus 常用 API 的 90%+，是开发阶段的理想选择。
