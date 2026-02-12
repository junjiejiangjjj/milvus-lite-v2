# 向量数据库存储设计调研

## 1. LanceDB

### 1.1 概述

LanceDB 是一个嵌入式向量数据库，核心是自研的 **Lance 列式文件格式**（VLDB 2025 论文）。整体架构借鉴 Apache Iceberg / Delta Lake 的表格式设计，走的是**数据湖格式**路线，没有 WAL 和 MemTable。

### 1.2 Lance 文件格式

Lance 格式的核心创新是**自适应结构编码（Adaptive Structural Encoding）**，根据数据宽度自动选择编码方式：

| 数据类型 | 编码方式 | 说明 |
|---------|---------|------|
| 大类型（≥128B，如向量） | **Full Zip** | 值缓冲区转置为行主序，"拉链式"合并为单缓冲区，随机访问仅需 1 次 IOP |
| 小类型（<128B，如标量） | **Mini-Block** | 2 的幂次大小的块（压缩后 4-8 KiB），支持不透明压缩算法 |

每个 `.lance` 文件的物理布局：

```
┌─────────────────────────┐
│     Data Pages          │  ← 变长数据页（推荐 8MB 每页）
├─────────────────────────┤
│     Column Metadatas    │  ← Protobuf 编码的列元数据
├─────────────────────────┤
│     Offset Tables       │  ← 偏移量表
├─────────────────────────┤
│     Global Buffers      │  ← 全局缓冲区（schema、索引、统计信息）
├─────────────────────────┤
│     Fixed Footer (32B)  │  ← 固定 32 字节的文件尾（含 "LANC" 魔数）
└─────────────────────────┘
```

读取元数据只需 **1-2 次 I/O**：先读 Footer，解析后确定元数据大小，再读取。

#### Lance vs Parquet 关键差异

| 维度 | Parquet | Lance v2 |
|------|---------|----------|
| Row Group | 必须有 | **完全消除**，文件即页的集合 |
| 编码 | 内建固定编码 | **插件式编码**，完全可扩展 |
| 页相邻性 | 行组内页必须相邻 | 页可以非连续存储 |
| 随机访问 | 默认极慢（~5,500 行/秒） | 无需特殊配置即可高效（~350K 行/秒） |
| 全扫描 | 优秀 | 9 项测试中 7 项超越 Parquet（1.3-2.0 倍） |
| 格式规范 | 数百行 | **不到 50 行 Protobuf** 定义 |

关键洞察：Parquet 经过精心调参可达到 60x 随机访问提升，但牺牲扫描性能和 RAM。Lance 在不做任何权衡的情况下达到同等或更好性能。

### 1.3 存储架构：Fragment + Manifest

```
my_table.lance/
├── _versions/              ← Manifest（每个版本一个，Protobuf 序列化）
│   ├── 1.manifest
│   ├── 2.manifest
├── _deletions/             ← 删除文件（按 Fragment 粒度）
│   ├── {version}-{fragment_id}.arrow
├── _indices/               ← 向量索引（UUID 命名的子目录）
├── _transactions/          ← 事务文件（OCC 乐观并发）
│   ├── {read_version}-{uuid}.txn
└── data/
    └── *.lance             ← 不可变数据文件
```

核心概念：

- **Fragment**：数据的水平分区（不可变，默认最大 ~90GB）。包含一个或多个 DataFile，各自存储不同的列子集。
- **Manifest**：某个版本的完整快照描述（Protobuf），包含 schema + fragment 列表 + 索引元数据。版本号单调递增。
- **DataFile**：Lance 格式文件，字段通过整数 ID 映射，-2 表示 tombstone 已删除，-1 未分配，≥0 有效字段。

### 1.4 CRUD 机制

| 操作 | 实现方式 |
|------|---------|
| **Insert** | 创建新 Fragment + 新 Manifest（追加，不改旧文件） |
| **Delete** | 在 `_deletions/` 写删除文件，两种格式：**ARROW_ARRAY**（稀疏删除）和 **BITMAP**（Roaring Bitmap，密集删除） |
| **Update** | = Delete 旧行 + Insert 新行 |
| **Upsert** | 存在则更新，不存在则插入。支持无冲突并发 upsert |

注意：**没有 WAL 和 MemTable**，直接写文件。适合批量写入，不适合高频单行写。

### 1.5 MVCC 版本控制

```
v1 ──── v2 ──── v3 ──── v4
│        │        │        │
新数据   新数据   删除     新数据
manifest manifest manifest manifest
```

- 每次操作产生新版本 Manifest，旧版本保留（默认 30 天）
- 并发写入用**乐观并发控制（OCC）**：先写数据文件，再竞争提交 Manifest。冲突时递增版本号重试
- 天然支持时间旅行查询（读旧版本 Manifest 即可）

### 1.6 Schema 演进

- 字段在创建时获得唯一整数 ID，新增列递增分配
- **Add Column**：在每个 Fragment 中追加新 DataFile（不重写旧文件），这是 Fragment 设计的一大优势
- **Alter Column**：重命名/变更可空性仅更新元数据；变更数据类型需重写
- **Drop Column**：字段 ID 标记为 tombstone（-2），compaction 时物理清除。不可撤销
- 缺失字段读取时返回 null

### 1.7 向量索引

- **磁盘优先**设计：得益于 Lance 的随机访问性能，索引可直接从磁盘查询
- **IVF-PQ**：倒排文件索引 + 乘积量化（~128x 内存需求降低）
- **IVF-HNSW**：在每个 IVF 分区内构建**子 HNSW 图**（不是全局单一 HNSW 图），更适合磁盘存储
- 行更新/删除后从索引中移出，需重建索引恢复最优性能

### 1.8 Compaction

- 将小 Fragment 合并为大 Fragment，减少文件数和元数据开销
- 物理清除已软删除的数据
- 建议保持 Fragment 数量在 100 以下
- 旧版本需显式 cleanup 操作回收空间

---

## 2. Turbopuffer

### 2.1 概述

Turbopuffer 是一个**对象存储优先（Object Storage-First）**的无服务器搜索引擎，由前 Shopify 基础设施负责人 Simon Eskildsen 创建。核心理念：**S3/GCS 是唯一的持久化状态，计算节点完全无状态**。

当前规模：2.5T+ 文档索引，10M+ 写入/秒，10k+ 查询/秒，自上线以来 99.99% 可用性。

### 2.2 三层存储架构

```
客户端请求
    │
    ▼
Rust 无状态计算集群（可跑 Spot 实例）
    │
    ├── RAM 缓存（热）      ~$5/GB      < 10ms
    ├── NVMe SSD 缓存（温）  ~$0.10/GB   ~10-50ms
    └── S3/GCS（冷）         ~$0.02/GB   ~200-500ms
```

- 传统方案 ~$3,600/TB/月 vs Turbopuffer ~$70/TB/月，**50x 成本差距**
- 节点完全无状态，任何节点可服务任何 namespace
- 节点故障后另一个节点一次冷查询（~500ms）即可接管
- 大量计算跑在 Spot 实例上

**"Pufferfish 效应"**：数据根据访问频率自动从 S3 "膨胀" 到 SSD 再到 RAM，类似 JIT 编译器。

### 2.3 存储设计

```
s3://bucket/namespace_a/
  ├── WAL/
  │   ├── 0000000001.bin     ← 顺序追加
  │   └── 0000000002.bin
  ├── index/                 ← SPFresh 聚类索引
  └── data/                  ← 数据向量
```

关键设计点：

- **每个租户独立 namespace 前缀**，不活跃时存储成本趋近于零
- **WAL 写入 S3**，用 S3 **条件写入**（`if-none-match: *`）防并发冲突，无需外部事务数据库
- 每个 namespace 限制每秒写入 1 个 WAL 条目，同秒并发写入批量合并
- 序列化格式用 **rkyv（零拷贝反序列化）**，磁盘字节与内存表示完全一致，相比 Parquet 解压解码可提供 **10x 查询加速**
- 写入延迟 ~285ms（p50），接受高延迟换取极低成本

**核心优化目标**：最小化 S3 API 调用次数（GET 请求 $12/百万次），而非最小化存储容量。

### 2.4 向量索引：SPFresh（放弃 HNSW）

这是 turbopuffer 最逆向的决策。HNSW/DiskANN 等图索引需要十几次随机读遍历图，每次 S3 round-trip ~100-250ms，总延迟数秒不可接受。

选择 **SPFresh**（SOSP 2023），基于质心聚类的层次化 ANN 索引：

```
      质心 L3 (DRAM)          ← 极少数据，常驻内存
       / | \
    质心 L2 (DRAM)            ← 量化后常驻内存
     / | \
   簇 L1 (SSD/S3)            ← posting list
    ↓
  数据向量 L0 (SSD/S3)        ← 全精度向量
```

**分支因子 ~100**，精确匹配 DRAM/SSD 容量比（10x-50x）。含义：如果数据向量能放入 SSD，那么质心向量就能放入 DRAM。1000 亿向量只需 ~3 层质心。

**冷查询最多 2-3 次 S3 round-trip**（vs HNSW 十几次）。

#### RaBitQ 向量量化

- 每维度压缩到 **1 bit**，实现 **32x 压缩**
- 计算距离估计区间，**不到 1% 的向量**需要全精度重排序
- 将瓶颈从带宽转为计算（64x 算术强度提升）
- AVX-512 `VPOPCNTDQ` 指令优化

**目标指标**（ANN v3）：1000 亿向量，200ms p99 延迟，1000+ QPS，92% 召回率。

#### 异步索引构建

```
查询结果 = merge(
    indexed_data   → 快速聚类查找 (~10ms),
    unindexed_data → 穷举扫描最近 WAL (~200ms)
)
```

索引更新与写入/查询异步进行，不阻塞。SPFresh 使用 **LIRE 增量更新**，动态拆分/合并簇，仅需 1% DRAM 和不到 10% CPU。

### 2.5 CRUD 机制

| 操作 | 实现 |
|------|------|
| **Upsert** | 写入 WAL（S3）→ 异步合并到索引 |
| **Patch** | WAL 记录部分更新，重放时合并（向量字段不支持 patch） |
| **Delete** | WAL 写入 tombstone（快速，不重写数据） |
| **条件写入** | 原子性评估条件 + 写入，Serializable 隔离级别 |
| **delete_by_filter** | 先查询匹配文档，再原子删除（Read Committed 隔离） |

写入 payload 上限 512MB，文档 ID 支持 64 位整数、UUID 或 64 字节字符串。

### 2.6 Native Filtering（不是前过滤也不是后过滤）

传统后过滤召回率可能趋近 0%，前过滤 O(dim × matches) 太慢。Turbopuffer 的做法是**向量索引和属性索引协同工作**：

- 每个文档有地址 `{cluster_id}:{local_id}`
- 属性索引：`(attribute_value, cluster_id) → Set<local_id>`（压缩为 bitmap）
- 查询时：找到包含匹配文档的最近簇 → 只评估那些簇中的匹配候选
- 不匹配的簇直接跳过
- 性能：~25ms 达到 90% 召回率（vs 无过滤 ~20ms）

属性索引用 LSM-Tree 存储，支持部分更新。

### 2.7 一致性与持久性

- 写入成功返回时数据**已持久化到 S3**
- 依赖 S3 强一致性（2020）+ S3 条件写入 CAS（2023.12）
- **无需任何共识协议或事务数据库**
- 超过 99.99% 查询返回一致数据
- CAP 定理中选择 **CP**（S3 不可达时优先一致性）
- 支持可选最终一致性模式获取 sub-10ms 延迟

### 2.8 缓存管理

- 首次查询从 S3 加载（p50 = 343ms / 1M 文档）
- 后续查询命中 SSD/RAM 缓存（p50 = 8ms / 1M 文档）
- 不活跃 namespace 数据从缓存驱逐，存储成本趋近于零
- 提供 **Warm Cache Hint** API：应用可提前触发缓存预热（Cursor 在用户打开代码库时触发，Notion 在用户打开搜索对话框时触发）

---

## 3. Milvus Partition 设计

### 3.1 数据层级

Milvus 的数据组织为严格的层次结构：**Database → Collection → Partition → Segment → Field Binlogs**。

```
Database
  └── Collection (schema 定义)
        ├── Partition "_default"
        │     ├── Segment 001 (Growing)
        │     ├── Segment 002 (Sealed)
        │     └── Segment 003 (Flushed)
        ├── Partition "2024_Q1"
        │     ├── Segment 004
        │     └── Segment 005
        └── Partition "2024_Q2"
              └── Segment 006
```

- **Segment 只属于一个 Partition**（N:1 关系），是最小的存储调度单位
- 一个 Partition 内可有多个 Segment（Growing + 多个 Sealed/Flushed）
- Segment 在创建时绑定 `(collectionID, partitionID, channelName)`，终生不变

### 4.2 对象存储路径

```
{rootPath}/{logType}/{collectionID}/{partitionID}/{segmentID}/{fieldID}/{logID}
```

| Log Type | 路径示例 |
|----------|---------|
| Insert Binlog | `root/insert_log/100/200/300/1/400` |
| Delta Log | `root/delta_log/100/200/300/400`（无 fieldID） |
| Stats Log | `root/stats_log/100/200/300/1/400` |

Partition 在路径中是独立的一层，Segment 挂在 Partition 下。

### 4.3 写入路径中的 Partition

#### Channel 共享

DML Channel（vChannel）是 **Collection 级**的，不是 Partition 级的。所有 Partition 共享同一组 vChannel：

```
Proxy                          vChannel (Collection 级)              DataNode
  │                                    │                                │
  │─ 确定目标 Partition ──→              │                                │
  │  (显式指定/PartitionKey/默认)        │                                │
  │                                    │                                │
  │─ 按 PK 哈希分配 Channel ──→          │                                │
  │                                    │                                │
  │─ 写入 WAL (Channel) ──────────────→ │                                │
  │                                    │ ─ 消费消息 ─────────────────→   │
  │                                    │   按 (partitionID, channelName) │
  │                                    │   路由到对应 Growing Segment     │
```

#### Partition 路由方式

| 模式 | 路由逻辑 |
|------|---------|
| **显式指定** | 用户在 insert 时传入 partition_name |
| **Partition Key** | 从指定字段提取值，哈希路由到自动创建的分区 |
| **默认** | 未指定且无 Partition Key → `_default` |

### 4.4 删除路径中的 Partition

Delta Log 存储在 **Segment 级**，路径含 partitionID：

```
delta_log/{collectionID}/{partitionID}/{segmentID}/{logID}
```

**L0 Segment（2.4+）**：专门存放删除记录的 Segment，解耦删除与具体 Segment 的绑定：

```
写入删除: delete(pk) → L0 Segment（全局删除缓冲，不关心 pk 在哪个 Segment）
L0 Compaction: 读 L0 → Bloom Filter 匹配 → 分发到具体 Segment 的 Delta Log
```

好处：写入时不需要查找 PK 所在的 Segment，由后台 Compaction 处理分发。

### 4.5 Drop Partition

Drop Partition 是**异步多阶段过程**，不需要 Compaction：

```
DropPartition("2024_Q1")
  │
  ├─ Phase 1: 广播 DropPartitionMessage → 停止写入
  ├─ Phase 2: 标记该 Partition 下所有 Segment 为 Dropped + 删除元数据
  └─ Phase 3: GC 异步清理 Segment 文件（直接删文件，不需 Compaction）
```

关键：因为 Segment 独立属于 Partition（不跨 Partition），可以直接按 Segment 粒度删除，无需重写其他数据。`_default` Partition 不可删除。

### 3.6 Search 中的 Partition Pruning

```
Search(vector, partition_names=["2024_Q1"])
  │
  ├─ Proxy: 解析 partition_names → partition IDs
  ├─ QueryNode: 只加载/扫描属于目标 Partition 的 Segment
  │     └─ 不属于目标 Partition 的 Segment 完全跳过（文件级剪枝）
  └─ Segment Pruner: 在目标 Partition 内按统计信息进一步剪枝
```

### 3.7 Partition Key（自动分区）

当 Collection Schema 中某个标量字段标记 `is_partition_key=True` 时，Milvus 自动按该字段的哈希值路由数据到不同 Partition：

```python
# Schema 定义
FieldSchema("tenant_id", DataType.VARCHAR, is_partition_key=True)

# 自动创建 16 个分区: _default_0, _default_1, ..., _default_15
# 路由算法:
#   VarChar → CRC32 hash → hash % 16 → 目标 Partition
#   Int64   → Murmur3 hash → hash % 16 → 目标 Partition
```

约束：
- 只有 Int64 和 VarChar 可作为 Partition Key
- Partition Key 和手动分区互斥
- 默认 16 个分区，最多 4096 个

### 3.8 关键设计决策总结

| 设计决策 | Milvus 做法 | 理由 |
|---------|------------|------|
| WAL/Channel 粒度 | Collection 级共享 | 减少 Channel 数量，简化管理 |
| Segment 归属 | 只属于一个 Partition | Drop Partition 时可直接删除，不影响其他 Partition |
| Delta Log 粒度 | Segment 级（+ L0 全局缓冲） | 删除时不需要查找目标 Segment，L0 Compaction 后台分发 |
| Drop Partition | 异步 GC，不需 Compaction | Segment 独立归属保证了这一点 |
| _default Partition | 自动创建，不可删除 | 保证所有数据有归属 |
| Partition Key | 哈希自动路由 | 用户无需手动管理分区 |

---

## 4. 对比总结

### 4.1 架构对比

| 维度 | LanceDB | Turbopuffer | Milvus | LiteVecDB (我们) |
|------|---------|-------------|--------|-----------------|
| **定位** | 嵌入式/数据湖 | 云原生 Serverless | 分布式向量数据库 | 本地嵌入式 |
| **存储后端** | 本地/对象存储 | 对象存储（唯一） | 对象存储 + etcd + MQ | 本地文件系统 |
| **文件格式** | Lance（自研） | rkyv（零拷贝） | Binlog（per-field） | Parquet |
| **写入路径** | 直接写 Fragment | WAL → S3（异步索引） | WAL → MemTable → Flush | WAL → MemTable → Flush |
| **删除方式** | 独立删除文件（Bitmap/Arrow） | WAL tombstone | Delta Binlog + Bloom Filter | Delta Log（独立 Parquet） |
| **版本控制** | 原生 MVCC（Manifest/版本） | S3 条件写入 | TSO 时间戳 | _seq 全局序号 |
| **向量索引** | IVF-PQ / IVF-HNSW | SPFresh 聚类树 + RaBitQ | HNSW / IVF / DiskANN | Brute-force → FAISS |
| **过滤方式** | 标准前/后过滤 | Native Filtering（协同） | Bitmap + 谓词下推 | Bitmap 管线（预留） |
| **Compaction** | Fragment 合并 | 面向 S3 的 LSM | Segment 合并 + Delta 清理 | Size-Tiered |
| **Schema** | 字段 ID + schema evolution | 属性自动索引 | CollectionSchema + FieldSchema | Collection Schema（对齐 Milvus） |
| **API 风格** | Python API，无严格对标 | REST + gRPC | MilvusClient / ORM 双层 | 内部引擎 API + gRPC 适配层（后续，pymilvus 兼容） |

### 4.2 删除机制对比

三者殊途同归：**独立的删除记录 + 不可变数据文件 + 后台清理**。

| 系统 | 删除记录存储 | 读取时过滤 | 物理清除 |
|------|------------|-----------|---------|
| LanceDB | `_deletions/` 目录下按 Fragment 的 Arrow/Roaring Bitmap 文件 | 查询时跳过删除行 | Compaction 合并 Fragment |
| Turbopuffer | WAL 中的 tombstone 记录 | 重放 WAL 时应用 | Compaction（面向 S3） |
| Milvus | Delta Binlog（独立文件） + Bloom Filter 匹配 | 内存 deleted set | Compaction（Delta 行数 > 20% 或 > 10MB） |
| LiteVecDB | Delta Log（独立 Parquet） + 内存 deleted_map | O(1) dict 查找 | Size-Tiered Compaction |

### 4.3 索引选择由存储层级决定

| 存储层级 | 延迟特征 | 适合的索引类型 | 代表系统 |
|---------|---------|--------------|---------|
| DRAM | ns 级 | HNSW（多次随机跳转无问题） | Milvus（内存模式） |
| NVMe SSD | μs 级 | HNSW / IVF-PQ / DiskANN | LanceDB、Milvus |
| 对象存储 (S3) | 100-250ms | **SPFresh 聚类树**（最少 round-trip） | Turbopuffer |
| 本地文件系统 | μs-ms 级 | IVF-PQ / HNSW / Brute-force (小数据) | LiteVecDB |

关键洞察：Turbopuffer 放弃 HNSW 选择 SPFresh 不是因为 HNSW 不好，而是因为对象存储的 round-trip 延迟使图遍历不可行。**索引结构应该匹配存储介质的延迟特征**。

### 4.4 写入路径对比

| 系统 | 路径 | 写入延迟 | 适合场景 |
|------|------|---------|---------|
| LanceDB | 直接写新 Fragment + 新 Manifest | 取决于文件大小 | 批量写入 |
| Turbopuffer | WAL → S3 | ~285ms (p50) | 高吞吐、可接受延迟 |
| Milvus | WAL → Growing Segment → Flush | 低延迟 | 高频实时写入 |
| LiteVecDB | WAL → MemTable → Flush | 低延迟 | 实时小写入 |

LanceDB 没有 WAL/MemTable 中间层，适合批量操作。Turbopuffer 接受高写入延迟换取极低存储成本。Milvus 和 LiteVecDB 走经典 LSM-Tree 路径，适合实时写入。

### 4.5 Partition / 数据分区对比

| 维度 | LanceDB | Turbopuffer | Milvus | LiteVecDB (我们) |
|------|---------|-------------|--------|-----------------|
| **分区概念** | Fragment（水平分区，非用户可控） | Namespace（租户级） | Partition（用户或自动） | Partition（用户手动） |
| **数据文件隔离** | Fragment 即隔离单元 | Namespace 级隔离 | Segment 只属于一个 Partition | 按 Partition 目录隔离 |
| **WAL 粒度** | 无 WAL | 每 Namespace 独立 WAL | Collection 级共享 Channel | Collection 级共享 WAL |
| **Drop Partition** | N/A（不支持用户分区） | 删除 Namespace 前缀 | 异步 GC，标记 Segment Dropped | 删子目录 + 更新 Manifest |
| **Search 剪枝** | Fragment 统计信息剪枝 | Namespace 级隔离 | 按 Partition 跳过 Segment | 按 Partition 跳过文件 |
| **自动分区** | 不支持 | 不支持 | Partition Key（哈希路由） | 后续支持 |
| **默认分区** | N/A | N/A | `_default`（不可删除） | `_default`（不可删除） |

**关键启示**：Milvus 的 "WAL/Channel Collection 级共享 + 数据文件 Partition 级隔离" 模式非常适合 LiteVecDB 的场景——共享 WAL 减少 IO 和内存开销，Partition 目录隔离保证 Drop Partition 和 Search Pruning 的高效性。

### 4.6 对 LiteVecDB 设计的启示

1. **删除设计已验证**：独立删除记录 + 不可变数据文件是行业共识，我们的 Delta Log 方案与三家一致
2. **Bitmap 管线方向正确**：Turbopuffer 的 Native Filtering 证明了向量索引和过滤协同的价值，我们的 bitmap 管线为此预留了扩展点
3. **Schema 对齐 Milvus 是合理的**：LanceDB 用字段 ID，Milvus 用 FieldSchema，都支持 schema evolution。我们的 Collection Schema 模型兼顾了类型安全和演进能力
4. **Partition 设计对齐 Milvus**：WAL/MemTable 在 Collection 级共享，数据文件按 Partition 目录隔离。Drop Partition = 删目录 + 更新 Manifest，Search 按 Partition 跳过无关文件。WAL 中通过 `_partition` 列记录归属，Parquet 通过目录隔离体现
5. **两层架构**：内部引擎 API 面向实现优化（`insert(records)` / `delete(pks)` / `get(pks)` / `search(vectors)`，输入始终 List）；后续加 gRPC 适配层实现 pymilvus 兼容（参数规范化、表达式解析、Milvus 协议返回值包装）。内部 insert 天然 upsert 语义（PK 唯一），delete 支持全局删除（partition_name=None）
6. **未来索引选择**：本地文件系统场景下 IVF-PQ 或 HNSW 都可行（不像 Turbopuffer 受限于 S3 延迟），FAISS 是合理的下一步
6. **Parquet 的局限性**：Lance 的出现说明 Parquet 在随机访问上确实有瓶颈，但对 MVP 阶段（brute-force 全扫描）影响不大，未来可评估是否需要更优格式

---

## 参考资料

### LanceDB / Lance

- [Lance: Efficient Random Access in Columnar Storage (VLDB 2025)](https://arxiv.org/html/2504.15247v1)
- [Lance File Format Specification](https://lance.org/format/file/)
- [Lance Table Format Specification](https://lance.org/format/table/)
- [Lance v2: A New Columnar Container Format](https://lancedb.com/blog/lance-v2/)
- [LanceDB Data Management Guide](https://lancedb.com/documentation/concepts/data.html)
- [Schema and Data Evolution - LanceDB](https://docs.lancedb.com/tables/schema)
- [Benchmarking Random Access in Lance](https://blog.lancedb.com/benchmarking-random-access-in-lance/)

### Turbopuffer

- [turbopuffer: fast search on object storage](https://turbopuffer.com/blog/turbopuffer)
- [ANN v3: 200ms p99 over 100 billion vectors](https://turbopuffer.com/blog/ann-v3)
- [Native filtering for high-recall vector search](https://turbopuffer.com/blog/native-filtering)
- [Why BM25 queries with more terms can be faster](https://turbopuffer.com/blog/bm25-latency-musings)
- [Architecture](https://turbopuffer.com/docs/architecture)
- [Guarantees](https://turbopuffer.com/docs/guarantees)
- [SPFresh: Incremental In-Place Update for Billion-Scale Vector Search (SOSP 2023)](https://dl.acm.org/doi/10.1145/3600006.3613166)
- [Cursor scales code retrieval to 100B+ vectors with turbopuffer](https://turbopuffer.com/customers/cursor)

### Milvus Partition

- [Milvus GitHub Repository](https://github.com/milvus-io/milvus)
- [Milvus Partition Documentation](https://milvus.io/docs/manage-partitions.md)
- [Milvus Timestamp / TSO Documentation](https://milvus.io/docs/timestamp.md)
- [Milvus Binlog Developer Guide](https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/chap08_binlog.md)
- [How Milvus Deletes Streaming Data in a Distributed Cluster](https://milvus.io/blog/2022-02-07-how-milvus-deletes-streaming-data-in-distributed-cluster.md)
- [Data Insertion and Persistence](https://milvus.io/blog/deep-dive-4-data-insertion-and-data-persistence.md)
- [MEP 9 -- Support delete entities by primary keys](https://wiki.lfaidata.foundation/display/MIL/MEP+9+--+Support+delete+entities+by+primary+keys)
- [MEP 16 -- Compaction Design](https://wiki.lfaidata.foundation/display/MIL/MEP+16+--+Compaction)
- [Level Zero Segment Feature (Issue #27349)](https://github.com/milvus-io/milvus/issues/27349)
