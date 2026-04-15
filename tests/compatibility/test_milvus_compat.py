"""
Milvus API 兼容性测试 — 通过 pymilvus MilvusClient 验证 LiteVecDB
与真实 Milvus 行为的一致性。

覆盖范围:
  1. 集合生命周期 (create / has / describe / list / drop)
  2. 数据插入 (insert / upsert)
  3. 数据查询 (get / query)
  4. 删除操作 (delete by pk / delete by filter)
  5. 向量搜索 (L2 / IP / COSINE + filter + output_fields)
  6. 索引管理 (create / describe / drop, HNSW / IVF_FLAT / AUTOINDEX)
  7. 分区管理 (create / has / list / drop + partition-scoped CRUD)
  8. 多种数据类型 (BOOL / INT / FLOAT / VARCHAR / JSON / ARRAY)
  9. Auto ID & Dynamic Field
 10. Load / Release 状态
 11. Upsert 语义 (insert-or-update)
 12. 批量操作 & 边界条件
 13. 错误处理 (重复集合 / 不存在的集合等)
 14. Hybrid Search (多向量列)
 15. 全文检索 (BM25 + text_match filter)
"""

from __future__ import annotations

import shutil
import tempfile
import time

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from litevecdb.adapter.grpc.server import start_server_in_thread

DIM = 32
SEED = 42


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Module-scoped gRPC server for all tests."""
    data_dir = tempfile.mkdtemp(prefix="compat_test_")
    server, db, port = start_server_in_thread(data_dir)
    yield port, data_dir
    server.stop(grace=2)
    db.close()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def client(server):
    """Per-test client that auto-cleans collections."""
    port, _ = server
    c = MilvusClient(uri=f"http://127.0.0.1:{port}")
    yield c
    # Cleanup: drop all collections created during the test
    for name in c.list_collections():
        c.drop_collection(name)


def random_vectors(n: int, dim: int = DIM, seed: int = SEED) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32).tolist()


# ====================================================================
# 1. 集合生命周期
# ====================================================================

class TestCollectionLifecycle:

    def test_create_and_has(self, client: MilvusClient):
        """创建集合后 has_collection 返回 True"""
        client.create_collection("lifecycle_test", dimension=DIM)
        assert client.has_collection("lifecycle_test") is True

    def test_list_collections(self, client: MilvusClient):
        """创建多个集合后 list_collections 包含所有"""
        client.create_collection("col_a", dimension=DIM)
        client.create_collection("col_b", dimension=DIM)
        names = client.list_collections()
        assert "col_a" in names
        assert "col_b" in names

    def test_describe_collection(self, client: MilvusClient):
        """describe_collection 返回正确的 schema 信息"""
        client.create_collection("describe_test", dimension=DIM)
        info = client.describe_collection("describe_test")
        assert info["collection_name"] == "describe_test"
        # 应至少包含 id 和 vector 字段
        field_names = [f["name"] for f in info["fields"]]
        assert "id" in field_names
        assert "vector" in field_names

    def test_drop_collection(self, client: MilvusClient):
        """drop 后 has_collection 返回 False"""
        client.create_collection("drop_me", dimension=DIM)
        assert client.has_collection("drop_me") is True
        client.drop_collection("drop_me")
        assert client.has_collection("drop_me") is False

    def test_drop_nonexistent_collection_no_error(self, client: MilvusClient):
        """drop 不存在的集合不应报错（Milvus 行为）"""
        client.drop_collection("no_such_collection")

    def test_create_duplicate_collection_error(self, client: MilvusClient):
        """重复创建同名集合应报错"""
        client.create_collection("dup_test", dimension=DIM)
        with pytest.raises(Exception):
            client.create_collection("dup_test", dimension=DIM)


# ====================================================================
# 2. 自定义 Schema 创建集合
# ====================================================================

class TestCustomSchema:

    def test_create_with_schema(self, client: MilvusClient):
        """通过显式 schema 创建集合"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("embedding", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", MilvusDataType.VARCHAR, max_length=256)

        client.create_collection("schema_test", schema=schema)
        info = client.describe_collection("schema_test")
        field_names = [f["name"] for f in info["fields"]]
        assert "pk" in field_names
        assert "embedding" in field_names
        assert "label" in field_names

    def test_multi_scalar_types(self, client: MilvusClient):
        """测试多种标量类型字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("flag", MilvusDataType.BOOL)
        schema.add_field("score_i32", MilvusDataType.INT32)
        schema.add_field("score_f64", MilvusDataType.DOUBLE)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=128)
        schema.add_field("meta", MilvusDataType.JSON)

        client.create_collection("multi_type", schema=schema)
        info = client.describe_collection("multi_type")
        field_names = [f["name"] for f in info["fields"]]
        for name in ["pk", "vec", "flag", "score_i32", "score_f64", "name", "meta"]:
            assert name in field_names

    def test_varchar_primary_key(self, client: MilvusClient):
        """VARCHAR 主键"""
        schema = client.create_schema()
        schema.add_field("id", MilvusDataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("varchar_pk", schema=schema)
        vecs = random_vectors(3)
        data = [
            {"id": "alpha", "vec": vecs[0]},
            {"id": "beta", "vec": vecs[1]},
            {"id": "gamma", "vec": vecs[2]},
        ]
        client.insert("varchar_pk", data)
        results = client.get("varchar_pk", ids=["beta"])
        assert len(results) == 1
        assert results[0]["id"] == "beta"

    def test_nullable_field(self, client: MilvusClient):
        """nullable 字段可以插入 None"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tag", MilvusDataType.VARCHAR, max_length=64, nullable=True)

        client.create_collection("nullable_test", schema=schema)
        vecs = random_vectors(2)
        client.insert("nullable_test", [
            {"pk": 1, "vec": vecs[0], "tag": "hello"},
            {"pk": 2, "vec": vecs[1], "tag": None},
        ])
        results = client.get("nullable_test", ids=[1, 2])
        assert len(results) == 2
        tags = {r["pk"]: r.get("tag") for r in results}
        assert tags[1] == "hello"
        assert tags[2] is None

    def test_array_field(self, client: MilvusClient):
        """ARRAY 类型字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tags", MilvusDataType.ARRAY,
                         element_type=MilvusDataType.VARCHAR,
                         max_capacity=10, max_length=64)

        client.create_collection("array_test", schema=schema)
        vecs = random_vectors(2)
        client.insert("array_test", [
            {"pk": 1, "vec": vecs[0], "tags": ["a", "b", "c"]},
            {"pk": 2, "vec": vecs[1], "tags": ["x"]},
        ])
        results = client.get("array_test", ids=[1])
        assert results[0]["tags"] == ["a", "b", "c"]


# ====================================================================
# 3. 数据插入 & 查询
# ====================================================================

class TestInsertAndQuery:

    def test_insert_and_get(self, client: MilvusClient):
        """insert 后 get 能正确返回"""
        client.create_collection("ig_test", dimension=DIM)
        vecs = random_vectors(5)
        data = [{"id": i, "vector": vecs[i]} for i in range(5)]
        res = client.insert("ig_test", data)
        assert res["insert_count"] == 5

        got = client.get("ig_test", ids=[0, 2, 4])
        assert len(got) == 3
        returned_ids = sorted([r["id"] for r in got])
        assert returned_ids == [0, 2, 4]

    def test_insert_with_extra_fields(self, client: MilvusClient):
        """带额外标量字段的 insert"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("color", MilvusDataType.VARCHAR, max_length=32)
        schema.add_field("price", MilvusDataType.FLOAT)

        client.create_collection("extra_fields", schema=schema)
        vecs = random_vectors(3)
        client.insert("extra_fields", [
            {"pk": 1, "vec": vecs[0], "color": "red", "price": 9.99},
            {"pk": 2, "vec": vecs[1], "color": "blue", "price": 19.99},
            {"pk": 3, "vec": vecs[2], "color": "red", "price": 29.99},
        ])

        results = client.query("extra_fields", filter="color == 'red'",
                               output_fields=["pk", "color", "price"])
        assert len(results) == 2
        prices = sorted([r["price"] for r in results])
        assert prices == pytest.approx([9.99, 29.99], rel=1e-3)

    def test_query_with_various_filters(self, client: MilvusClient):
        """各种 filter 表达式"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("age", MilvusDataType.INT64)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=64)

        client.create_collection("filter_test", schema=schema)
        vecs = random_vectors(10)
        data = [
            {"pk": i, "vec": vecs[i], "age": 20 + i, "name": f"user_{i}"}
            for i in range(10)
        ]
        client.insert("filter_test", data)

        # 大于
        r = client.query("filter_test", filter="age > 25", output_fields=["pk", "age"])
        assert all(x["age"] > 25 for x in r)
        assert len(r) == 4  # ages 26,27,28,29

        # 范围
        r = client.query("filter_test", filter="age >= 22 and age < 25",
                         output_fields=["pk", "age"])
        assert len(r) == 3  # ages 22,23,24

        # IN 操作
        r = client.query("filter_test", filter="pk in [0, 5, 9]",
                         output_fields=["pk"])
        assert sorted([x["pk"] for x in r]) == [0, 5, 9]

        # 字符串比较
        r = client.query("filter_test", filter='name == "user_3"',
                         output_fields=["pk", "name"])
        assert len(r) == 1
        assert r[0]["name"] == "user_3"

    def test_query_with_limit(self, client: MilvusClient):
        """query 带 limit"""
        client.create_collection("limit_test", dimension=DIM)
        vecs = random_vectors(20)
        data = [{"id": i, "vector": vecs[i]} for i in range(20)]
        client.insert("limit_test", data)

        r = client.query("limit_test", filter="id >= 0", limit=5,
                         output_fields=["id"])
        assert len(r) == 5

    def test_get_nonexistent_ids(self, client: MilvusClient):
        """get 不存在的 id 返回空"""
        client.create_collection("get_empty", dimension=DIM)
        vecs = random_vectors(1)
        client.insert("get_empty", [{"id": 1, "vector": vecs[0]}])
        got = client.get("get_empty", ids=[999])
        assert len(got) == 0


# ====================================================================
# 4. 删除操作
# ====================================================================

class TestDelete:

    def test_delete_by_ids(self, client: MilvusClient):
        """按主键删除"""
        client.create_collection("del_test", dimension=DIM)
        vecs = random_vectors(5)
        data = [{"id": i, "vector": vecs[i]} for i in range(5)]
        client.insert("del_test", data)

        client.delete("del_test", ids=[1, 3])
        remaining = client.query("del_test", filter="id >= 0",
                                 output_fields=["id"])
        remaining_ids = sorted([r["id"] for r in remaining])
        assert remaining_ids == [0, 2, 4]

    def test_delete_by_filter(self, client: MilvusClient):
        """按 filter 删除"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", MilvusDataType.VARCHAR, max_length=32)

        client.create_collection("del_filter", schema=schema)
        vecs = random_vectors(6)
        client.insert("del_filter", [
            {"pk": i, "vec": vecs[i], "status": "active" if i % 2 == 0 else "inactive"}
            for i in range(6)
        ])

        client.delete("del_filter", filter='status == "inactive"')
        remaining = client.query("del_filter", filter="pk >= 0",
                                 output_fields=["pk", "status"])
        assert all(r["status"] == "active" for r in remaining)
        assert len(remaining) == 3

    def test_delete_then_insert_same_pk(self, client: MilvusClient):
        """删除后用相同 pk 重新插入"""
        client.create_collection("del_reinsert", dimension=DIM)
        vecs = random_vectors(3)
        client.insert("del_reinsert", [{"id": 1, "vector": vecs[0]}])
        client.delete("del_reinsert", ids=[1])
        client.insert("del_reinsert", [{"id": 1, "vector": vecs[1]}])

        got = client.get("del_reinsert", ids=[1])
        assert len(got) == 1


# ====================================================================
# 5. 向量搜索
# ====================================================================

class TestVectorSearch:

    def _setup_search_collection(self, client: MilvusClient, name: str,
                                 metric: str = "COSINE"):
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("category", MilvusDataType.VARCHAR, max_length=32)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec",
                               index_type="HNSW",
                               metric_type=metric,
                               params={"M": 16, "efConstruction": 64})

        client.create_collection(name, schema=schema,
                                 index_params=index_params)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((100, DIM)).astype(np.float32)
        data = [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": "cat_A" if i < 50 else "cat_B"}
            for i in range(100)
        ]
        client.insert(name, data)
        client.load_collection(name)
        return vecs

    def test_search_cosine(self, client: MilvusClient):
        """COSINE 向量搜索"""
        vecs = self._setup_search_collection(client, "search_cos", "COSINE")
        query = vecs[0:1].tolist()
        results = client.search("search_cos", data=query, limit=5,
                                output_fields=["pk", "category"])
        assert len(results) == 1
        assert len(results[0]) == 5
        # 最近邻应该是自己
        assert results[0][0]["entity"]["pk"] == 0

    def test_search_l2(self, client: MilvusClient):
        """L2 向量搜索"""
        vecs = self._setup_search_collection(client, "search_l2", "L2")
        query = vecs[10:11].tolist()
        results = client.search("search_l2", data=query, limit=3,
                                output_fields=["pk"])
        assert results[0][0]["entity"]["pk"] == 10

    def test_search_ip(self, client: MilvusClient):
        """IP (内积) 向量搜索"""
        vecs = self._setup_search_collection(client, "search_ip", "IP")
        query = vecs[50:51].tolist()
        results = client.search("search_ip", data=query, limit=3,
                                output_fields=["pk"])
        assert results[0][0]["entity"]["pk"] == 50

    def test_search_with_filter(self, client: MilvusClient):
        """搜索时带 filter 过滤"""
        vecs = self._setup_search_collection(client, "search_filter", "COSINE")
        # 用前50个向量中的一个查询，但只搜 cat_B (后50个)
        query = vecs[0:1].tolist()
        results = client.search("search_filter", data=query, limit=5,
                                filter='category == "cat_B"',
                                output_fields=["pk", "category"])
        assert len(results[0]) == 5
        for hit in results[0]:
            assert hit["entity"]["category"] == "cat_B"
            assert hit["entity"]["pk"] >= 50

    def test_search_with_output_fields(self, client: MilvusClient):
        """搜索结果包含指定的 output_fields"""
        self._setup_search_collection(client, "search_output", "COSINE")
        query = random_vectors(1)
        results = client.search("search_output", data=query, limit=3,
                                output_fields=["pk", "category"])
        for hit in results[0]:
            assert "pk" in hit["entity"]
            assert "category" in hit["entity"]

    def test_search_multiple_queries(self, client: MilvusClient):
        """批量查询 (多个 query vector)"""
        vecs = self._setup_search_collection(client, "search_batch", "COSINE")
        queries = vecs[0:3].tolist()
        results = client.search("search_batch", data=queries, limit=5,
                                output_fields=["pk"])
        assert len(results) == 3
        for i, res in enumerate(results):
            assert len(res) == 5
            assert res[0]["entity"]["pk"] == i  # 最近邻是自己

    def test_search_distance_ordering(self, client: MilvusClient):
        """搜索结果按距离排序"""
        self._setup_search_collection(client, "search_order", "L2")
        query = random_vectors(1, seed=99)
        results = client.search("search_order", data=query, limit=10,
                                output_fields=["pk"])
        distances = [hit["distance"] for hit in results[0]]
        # L2: 距离应该递增
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-6


# ====================================================================
# 6. 索引管理
# ====================================================================

class TestIndexManagement:

    def test_create_hnsw_index(self, client: MilvusClient):
        """创建 HNSW 索引"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("idx_hnsw", schema=schema)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec",
                               index_type="HNSW",
                               metric_type="COSINE",
                               params={"M": 16, "efConstruction": 64})
        client.create_index("idx_hnsw", index_params)

        indexes = client.describe_index("idx_hnsw", index_name="vec")
        assert indexes is not None

    def test_create_ivf_flat_index(self, client: MilvusClient):
        """创建 IVF_FLAT 索引"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("idx_ivf", schema=schema)

        # 先插入足够的数据供 IVF 训练
        vecs = random_vectors(200)
        data = [{"pk": i, "vec": vecs[i]} for i in range(200)]
        client.insert("idx_ivf", data)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec",
                               index_type="IVF_FLAT",
                               metric_type="L2",
                               params={"nlist": 8})
        client.create_index("idx_ivf", index_params)

        indexes = client.describe_index("idx_ivf", index_name="vec")
        assert indexes is not None

    def test_drop_index(self, client: MilvusClient):
        """删除索引"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("idx_drop", schema=schema)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec",
                               index_type="HNSW",
                               metric_type="COSINE",
                               params={"M": 16, "efConstruction": 64})
        client.create_index("idx_drop", index_params)
        client.drop_index("idx_drop", index_name="vec")

    def test_autoindex(self, client: MilvusClient):
        """AUTOINDEX 类型"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("idx_auto", schema=schema)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec",
                               index_type="AUTOINDEX",
                               metric_type="COSINE")
        client.create_index("idx_auto", index_params)


# ====================================================================
# 7. 分区管理
# ====================================================================

class TestPartitions:

    def test_create_and_list_partitions(self, client: MilvusClient):
        """创建分区并列出"""
        client.create_collection("part_test", dimension=DIM)
        client.create_partition("part_test", "region_us")
        client.create_partition("part_test", "region_eu")

        parts = client.list_partitions("part_test")
        assert "region_us" in parts
        assert "region_eu" in parts
        assert "_default" in parts

    def test_has_partition(self, client: MilvusClient):
        """检查分区是否存在"""
        client.create_collection("part_has", dimension=DIM)
        client.create_partition("part_has", "existing")

        assert client.has_partition("part_has", "existing") is True
        assert client.has_partition("part_has", "nonexistent") is False

    def test_drop_partition(self, client: MilvusClient):
        """删除分区"""
        client.create_collection("part_drop", dimension=DIM)
        client.create_partition("part_drop", "temp")
        assert client.has_partition("part_drop", "temp") is True
        client.drop_partition("part_drop", "temp")
        assert client.has_partition("part_drop", "temp") is False

    def test_insert_into_partition(self, client: MilvusClient):
        """向指定分区插入数据"""
        client.create_collection("part_insert", dimension=DIM)
        client.create_partition("part_insert", "shard_a")

        vecs = random_vectors(5)
        data = [{"id": i, "vector": vecs[i]} for i in range(5)]
        client.insert("part_insert", data, partition_name="shard_a")

        # 应该能查到
        got = client.get("part_insert", ids=[0, 1, 2])
        assert len(got) == 3

    def test_search_in_partition(self, client: MilvusClient):
        """在指定分区中搜索"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec", index_type="HNSW",
                               metric_type="COSINE",
                               params={"M": 16, "efConstruction": 64})

        client.create_collection("part_search", schema=schema,
                                 index_params=index_params)
        client.create_partition("part_search", "group_a")
        client.create_partition("part_search", "group_b")

        rng = np.random.default_rng(SEED)
        vecs_a = rng.standard_normal((20, DIM)).astype(np.float32)
        vecs_b = rng.standard_normal((20, DIM)).astype(np.float32)

        client.insert("part_search",
                      [{"pk": i, "vec": vecs_a[i].tolist()} for i in range(20)],
                      partition_name="group_a")
        client.insert("part_search",
                      [{"pk": 100 + i, "vec": vecs_b[i].tolist()} for i in range(20)],
                      partition_name="group_b")

        client.load_collection("part_search")

        # 在 group_a 搜索：结果的 pk 应该都 < 100
        query = vecs_a[0:1].tolist()
        results = client.search("part_search", data=query, limit=5,
                                partition_names=["group_a"],
                                output_fields=["pk"])
        for hit in results[0]:
            assert hit["entity"]["pk"] < 100


# ====================================================================
# 8. Load / Release
# ====================================================================

class TestLoadRelease:

    def test_load_and_release(self, client: MilvusClient):
        """load 和 release 不报错"""
        client.create_collection("lr_test", dimension=DIM)
        client.load_collection("lr_test")
        client.release_collection("lr_test")

    def test_get_load_state(self, client: MilvusClient):
        """获取 load 状态"""
        client.create_collection("load_state", dimension=DIM)
        state = client.get_load_state("load_state")
        # 应返回有效的状态信息
        assert state is not None


# ====================================================================
# 9. Upsert 语义
# ====================================================================

class TestUpsert:

    def test_upsert_new_records(self, client: MilvusClient):
        """upsert 新记录 = insert"""
        client.create_collection("ups_new", dimension=DIM)
        vecs = random_vectors(3)
        data = [{"id": i, "vector": vecs[i]} for i in range(3)]
        res = client.upsert("ups_new", data)
        assert res["upsert_count"] == 3

        got = client.get("ups_new", ids=[0, 1, 2])
        assert len(got) == 3

    def test_upsert_updates_existing(self, client: MilvusClient):
        """upsert 已有主键 = 覆盖"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", MilvusDataType.VARCHAR, max_length=64)

        client.create_collection("ups_update", schema=schema)
        vecs = random_vectors(2)
        client.insert("ups_update", [
            {"pk": 1, "vec": vecs[0], "label": "old"},
        ])

        # upsert 覆盖 pk=1
        new_vecs = random_vectors(2, seed=99)
        client.upsert("ups_update", [
            {"pk": 1, "vec": new_vecs[0], "label": "new"},
        ])

        got = client.get("ups_update", ids=[1])
        assert len(got) == 1
        assert got[0]["label"] == "new"


# ====================================================================
# 10. Auto ID
# ====================================================================

class TestAutoId:

    def test_auto_id_int64(self, client: MilvusClient):
        """auto_id 自动生成 INT64 主键"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("text", MilvusDataType.VARCHAR, max_length=128)

        client.create_collection("auto_id_test", schema=schema)
        vecs = random_vectors(3)
        res = client.insert("auto_id_test", [
            {"vec": vecs[0], "text": "hello"},
            {"vec": vecs[1], "text": "world"},
            {"vec": vecs[2], "text": "foo"},
        ])
        assert res["insert_count"] == 3

        # 查询所有记录
        results = client.query("auto_id_test", filter="pk >= 0",
                               output_fields=["pk", "text"])
        assert len(results) == 3
        # 每条记录都应该有唯一的 pk
        pks = [r["pk"] for r in results]
        assert len(set(pks)) == 3


# ====================================================================
# 11. Dynamic Field
# ====================================================================

class TestDynamicField:

    def test_dynamic_field_insert_and_query(self, client: MilvusClient):
        """动态字段：插入 schema 中未定义的字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.enable_dynamic_field = True

        client.create_collection("dynamic_test", schema=schema)
        vecs = random_vectors(3)
        client.insert("dynamic_test", [
            {"pk": 1, "vec": vecs[0], "color": "red", "score": 95},
            {"pk": 2, "vec": vecs[1], "color": "blue", "score": 80},
            {"pk": 3, "vec": vecs[2], "color": "red", "score": 70},
        ])

        # 用动态字段过滤
        results = client.query("dynamic_test", filter='color == "red"',
                               output_fields=["pk", "color", "score"])
        assert len(results) == 2
        for r in results:
            assert r["color"] == "red"


# ====================================================================
# 12. JSON 字段
# ====================================================================

class TestJsonField:

    def test_json_insert_and_query(self, client: MilvusClient):
        """JSON 字段的读写"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("meta", MilvusDataType.JSON)

        client.create_collection("json_test", schema=schema)
        vecs = random_vectors(3)
        client.insert("json_test", [
            {"pk": 1, "vec": vecs[0], "meta": {"env": "prod", "version": 3}},
            {"pk": 2, "vec": vecs[1], "meta": {"env": "staging", "version": 2}},
            {"pk": 3, "vec": vecs[2], "meta": {"env": "prod", "version": 1}},
        ])

        got = client.get("json_test", ids=[1])
        assert got[0]["meta"]["env"] == "prod"
        assert got[0]["meta"]["version"] == 3

    def test_json_field_filter(self, client: MilvusClient):
        """JSON 字段的 filter 查询"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("info", MilvusDataType.JSON)

        client.create_collection("json_filter", schema=schema)
        vecs = random_vectors(4)
        client.insert("json_filter", [
            {"pk": 1, "vec": vecs[0], "info": {"level": 5, "active": True}},
            {"pk": 2, "vec": vecs[1], "info": {"level": 3, "active": False}},
            {"pk": 3, "vec": vecs[2], "info": {"level": 8, "active": True}},
            {"pk": 4, "vec": vecs[3], "info": {"level": 1, "active": True}},
        ])

        results = client.query("json_filter", filter='info["level"] >= 5',
                               output_fields=["pk", "info"])
        pks = sorted([r["pk"] for r in results])
        assert pks == [1, 3]


# ====================================================================
# 13. 批量操作
# ====================================================================

class TestBulkOperations:

    def test_large_batch_insert(self, client: MilvusClient):
        """插入 1000 条数据"""
        client.create_collection("bulk_test", dimension=DIM)
        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((1000, DIM)).astype(np.float32).tolist()
        data = [{"id": i, "vector": vecs[i]} for i in range(1000)]
        res = client.insert("bulk_test", data)
        assert res["insert_count"] == 1000

        # 统计
        stats = client.get_collection_stats("bulk_test")
        assert int(stats["row_count"]) == 1000

    def test_search_after_large_insert(self, client: MilvusClient):
        """大批量插入后搜索正确"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec", index_type="HNSW",
                               metric_type="L2",
                               params={"M": 16, "efConstruction": 64})

        client.create_collection("bulk_search", schema=schema,
                                 index_params=index_params)

        rng = np.random.default_rng(SEED)
        n = 500
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        data = [{"pk": i, "vec": vecs[i].tolist()} for i in range(n)]
        client.insert("bulk_search", data)
        client.load_collection("bulk_search")

        query = vecs[0:1].tolist()
        results = client.search("bulk_search", data=query, limit=1,
                                output_fields=["pk"])
        assert results[0][0]["entity"]["pk"] == 0


# ====================================================================
# 14. 错误处理
# ====================================================================

class TestErrorHandling:

    def test_insert_to_nonexistent_collection(self, client: MilvusClient):
        """向不存在的集合插入数据应报错"""
        with pytest.raises(Exception):
            client.insert("no_such_col", [{"id": 1, "vector": [0.0] * DIM}])

    def test_search_nonexistent_collection(self, client: MilvusClient):
        """搜索不存在的集合应报错"""
        with pytest.raises(Exception):
            client.search("no_such_col", data=[[0.0] * DIM], limit=5)

    def test_insert_wrong_dimension(self, client: MilvusClient):
        """向量维度不匹配应报错"""
        client.create_collection("dim_err", dimension=DIM)
        with pytest.raises(Exception):
            client.insert("dim_err", [{"id": 1, "vector": [0.0] * (DIM + 10)}])

    def test_describe_nonexistent_collection(self, client: MilvusClient):
        """describe 不存在的集合应报错"""
        with pytest.raises(Exception):
            client.describe_collection("ghost_collection")


# ====================================================================
# 15. Collection Statistics
# ====================================================================

class TestStatistics:

    def test_get_collection_stats(self, client: MilvusClient):
        """获取集合统计信息"""
        client.create_collection("stats_test", dimension=DIM)
        vecs = random_vectors(10)
        data = [{"id": i, "vector": vecs[i]} for i in range(10)]
        client.insert("stats_test", data)

        stats = client.get_collection_stats("stats_test")
        assert int(stats["row_count"]) == 10


# ====================================================================
# 16. 端到端典型使用流程
# ====================================================================

class TestEndToEnd:

    def test_full_workflow(self, client: MilvusClient):
        """
        完整的 pymilvus 工作流:
        create schema -> create collection -> insert -> create index
        -> load -> search -> query -> delete -> query again
        """
        # 1) 创建 schema
        schema = client.create_schema()
        schema.add_field("id", MilvusDataType.INT64, is_primary=True)
        schema.add_field("embedding", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("title", MilvusDataType.VARCHAR, max_length=256)
        schema.add_field("rating", MilvusDataType.FLOAT)

        # 2) 创建集合 + 索引
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding",
                               index_type="HNSW",
                               metric_type="COSINE",
                               params={"M": 16, "efConstruction": 64})

        client.create_collection("e2e_test", schema=schema,
                                 index_params=index_params)

        # 3) 插入数据
        rng = np.random.default_rng(SEED)
        n = 50
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        data = [
            {
                "id": i,
                "embedding": vecs[i].tolist(),
                "title": f"Document {i}",
                "rating": float(i % 5) + 0.5,
            }
            for i in range(n)
        ]
        res = client.insert("e2e_test", data)
        assert res["insert_count"] == n

        # 4) Load
        client.load_collection("e2e_test")

        # 5) 搜索
        query = vecs[0:1].tolist()
        search_res = client.search("e2e_test", data=query, limit=10,
                                   filter="rating >= 3.0",
                                   output_fields=["id", "title", "rating"])
        assert len(search_res[0]) > 0
        for hit in search_res[0]:
            assert hit["entity"]["rating"] >= 3.0

        # 6) Query
        query_res = client.query("e2e_test", filter="rating == 0.5",
                                 output_fields=["id", "title", "rating"])
        assert all(r["rating"] == pytest.approx(0.5) for r in query_res)

        # 7) 删除部分数据
        ids_to_delete = [0, 1, 2, 3, 4]
        client.delete("e2e_test", ids=ids_to_delete)

        # 8) 验证删除生效
        got = client.get("e2e_test", ids=ids_to_delete)
        assert len(got) == 0

        # 9) 剩余数据完整
        remaining = client.query("e2e_test", filter="id >= 0",
                                 output_fields=["id"], limit=100)
        assert len(remaining) == n - len(ids_to_delete)

    def test_quickstart_api(self, client: MilvusClient):
        """
        pymilvus quickstart 风格 API:
        create_collection(name, dimension=N) -> insert -> search
        """
        client.create_collection("quickstart", dimension=DIM)

        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((20, DIM)).astype(np.float32).tolist()
        data = [{"id": i, "vector": vecs[i]} for i in range(20)]
        client.insert("quickstart", data)

        results = client.search("quickstart", data=[vecs[0]], limit=3,
                                output_fields=["id"])
        assert len(results) == 1
        assert len(results[0]) == 3
        # 最近邻是自己
        assert results[0][0]["entity"]["id"] == 0


# ====================================================================
# 17. Hybrid Search
# ====================================================================

class TestHybridSearch:

    def test_hybrid_search_basic(self, client: MilvusClient):
        """Hybrid Search: 多个 ANN 请求 + RRF 融合"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("dense", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("title", MilvusDataType.VARCHAR, max_length=128)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="HNSW",
                               metric_type="COSINE",
                               params={"M": 16, "efConstruction": 64})

        client.create_collection("hybrid_test", schema=schema,
                                 index_params=index_params)

        rng = np.random.default_rng(SEED)
        n = 50
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        data = [
            {"pk": i, "dense": vecs[i].tolist(), "title": f"doc_{i}"}
            for i in range(n)
        ]
        client.insert("hybrid_test", data)
        client.load_collection("hybrid_test")

        from pymilvus import AnnSearchRequest, RRFRanker

        q = vecs[0:1].tolist()
        req1 = AnnSearchRequest(data=q, anns_field="dense", param={}, limit=10)
        req2 = AnnSearchRequest(data=q, anns_field="dense", param={}, limit=10)

        results = client.hybrid_search(
            "hybrid_test",
            reqs=[req1, req2],
            ranker=RRFRanker(k=60),
            limit=5,
            output_fields=["pk", "title"],
        )
        assert len(results) == 1
        assert len(results[0]) == 5
        # 第一个结果应该是自己
        assert results[0][0]["entity"]["pk"] == 0


# ====================================================================
# 18. 全文检索 (BM25)
# ====================================================================

class TestFullTextSearch:

    def test_bm25_text_search(self, client: MilvusClient):
        """BM25 全文检索"""
        from pymilvus import Function, FunctionType

        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", MilvusDataType.VARCHAR, max_length=1024,
                         enable_analyzer=True)
        schema.add_field("sparse", MilvusDataType.SPARSE_FLOAT_VECTOR)

        bm25 = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
        schema.add_function(bm25)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="sparse",
                               index_type="SPARSE_INVERTED_INDEX",
                               metric_type="BM25")

        client.create_collection("fts_test", schema=schema,
                                 index_params=index_params)

        docs = [
            {"text": "machine learning algorithms for natural language processing"},
            {"text": "deep learning neural network architecture design"},
            {"text": "python programming language tutorial for beginners"},
            {"text": "database management system and SQL query optimization"},
            {"text": "machine learning model training and evaluation metrics"},
        ]
        client.insert("fts_test", docs)
        client.load_collection("fts_test")

        # 搜索 "machine learning"
        results = client.search(
            "fts_test",
            data=["machine learning"],
            anns_field="sparse",
            limit=3,
            output_fields=["text"],
        )
        assert len(results[0]) >= 2
        # 包含 "machine learning" 的文档应该排在前面
        top_texts = [hit["entity"]["text"] for hit in results[0]]
        assert any("machine learning" in t for t in top_texts)

    def test_text_match_filter(self, client: MilvusClient):
        """TEXT_MATCH filter 配合向量搜索"""
        from pymilvus import Function, FunctionType

        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("content", MilvusDataType.VARCHAR, max_length=1024,
                         enable_analyzer=True, enable_match=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)

        client.create_collection("text_match_test", schema=schema)

        rng = np.random.default_rng(SEED)
        n = 10
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        docs = [
            {"content": "apple banana cherry", "vec": vecs[0].tolist()},
            {"content": "banana date elderberry", "vec": vecs[1].tolist()},
            {"content": "cherry fig grape", "vec": vecs[2].tolist()},
            {"content": "apple fig kiwi", "vec": vecs[3].tolist()},
            {"content": "banana grape lemon", "vec": vecs[4].tolist()},
            {"content": "apple cherry mango", "vec": vecs[5].tolist()},
            {"content": "date fig orange", "vec": vecs[6].tolist()},
            {"content": "elderberry kiwi peach", "vec": vecs[7].tolist()},
            {"content": "fig grape raspberry", "vec": vecs[8].tolist()},
            {"content": "apple banana strawberry", "vec": vecs[9].tolist()},
        ]
        client.insert("text_match_test", docs)

        # 用 text_match 查询包含 "apple" 的文档
        results = client.query("text_match_test",
                               filter='TEXT_MATCH(content, "apple")',
                               output_fields=["content"])
        assert len(results) == 4  # docs 0, 3, 5, 9
        for r in results:
            assert "apple" in r["content"]


# ====================================================================
# 19. rename collection
# ====================================================================

class TestRenameCollection:

    def test_rename(self, client: MilvusClient):
        """重命名集合"""
        client.create_collection("old_name", dimension=DIM)
        vecs = random_vectors(3)
        client.insert("old_name", [{"id": i, "vector": vecs[i]} for i in range(3)])

        client.rename_collection("old_name", "new_name")
        assert client.has_collection("old_name") is False
        assert client.has_collection("new_name") is True

        # rename 后需要 load 才能查询（与 Milvus 行为一致）
        client.load_collection("new_name")
        got = client.get("new_name", ids=[0, 1, 2])
        assert len(got) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
