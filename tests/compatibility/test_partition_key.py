"""
Partition Key 功能兼容性测试 — 通过 pymilvus MilvusClient 验证。

Partition Key 语义 (与 Milvus 一致):
  - schema 中某个标量字段标记 is_partition_key=True
  - insert 时自动按该字段值 hash 路由到内部 bucket 分区
  - 用户无需手动管理分区，查询/搜索时自动扫描所有 bucket
  - 用户不可手动创建/删除分区 (partition_key 集合禁止手动分区操作)

测试覆盖:
  1.  基本创建 + 插入 + 查询
  2.  DescribeCollection 返回 partition_key 信息
  3.  搜索正确 (跨 bucket 搜索)
  4.  按 partition_key 字段 filter 查询
  5.  不同 partition_key 值的数据隔离性
  6.  相同 partition_key 值的数据聚合
  7.  delete 操作 (跨 bucket 删除)
  8.  upsert 操作
  9.  count(*) 统计
 10.  partition_key + 其他标量 filter 组合
 11.  partition_key + 向量搜索 + filter
 12.  大量不同 partition_key 值
 13.  partition_key 为 INT64 类型
 14.  partition_key + auto_id
 15.  partition_key + dynamic field
 16.  partition_key + nullable 字段
 17.  partition_key + BM25 全文检索
 18.  flush 后 partition_key 数据仍正确
 19.  partition_key 集合禁止手动分区操作
 20.  partition_key 值为空字符串 / 极端值
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from litevecdb.adapter.grpc.server import start_server_in_thread

DIM = 8
SEED = 88


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="pkey_test_")
    server, db, port = start_server_in_thread(data_dir)
    yield port, data_dir
    server.stop(grace=2)
    db.close()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def client(server):
    port, _ = server
    c = MilvusClient(uri=f"http://127.0.0.1:{port}")
    yield c
    for name in c.list_collections():
        c.drop_collection(name)


def rvecs(n, dim=DIM, seed=SEED):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32).tolist()


def make_pkey_collection(client, name, pkey_type=MilvusDataType.VARCHAR):
    """创建带 partition_key 的集合"""
    schema = client.create_schema()
    schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
    schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
    if pkey_type == MilvusDataType.VARCHAR:
        schema.add_field("category", pkey_type, max_length=64,
                         is_partition_key=True)
    else:
        schema.add_field("category", pkey_type, is_partition_key=True)
    schema.add_field("score", MilvusDataType.FLOAT, nullable=True)

    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                  params={"M": 16, "efConstruction": 64})
    client.create_collection(name, schema=schema, index_params=idx)


# ====================================================================
# 1. 基本创建 + 插入 + 查询
# ====================================================================

class TestBasicPartitionKey:

    def test_insert_and_get(self, client: MilvusClient):
        """partition_key 集合的基本 insert + get"""
        make_pkey_collection(client, "pkey_basic")
        vecs = rvecs(5)
        client.insert("pkey_basic", [
            {"pk": i, "vec": vecs[i], "category": f"cat_{i % 3}", "score": float(i)}
            for i in range(5)
        ])

        got = client.get("pkey_basic", ids=[0, 1, 2, 3, 4])
        assert len(got) == 5
        cats = {r["pk"]: r["category"] for r in got}
        assert cats[0] == "cat_0"
        assert cats[1] == "cat_1"
        assert cats[4] == "cat_1"

    def test_query_all(self, client: MilvusClient):
        """partition_key 集合 query 返回全部数据"""
        make_pkey_collection(client, "pkey_query")
        vecs = rvecs(10)
        client.insert("pkey_query", [
            {"pk": i, "vec": vecs[i], "category": f"cat_{i % 4}", "score": float(i)}
            for i in range(10)
        ])

        r = client.query("pkey_query", filter="pk >= 0", output_fields=["pk"],
                         limit=100)
        assert len(r) == 10


# ====================================================================
# 2. DescribeCollection 返回 partition_key 信息
# ====================================================================

class TestDescribePartitionKey:

    def test_describe_shows_partition_key(self, client: MilvusClient):
        """describe 应显示 partition_key 字段"""
        make_pkey_collection(client, "pkey_desc")
        info = client.describe_collection("pkey_desc")

        # 找到 category 字段
        cat_field = next(f for f in info["fields"] if f["name"] == "category")
        assert cat_field.get("is_partition_key") is True


# ====================================================================
# 3. 搜索正确 (跨 bucket 搜索)
# ====================================================================

class TestPartitionKeySearch:

    def test_search_across_buckets(self, client: MilvusClient):
        """搜索自动跨所有 bucket 分区"""
        make_pkey_collection(client, "pkey_search")
        rng = np.random.default_rng(SEED)
        n = 50
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        client.insert("pkey_search", [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": f"cat_{i % 5}", "score": float(i)}
            for i in range(n)
        ])
        client.load_collection("pkey_search")

        # 搜索应该能找到任何 bucket 中的数据
        results = client.search("pkey_search", data=vecs[0:1].tolist(), limit=5,
                                output_fields=["pk", "category"])
        assert len(results[0]) == 5
        assert results[0][0]["entity"]["pk"] == 0

    def test_search_with_different_categories(self, client: MilvusClient):
        """不同 category 的向量都能被搜索到"""
        make_pkey_collection(client, "pkey_search_cat")
        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((6, DIM)).astype(np.float32)
        client.insert("pkey_search_cat", [
            {"pk": 0, "vec": vecs[0].tolist(), "category": "alpha", "score": 1.0},
            {"pk": 1, "vec": vecs[1].tolist(), "category": "beta",  "score": 2.0},
            {"pk": 2, "vec": vecs[2].tolist(), "category": "gamma", "score": 3.0},
            {"pk": 3, "vec": vecs[3].tolist(), "category": "alpha", "score": 4.0},
            {"pk": 4, "vec": vecs[4].tolist(), "category": "beta",  "score": 5.0},
            {"pk": 5, "vec": vecs[5].tolist(), "category": "gamma", "score": 6.0},
        ])
        client.load_collection("pkey_search_cat")

        # 搜索每个 category 的向量都能作为最近邻返回
        for i in range(6):
            r = client.search("pkey_search_cat", data=vecs[i:i+1].tolist(),
                              limit=1, output_fields=["pk"])
            assert r[0][0]["entity"]["pk"] == i


# ====================================================================
# 4. 按 partition_key 字段 filter 查询
# ====================================================================

class TestPartitionKeyFilter:

    def test_filter_by_partition_key(self, client: MilvusClient):
        """按 partition_key 字段值过滤"""
        make_pkey_collection(client, "pkey_filter")
        vecs = rvecs(12)
        client.insert("pkey_filter", [
            {"pk": i, "vec": vecs[i],
             "category": ["red", "green", "blue"][i % 3], "score": float(i)}
            for i in range(12)
        ])

        r = client.query("pkey_filter", filter='category == "red"',
                         output_fields=["pk", "category"])
        assert len(r) == 4  # 0, 3, 6, 9
        assert all(x["category"] == "red" for x in r)

    def test_search_with_partition_key_filter(self, client: MilvusClient):
        """search + partition_key 字段 filter"""
        make_pkey_collection(client, "pkey_sf")
        rng = np.random.default_rng(SEED)
        n = 30
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        client.insert("pkey_sf", [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": "A" if i < 15 else "B", "score": float(i)}
            for i in range(n)
        ])
        client.load_collection("pkey_sf")

        r = client.search("pkey_sf", data=vecs[0:1].tolist(), limit=5,
                          filter='category == "A"',
                          output_fields=["pk", "category"])
        assert len(r[0]) == 5
        for hit in r[0]:
            assert hit["entity"]["category"] == "A"


# ====================================================================
# 5. 不同 partition_key 值的数据隔离性
# ====================================================================

class TestPartitionKeyIsolation:

    def test_data_isolation_via_filter(self, client: MilvusClient):
        """不同 partition_key 值的数据通过 filter 完全隔离"""
        make_pkey_collection(client, "pkey_iso")
        vecs = rvecs(20)
        client.insert("pkey_iso", [
            {"pk": i, "vec": vecs[i],
             "category": f"tenant_{i % 4}", "score": float(i)}
            for i in range(20)
        ])

        for t in range(4):
            r = client.query("pkey_iso", filter=f'category == "tenant_{t}"',
                             output_fields=["pk", "category"])
            assert len(r) == 5
            assert all(x["category"] == f"tenant_{t}" for x in r)


# ====================================================================
# 6. 相同 partition_key 值的数据聚合
# ====================================================================

class TestPartitionKeyAggregation:

    def test_same_key_data_queryable(self, client: MilvusClient):
        """相同 partition_key 的数据可以一起查询"""
        make_pkey_collection(client, "pkey_agg")
        vecs = rvecs(10)
        # 所有数据都用同一个 partition_key
        client.insert("pkey_agg", [
            {"pk": i, "vec": vecs[i], "category": "single_tenant", "score": float(i)}
            for i in range(10)
        ])

        r = client.query("pkey_agg", filter='category == "single_tenant"',
                         output_fields=["pk"])
        assert len(r) == 10


# ====================================================================
# 7. delete 操作
# ====================================================================

class TestPartitionKeyDelete:

    def test_delete_by_pk(self, client: MilvusClient):
        """partition_key 集合按 PK 删除"""
        make_pkey_collection(client, "pkey_del")
        vecs = rvecs(6)
        client.insert("pkey_del", [
            {"pk": i, "vec": vecs[i], "category": f"c{i % 3}", "score": float(i)}
            for i in range(6)
        ])

        client.delete("pkey_del", ids=[1, 3, 5])
        remaining = client.query("pkey_del", filter="pk >= 0",
                                 output_fields=["pk"])
        pks = sorted([r["pk"] for r in remaining])
        assert pks == [0, 2, 4]

    def test_delete_by_filter(self, client: MilvusClient):
        """partition_key 集合按 filter 删除"""
        make_pkey_collection(client, "pkey_del_f")
        vecs = rvecs(9)
        client.insert("pkey_del_f", [
            {"pk": i, "vec": vecs[i],
             "category": ["x", "y", "z"][i % 3], "score": float(i)}
            for i in range(9)
        ])

        client.delete("pkey_del_f", filter='category == "y"')
        remaining = client.query("pkey_del_f", filter="pk >= 0",
                                 output_fields=["pk", "category"])
        assert all(r["category"] != "y" for r in remaining)
        assert len(remaining) == 6


# ====================================================================
# 8. upsert 操作
# ====================================================================

class TestPartitionKeyUpsert:

    def test_upsert_existing(self, client: MilvusClient):
        """partition_key 集合 upsert 已存在的记录"""
        make_pkey_collection(client, "pkey_ups")
        vecs = rvecs(3)
        client.insert("pkey_ups", [
            {"pk": 1, "vec": vecs[0], "category": "alpha", "score": 10.0},
            {"pk": 2, "vec": vecs[1], "category": "beta",  "score": 20.0},
        ])

        new_vec = rvecs(1, seed=99)[0]
        client.upsert("pkey_ups", [
            {"pk": 1, "vec": new_vec, "category": "alpha", "score": 99.0},
        ])

        got = client.get("pkey_ups", ids=[1])
        assert got[0]["score"] == pytest.approx(99.0)

    def test_upsert_new(self, client: MilvusClient):
        """partition_key 集合 upsert 新记录"""
        make_pkey_collection(client, "pkey_ups_new")
        vecs = rvecs(2)
        client.insert("pkey_ups_new", [
            {"pk": 1, "vec": vecs[0], "category": "alpha", "score": 1.0},
        ])

        client.upsert("pkey_ups_new", [
            {"pk": 2, "vec": vecs[1], "category": "beta", "score": 2.0},
        ])

        got = client.get("pkey_ups_new", ids=[1, 2])
        assert len(got) == 2


# ====================================================================
# 9. count(*)
# ====================================================================

class TestPartitionKeyCount:

    def test_count_all(self, client: MilvusClient):
        make_pkey_collection(client, "pkey_cnt")
        vecs = rvecs(15)
        client.insert("pkey_cnt", [
            {"pk": i, "vec": vecs[i], "category": f"c{i % 5}", "score": 0.0}
            for i in range(15)
        ])

        r = client.query("pkey_cnt", filter="", output_fields=["count(*)"])
        assert r[0]["count(*)"] == 15

    def test_count_by_category(self, client: MilvusClient):
        make_pkey_collection(client, "pkey_cnt_cat")
        vecs = rvecs(12)
        client.insert("pkey_cnt_cat", [
            {"pk": i, "vec": vecs[i],
             "category": ["a", "b", "c"][i % 3], "score": 0.0}
            for i in range(12)
        ])

        r = client.query("pkey_cnt_cat", filter='category == "a"',
                          output_fields=["count(*)"])
        assert r[0]["count(*)"] == 4


# ====================================================================
# 10. partition_key + 其他标量 filter 组合
# ====================================================================

class TestPartitionKeyComboFilter:

    def test_partition_key_and_scalar_filter(self, client: MilvusClient):
        """partition_key filter + 其他标量条件"""
        make_pkey_collection(client, "pkey_combo")
        vecs = rvecs(20)
        client.insert("pkey_combo", [
            {"pk": i, "vec": vecs[i],
             "category": "hot" if i < 10 else "cold",
             "score": float(i * 10)}
            for i in range(20)
        ])

        # category == "hot" AND score >= 50
        r = client.query("pkey_combo",
                         filter='category == "hot" and score >= 50',
                         output_fields=["pk", "category", "score"])
        for x in r:
            assert x["category"] == "hot"
            assert x["score"] >= 50
        assert len(r) == 5  # pk 5,6,7,8,9


# ====================================================================
# 11. 大量不同 partition_key 值
# ====================================================================

class TestManyPartitionKeys:

    def test_100_different_keys(self, client: MilvusClient):
        """100 个不同的 partition_key 值"""
        make_pkey_collection(client, "pkey_many")
        rng = np.random.default_rng(SEED)
        n = 100
        vecs = rng.standard_normal((n, DIM)).astype(np.float32).tolist()
        client.insert("pkey_many", [
            {"pk": i, "vec": vecs[i], "category": f"key_{i}", "score": 0.0}
            for i in range(n)
        ])

        stats = client.get_collection_stats("pkey_many")
        assert int(stats["row_count"]) == n

        # 随机抽查
        got = client.get("pkey_many", ids=[0, 50, 99])
        assert len(got) == 3
        cats = {r["pk"]: r["category"] for r in got}
        assert cats[0] == "key_0"
        assert cats[50] == "key_50"
        assert cats[99] == "key_99"


# ====================================================================
# 12. INT64 partition_key
# ====================================================================

class TestIntPartitionKey:

    def test_int64_partition_key(self, client: MilvusClient):
        """INT64 类型的 partition_key"""
        make_pkey_collection(client, "pkey_int", pkey_type=MilvusDataType.INT64)
        vecs = rvecs(10)
        client.insert("pkey_int", [
            {"pk": i, "vec": vecs[i], "category": i % 3, "score": float(i)}
            for i in range(10)
        ])

        got = client.get("pkey_int", ids=list(range(10)))
        assert len(got) == 10

        # filter by int partition_key
        r = client.query("pkey_int", filter="category == 1",
                         output_fields=["pk", "category"])
        assert all(x["category"] == 1 for x in r)


# ====================================================================
# 13. partition_key + auto_id
# ====================================================================

class TestPartitionKeyAutoId:

    def test_auto_id_with_partition_key(self, client: MilvusClient):
        """auto_id + partition_key"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tenant", MilvusDataType.VARCHAR, max_length=64,
                         is_partition_key=True)

        client.create_collection("pkey_autoid", schema=schema)
        vecs = rvecs(6)
        client.insert("pkey_autoid", [
            {"vec": vecs[i], "tenant": f"t{i % 2}"} for i in range(6)
        ])

        r = client.query("pkey_autoid", filter='tenant == "t0"',
                         output_fields=["pk", "tenant"])
        assert len(r) == 3
        assert all(x["tenant"] == "t0" for x in r)


# ====================================================================
# 14. partition_key + dynamic field
# ====================================================================

class TestPartitionKeyDynamic:

    def test_partition_key_with_dynamic_fields(self, client: MilvusClient):
        """partition_key + 动态字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("region", MilvusDataType.VARCHAR, max_length=32,
                         is_partition_key=True)
        schema.enable_dynamic_field = True

        client.create_collection("pkey_dyn", schema=schema)
        vecs = rvecs(4)
        client.insert("pkey_dyn", [
            {"pk": 0, "vec": vecs[0], "region": "us", "color": "red"},
            {"pk": 1, "vec": vecs[1], "region": "eu", "color": "blue"},
            {"pk": 2, "vec": vecs[2], "region": "us", "color": "green"},
            {"pk": 3, "vec": vecs[3], "region": "eu", "color": "red"},
        ])

        # filter by dynamic field
        r = client.query("pkey_dyn", filter='color == "red"',
                         output_fields=["pk", "region", "color"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 3]

        # filter by partition_key + dynamic
        r = client.query("pkey_dyn",
                         filter='region == "us" and color == "red"',
                         output_fields=["pk"])
        assert len(r) == 1
        assert r[0]["pk"] == 0


# ====================================================================
# 15. flush 后数据仍正确
# ====================================================================

class TestPartitionKeyFlush:

    def test_data_survives_flush(self, client: MilvusClient):
        """flush 后 partition_key 数据完整"""
        make_pkey_collection(client, "pkey_flush")
        vecs = rvecs(10)
        client.insert("pkey_flush", [
            {"pk": i, "vec": vecs[i], "category": f"c{i % 3}", "score": float(i)}
            for i in range(10)
        ])

        client.flush("pkey_flush")

        got = client.get("pkey_flush", ids=list(range(10)))
        assert len(got) == 10

        r = client.query("pkey_flush", filter='category == "c0"',
                         output_fields=["pk"])
        assert len(r) == 4  # 0, 3, 6, 9


# ====================================================================
# 16. partition_key 集合禁止手动分区操作
# ====================================================================

class TestPartitionKeyNoManualPartition:

    def test_create_partition_forbidden(self, client: MilvusClient):
        """partition_key 集合不允许手动创建分区"""
        make_pkey_collection(client, "pkey_no_part")
        with pytest.raises(Exception):
            client.create_partition("pkey_no_part", "manual_partition")

    def test_drop_partition_forbidden(self, client: MilvusClient):
        """partition_key 集合不允许手动删除分区"""
        make_pkey_collection(client, "pkey_no_drop")
        with pytest.raises(Exception):
            client.drop_partition("pkey_no_drop", "_pk_0")


# ====================================================================
# 17. partition_key 值为空字符串
# ====================================================================

class TestPartitionKeyEdgeValues:

    def test_empty_string_partition_key(self, client: MilvusClient):
        """空字符串作为 partition_key 值"""
        make_pkey_collection(client, "pkey_empty")
        vecs = rvecs(3)
        client.insert("pkey_empty", [
            {"pk": 0, "vec": vecs[0], "category": "", "score": 1.0},
            {"pk": 1, "vec": vecs[1], "category": "normal", "score": 2.0},
            {"pk": 2, "vec": vecs[2], "category": "", "score": 3.0},
        ])

        r = client.query("pkey_empty", filter='category == ""',
                         output_fields=["pk"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [0, 2]

    def test_special_chars_partition_key(self, client: MilvusClient):
        """特殊字符作为 partition_key 值"""
        make_pkey_collection(client, "pkey_special")
        vecs = rvecs(3)
        client.insert("pkey_special", [
            {"pk": 0, "vec": vecs[0], "category": "hello world", "score": 1.0},
            {"pk": 1, "vec": vecs[1], "category": "中文分区", "score": 2.0},
            {"pk": 2, "vec": vecs[2], "category": "a/b/c", "score": 3.0},
        ])

        got = client.get("pkey_special", ids=[0, 1, 2])
        assert len(got) == 3


# ====================================================================
# 18. 端到端完整流程
# ====================================================================

class TestPartitionKeyE2E:

    def test_full_workflow(self, client: MilvusClient):
        """partition_key 完整工作流"""
        make_pkey_collection(client, "pkey_e2e")
        rng = np.random.default_rng(SEED)

        # 1. 插入
        n = 30
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        tenants = ["acme", "globex", "initech"]
        client.insert("pkey_e2e", [
            {"pk": i, "vec": vecs[i].tolist(),
             "category": tenants[i % 3], "score": float(i)}
            for i in range(n)
        ])

        # 2. 各 tenant 数据量正确
        for t in tenants:
            r = client.query("pkey_e2e", filter=f'category == "{t}"',
                             output_fields=["count(*)"])
            assert r[0]["count(*)"] == 10

        # 3. 搜索
        client.load_collection("pkey_e2e")
        results = client.search("pkey_e2e", data=vecs[0:1].tolist(), limit=3,
                                filter='category == "acme"',
                                output_fields=["pk", "category"])
        assert all(h["entity"]["category"] == "acme" for h in results[0])

        # 4. 删除一个 tenant 的数据
        client.delete("pkey_e2e", filter='category == "globex"')

        r = client.query("pkey_e2e", filter="", output_fields=["count(*)"])
        assert r[0]["count(*)"] == 20  # 30 - 10

        # 5. upsert
        client.upsert("pkey_e2e", [
            {"pk": 0, "vec": vecs[0].tolist(), "category": "acme", "score": 999.0},
        ])
        got = client.get("pkey_e2e", ids=[0])
        assert got[0]["score"] == pytest.approx(999.0)

        # 6. flush + 再查
        client.flush("pkey_e2e")
        got = client.get("pkey_e2e", ids=[0])
        assert got[0]["score"] == pytest.approx(999.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
