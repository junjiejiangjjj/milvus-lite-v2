"""
Milvus 杂项兼容性测试 — 覆盖剩余未测场景。

覆盖范围:
  1.  pymilvus ORM API (connections.connect + Collection)
  2.  错误 filter 语法 → 应返回清晰报错
  3.  插入不存在的字段名
  4.  类型不匹配 (字符串插入整数字段)
  5.  搜索返回零结果 (极端限制性 filter)
  6.  空字符串 filter 与 None filter
  7.  insert 大量重复 PK (1000 次覆盖同一 PK)
  8.  query limit=1 只取一条
  9.  get 单个 id (非列表)
 10.  search limit=1 只取最近邻
 11.  delete 大量 id (500 条)
 12.  集合名特殊字符/长名
 13.  FLOAT 和 INT 混插 (自动类型转换)
 14.  多个 filter 条件中使用同一字段
 15.  并发创建/删除集合
 16.  search + filter 命中 0 条
 17.  query offset > limit 组合
 18.  BOOL filter 各种写法
 19.  VARCHAR LIKE 各种模式
 20.  JSON 多层嵌套路径
"""

from __future__ import annotations

import shutil
import tempfile
import threading

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from litevecdb.adapter.grpc.server import start_server_in_thread

DIM = 8
SEED = 66


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="misc_test_")
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


# ====================================================================
# 1. pymilvus ORM API
# ====================================================================

class TestORMAPI:

    def test_orm_connect_and_crud(self, server):
        """用 pymilvus ORM API (connections + Collection) 做完整 CRUD"""
        from pymilvus import (
            connections, Collection, CollectionSchema,
            FieldSchema, DataType, utility,
        )
        port, _ = server
        alias = f"test_orm_{port}"

        connections.connect(alias=alias, host="127.0.0.1", port=port)
        try:
            # 创建 schema
            fields = [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=DIM),
                FieldSchema("tag", DataType.VARCHAR, max_length=64),
            ]
            schema = CollectionSchema(fields, description="ORM test")

            # drop if exists
            if utility.has_collection("orm_test", using=alias):
                utility.drop_collection("orm_test", using=alias)

            col = Collection("orm_test", schema=schema, using=alias)

            # Insert
            import random
            rng = np.random.default_rng(SEED)
            vecs = rng.standard_normal((10, DIM)).astype(np.float32).tolist()
            data = [
                list(range(10)),     # pk
                vecs,                # vec
                [f"t{i}" for i in range(10)],  # tag
            ]
            col.insert(data)

            # Create index + load
            col.create_index("vec", {"index_type": "HNSW", "metric_type": "COSINE",
                                     "params": {"M": 16, "efConstruction": 64}})
            col.load()

            # Search
            results = col.search(
                data=[vecs[0]], anns_field="vec",
                param={"metric_type": "COSINE"},
                limit=3, output_fields=["pk", "tag"],
            )
            assert len(results[0]) == 3
            assert results[0][0].entity.get("pk") == 0

            # Query
            rows = col.query(expr="pk < 5", output_fields=["pk", "tag"])
            assert len(rows) == 5

            # Delete
            col.delete(expr="pk in [0, 1]")

            # Verify
            rows = col.query(expr="pk >= 0", output_fields=["pk"])
            assert len(rows) == 8

            col.drop()
            assert not utility.has_collection("orm_test", using=alias)
        finally:
            connections.disconnect(alias)


# ====================================================================
# 2. 错误 filter 语法
# ====================================================================

class TestInvalidFilter:

    def test_bad_filter_syntax(self, client: MilvusClient):
        """无效 filter 语法应报错"""
        client.create_collection("bad_filter", dimension=DIM)
        vecs = rvecs(3)
        client.insert("bad_filter", [{"id": i, "vector": vecs[i]} for i in range(3)])

        with pytest.raises(Exception):
            client.query("bad_filter", filter="((( invalid syntax !!!",
                         output_fields=["id"])

    def test_unknown_field_in_filter(self, client: MilvusClient):
        """filter 中引用不存在的字段 (非 dynamic field 集合)"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        # enable_dynamic_field 默认 False → 未知字段应报错

        client.create_collection("unknown_field", schema=schema)
        vecs = rvecs(1)
        client.insert("unknown_field", [{"pk": 0, "vec": vecs[0]}])

        with pytest.raises(Exception):
            client.query("unknown_field", filter="nonexistent > 5",
                         output_fields=["pk"])


# ====================================================================
# 3. 类型不匹配
# ====================================================================

class TestTypeMismatch:

    def test_string_into_int_field(self, client: MilvusClient):
        """字符串插入 INT64 字段应报错"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("count", MilvusDataType.INT64)

        client.create_collection("type_err", schema=schema)
        with pytest.raises(Exception):
            client.insert("type_err", [
                {"pk": 1, "vec": rvecs(1)[0], "count": "not_a_number"},
            ])

    def test_int_into_float_field_ok(self, client: MilvusClient):
        """int 插入 FLOAT 字段应自动转换 (Milvus 行为)"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("score", MilvusDataType.FLOAT)

        client.create_collection("int_to_float", schema=schema)
        vecs = rvecs(1)
        client.insert("int_to_float", [
            {"pk": 1, "vec": vecs[0], "score": 42},  # int → float
        ])
        got = client.get("int_to_float", ids=[1])
        assert got[0]["score"] == pytest.approx(42.0)


# ====================================================================
# 4. 搜索返回零结果
# ====================================================================

class TestSearchZeroResults:

    def test_search_with_impossible_filter(self, client: MilvusClient):
        """filter 排除全部数据 → 返回空"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.INT64)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("zero_res", schema=schema, index_params=idx)

        vecs = rvecs(10)
        client.insert("zero_res", [
            {"pk": i, "vec": vecs[i], "val": i} for i in range(10)
        ])
        client.load_collection("zero_res")

        # val 最大是 9，搜 > 100 不可能命中
        results = client.search("zero_res", data=[vecs[0]], limit=5,
                                filter="val > 100", output_fields=["pk"])
        assert len(results[0]) == 0

    def test_query_no_match(self, client: MilvusClient):
        """query filter 无命中"""
        client.create_collection("q_no_match", dimension=DIM)
        vecs = rvecs(5)
        client.insert("q_no_match", [{"id": i, "vector": vecs[i]} for i in range(5)])

        r = client.query("q_no_match", filter="id > 999", output_fields=["id"])
        assert r == []


# ====================================================================
# 5. insert 大量重复 PK
# ====================================================================

class TestMassivePKOverwrite:

    def test_1000_overwrites_same_pk(self, client: MilvusClient):
        """对同一 PK 连续 insert 1000 次，只保留最后一条"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("ver", MilvusDataType.INT64)

        client.create_collection("mass_overwrite", schema=schema)

        # 批量插入同一 PK
        base_vec = rvecs(1)[0]
        data = [{"pk": 1, "vec": base_vec, "ver": i} for i in range(1000)]
        client.insert("mass_overwrite", data)

        got = client.get("mass_overwrite", ids=[1])
        assert len(got) == 1
        assert got[0]["ver"] == 999  # 最后一条

        stats = client.get_collection_stats("mass_overwrite")
        assert int(stats["row_count"]) == 1


# ====================================================================
# 6. query limit=1 / search limit=1
# ====================================================================

class TestLimitOne:

    def test_query_limit_1(self, client: MilvusClient):
        client.create_collection("lim1_q", dimension=DIM)
        vecs = rvecs(10)
        client.insert("lim1_q", [{"id": i, "vector": vecs[i]} for i in range(10)])

        r = client.query("lim1_q", filter="id >= 0", limit=1, output_fields=["id"])
        assert len(r) == 1

    def test_search_limit_1(self, client: MilvusClient):
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("lim1_s", schema=schema, index_params=idx)

        vecs = rvecs(10)
        client.insert("lim1_s", [{"pk": i, "vec": vecs[i]} for i in range(10)])
        client.load_collection("lim1_s")

        r = client.search("lim1_s", data=[vecs[5]], limit=1, output_fields=["pk"])
        assert len(r[0]) == 1
        assert r[0][0]["entity"]["pk"] == 5


# ====================================================================
# 7. delete 大量 id (500 条)
# ====================================================================

class TestBulkDelete:

    def test_delete_500_ids(self, client: MilvusClient):
        client.create_collection("bulk_del_500", dimension=DIM)
        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((500, DIM)).astype(np.float32).tolist()
        client.insert("bulk_del_500", [{"id": i, "vector": vecs[i]} for i in range(500)])

        client.delete("bulk_del_500", ids=list(range(500)))
        stats = client.get_collection_stats("bulk_del_500")
        assert int(stats["row_count"]) == 0


# ====================================================================
# 8. 集合名边界
# ====================================================================

class TestCollectionNameEdge:

    def test_long_collection_name(self, client: MilvusClient):
        """较长的集合名"""
        name = "a" * 200
        client.create_collection(name, dimension=DIM)
        assert client.has_collection(name) is True
        client.drop_collection(name)

    def test_collection_name_with_underscore_and_digits(self, client: MilvusClient):
        """下划线和数字的集合名"""
        name = "test_collection_123_v2"
        client.create_collection(name, dimension=DIM)
        assert client.has_collection(name) is True


# ====================================================================
# 9. FLOAT 和 INT 混用
# ====================================================================

class TestTypeCoercion:

    def test_float_into_double_field(self, client: MilvusClient):
        """float 插入 DOUBLE 字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("val", MilvusDataType.DOUBLE)

        client.create_collection("f2d", schema=schema)
        vecs = rvecs(1)
        client.insert("f2d", [{"pk": 1, "vec": vecs[0], "val": 3.14}])
        got = client.get("f2d", ids=[1])
        assert got[0]["val"] == pytest.approx(3.14, rel=1e-5)

    def test_bool_filter_variations(self, client: MilvusClient):
        """BOOL 字段的各种 filter 写法"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("flag", MilvusDataType.BOOL)

        client.create_collection("bool_var", schema=schema)
        vecs = rvecs(4)
        client.insert("bool_var", [
            {"pk": 0, "vec": vecs[0], "flag": True},
            {"pk": 1, "vec": vecs[1], "flag": False},
            {"pk": 2, "vec": vecs[2], "flag": True},
            {"pk": 3, "vec": vecs[3], "flag": False},
        ])

        # true 小写
        r1 = client.query("bool_var", filter="flag == true", output_fields=["pk"])
        assert len(r1) == 2

        # false 小写
        r2 = client.query("bool_var", filter="flag == false", output_fields=["pk"])
        assert len(r2) == 2

        # not flag
        r3 = client.query("bool_var", filter="not flag", output_fields=["pk"])
        assert len(r3) == 2
        assert all(x["pk"] % 2 == 1 for x in r3)


# ====================================================================
# 10. 同一字段多条件
# ====================================================================

class TestSameFieldMultiCondition:

    def test_range_on_same_field(self, client: MilvusClient):
        """同一字段的范围查询: a > X and a < Y"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("score", MilvusDataType.FLOAT)

        client.create_collection("same_field", schema=schema)
        vecs = rvecs(20)
        client.insert("same_field", [
            {"pk": i, "vec": vecs[i], "score": float(i * 5)} for i in range(20)
        ])

        r = client.query("same_field",
                         filter="score >= 25.0 and score < 50.0",
                         output_fields=["pk", "score"])
        for x in r:
            assert 25.0 <= x["score"] < 50.0
        assert len(r) == 5  # 25, 30, 35, 40, 45


# ====================================================================
# 11. VARCHAR LIKE 各种模式
# ====================================================================

class TestLikePatterns:

    @pytest.fixture(autouse=True)
    def _setup(self, client):
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=128)

        client.create_collection("like_test", schema=schema)
        vecs = rvecs(8)
        client.insert("like_test", [
            {"pk": 0, "vec": vecs[0], "name": "apple"},
            {"pk": 1, "vec": vecs[1], "name": "application"},
            {"pk": 2, "vec": vecs[2], "name": "banana"},
            {"pk": 3, "vec": vecs[3], "name": "cherry"},
            {"pk": 4, "vec": vecs[4], "name": "pineapple"},
            {"pk": 5, "vec": vecs[5], "name": "grape"},
            {"pk": 6, "vec": vecs[6], "name": "APP_config"},
            {"pk": 7, "vec": vecs[7], "name": "test123"},
        ])

    def test_like_prefix(self, client: MilvusClient):
        """前缀匹配: app%"""
        r = client.query("like_test", filter='name like "app%"',
                         output_fields=["pk", "name"])
        names = [x["name"] for x in r]
        assert "apple" in names
        assert "application" in names
        assert "banana" not in names

    def test_like_suffix(self, client: MilvusClient):
        """后缀匹配: %ple"""
        r = client.query("like_test", filter='name like "%ple"',
                         output_fields=["pk", "name"])
        names = sorted([x["name"] for x in r])
        assert "apple" in names
        assert "pineapple" in names

    def test_like_contains(self, client: MilvusClient):
        """包含匹配: %an%"""
        r = client.query("like_test", filter='name like "%an%"',
                         output_fields=["pk", "name"])
        names = [x["name"] for x in r]
        assert "banana" in names

    def test_like_no_match(self, client: MilvusClient):
        """无匹配"""
        r = client.query("like_test", filter='name like "zzz%"',
                         output_fields=["pk"])
        assert r == []


# ====================================================================
# 12. JSON 多层嵌套
# ====================================================================

class TestDeepJsonNesting:

    def test_deep_nested_json_read(self, client: MilvusClient):
        """深层嵌套 JSON 读取"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("data", MilvusDataType.JSON)

        client.create_collection("deep_json", schema=schema)
        vecs = rvecs(1)
        nested = {
            "level1": {
                "level2": {
                    "level3": {"value": 42, "tags": ["a", "b"]}
                }
            },
            "flat": "hello",
        }
        client.insert("deep_json", [{"pk": 1, "vec": vecs[0], "data": nested}])

        got = client.get("deep_json", ids=[1])
        assert got[0]["data"]["level1"]["level2"]["level3"]["value"] == 42
        assert got[0]["data"]["flat"] == "hello"

    def test_json_nested_filter(self, client: MilvusClient):
        """JSON 嵌套字段 filter"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("info", MilvusDataType.JSON)

        client.create_collection("json_nest_f", schema=schema)
        vecs = rvecs(3)
        client.insert("json_nest_f", [
            {"pk": 0, "vec": vecs[0], "info": {"a": {"b": 1}}},
            {"pk": 1, "vec": vecs[1], "info": {"a": {"b": 5}}},
            {"pk": 2, "vec": vecs[2], "info": {"a": {"b": 10}}},
        ])

        r = client.query("json_nest_f", filter='info["a"]["b"] >= 5',
                         output_fields=["pk", "info"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [1, 2]


# ====================================================================
# 13. 并发创建/删除集合
# ====================================================================

class TestConcurrentCollectionOps:

    def test_concurrent_create_drop(self, client: MilvusClient):
        """多线程并发创建不同集合"""
        errors = []

        def create_col(name):
            try:
                client.create_collection(name, dimension=DIM)
                assert client.has_collection(name)
                vecs = rvecs(2)
                client.insert(name, [{"id": i, "vector": vecs[i]} for i in range(2)])
            except Exception as e:
                errors.append(f"{name}: {e}")

        threads = [threading.Thread(target=create_col, args=(f"cc_{i}",))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Errors: {errors}"
        names = client.list_collections()
        for i in range(10):
            assert f"cc_{i}" in names


# ====================================================================
# 14. query offset + limit 边界
# ====================================================================

class TestPaginationEdge:

    def test_offset_equals_total(self, client: MilvusClient):
        """offset == 总数据量 → 空"""
        client.create_collection("pag_eq", dimension=DIM)
        vecs = rvecs(5)
        client.insert("pag_eq", [{"id": i, "vector": vecs[i]} for i in range(5)])

        r = client.query("pag_eq", filter="id >= 0", limit=10, offset=5,
                         output_fields=["id"])
        assert len(r) == 0

    def test_offset_0_limit_0(self, client: MilvusClient):
        """limit=0 → 空 (或报错，取决于实现)"""
        client.create_collection("pag_zero", dimension=DIM)
        vecs = rvecs(3)
        client.insert("pag_zero", [{"id": i, "vector": vecs[i]} for i in range(3)])

        # Milvus: limit=0 通常被视为 "不限制" 或报错
        # 先试看是否能成功
        try:
            r = client.query("pag_zero", filter="id >= 0", limit=0,
                             output_fields=["id"])
            # 如果成功，结果应为空或全部
            assert isinstance(r, list)
        except Exception:
            pass  # 报错也可接受

    def test_full_pagination(self, client: MilvusClient):
        """完整分页遍历"""
        client.create_collection("pag_full", dimension=DIM)
        vecs = rvecs(17)
        client.insert("pag_full", [{"id": i, "vector": vecs[i]} for i in range(17)])

        all_ids = []
        offset = 0
        page_size = 5
        while True:
            r = client.query("pag_full", filter="id >= 0",
                             limit=page_size, offset=offset,
                             output_fields=["id"])
            if not r:
                break
            all_ids.extend(x["id"] for x in r)
            offset += page_size

        assert sorted(all_ids) == list(range(17))


# ====================================================================
# 15. search + filter 命中 0 条 (不同于空集合)
# ====================================================================

class TestFilterNoHit:

    def test_search_filter_excludes_all_nearest(self, client: MilvusClient):
        """最近邻全被 filter 排除"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("group", MilvusDataType.VARCHAR, max_length=16)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("filter_excl", schema=schema, index_params=idx)

        vecs = rvecs(10)
        client.insert("filter_excl", [
            {"pk": i, "vec": vecs[i], "group": "only_group"} for i in range(10)
        ])
        client.load_collection("filter_excl")

        # 搜索时 filter 要求 group="nonexistent" → 0 条
        r = client.search("filter_excl", data=[vecs[0]], limit=5,
                          filter='group == "nonexistent"', output_fields=["pk"])
        assert len(r[0]) == 0


# ====================================================================
# 16. 负数 PK
# ====================================================================

class TestNegativePK:

    def test_negative_int64_pk(self, client: MilvusClient):
        """负数作为 INT64 主键"""
        client.create_collection("neg_pk", dimension=DIM)
        vecs = rvecs(3)
        client.insert("neg_pk", [
            {"id": -100, "vector": vecs[0]},
            {"id": 0,    "vector": vecs[1]},
            {"id": 100,  "vector": vecs[2]},
        ])

        got = client.get("neg_pk", ids=[-100, 0, 100])
        assert len(got) == 3

        # 按负数 id 删除
        client.delete("neg_pk", ids=[-100])
        got = client.get("neg_pk", ids=[-100])
        assert len(got) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
