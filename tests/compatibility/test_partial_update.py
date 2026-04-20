"""
Partial Update (Upsert) 兼容性测试 — 通过 pymilvus MilvusClient 验证
MilvusLite 的 partial upsert 功能。

Partial Update 语义:
  - upsert 时只提供部分字段 + 主键
  - 若 pk 已存在: 合并旧记录，新字段覆盖旧字段，未提供的字段保持不变
  - 若 pk 不存在: 当作新记录插入 (必须提供全部必填字段)

测试覆盖:
  1. 基本 partial update — 只更新部分标量字段
  2. partial update 保持向量不变
  3. partial update 更新向量但保持标量不变
  4. 混合批次 — 部分已存在 + 部分新记录
  5. 多次连续 partial update
  6. flush 后的 partial update (读旧记录需走 segment)
  7. 动态字段的 partial update
  8. JSON 字段的 partial update
  9. ARRAY 字段的 partial update
 10. nullable 字段的 partial update (置为 None)
 11. partial update 后搜索结果正确
 12. partial update 后 filter 查询正确
 13. partial update + delete 交互
 14. VARCHAR 主键的 partial update
 15. partial update 不影响其他记录
 16. 全字段 upsert 退化为完整覆盖
 17. partial update 后 count 不变
 18. 分区内的 partial update
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest
from pymilvus import MilvusClient, DataType as MilvusDataType

from milvus_lite.adapter.grpc.server import start_server_in_thread

DIM = 8
SEED = 42


@pytest.fixture(scope="module")
def server():
    data_dir = tempfile.mkdtemp(prefix="partial_test_")
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


def rvecs(n: int, dim: int = DIM, seed: int = SEED) -> list[list[float]]:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32).tolist()


def make_standard_collection(client, name):
    """创建标准测试集合: pk(INT64) + vec + title(nullable) + score(nullable)"""
    schema = client.create_schema()
    schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
    schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", MilvusDataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field("score", MilvusDataType.FLOAT, nullable=True)
    client.create_collection(name, schema=schema)


# ====================================================================
# 1. 基本 partial update — 只更新部分标量字段
# ====================================================================

class TestBasicPartialUpdate:

    def test_update_single_scalar_field(self, client: MilvusClient):
        """只更新 title，保持 vec 和 score 不变"""
        make_standard_collection(client, "partial_basic")
        vecs = rvecs(1)
        client.insert("partial_basic", [
            {"pk": 1, "vec": vecs[0], "title": "original", "score": 88.5},
        ])

        client.upsert("partial_basic", [
            {"pk": 1, "title": "updated"},
        ], partial_update=True)

        got = client.get("partial_basic", ids=[1])
        assert len(got) == 1
        assert got[0]["title"] == "updated"
        assert got[0]["score"] == pytest.approx(88.5)
        # 向量也应该保持不变
        assert got[0]["vec"] == pytest.approx(vecs[0], rel=1e-5)

    def test_update_multiple_scalar_fields(self, client: MilvusClient):
        """同时更新 title 和 score"""
        make_standard_collection(client, "partial_multi")
        vecs = rvecs(1)
        client.insert("partial_multi", [
            {"pk": 1, "vec": vecs[0], "title": "old", "score": 10.0},
        ])

        client.upsert("partial_multi", [
            {"pk": 1, "title": "new", "score": 99.0},
        ], partial_update=True)

        got = client.get("partial_multi", ids=[1])
        assert got[0]["title"] == "new"
        assert got[0]["score"] == pytest.approx(99.0)
        assert got[0]["vec"] == pytest.approx(vecs[0], rel=1e-5)


# ====================================================================
# 2. partial update 保持向量不变
# ====================================================================

class TestVectorPreservation:

    def test_vector_unchanged_after_partial_update(self, client: MilvusClient):
        """partial update 标量字段后，向量搜索仍能正确找到记录"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("label", MilvusDataType.VARCHAR, max_length=64, nullable=True)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="L2",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("partial_vec_keep", schema=schema, index_params=idx)

        vecs = rvecs(5)
        client.insert("partial_vec_keep", [
            {"pk": i, "vec": vecs[i], "label": f"orig_{i}"} for i in range(5)
        ])

        # partial update pk=2 的 label
        client.upsert("partial_vec_keep", [{"pk": 2, "label": "changed"}],
                      partial_update=True)

        client.load_collection("partial_vec_keep")

        # 用 pk=2 的原始向量搜索，应该还是最近邻
        results = client.search("partial_vec_keep", data=[vecs[2]], limit=1,
                                output_fields=["pk", "label"])
        assert results[0][0]["entity"]["pk"] == 2
        assert results[0][0]["entity"]["label"] == "changed"
        assert results[0][0]["distance"] < 1e-4  # 距离应接近 0


# ====================================================================
# 3. partial update 更新向量但保持标量不变
# ====================================================================

class TestVectorUpdate:

    def test_update_vector_only(self, client: MilvusClient):
        """只更新向量，标量字段保持不变"""
        make_standard_collection(client, "partial_vec_update")
        old_vec = rvecs(1, seed=1)[0]
        client.insert("partial_vec_update", [
            {"pk": 1, "vec": old_vec, "title": "keep_me", "score": 42.0},
        ])

        new_vec = rvecs(1, seed=999)[0]
        client.upsert("partial_vec_update", [
            {"pk": 1, "vec": new_vec},
        ], partial_update=True)

        got = client.get("partial_vec_update", ids=[1])
        assert got[0]["title"] == "keep_me"
        assert got[0]["score"] == pytest.approx(42.0)
        assert got[0]["vec"] == pytest.approx(new_vec, rel=1e-5)


# ====================================================================
# 4. 混合批次 — 部分已存在 + 部分新记录
# ====================================================================

class TestMixedBatch:

    def test_mixed_existing_and_new(self, client: MilvusClient):
        """partial update 已有记录 + 全量 upsert 新记录 (需分两批)"""
        make_standard_collection(client, "partial_mixed")
        vecs = rvecs(3)
        client.insert("partial_mixed", [
            {"pk": 1, "vec": vecs[0], "title": "one", "score": 1.0},
            {"pk": 2, "vec": vecs[1], "title": "two", "score": 2.0},
        ])

        # pymilvus partial_update 要求同批次字段数一致，需拆两批
        # Batch 1: partial update 已有记录
        client.upsert("partial_mixed", [
            {"pk": 1, "title": "one_updated"},
        ], partial_update=True)

        # Batch 2: 全量 upsert 新记录
        new_vecs = rvecs(2, seed=88)
        client.upsert("partial_mixed", [
            {"pk": 3, "vec": new_vecs[0], "title": "three", "score": 3.0},
        ])

        r1 = client.get("partial_mixed", ids=[1])[0]
        assert r1["title"] == "one_updated"
        assert r1["score"] == pytest.approx(1.0)  # 保持不变
        assert r1["vec"] == pytest.approx(vecs[0], rel=1e-5)  # 保持不变

        r2 = client.get("partial_mixed", ids=[2])[0]
        assert r2["title"] == "two"  # 未被触及

        r3 = client.get("partial_mixed", ids=[3])[0]
        assert r3["title"] == "three"


# ====================================================================
# 5. 多次连续 partial update
# ====================================================================

class TestConsecutiveUpdates:

    def test_multiple_partial_updates(self, client: MilvusClient):
        """对同一条记录做多次 partial update"""
        make_standard_collection(client, "partial_multi_round")
        vecs = rvecs(1)
        client.insert("partial_multi_round", [
            {"pk": 1, "vec": vecs[0], "title": "v1", "score": 10.0},
        ])

        # Round 1: 只更新 title
        client.upsert("partial_multi_round", [{"pk": 1, "title": "v2"}],
                      partial_update=True)
        got = client.get("partial_multi_round", ids=[1])[0]
        assert got["title"] == "v2"
        assert got["score"] == pytest.approx(10.0)

        # Round 2: 只更新 score
        client.upsert("partial_multi_round", [{"pk": 1, "score": 20.0}],
                      partial_update=True)
        got = client.get("partial_multi_round", ids=[1])[0]
        assert got["title"] == "v2"  # 上一轮的 title 仍在
        assert got["score"] == pytest.approx(20.0)

        # Round 3: 更新 title + score
        client.upsert("partial_multi_round", [{"pk": 1, "title": "v3", "score": 30.0}],
                      partial_update=True)
        got = client.get("partial_multi_round", ids=[1])[0]
        assert got["title"] == "v3"
        assert got["score"] == pytest.approx(30.0)
        assert got["vec"] == pytest.approx(vecs[0], rel=1e-5)  # 向量始终不变


# ====================================================================
# 6. flush 后的 partial update
# ====================================================================

class TestPartialUpdateAfterFlush:

    def test_partial_update_reads_from_segments(self, client: MilvusClient):
        """flush 后旧记录落到 segment，partial update 仍能正确合并"""
        make_standard_collection(client, "partial_flush")
        vecs = rvecs(1)
        client.insert("partial_flush", [
            {"pk": 1, "vec": vecs[0], "title": "persisted", "score": 77.0},
        ])
        client.flush("partial_flush")

        # partial update
        client.upsert("partial_flush", [{"pk": 1, "score": 88.0}],
                      partial_update=True)

        got = client.get("partial_flush", ids=[1])[0]
        assert got["title"] == "persisted"  # 从 segment 读取
        assert got["score"] == pytest.approx(88.0)


# ====================================================================
# 7. 动态字段的 partial update
# ====================================================================

class TestDynamicFieldPartialUpdate:

    def test_partial_update_preserves_dynamic_fields(self, client: MilvusClient):
        """partial update 保持旧的动态字段，更新指定字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.enable_dynamic_field = True

        client.create_collection("partial_dyn", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_dyn", [
            {"pk": 1, "vec": vecs[0], "color": "red", "size": 42, "tag": "important"},
        ])

        # partial update: 只更新 color
        client.upsert("partial_dyn", [{"pk": 1, "color": "blue"}],
                      partial_update=True)

        # 动态字段需要显式指定 output_fields
        got = client.get("partial_dyn", ids=[1],
                         output_fields=["pk", "vec", "color", "size", "tag"])[0]
        assert got["color"] == "blue"
        assert int(got["size"]) == 42   # 动态字段数字可能被 JSON 序列化为字符串
        assert got["tag"] == "important"  # 保持不变

    def test_partial_update_adds_new_dynamic_field(self, client: MilvusClient):
        """partial update 可以添加新的动态字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.enable_dynamic_field = True

        client.create_collection("partial_dyn_new", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_dyn_new", [
            {"pk": 1, "vec": vecs[0], "color": "red"},
        ])

        # partial update: 添加新字段 priority
        client.upsert("partial_dyn_new", [{"pk": 1, "priority": "high"}],
                      partial_update=True)

        got = client.get("partial_dyn_new", ids=[1],
                         output_fields=["pk", "vec", "color", "priority"])[0]
        assert got["color"] == "red"       # 保持
        assert got["priority"] == "high"   # 新增


# ====================================================================
# 8. JSON 字段的 partial update
# ====================================================================

class TestJsonPartialUpdate:

    def test_json_field_replacement(self, client: MilvusClient):
        """partial update 替换整个 JSON 字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("meta", MilvusDataType.JSON, nullable=True)
        schema.add_field("name", MilvusDataType.VARCHAR, max_length=64, nullable=True)

        client.create_collection("partial_json", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_json", [
            {"pk": 1, "vec": vecs[0], "meta": {"env": "dev", "v": 1}, "name": "test"},
        ])

        # 只更新 meta
        client.upsert("partial_json", [
            {"pk": 1, "meta": {"env": "prod", "v": 2, "region": "us"}},
        ], partial_update=True)

        got = client.get("partial_json", ids=[1])[0]
        assert got["meta"]["env"] == "prod"
        assert got["meta"]["v"] == 2
        assert got["meta"]["region"] == "us"
        assert got["name"] == "test"  # 保持不变


# ====================================================================
# 9. ARRAY 字段的 partial update
# ====================================================================

class TestArrayPartialUpdate:

    def test_array_field_replacement(self, client: MilvusClient):
        """partial update 替换整个 ARRAY 字段"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("tags", MilvusDataType.ARRAY,
                         element_type=MilvusDataType.VARCHAR,
                         max_capacity=10, max_length=64,
                         nullable=True)
        schema.add_field("score", MilvusDataType.FLOAT, nullable=True)

        client.create_collection("partial_arr", schema=schema)
        vecs = rvecs(1)
        client.insert("partial_arr", [
            {"pk": 1, "vec": vecs[0], "tags": ["a", "b"], "score": 1.0},
        ])

        client.upsert("partial_arr", [
            {"pk": 1, "tags": ["x", "y", "z"]},
        ], partial_update=True)

        got = client.get("partial_arr", ids=[1])[0]
        assert got["tags"] == ["x", "y", "z"]
        assert got["score"] == pytest.approx(1.0)  # 保持不变


# ====================================================================
# 10. nullable 字段置为 None
# ====================================================================

class TestNullablePartialUpdate:

    def test_set_field_to_none(self, client: MilvusClient):
        """partial update 将 nullable 字段显式设为 None"""
        make_standard_collection(client, "partial_null")
        vecs = rvecs(1)
        client.insert("partial_null", [
            {"pk": 1, "vec": vecs[0], "title": "has_value", "score": 50.0},
        ])

        # 将 title 设为 None
        client.upsert("partial_null", [
            {"pk": 1, "title": None},
        ], partial_update=True)

        got = client.get("partial_null", ids=[1])[0]
        assert got["title"] is None
        assert got["score"] == pytest.approx(50.0)  # 保持不变

    def test_set_none_to_value(self, client: MilvusClient):
        """将原来为 None 的字段更新为有值"""
        make_standard_collection(client, "partial_fill")
        vecs = rvecs(1)
        client.insert("partial_fill", [
            {"pk": 1, "vec": vecs[0], "title": None, "score": None},
        ])

        client.upsert("partial_fill", [
            {"pk": 1, "title": "now_has_value"},
        ], partial_update=True)

        got = client.get("partial_fill", ids=[1])[0]
        assert got["title"] == "now_has_value"
        assert got["score"] is None  # 保持不变


# ====================================================================
# 11. partial update 后搜索结果正确
# ====================================================================

class TestSearchAfterPartialUpdate:

    def test_search_returns_updated_fields(self, client: MilvusClient):
        """partial update 后搜索返回更新后的字段值"""
        schema = client.create_schema()
        schema.add_field("pk", MilvusDataType.INT64, is_primary=True)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("category", MilvusDataType.VARCHAR, max_length=32, nullable=True)

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 64})
        client.create_collection("partial_search", schema=schema, index_params=idx)

        vecs = rvecs(5)
        client.insert("partial_search", [
            {"pk": i, "vec": vecs[i], "category": "old"} for i in range(5)
        ])

        # partial update pk=0 的 category
        client.upsert("partial_search", [{"pk": 0, "category": "new"}],
                      partial_update=True)

        client.load_collection("partial_search")

        results = client.search("partial_search", data=[vecs[0]], limit=1,
                                output_fields=["pk", "category"])
        assert results[0][0]["entity"]["pk"] == 0
        assert results[0][0]["entity"]["category"] == "new"


# ====================================================================
# 12. partial update 后 filter 查询正确
# ====================================================================

class TestFilterAfterPartialUpdate:

    def test_filter_sees_updated_values(self, client: MilvusClient):
        """partial update 后 filter 能查到更新后的值"""
        make_standard_collection(client, "partial_filter")
        vecs = rvecs(3)
        client.insert("partial_filter", [
            {"pk": 1, "vec": vecs[0], "title": "a", "score": 10.0},
            {"pk": 2, "vec": vecs[1], "title": "b", "score": 20.0},
            {"pk": 3, "vec": vecs[2], "title": "c", "score": 30.0},
        ])

        # 将 pk=2 的 score 从 20 改为 99
        client.upsert("partial_filter", [{"pk": 2, "score": 99.0}],
                      partial_update=True)

        r = client.query("partial_filter", filter="score > 50",
                         output_fields=["pk", "score"])
        pks = sorted([x["pk"] for x in r])
        assert pks == [2]  # 只有 pk=2 的 score=99 > 50


# ====================================================================
# 13. partial update + delete 交互
# ====================================================================

class TestPartialUpdateAndDelete:

    def test_partial_update_deleted_record(self, client: MilvusClient):
        """删除后 upsert 同一 pk: 应当作全新插入"""
        make_standard_collection(client, "partial_del")
        vecs = rvecs(2)
        client.insert("partial_del", [
            {"pk": 1, "vec": vecs[0], "title": "alive", "score": 1.0},
        ])

        client.delete("partial_del", ids=[1])

        # upsert 已删除的 pk — 由于找不到旧记录，当作新记录
        new_vec = rvecs(1, seed=99)[0]
        client.upsert("partial_del", [
            {"pk": 1, "vec": new_vec, "title": "resurrected", "score": 2.0},
        ])

        got = client.get("partial_del", ids=[1])
        assert len(got) == 1
        assert got[0]["title"] == "resurrected"

    def test_delete_after_partial_update(self, client: MilvusClient):
        """partial update 后再 delete，记录应消失"""
        make_standard_collection(client, "partial_then_del")
        vecs = rvecs(1)
        client.insert("partial_then_del", [
            {"pk": 1, "vec": vecs[0], "title": "orig", "score": 1.0},
        ])

        client.upsert("partial_then_del", [{"pk": 1, "title": "updated"}],
                      partial_update=True)
        client.delete("partial_then_del", ids=[1])

        got = client.get("partial_then_del", ids=[1])
        assert len(got) == 0


# ====================================================================
# 14. VARCHAR 主键的 partial update
# ====================================================================

class TestVarcharPKPartialUpdate:

    def test_varchar_pk_partial_update(self, client: MilvusClient):
        """VARCHAR 主键的 partial update"""
        schema = client.create_schema()
        schema.add_field("id", MilvusDataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vec", MilvusDataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("status", MilvusDataType.VARCHAR, max_length=32, nullable=True)
        schema.add_field("count", MilvusDataType.INT64, nullable=True)

        client.create_collection("partial_vpk", schema=schema)
        vecs = rvecs(2)
        client.insert("partial_vpk", [
            {"id": "doc_a", "vec": vecs[0], "status": "draft", "count": 0},
            {"id": "doc_b", "vec": vecs[1], "status": "published", "count": 100},
        ])

        # partial update doc_a
        client.upsert("partial_vpk", [{"id": "doc_a", "status": "published"}],
                      partial_update=True)

        got = client.get("partial_vpk", ids=["doc_a"])[0]
        assert got["status"] == "published"
        assert got["count"] == 0  # 保持不变
        assert got["vec"] == pytest.approx(vecs[0], rel=1e-5)

        # doc_b 不受影响
        got_b = client.get("partial_vpk", ids=["doc_b"])[0]
        assert got_b["status"] == "published"
        assert got_b["count"] == 100


# ====================================================================
# 15. partial update 不影响其他记录
# ====================================================================

class TestNoSideEffects:

    def test_other_records_unchanged(self, client: MilvusClient):
        """partial update 一条记录不影响其他记录"""
        make_standard_collection(client, "partial_no_side")
        vecs = rvecs(5)
        original = [
            {"pk": i, "vec": vecs[i], "title": f"t{i}", "score": float(i * 10)}
            for i in range(5)
        ]
        client.insert("partial_no_side", original)

        # 只更新 pk=2
        client.upsert("partial_no_side", [{"pk": 2, "title": "changed"}],
                      partial_update=True)

        # 检查其他记录没变
        for i in [0, 1, 3, 4]:
            got = client.get("partial_no_side", ids=[i])[0]
            assert got["title"] == f"t{i}"
            assert got["score"] == pytest.approx(float(i * 10))


# ====================================================================
# 16. 全字段 upsert 退化为完整覆盖
# ====================================================================

class TestFullUpsertFallback:

    def test_full_upsert_replaces_all(self, client: MilvusClient):
        """提供所有字段的 upsert 等价于完整覆盖"""
        make_standard_collection(client, "partial_full")
        vecs = rvecs(2)
        client.insert("partial_full", [
            {"pk": 1, "vec": vecs[0], "title": "old", "score": 1.0},
        ])

        new_vec = rvecs(1, seed=88)[0]
        client.upsert("partial_full", [
            {"pk": 1, "vec": new_vec, "title": "new", "score": 99.0},
        ])

        got = client.get("partial_full", ids=[1])[0]
        assert got["title"] == "new"
        assert got["score"] == pytest.approx(99.0)
        assert got["vec"] == pytest.approx(new_vec, rel=1e-5)


# ====================================================================
# 17. partial update 后 count 不变
# ====================================================================

class TestCountAfterPartialUpdate:

    def test_row_count_stable(self, client: MilvusClient):
        """partial update 已有记录不增加 row_count"""
        make_standard_collection(client, "partial_count")
        vecs = rvecs(5)
        client.insert("partial_count", [
            {"pk": i, "vec": vecs[i], "title": f"t{i}", "score": 0.0}
            for i in range(5)
        ])

        stats_before = client.get_collection_stats("partial_count")
        assert int(stats_before["row_count"]) == 5

        # partial update 3 条
        client.upsert("partial_count", [
            {"pk": 0, "title": "u0"},
            {"pk": 2, "title": "u2"},
            {"pk": 4, "title": "u4"},
        ], partial_update=True)

        stats_after = client.get_collection_stats("partial_count")
        assert int(stats_after["row_count"]) == 5  # 仍然是 5


# ====================================================================
# 18. 分区内的 partial update
# ====================================================================

class TestPartitionPartialUpdate:

    def test_partial_update_in_partition(self, client: MilvusClient):
        """在指定分区中进行 partial update"""
        make_standard_collection(client, "partial_part")
        client.create_partition("partial_part", "region_a")

        vecs = rvecs(2)
        client.insert("partial_part", [
            {"pk": 1, "vec": vecs[0], "title": "orig", "score": 5.0},
        ], partition_name="region_a")

        client.upsert("partial_part", [
            {"pk": 1, "title": "updated_in_partition"},
        ], partition_name="region_a", partial_update=True)

        got = client.get("partial_part", ids=[1])
        assert got[0]["title"] == "updated_in_partition"
        assert got[0]["score"] == pytest.approx(5.0)


# ====================================================================
# 19. 大批量 partial update
# ====================================================================

class TestBulkPartialUpdate:

    def test_bulk_partial_update(self, client: MilvusClient):
        """批量 partial update 100 条记录"""
        make_standard_collection(client, "partial_bulk")
        vecs = rvecs(100, seed=1)
        client.insert("partial_bulk", [
            {"pk": i, "vec": vecs[i], "title": f"orig_{i}", "score": float(i)}
            for i in range(100)
        ])

        # partial update 所有偶数 pk 的 title
        updates = [{"pk": i, "title": f"updated_{i}"} for i in range(0, 100, 2)]
        client.upsert("partial_bulk", updates, partial_update=True)

        # 验证偶数 pk 更新了，奇数没变
        for i in [0, 10, 50, 98]:
            got = client.get("partial_bulk", ids=[i])[0]
            assert got["title"] == f"updated_{i}"
            assert got["score"] == pytest.approx(float(i))

        for i in [1, 11, 51, 99]:
            got = client.get("partial_bulk", ids=[i])[0]
            assert got["title"] == f"orig_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
