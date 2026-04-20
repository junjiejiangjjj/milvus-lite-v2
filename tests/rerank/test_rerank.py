"""RERANK function tests.

Uses a mock RerankProvider to avoid real API calls. Tests the full
search → rerank pipeline and factory/schema validation.

Also tests the DecayReranker (local computation, no API calls).
"""

import math
import tempfile
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from milvus_lite.rerank.protocol import RerankProvider, RerankResult
from milvus_lite.rerank.decay import DecayReranker
from milvus_lite.rerank.factory import create_rerank_provider
from milvus_lite.embedding.protocol import EmbeddingProvider
from milvus_lite.schema.types import (
    CollectionSchema, DataType, FieldSchema, Function, FunctionType,
)
from milvus_lite.engine.collection import Collection


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

class MockRerankProvider(RerankProvider):
    """Deterministic mock: scores by substring overlap with the query."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        query_words = set(query.lower().split())
        scored = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words | doc_words), 1)
            scored.append(RerankResult(index=i, relevance_score=score))
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return scored


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock: hashes text to a fixed-dim vector."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_text(text)

    def _hash_text(self, text: str) -> List[float]:
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def _mock_embedding_factory(params):
    return MockEmbeddingProvider(dim=params.get("dimensions", 8))


def _mock_rerank_factory(params):
    return MockRerankProvider()


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

def _make_schema_with_rerank(dim=8):
    """Schema with TEXT_EMBEDDING + RERANK functions."""
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim,
                    is_function_output=True),
    ], functions=[
        Function(
            name="text_emb",
            function_type=FunctionType.TEXT_EMBEDDING,
            input_field_names=["text"],
            output_field_names=["vec"],
            params={"provider": "openai", "model_name": "mock", "dimensions": dim},
        ),
        Function(
            name="my_reranker",
            function_type=FunctionType.RERANK,
            input_field_names=["text"],
            output_field_names=[],
            params={"provider": "cohere"},
        ),
    ])


def _make_schema_no_rerank(dim=8):
    """Schema with TEXT_EMBEDDING only, no rerank."""
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim,
                    is_function_output=True),
    ], functions=[
        Function(
            name="text_emb",
            function_type=FunctionType.TEXT_EMBEDDING,
            input_field_names=["text"],
            output_field_names=["vec"],
            params={"provider": "openai", "model_name": "mock", "dimensions": dim},
        ),
    ])


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_missing_provider(self):
        with pytest.raises(ValueError, match="requires 'provider'"):
            create_rerank_provider({})

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown rerank provider"):
            create_rerank_provider({"provider": "foobar"})

    def test_cohere_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            old = os.environ.pop("COHERE_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="API key is required"):
                    create_rerank_provider({"provider": "cohere"})
            finally:
                if old is not None:
                    os.environ["COHERE_API_KEY"] = old


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_rerank_valid(self):
        """A valid RERANK function should pass validation."""
        from milvus_lite.schema.validation import validate_schema
        schema = _make_schema_with_rerank()
        validate_schema(schema)  # should not raise

    def test_rerank_wrong_input_type(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="num", dtype=DataType.INT64),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="bad_rerank",
                function_type=FunctionType.RERANK,
                input_field_names=["num"],
                output_field_names=[],
                params={"provider": "cohere"},
            ),
        ])
        with pytest.raises(Exception, match="must be VARCHAR"):
            validate_schema(schema)

    def test_rerank_non_empty_output(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="bad_rerank",
                function_type=FunctionType.RERANK,
                input_field_names=["text"],
                output_field_names=["vec"],
                params={"provider": "cohere"},
            ),
        ])
        with pytest.raises(Exception, match="empty output_field_names"):
            validate_schema(schema)

    def test_rerank_missing_provider_param(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="bad_rerank",
                function_type=FunctionType.RERANK,
                input_field_names=["text"],
                output_field_names=[],
                params={},
            ),
        ])
        with pytest.raises(Exception, match="requires 'provider'"):
            validate_schema(schema)


# ---------------------------------------------------------------------------
# Mock provider unit tests
# ---------------------------------------------------------------------------

class TestMockRerankProvider:
    def test_rerank_ordering(self):
        provider = MockRerankProvider()
        results = provider.rerank(
            query="machine learning",
            documents=[
                "web development frameworks",
                "machine learning algorithms",
                "deep learning neural networks",
            ],
        )
        # "machine learning algorithms" has the most overlap with query
        assert results[0].index == 1

    def test_rerank_empty_documents(self):
        provider = MockRerankProvider()
        results = provider.rerank("test", [])
        assert results == []

    def test_rerank_top_n(self):
        provider = MockRerankProvider()
        results = provider.rerank(
            query="test",
            documents=["test one", "test two", "other"],
            top_n=2,
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Engine-level integration tests
# ---------------------------------------------------------------------------

class TestRerankSearch:
    """Test reranking during search."""

    @patch("milvus_lite.rerank.factory.create_rerank_provider", side_effect=_mock_rerank_factory)
    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_embedding_factory)
    def test_text_query_triggers_rerank(self, mock_emb, mock_rerank):
        """Text query with RERANK function should reorder results."""
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d,
                             schema=_make_schema_with_rerank())
            col.insert([
                {"id": 1, "text": "web development frameworks"},
                {"id": 2, "text": "machine learning algorithms"},
                {"id": 3, "text": "deep learning neural networks"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=["machine learning"],
                top_k=3,
                metric_type="COSINE",
                anns_field="vec",
            )
            assert len(results) == 1
            assert len(results[0]) == 3
            # After reranking, "machine learning algorithms" should be first
            assert results[0][0]["id"] == 2
            # distance should be the reranker relevance_score
            assert results[0][0]["distance"] > 0

    @patch("milvus_lite.rerank.factory.create_rerank_provider", side_effect=_mock_rerank_factory)
    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_embedding_factory)
    def test_vector_query_skips_rerank(self, mock_emb, mock_rerank):
        """Float vector query should skip reranking (no text available)."""
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d,
                             schema=_make_schema_with_rerank())
            col.insert([
                {"id": 1, "text": "hello"},
                {"id": 2, "text": "world"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            # Search with float vector — rerank should be skipped
            mock_vec = MockEmbeddingProvider(dim=8).embed_query("hello")
            results = col.search(
                query_vectors=[mock_vec],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
            )
            assert len(results[0]) == 2

    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_embedding_factory)
    def test_no_rerank_function_normal_search(self, mock_emb):
        """Schema without RERANK function should search normally."""
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d,
                             schema=_make_schema_no_rerank())
            col.insert([
                {"id": 1, "text": "machine learning"},
                {"id": 2, "text": "deep learning"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=["machine learning"],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
            )
            assert len(results) == 1
            assert len(results[0]) == 2
            # Without reranking, result 1 should match itself at rank 1
            assert results[0][0]["id"] == 1

    @patch("milvus_lite.rerank.factory.create_rerank_provider", side_effect=_mock_rerank_factory)
    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_embedding_factory)
    def test_rerank_strips_injected_field(self, mock_emb, mock_rerank):
        """If user doesn't request the rerank input field, it should be stripped."""
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d,
                             schema=_make_schema_with_rerank())
            col.insert([
                {"id": 1, "text": "hello world"},
                {"id": 2, "text": "goodbye world"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            # Search with output_fields that does NOT include "text"
            results = col.search(
                query_vectors=["hello"],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
                output_fields=["id"],
            )
            # "text" should NOT appear in entity (was injected for reranking only)
            for hit in results[0]:
                assert "text" not in hit.get("entity", {})

    @patch("milvus_lite.rerank.factory.create_rerank_provider", side_effect=_mock_rerank_factory)
    @patch("milvus_lite.embedding.factory.create_embedding_provider", side_effect=_mock_embedding_factory)
    def test_rerank_keeps_requested_field(self, mock_emb, mock_rerank):
        """If user explicitly requests the rerank input field, it should remain."""
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d,
                             schema=_make_schema_with_rerank())
            col.insert([
                {"id": 1, "text": "hello world"},
                {"id": 2, "text": "goodbye world"},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=["hello"],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
                output_fields=["text"],
            )
            # "text" should appear in entity
            for hit in results[0]:
                assert "text" in hit.get("entity", {})


# ---------------------------------------------------------------------------
# DecayReranker unit tests
# ---------------------------------------------------------------------------

class TestDecayReranker:
    """Test the decay math for all three functions."""

    def test_gauss_at_origin(self):
        dr = DecayReranker("gauss", origin=0, scale=100, decay=0.5)
        assert dr.compute_factor(0.0) == 1.0

    def test_gauss_at_scale(self):
        dr = DecayReranker("gauss", origin=0, scale=100, decay=0.5)
        assert abs(dr.compute_factor(100.0) - 0.5) < 1e-9

    def test_gauss_symmetric(self):
        dr = DecayReranker("gauss", origin=50, scale=100, decay=0.5)
        assert abs(dr.compute_factor(50 + 30) - dr.compute_factor(50 - 30)) < 1e-9

    def test_exp_at_origin(self):
        dr = DecayReranker("exp", origin=0, scale=100, decay=0.5)
        assert dr.compute_factor(0.0) == 1.0

    def test_exp_at_scale(self):
        dr = DecayReranker("exp", origin=0, scale=100, decay=0.5)
        assert abs(dr.compute_factor(100.0) - 0.5) < 1e-9

    def test_exp_monotone_decrease(self):
        dr = DecayReranker("exp", origin=0, scale=100, decay=0.5)
        factors = [dr.compute_factor(d) for d in [0, 25, 50, 75, 100, 200]]
        for i in range(len(factors) - 1):
            assert factors[i] > factors[i + 1]

    def test_linear_at_origin(self):
        dr = DecayReranker("linear", origin=0, scale=100, decay=0.5)
        assert dr.compute_factor(0.0) == 1.0

    def test_linear_at_scale(self):
        dr = DecayReranker("linear", origin=0, scale=100, decay=0.5)
        assert abs(dr.compute_factor(100.0) - 0.5) < 1e-9

    def test_linear_clamps_to_zero(self):
        dr = DecayReranker("linear", origin=0, scale=100, decay=0.5)
        # Beyond scale/(1-decay) = 200, factor should be 0
        assert dr.compute_factor(300.0) == 0.0

    def test_offset_creates_safe_zone(self):
        dr = DecayReranker("gauss", origin=0, scale=100, offset=20, decay=0.5)
        # Within offset: factor = 1.0
        assert dr.compute_factor(10.0) == 1.0
        assert dr.compute_factor(20.0) == 1.0
        # Beyond offset: factor < 1.0
        assert dr.compute_factor(21.0) < 1.0

    def test_offset_shifts_scale(self):
        dr = DecayReranker("exp", origin=0, scale=100, offset=50, decay=0.5)
        # At origin + offset + scale = 150, factor should be decay
        assert abs(dr.compute_factor(150.0) - 0.5) < 1e-9

    def test_invalid_function(self):
        with pytest.raises(ValueError, match="invalid function"):
            DecayReranker("cubic", origin=0, scale=100)

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="scale must be > 0"):
            DecayReranker("gauss", origin=0, scale=-1)

    def test_invalid_offset(self):
        with pytest.raises(ValueError, match="offset must be >= 0"):
            DecayReranker("gauss", origin=0, scale=100, offset=-5)

    def test_invalid_decay(self):
        with pytest.raises(ValueError, match="decay must be 0 < decay < 1"):
            DecayReranker("gauss", origin=0, scale=100, decay=1.0)
        with pytest.raises(ValueError, match="decay must be 0 < decay < 1"):
            DecayReranker("gauss", origin=0, scale=100, decay=0.0)


# ---------------------------------------------------------------------------
# Decay schema validation tests
# ---------------------------------------------------------------------------

class TestDecaySchemaValidation:
    def test_decay_valid(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["score"],
                output_field_names=[],
                params={"reranker": "decay", "function": "gauss",
                        "origin": 0, "scale": 100},
            ),
        ])
        validate_schema(schema)  # should not raise

    def test_decay_varchar_input_rejected(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["text"],
                output_field_names=[],
                params={"reranker": "decay", "function": "gauss",
                        "origin": 0, "scale": 100},
            ),
        ])
        with pytest.raises(Exception, match="must be numeric"):
            validate_schema(schema)

    def test_decay_missing_function(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["score"],
                output_field_names=[],
                params={"reranker": "decay", "origin": 0, "scale": 100},
            ),
        ])
        with pytest.raises(Exception, match="must be one of"):
            validate_schema(schema)

    def test_decay_missing_origin(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["score"],
                output_field_names=[],
                params={"reranker": "decay", "function": "gauss", "scale": 100},
            ),
        ])
        with pytest.raises(Exception, match="origin not specified"):
            validate_schema(schema)

    def test_decay_missing_scale(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["score"],
                output_field_names=[],
                params={"reranker": "decay", "function": "gauss", "origin": 0},
            ),
        ])
        with pytest.raises(Exception, match="scale not specified"):
            validate_schema(schema)

    def test_decay_invalid_scale(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["score"],
                output_field_names=[],
                params={"reranker": "decay", "function": "gauss",
                        "origin": 0, "scale": -10},
            ),
        ])
        with pytest.raises(Exception, match="scale must be > 0"):
            validate_schema(schema)

    def test_no_provider_no_reranker(self):
        from milvus_lite.schema.validation import validate_schema
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ], functions=[
            Function(
                name="bad",
                function_type=FunctionType.RERANK,
                input_field_names=["text"],
                output_field_names=[],
                params={},
            ),
        ])
        with pytest.raises(Exception, match="requires 'provider' or 'reranker'"):
            validate_schema(schema)


# ---------------------------------------------------------------------------
# Decay engine integration tests
# ---------------------------------------------------------------------------

def _make_schema_with_decay(dim=8):
    """Schema with a numeric field + decay reranker."""
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="priority", dtype=DataType.FLOAT),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ], functions=[
        Function(
            name="decay_fn",
            function_type=FunctionType.RERANK,
            input_field_names=["priority"],
            output_field_names=[],
            params={
                "reranker": "decay",
                "function": "gauss",
                "origin": 100,
                "scale": 50,
                "decay": 0.5,
            },
        ),
    ])


class TestDecaySearch:
    """Test decay reranking during search."""

    def test_decay_reorders_by_field_proximity(self):
        """Items closer to origin should rank higher after decay."""
        with tempfile.TemporaryDirectory() as d:
            schema = _make_schema_with_decay()
            col = Collection(name="test", data_dir=d, schema=schema)

            # Insert: all have similar vectors but different priority values
            rng = np.random.default_rng(42)
            base_vec = rng.standard_normal(8).astype(np.float32)
            base_vec = base_vec / np.linalg.norm(base_vec)

            col.insert([
                {"id": 1, "priority": 0.0, "vec": base_vec.tolist()},    # far from 100
                {"id": 2, "priority": 100.0, "vec": base_vec.tolist()},  # at origin
                {"id": 3, "priority": 50.0, "vec": base_vec.tolist()},   # medium
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[base_vec.tolist()],
                top_k=3,
                metric_type="COSINE",
                anns_field="vec",
            )
            assert len(results[0]) == 3
            # id=2 (priority=100, at origin) should be first
            assert results[0][0]["id"] == 2
            # id=3 (priority=50) should be second
            assert results[0][1]["id"] == 3
            # id=1 (priority=0, farthest) should be last
            assert results[0][2]["id"] == 1
            # All scores should be positive
            for hit in results[0]:
                assert hit["distance"] > 0

    def test_decay_strips_injected_field(self):
        """Decay input field should be stripped if not requested."""
        with tempfile.TemporaryDirectory() as d:
            schema = _make_schema_with_decay()
            col = Collection(name="test", data_dir=d, schema=schema)
            rng = np.random.default_rng(42)
            vec = rng.standard_normal(8).astype(np.float32).tolist()
            col.insert([
                {"id": 1, "priority": 100.0, "vec": vec},
                {"id": 2, "priority": 0.0, "vec": vec},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[vec],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
                output_fields=["id"],
            )
            for hit in results[0]:
                assert "priority" not in hit.get("entity", {})

    def test_decay_keeps_requested_field(self):
        """Decay input field should remain if user requests it."""
        with tempfile.TemporaryDirectory() as d:
            schema = _make_schema_with_decay()
            col = Collection(name="test", data_dir=d, schema=schema)
            rng = np.random.default_rng(42)
            vec = rng.standard_normal(8).astype(np.float32).tolist()
            col.insert([
                {"id": 1, "priority": 100.0, "vec": vec},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[vec],
                top_k=1,
                metric_type="COSINE",
                anns_field="vec",
                output_fields=["priority"],
            )
            assert "priority" in results[0][0].get("entity", {})

    @pytest.mark.parametrize("func_name", ["gauss", "exp", "linear"])
    def test_decay_all_functions(self, func_name):
        """All three decay functions should produce valid results."""
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="val", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["val"],
                output_field_names=[],
                params={
                    "reranker": "decay",
                    "function": func_name,
                    "origin": 0,
                    "scale": 10,
                    "decay": 0.5,
                },
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            vec = [1.0, 0.0, 0.0, 0.0]
            col.insert([
                {"id": 1, "val": 0.0, "vec": vec},    # at origin
                {"id": 2, "val": 10.0, "vec": vec},   # at scale
                {"id": 3, "val": 100.0, "vec": vec},  # far away
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[vec],
                top_k=3,
                metric_type="COSINE",
                anns_field="vec",
            )
            # id=1 at origin should be first
            assert results[0][0]["id"] == 1
            # Scores should be monotonically decreasing
            scores = [h["distance"] for h in results[0]]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]


# ---------------------------------------------------------------------------
# Milvus-ported decay integration tests
# ---------------------------------------------------------------------------

def _gauss_decay(origin, scale, decay, offset, value):
    adj = max(0, abs(value - origin) - offset)
    sigma_sq = scale ** 2 / math.log(decay)
    return math.exp(adj ** 2 / sigma_sq)


def _exp_decay(origin, scale, decay, offset, value):
    adj = max(0, abs(value - origin) - offset)
    lam = math.log(decay) / scale
    return math.exp(lam * adj)


def _linear_decay(origin, scale, decay, offset, value):
    adj = max(0, abs(value - origin) - offset)
    slope = (1 - decay) / scale
    return max(decay, 1 - slope * adj)


_DECAY_FUNCS = {"gauss": _gauss_decay, "exp": _exp_decay, "linear": _linear_decay}


class TestDecayScoreOrdering:
    """Ported from Milvus: test_milvus_client_search_reranker_decay_score_ordering.

    Insert rows with identical vectors and varying field values,
    search with decay reranker. Verify scores are DESC and distances
    from origin are ASC.
    """

    @pytest.mark.parametrize("function", ["gauss", "linear", "exp"])
    @pytest.mark.parametrize("decay_val", [0.1, 0.5, 0.9])
    def test_score_ordering(self, function, decay_val):
        dim = 5
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="rf", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ], functions=[
            Function(
                name="my_reranker",
                function_type=FunctionType.RERANK,
                input_field_names=["rf"],
                output_field_names=[],
                params={
                    "reranker": "decay",
                    "function": function,
                    "origin": 0,
                    "offset": 0,
                    "decay": decay_val,
                    "scale": 100,
                },
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            fixed_vec = [0.5] * dim
            field_values = [0, 10, 50, 100, 200, 500]
            col.insert([
                {"id": i, "rf": float(field_values[i]), "vec": fixed_vec}
                for i in range(len(field_values))
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[fixed_vec],
                top_k=len(field_values),
                metric_type="COSINE",
                anns_field="vec",
                output_fields=["rf"],
            )
            hits = results[0]
            assert len(hits) == len(field_values)

            scores = [h["distance"] for h in hits]
            rf_values = [h["entity"]["rf"] for h in hits]

            # All scores non-negative (linear can clamp to 0 at large distances)
            for s in scores:
                assert s >= 0
            # Score at origin must be positive
            assert scores[0] > 0

            # Scores in descending order
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"scores[{i}]={scores[i]} < scores[{i+1}]={scores[i+1]}"
                )

            # Distances from origin in ascending order
            distances = [abs(v) for v in rf_values]
            for i in range(len(distances) - 1):
                assert distances[i] <= distances[i + 1], (
                    f"dist[{i}]={distances[i]} > dist[{i+1}]={distances[i+1]}"
                )


class TestDecayScoreRatio:
    """Ported from Milvus: test_milvus_client_search_reranker_decay_score_ratio.

    Verify score ratios match Python-computed decay formulas within epsilon.
    """

    @pytest.mark.parametrize("function", ["gauss", "linear", "exp"])
    def test_score_ratio(self, function):
        dim = 5
        origin = 0
        scale = 100
        decay_param = 0.5
        offset = 0
        field_values = [0, 25, 50, 75, 100]

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="rf", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ], functions=[
            Function(
                name="my_reranker",
                function_type=FunctionType.RERANK,
                input_field_names=["rf"],
                output_field_names=[],
                params={
                    "reranker": "decay",
                    "function": function,
                    "origin": origin,
                    "offset": offset,
                    "decay": decay_param,
                    "scale": scale,
                },
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            fixed_vec = [0.5] * dim
            col.insert([
                {"id": i, "rf": float(field_values[i]), "vec": fixed_vec}
                for i in range(len(field_values))
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[fixed_vec],
                top_k=len(field_values),
                metric_type="COSINE",
                anns_field="vec",
                output_fields=["rf"],
            )
            hits = results[0]

            # Build mapping: field_value -> actual score
            actual_scores = {}
            for h in hits:
                actual_scores[h["entity"]["rf"]] = h["distance"]

            # Compute expected decay scores
            decay_fn = _DECAY_FUNCS[function]
            expected_scores = {
                v: decay_fn(origin, scale, decay_param, offset, v)
                for v in field_values
            }

            # Compare ratios relative to the reference point (value=0)
            ref = 0
            ref_actual = actual_scores[ref]
            ref_expected = expected_scores[ref]
            epsilon = 0.01

            for v in field_values:
                if v == ref:
                    continue
                actual_ratio = actual_scores[v] / ref_actual
                expected_ratio = expected_scores[v] / ref_expected
                assert abs(actual_ratio - expected_ratio) < epsilon, (
                    f"function={function}, value={v}: "
                    f"actual_ratio={actual_ratio:.6f}, "
                    f"expected_ratio={expected_ratio:.6f}"
                )

            # At distance=scale, ratio should equal decay param
            if scale in actual_scores:
                ratio_at_scale = actual_scores[scale] / actual_scores[ref]
                assert abs(ratio_at_scale - decay_param) < epsilon, (
                    f"At scale={scale}, expected ratio≈{decay_param}, "
                    f"got {ratio_at_scale:.6f}"
                )


class TestDecayOffsetEffect:
    """Ported from Milvus: test_milvus_client_search_reranker_decay_offset_effect.

    Items within the offset zone should have equal scores; items beyond
    should have strictly decreasing scores.
    """

    def test_offset_zone_equal_scores(self):
        dim = 5
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="rf", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ], functions=[
            Function(
                name="my_reranker",
                function_type=FunctionType.RERANK,
                input_field_names=["rf"],
                output_field_names=[],
                params={
                    "reranker": "decay",
                    "function": "gauss",
                    "origin": 0,
                    "offset": 10,
                    "decay": 0.5,
                    "scale": 100,
                },
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            fixed_vec = [0.5] * dim
            field_values = [0, 5, 10, 15, 50, 100]
            col.insert([
                {"id": i, "rf": float(field_values[i]), "vec": fixed_vec}
                for i in range(len(field_values))
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[fixed_vec],
                top_k=len(field_values),
                metric_type="COSINE",
                anns_field="vec",
                output_fields=["rf"],
            )
            hits = results[0]
            score_map = {h["entity"]["rf"]: h["distance"] for h in hits}

            # Items within offset (distance <= 10) should have equal scores
            within_offset = [0.0, 5.0, 10.0]
            epsilon = 1e-4
            ref_score = score_map[within_offset[0]]
            for v in within_offset:
                assert abs(score_map[v] - ref_score) < epsilon, (
                    f"Within offset: score({v})={score_map[v]} != "
                    f"score(0)={ref_score}"
                )

            # Items beyond offset should have lower scores
            beyond_offset = [15.0, 50.0, 100.0]
            for v in beyond_offset:
                assert score_map[v] < ref_score, (
                    f"Beyond offset: score({v})={score_map[v]} should be "
                    f"< ref={ref_score}"
                )

            # Beyond-offset scores should be strictly decreasing
            for i in range(len(beyond_offset) - 1):
                assert score_map[beyond_offset[i]] > score_map[beyond_offset[i + 1]], (
                    f"score({beyond_offset[i]})={score_map[beyond_offset[i]]} <= "
                    f"score({beyond_offset[i+1]})={score_map[beyond_offset[i+1]]}"
                )


class TestDecayL2Metric:
    """Ported from Milvus: test_milvus_client_search_decay_rerank_l2_metric_no_norm_score.

    With L2 metric, the item with smallest L2 distance must still rank
    first after decay reranking. Tests that the distance→score conversion
    handles L2 correctly (smaller L2 = better).
    """

    def test_l2_best_match_ranks_first(self):
        dim = 8
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="ts", dtype=DataType.INT64),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ], functions=[
            Function(
                name="decay_l2",
                function_type=FunctionType.RERANK,
                input_field_names=["ts"],
                output_field_names=[],
                params={
                    "reranker": "decay",
                    "function": "gauss",
                    "origin": 1000,
                    "scale": 100,
                    "decay": 0.5,
                },
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            # row i: vector [0.1*i]*dim, all ts=1000 (at origin → factor=1.0)
            nrows = 5
            col.insert([
                {"id": i, "ts": 1000, "vec": [0.1 * i] * dim}
                for i in range(nrows)
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "L2",
                "params": {},
            })
            col.load()

            # Query matches row 0 exactly → L2=0
            results = col.search(
                query_vectors=[[0.0] * dim],
                top_k=nrows,
                metric_type="L2",
                anns_field="vec",
                output_fields=["id"],
            )
            hits = results[0]
            ids = [h["id"] for h in hits]
            scores = [h["distance"] for h in hits]

            assert len(hits) == nrows
            # Row 0 (L2=0, decay=1.0) must rank first
            assert ids[0] == 0, f"Row 0 must rank first, got order {ids}"
            # Scores must be non-increasing
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"scores[{i}]={scores[i]} < scores[{i+1}]={scores[i+1]}"
                )


class TestDecayAllNumericTypes:
    """Ported from Milvus: test_milvus_client_search_reranker_decay_nullable_all_types.

    Verify decay reranker works with all numeric field types.
    """

    @pytest.mark.parametrize("dtype", [
        DataType.INT8, DataType.INT16, DataType.INT32,
        DataType.INT64, DataType.FLOAT, DataType.DOUBLE,
    ])
    def test_numeric_type(self, dtype):
        dim = 4
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="val", dtype=dtype),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["val"],
                output_field_names=[],
                params={
                    "reranker": "decay",
                    "function": "gauss",
                    "origin": 0,
                    "scale": 50,
                    "decay": 0.5,
                },
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            vec = [1.0, 0.0, 0.0, 0.0]
            col.insert([
                {"id": 1, "val": 0, "vec": vec},   # at origin
                {"id": 2, "val": 50, "vec": vec},   # at scale
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[vec],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
            )
            # At origin should rank first
            assert results[0][0]["id"] == 1
            assert results[0][0]["distance"] > results[0][1]["distance"]


class TestDecayDefaultParams:
    """Ported from Milvus: test_milvus_client_search_with_reranker_default_offset_decay.

    Verify decay works with default offset=0 and decay=0.5 (omitted from params).
    """

    @pytest.mark.parametrize("function", ["gauss", "exp", "linear"])
    def test_default_offset_and_decay(self, function):
        dim = 4
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="val", dtype=DataType.FLOAT),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ], functions=[
            Function(
                name="decay_fn",
                function_type=FunctionType.RERANK,
                input_field_names=["val"],
                output_field_names=[],
                params={
                    "reranker": "decay",
                    "function": function,
                    "origin": 0,
                    "scale": 100,
                    # offset and decay omitted — use defaults
                },
            ),
        ])
        with tempfile.TemporaryDirectory() as d:
            col = Collection(name="test", data_dir=d, schema=schema)
            vec = [1.0, 0.0, 0.0, 0.0]
            col.insert([
                {"id": 1, "val": 0.0, "vec": vec},
                {"id": 2, "val": 100.0, "vec": vec},
            ])
            col.create_index("vec", {
                "index_type": "BRUTE_FORCE",
                "metric_type": "COSINE",
                "params": {},
            })
            col.load()

            results = col.search(
                query_vectors=[vec],
                top_k=2,
                metric_type="COSINE",
                anns_field="vec",
            )
            # At origin should be first
            assert results[0][0]["id"] == 1
            # Score at origin > score at scale
            assert results[0][0]["distance"] > results[0][1]["distance"]
