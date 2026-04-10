"""Phase 11.4 — anns_field parameter tests.

Covers:
- Default anns_field (None → first FLOAT_VECTOR)
- Explicit anns_field pointing to FLOAT_VECTOR works
- Explicit anns_field pointing to SPARSE_FLOAT_VECTOR raises NotImplementedError
- Invalid anns_field raises SchemaValidationError
- Non-vector anns_field raises SchemaValidationError
"""

import tempfile

import pytest

from litevecdb.engine.collection import Collection
from litevecdb.exceptions import SchemaValidationError
from litevecdb.schema.types import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
)


def _mixed_collection(tmpdir):
    """Collection with FLOAT_VECTOR + BM25 SPARSE_FLOAT_VECTOR."""
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR,
                enable_analyzer=True,
                analyzer_params={"tokenizer": "standard"},
            ),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=4),
            FieldSchema(
                name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR,
                is_function_output=True,
            ),
        ],
        functions=[
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            ),
        ],
    )
    col = Collection(name="test", data_dir=tmpdir, schema=schema)
    col.insert([
        {"id": 1, "text": "hello world", "dense": [1.0, 0.0, 0.0, 0.0]},
        {"id": 2, "text": "machine learning", "dense": [0.0, 1.0, 0.0, 0.0]},
        {"id": 3, "text": "hello machine", "dense": [0.0, 0.0, 1.0, 0.0]},
    ])
    return col


class TestAnnsField:
    def test_default_uses_float_vector(self):
        """anns_field=None defaults to the FLOAT_VECTOR field."""
        with tempfile.TemporaryDirectory() as d:
            col = _mixed_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=2,
                metric_type="COSINE",
            )
            assert len(results) == 1
            assert len(results[0]) == 2

    def test_explicit_float_vector(self):
        """Explicitly naming the FLOAT_VECTOR field works."""
        with tempfile.TemporaryDirectory() as d:
            col = _mixed_collection(d)
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=2,
                metric_type="COSINE",
                anns_field="dense",
            )
            assert len(results) == 1
            assert len(results[0]) == 2

    def test_sparse_raises_not_implemented(self):
        """Searching on SPARSE_FLOAT_VECTOR raises NotImplementedError (Phase 11.5)."""
        with tempfile.TemporaryDirectory() as d:
            col = _mixed_collection(d)
            with pytest.raises(NotImplementedError, match="Sparse vector search"):
                col.search(
                    query_vectors=[{"hello": 1.0}],
                    top_k=2,
                    metric_type="BM25",
                    anns_field="sparse",
                )

    def test_invalid_field_name(self):
        """anns_field pointing to a non-existent field raises error."""
        with tempfile.TemporaryDirectory() as d:
            col = _mixed_collection(d)
            with pytest.raises(SchemaValidationError, match="not found"):
                col.search(
                    query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                    top_k=2,
                    anns_field="nonexistent",
                )

    def test_non_vector_field(self):
        """anns_field pointing to a scalar field raises error."""
        with tempfile.TemporaryDirectory() as d:
            col = _mixed_collection(d)
            with pytest.raises(SchemaValidationError, match="not a vector"):
                col.search(
                    query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                    top_k=2,
                    anns_field="text",
                )

    def test_backward_compatible(self):
        """Existing code without anns_field still works."""
        with tempfile.TemporaryDirectory() as d:
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
                ],
            )
            col = Collection(name="compat", data_dir=d, schema=schema)
            col.insert([
                {"id": 1, "vec": [1.0, 0.0, 0.0, 0.0]},
            ])
            results = col.search(
                query_vectors=[[1.0, 0.0, 0.0, 0.0]],
                top_k=1,
            )
            assert len(results) == 1
            assert results[0][0]["id"] == 1
