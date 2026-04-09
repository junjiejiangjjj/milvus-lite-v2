"""Schema translation: Milvus proto ↔ LiteVecDB CollectionSchema.

Two functions, both lossless within the supported type subset:

    milvus_to_litevecdb_schema(proto_schema) → CollectionSchema
    litevecdb_to_milvus_schema(name, schema) → schema_pb2.CollectionSchema

Type mapping (LiteVecDB ↔ Milvus DataType enum):
    INT8         ↔ Int8 (2)
    INT16        ↔ Int16 (3)
    INT32        ↔ Int32 (4)
    INT64        ↔ Int64 (5)
    FLOAT        ↔ Float (10)
    DOUBLE       ↔ Double (11)
    VARCHAR      ↔ VarChar (21)
    BOOL         ↔ Bool (1)
    JSON         ↔ JSON (23)
    FLOAT_VECTOR ↔ FloatVector (101)

Unsupported Milvus types raise UnsupportedFieldTypeError. Phase 10.2 MVP
intentionally rejects rather than silently mapping (e.g.) Float16Vector
to FloatVector — silent type loss is the kind of subtle bug that ruins
trust in the protocol layer.

Field-level params we preserve:
    dim         (FloatVector — required)
    max_length  (VarChar — required)
    is_primary  (any field)
    nullable    (any field)

Fields we IGNORE on incoming protos (Phase 10.2 has no engine support):
    autoID, is_partition_key, is_clustering_key, default_value, element_type
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymilvus.grpc_gen import schema_pb2

from litevecdb.exceptions import SchemaValidationError
from litevecdb.schema.types import CollectionSchema, DataType, FieldSchema

if TYPE_CHECKING:
    pass


# Pymilvus DataType enum value → LiteVecDB DataType enum.
# These are the integer values from schema_pb2.DataType (NOT the
# pymilvus.DataType Python enum, which uses the same numbers but is
# a separate type).
_MILVUS_TO_LITEVECDB: dict[int, DataType] = {
    1:  DataType.BOOL,           # Bool
    2:  DataType.INT8,           # Int8
    3:  DataType.INT16,          # Int16
    4:  DataType.INT32,          # Int32
    5:  DataType.INT64,          # Int64
    10: DataType.FLOAT,          # Float
    11: DataType.DOUBLE,         # Double
    21: DataType.VARCHAR,        # VarChar
    23: DataType.JSON,           # JSON
    101: DataType.FLOAT_VECTOR,  # FloatVector
}

# Inverse — LiteVecDB → Milvus enum int.
_LITEVECDB_TO_MILVUS: dict[DataType, int] = {
    v: k for k, v in _MILVUS_TO_LITEVECDB.items()
}


# Milvus type names for clearer error messages.
_MILVUS_TYPE_NAMES: dict[int, str] = {
    0: "None",
    1: "Bool",
    2: "Int8",
    3: "Int16",
    4: "Int32",
    5: "Int64",
    10: "Float",
    11: "Double",
    20: "String",
    21: "VarChar",
    22: "Array",
    23: "JSON",
    24: "Geometry",
    25: "Text",
    100: "BinaryVector",
    101: "FloatVector",
    102: "Float16Vector",
    103: "BFloat16Vector",
    104: "SparseFloatVector",
    105: "Int8Vector",
}


# ── Milvus → LiteVecDB ──────────────────────────────────────────────

def milvus_to_litevecdb_schema(
    proto_schema: schema_pb2.CollectionSchema,
) -> CollectionSchema:
    """Decode a serialized Milvus schema into a LiteVecDB CollectionSchema.

    Raises:
        SchemaValidationError: if any field uses an unsupported type
            or is missing a required parameter (dim / max_length).
    """
    fields: list[FieldSchema] = []
    for f in proto_schema.fields:
        fields.append(_decode_field(f))

    return CollectionSchema(
        fields=fields,
        enable_dynamic_field=proto_schema.enable_dynamic_field,
    )


def _decode_field(proto_field: schema_pb2.FieldSchema) -> FieldSchema:
    """Translate one milvus FieldSchema proto into a litevecdb FieldSchema.

    Pulls dim / max_length out of the type_params KeyValuePair list."""
    milvus_dtype_int = int(proto_field.data_type)

    if milvus_dtype_int not in _MILVUS_TO_LITEVECDB:
        type_name = _MILVUS_TYPE_NAMES.get(milvus_dtype_int, f"unknown({milvus_dtype_int})")
        raise SchemaValidationError(
            f"field {proto_field.name!r} uses Milvus type {type_name} "
            f"which LiteVecDB does not support. "
            f"Phase 10.2 supports: BOOL, INT8/16/32/64, FLOAT, DOUBLE, "
            f"VARCHAR, JSON, FLOAT_VECTOR."
        )

    dtype = _MILVUS_TO_LITEVECDB[milvus_dtype_int]

    # Pull params out of type_params (it's a repeated KeyValuePair).
    params: dict[str, str] = {}
    for kv in proto_field.type_params:
        params[kv.key] = kv.value

    dim: int | None = None
    max_length: int | None = None

    if dtype == DataType.FLOAT_VECTOR:
        dim_str = params.get("dim")
        if not dim_str:
            raise SchemaValidationError(
                f"FLOAT_VECTOR field {proto_field.name!r} is missing required "
                f"'dim' type_param"
            )
        try:
            dim = int(dim_str)
        except ValueError as e:
            raise SchemaValidationError(
                f"FLOAT_VECTOR field {proto_field.name!r} has non-integer dim "
                f"{dim_str!r}"
            ) from e

    if dtype == DataType.VARCHAR:
        max_length_str = params.get("max_length")
        if max_length_str:
            try:
                max_length = int(max_length_str)
            except ValueError as e:
                raise SchemaValidationError(
                    f"VARCHAR field {proto_field.name!r} has non-integer "
                    f"max_length {max_length_str!r}"
                ) from e

    return FieldSchema(
        name=proto_field.name,
        dtype=dtype,
        is_primary=bool(proto_field.is_primary_key),
        dim=dim,
        max_length=max_length,
        nullable=bool(proto_field.nullable),
    )


# ── LiteVecDB → Milvus ──────────────────────────────────────────────

def litevecdb_to_milvus_schema(
    name: str,
    schema: CollectionSchema,
) -> schema_pb2.CollectionSchema:
    """Encode a LiteVecDB CollectionSchema as a Milvus proto schema.

    Used by DescribeCollection responses so pymilvus clients see a
    schema shape they recognize.
    """
    proto = schema_pb2.CollectionSchema()
    proto.name = name
    proto.enable_dynamic_field = schema.enable_dynamic_field

    for f in schema.fields:
        if f.dtype not in _LITEVECDB_TO_MILVUS:
            raise SchemaValidationError(
                f"field {f.name!r} has type {f.dtype.name} which has no "
                f"Milvus equivalent in the Phase 10.2 type subset"
            )
        pf = proto.fields.add()
        pf.name = f.name
        pf.data_type = _LITEVECDB_TO_MILVUS[f.dtype]
        pf.is_primary_key = bool(f.is_primary)
        pf.nullable = bool(f.nullable)

        if f.dtype == DataType.FLOAT_VECTOR:
            if f.dim is None:
                raise SchemaValidationError(
                    f"FLOAT_VECTOR field {f.name!r} is missing dim"
                )
            kv = pf.type_params.add()
            kv.key = "dim"
            kv.value = str(f.dim)

        if f.dtype == DataType.VARCHAR and f.max_length is not None:
            kv = pf.type_params.add()
            kv.key = "max_length"
            kv.value = str(f.max_length)

    return proto
