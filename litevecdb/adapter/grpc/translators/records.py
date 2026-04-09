"""FieldData ↔ records translation.

This is the most error-prone translator in Phase 10 — Milvus's
``InsertRequest.fields_data`` is COLUMNAR (one FieldData per field,
each carrying a length-N value array), while LiteVecDB engine takes
ROW-WISE list[dict]. The two views must be transposed bidirectionally
without losing any field, type, or null information.

Phase 10.3 supported types (matches translators/schema.py):
    Bool / Int8 / Int16 / Int32 / Int64 / Float / Double / VarChar
    JSON / FloatVector

Unsupported (raise UnsupportedFieldTypeError):
    BinaryVector / Float16Vector / BFloat16Vector / SparseFloatVector
    Int8Vector / Array / Geometry / Text / Timestamptz / ArrayOfVector

Two functions:

    fields_data_to_records(fields_data, num_rows)
        list[FieldData] + num_rows → list[dict]
        Used by Insert / Upsert RPCs.

    records_to_fields_data(records, schema, output_fields)
        list[dict] + CollectionSchema → list[FieldData]
        Used by Query / Get / Search response builders.

valid_data semantics: Milvus uses ``FieldData.valid_data`` (a parallel
bool array) to mark per-row null values for nullable fields. We
translate this to Python ``None`` in the records form, and rebuild
the valid_data array on the way back.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pymilvus.grpc_gen import schema_pb2

from litevecdb.exceptions import SchemaValidationError
from litevecdb.schema.types import CollectionSchema, DataType


# ── Milvus DataType enum (int) → name and category. We use the int
# values directly to avoid importing pymilvus.DataType (the Python
# enum that wraps the same numbers).

# Scalar types we handle, mapped to the ScalarField oneof slot name.
_SCALAR_TYPE_TO_SLOT: Dict[int, str] = {
    1:  "bool_data",     # Bool
    2:  "int_data",      # Int8 (uses IntArray)
    3:  "int_data",      # Int16
    4:  "int_data",      # Int32
    5:  "long_data",     # Int64
    10: "float_data",    # Float
    11: "double_data",   # Double
    21: "string_data",   # VarChar
    23: "json_data",     # JSON
}

_VECTOR_TYPES = frozenset({100, 101, 102, 103, 104, 105})  # Binary/Float/F16/BF16/Sparse/Int8


def _milvus_type_name(dtype_int: int) -> str:
    """Pretty name for error messages."""
    names = {
        1: "Bool", 2: "Int8", 3: "Int16", 4: "Int32", 5: "Int64",
        10: "Float", 11: "Double", 20: "String", 21: "VarChar",
        22: "Array", 23: "JSON", 24: "Geometry", 25: "Text",
        100: "BinaryVector", 101: "FloatVector",
        102: "Float16Vector", 103: "BFloat16Vector",
        104: "SparseFloatVector", 105: "Int8Vector",
    }
    return names.get(dtype_int, f"Unknown({dtype_int})")


# ── Milvus → records (Insert path) ──────────────────────────────────

def fields_data_to_records(
    fields_data,
    num_rows: int,
) -> List[Dict[str, Any]]:
    """Transpose Milvus columnar fields_data into row-wise records.

    Args:
        fields_data: iterable of FieldData proto messages
        num_rows: declared row count from InsertRequest.num_rows.
            We use this as the authoritative length and validate every
            FieldData against it.

    Returns:
        List of length num_rows. Each element is a dict mapping
        field_name → Python value (or None for null entries).

    Raises:
        SchemaValidationError: any FieldData length mismatches num_rows
            or uses an unsupported type
    """
    if num_rows == 0:
        return []

    records: List[Dict[str, Any]] = [{} for _ in range(num_rows)]

    for fd in fields_data:
        column = _extract_column(fd, num_rows)
        for i in range(num_rows):
            records[i][fd.field_name] = column[i]

    return records


def _extract_column(fd, num_rows: int) -> List[Any]:
    """Pull a single FieldData out as a length-num_rows Python list.

    Handles the scalar/vector dispatch, validates length, and
    overlays valid_data nulls if present.
    """
    dtype_int = int(fd.type)

    if fd.HasField("scalars"):
        column = _extract_scalar_column(fd, dtype_int)
    elif fd.HasField("vectors"):
        column = _extract_vector_column(fd, dtype_int, num_rows)
    else:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} has neither scalars nor vectors"
        )

    if len(column) != num_rows:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} has {len(column)} rows, "
            f"expected {num_rows}"
        )

    # Apply valid_data null mask if present (nullable fields).
    if list(fd.valid_data):
        valid = list(fd.valid_data)
        if len(valid) != num_rows:
            raise SchemaValidationError(
                f"FieldData {fd.field_name!r} valid_data length "
                f"{len(valid)} != num_rows {num_rows}"
            )
        column = [v if valid[i] else None for i, v in enumerate(column)]

    return column


def _extract_scalar_column(fd, dtype_int: int) -> List[Any]:
    scalars = fd.scalars

    if dtype_int not in _SCALAR_TYPE_TO_SLOT:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} uses scalar type "
            f"{_milvus_type_name(dtype_int)} which LiteVecDB does not support"
        )

    slot = _SCALAR_TYPE_TO_SLOT[dtype_int]
    sub = getattr(scalars, slot)
    raw = list(sub.data)

    if dtype_int == 23:  # JSON
        # JSON values arrive as bytes; decode + parse each.
        out = []
        for b in raw:
            if isinstance(b, bytes):
                b = b.decode("utf-8")
            try:
                out.append(json.loads(b))
            except (json.JSONDecodeError, ValueError):
                # Tolerate malformed JSON: pass through as a string.
                out.append(b)
        return out

    return raw


def _extract_vector_column(fd, dtype_int: int, num_rows: int) -> List[Any]:
    vectors = fd.vectors

    if dtype_int != 101:  # FloatVector — the only vector we support
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} uses vector type "
            f"{_milvus_type_name(dtype_int)} which LiteVecDB does not "
            f"support (Phase 10.3 supports FloatVector only)"
        )

    if not vectors.HasField("float_vector"):
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} declared FloatVector but no "
            f"float_vector data is set"
        )

    dim = int(vectors.dim)
    if dim <= 0:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} has invalid dim {dim}"
        )

    flat = list(vectors.float_vector.data)
    expected_total = num_rows * dim
    if len(flat) != expected_total:
        raise SchemaValidationError(
            f"FieldData {fd.field_name!r} float_vector data has "
            f"{len(flat)} elements, expected num_rows({num_rows}) * dim({dim}) "
            f"= {expected_total}"
        )

    # Slice into per-row lists.
    return [flat[i * dim:(i + 1) * dim] for i in range(num_rows)]


# ── records → Milvus (Query / Get / Search response path) ───────────

def records_to_fields_data(
    records: List[Dict[str, Any]],
    schema: CollectionSchema,
    output_fields: Optional[List[str]] = None,
) -> List:
    """Build columnar FieldData list from row-wise records.

    Args:
        records: list of dicts (engine output)
        schema: source CollectionSchema — needed for per-field type info
        output_fields: optional whitelist; None → emit every schema field.
            Pk is always emitted.

    Returns:
        List of FieldData proto messages, one per emitted field. When
        *records* is empty, the FieldData list still contains one entry
        per emitted field with an empty data slot — pymilvus's query
        client raises "No fields returned" if we send back an entirely
        empty fields_data list.
    """
    pk_field = next((f for f in schema.fields if f.is_primary), None)
    pk_name = pk_field.name if pk_field else None

    if output_fields is None:
        emit_names = [f.name for f in schema.fields]
    else:
        emit = set(output_fields)
        if pk_name:
            emit.add(pk_name)
        # Preserve schema order for determinism.
        emit_names = [f.name for f in schema.fields if f.name in emit]

    field_by_name = {f.name: f for f in schema.fields}

    fields_data: List = []
    for fname in emit_names:
        fschema = field_by_name[fname]
        column = [r.get(fname) for r in records]
        fd = _build_field_data(fname, fschema, column)
        fields_data.append(fd)

    return fields_data


def _build_field_data(name, fschema, column):
    """Build one FieldData proto from a (name, FieldSchema, column) triple.

    Handles per-type oneof slot population, valid_data null mask
    construction, and float_vector flattening.
    """
    fd = schema_pb2.FieldData()
    fd.field_name = name

    dtype = fschema.dtype

    # Build the null-aware column: replace None with a type-appropriate
    # default and emit a parallel valid_data list.
    has_nulls = any(v is None for v in column)
    if has_nulls:
        valid = [v is not None for v in column]
        fd.valid_data.extend(valid)

    if dtype == DataType.FLOAT_VECTOR:
        fd.type = 101  # FloatVector
        dim = int(fschema.dim or 0)
        if dim <= 0:
            raise SchemaValidationError(
                f"FLOAT_VECTOR field {name!r} has missing or invalid dim"
            )
        fd.vectors.dim = dim
        flat: List[float] = []
        zero_row = [0.0] * dim
        for v in column:
            if v is None:
                flat.extend(zero_row)  # placeholder; valid_data marks it null
            else:
                if len(v) != dim:
                    raise SchemaValidationError(
                        f"vector value for {name!r} has length {len(v)}, "
                        f"expected dim {dim}"
                    )
                flat.extend(float(x) for x in v)
        fd.vectors.float_vector.data.extend(flat)
        return fd

    # Scalar types
    milvus_type_int = _LITEVECDB_TO_MILVUS_INT.get(dtype)
    if milvus_type_int is None:
        raise SchemaValidationError(
            f"field {name!r} has type {dtype.name} which has no Milvus "
            f"FieldData equivalent"
        )

    fd.type = milvus_type_int
    slot = _SCALAR_TYPE_TO_SLOT[milvus_type_int]
    sub = getattr(fd.scalars, slot)

    if dtype == DataType.JSON:
        # Encode each value as bytes JSON
        encoded = []
        for v in column:
            if v is None:
                encoded.append(b"null")  # placeholder
            elif isinstance(v, (str, bytes)):
                # Already-serialized: pass through (decode if str)
                encoded.append(v.encode("utf-8") if isinstance(v, str) else v)
            else:
                encoded.append(json.dumps(v).encode("utf-8"))
        sub.data.extend(encoded)
        return fd

    # Numeric / bool / string scalar
    out = []
    for v in column:
        if v is None:
            out.append(_default_for(dtype))
        else:
            out.append(_coerce_for(dtype, v))
    sub.data.extend(out)
    return fd


# litevecdb.DataType → Milvus enum int
_LITEVECDB_TO_MILVUS_INT: Dict[DataType, int] = {
    DataType.BOOL:    1,
    DataType.INT8:    2,
    DataType.INT16:   3,
    DataType.INT32:   4,
    DataType.INT64:   5,
    DataType.FLOAT:   10,
    DataType.DOUBLE:  11,
    DataType.VARCHAR: 21,
    DataType.JSON:    23,
    DataType.FLOAT_VECTOR: 101,
}


def _default_for(dtype: DataType) -> Any:
    """Type-appropriate placeholder for null entries. The valid_data
    array marks them as null on the wire."""
    if dtype == DataType.BOOL:
        return False
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return 0
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return 0.0
    if dtype == DataType.VARCHAR:
        return ""
    return None


def _coerce_for(dtype: DataType, v: Any) -> Any:
    """Coerce a Python value into the type its proto slot expects."""
    if dtype == DataType.BOOL:
        return bool(v)
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return int(v)
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return float(v)
    if dtype == DataType.VARCHAR:
        return str(v)
    return v
