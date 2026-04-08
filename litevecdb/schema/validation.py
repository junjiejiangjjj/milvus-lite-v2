"""Schema and record validation."""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from litevecdb.exceptions import SchemaValidationError
from litevecdb.schema.types import CollectionSchema, DataType, FieldSchema

# Reserved column names — users may not name fields these.
RESERVED_FIELD_NAMES = frozenset({"_seq", "_partition", "$meta"})

# pk dtype must be one of these.
_PK_ALLOWED_DTYPES = frozenset({DataType.VARCHAR, DataType.INT64})

# Per-DataType Python-type predicate. We accept the canonical Python types
# plus a couple of widening cases (int → float, bool → int handled below).
_DTYPE_PYTHON_CHECK = {
    DataType.BOOL: lambda v: isinstance(v, bool),
    DataType.INT8: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**7) <= v < 2**7,
    DataType.INT16: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**15) <= v < 2**15,
    DataType.INT32: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**31) <= v < 2**31,
    DataType.INT64: lambda v: isinstance(v, int) and not isinstance(v, bool) and -(2**63) <= v < 2**63,
    DataType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    DataType.DOUBLE: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    DataType.VARCHAR: lambda v: isinstance(v, str),
    DataType.JSON: lambda v: isinstance(v, (dict, list, str, int, float, bool)) or v is None,
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(schema: CollectionSchema) -> None:
    """Validate a CollectionSchema definition.

    Rules:
    - exactly one is_primary=True field; dtype is VARCHAR or INT64
    - exactly one FLOAT_VECTOR field (MVP limitation)
    - vector field must have dim > 0
    - primary key field must not be nullable
    - field names must be unique
    - field names must not collide with reserved names (_seq, _partition, $meta)
    """
    if not schema.fields:
        raise SchemaValidationError("schema has no fields")

    seen_names: set[str] = set()
    pk_fields: list[FieldSchema] = []
    vector_fields: list[FieldSchema] = []

    for f in schema.fields:
        if not f.name:
            raise SchemaValidationError("field name must not be empty")
        if f.name in RESERVED_FIELD_NAMES:
            raise SchemaValidationError(
                f"field name {f.name!r} is reserved (one of {sorted(RESERVED_FIELD_NAMES)})"
            )
        if f.name in seen_names:
            raise SchemaValidationError(f"duplicate field name: {f.name!r}")
        seen_names.add(f.name)

        if f.is_primary:
            pk_fields.append(f)
        if f.dtype == DataType.FLOAT_VECTOR:
            vector_fields.append(f)
            if f.dim is None or f.dim <= 0:
                raise SchemaValidationError(
                    f"vector field {f.name!r} requires dim > 0, got {f.dim}"
                )

    if len(pk_fields) == 0:
        raise SchemaValidationError("schema has no primary key field")
    if len(pk_fields) > 1:
        names = [f.name for f in pk_fields]
        raise SchemaValidationError(
            f"schema must have exactly one primary key field, got {names}"
        )
    pk = pk_fields[0]
    if pk.dtype not in _PK_ALLOWED_DTYPES:
        raise SchemaValidationError(
            f"primary key {pk.name!r} must be VARCHAR or INT64, got {pk.dtype}"
        )
    if pk.nullable:
        raise SchemaValidationError(
            f"primary key {pk.name!r} must not be nullable"
        )

    if len(vector_fields) == 0:
        raise SchemaValidationError("schema has no FLOAT_VECTOR field")
    if len(vector_fields) > 1:
        names = [f.name for f in vector_fields]
        raise SchemaValidationError(
            f"MVP supports exactly one FLOAT_VECTOR field, got {names}"
        )


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------

def validate_record(record: dict, schema: CollectionSchema) -> None:
    """Validate a single record dict against the schema.

    Rules:
    - pk field present and non-None
    - vector field present and len(vector) == field.dim, every element numeric
    - declared field values match their dtype
    - non-nullable fields are not None
    - if enable_dynamic_field=False, no fields outside the schema
    """
    if not isinstance(record, dict):
        raise SchemaValidationError(
            f"record must be a dict, got {type(record).__name__}"
        )

    schema_field_names = {f.name for f in schema.fields}
    pk = _find_pk(schema)
    vec = _find_vector(schema)

    # pk presence
    if pk.name not in record or record[pk.name] is None:
        raise SchemaValidationError(
            f"primary key {pk.name!r} missing or None"
        )

    # vector presence + shape
    if vec.name not in record or record[vec.name] is None:
        raise SchemaValidationError(
            f"vector field {vec.name!r} missing or None"
        )
    vector_value = record[vec.name]
    if not isinstance(vector_value, (list, tuple)):
        raise SchemaValidationError(
            f"vector field {vec.name!r} must be list/tuple, got {type(vector_value).__name__}"
        )
    if len(vector_value) != vec.dim:
        raise SchemaValidationError(
            f"vector field {vec.name!r} expected dim {vec.dim}, got {len(vector_value)}"
        )
    for i, x in enumerate(vector_value):
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise SchemaValidationError(
                f"vector field {vec.name!r}[{i}] must be numeric, got {type(x).__name__}"
            )

    # per-field type / nullability
    for f in schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR:
            continue  # already checked
        if f.name not in record:
            if f.nullable or f.default_value is not None:
                continue
            raise SchemaValidationError(
                f"field {f.name!r} missing and not nullable / no default"
            )
        value = record[f.name]
        if value is None:
            if not f.nullable:
                raise SchemaValidationError(
                    f"field {f.name!r} is None but not nullable"
                )
            continue
        check = _DTYPE_PYTHON_CHECK.get(f.dtype)
        if check is None:
            continue
        if not check(value):
            raise SchemaValidationError(
                f"field {f.name!r} value {value!r} does not match dtype {f.dtype}"
            )

    # dynamic-field policy
    if not schema.enable_dynamic_field:
        extras = set(record.keys()) - schema_field_names
        if extras:
            raise SchemaValidationError(
                f"record has fields {sorted(extras)} not in schema "
                f"(enable_dynamic_field is False)"
            )


# ---------------------------------------------------------------------------
# Dynamic field separation
# ---------------------------------------------------------------------------

def separate_dynamic_fields(
    record: dict, schema: CollectionSchema
) -> Tuple[dict, Optional[str]]:
    """Split a record into (schema_fields, meta_json).

    schema_fields contains only fields declared in the schema, with default
    values filled in for missing nullable fields.

    meta_json is a JSON string of the extra fields, or None if no extras.

    Raises SchemaValidationError if enable_dynamic_field=False and there
    are extra fields.
    """
    if not isinstance(record, dict):
        raise SchemaValidationError(
            f"record must be a dict, got {type(record).__name__}"
        )

    schema_field_names = {f.name for f in schema.fields}

    schema_part: dict = {}
    extras: dict = {}
    for key, value in record.items():
        if key in schema_field_names:
            schema_part[key] = value
        else:
            extras[key] = value

    # Fill defaults for missing schema fields.
    for f in schema.fields:
        if f.name not in schema_part:
            if f.default_value is not None:
                schema_part[f.name] = f.default_value
            elif f.nullable:
                schema_part[f.name] = None
            # else: leave missing — validate_record will have caught it

    if extras and not schema.enable_dynamic_field:
        raise SchemaValidationError(
            f"record has fields {sorted(extras)} not in schema "
            f"(enable_dynamic_field is False)"
        )

    meta_json = json.dumps(extras, sort_keys=True) if extras else None
    return schema_part, meta_json


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_pk(schema: CollectionSchema) -> FieldSchema:
    for f in schema.fields:
        if f.is_primary:
            return f
    raise SchemaValidationError("schema has no primary key field")


def _find_vector(schema: CollectionSchema) -> FieldSchema:
    for f in schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR:
            return f
    raise SchemaValidationError("schema has no FLOAT_VECTOR field")
