"""Data type definitions for LiteVecDB schema layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pyarrow as pa


class DataType(Enum):
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    DOUBLE = "double"
    VARCHAR = "varchar"
    JSON = "json"
    FLOAT_VECTOR = "float_vector"


@dataclass
class FieldSchema:
    name: str
    dtype: DataType
    is_primary: bool = False
    dim: Optional[int] = None
    max_length: Optional[int] = None
    nullable: bool = False
    default_value: Any = None


@dataclass
class CollectionSchema:
    fields: List[FieldSchema]
    version: int = 1
    enable_dynamic_field: bool = False


# DataType -> PyArrow type mapping.
# FLOAT_VECTOR is None here; at runtime use pa.list_(pa.float32(), dim).
TYPE_MAP: Dict[DataType, Any] = {
    DataType.BOOL: pa.bool_(),
    DataType.INT8: pa.int8(),
    DataType.INT16: pa.int16(),
    DataType.INT32: pa.int32(),
    DataType.INT64: pa.int64(),
    DataType.FLOAT: pa.float32(),
    DataType.DOUBLE: pa.float64(),
    DataType.VARCHAR: pa.string(),
    DataType.JSON: pa.string(),
    DataType.FLOAT_VECTOR: None,
}
