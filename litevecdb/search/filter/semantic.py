"""Schema-bound semantic check + type inference for filter expressions.

Phase 8 stage 2: parse_expr produces an AST that's schema-agnostic;
compile_expr binds it to a CollectionSchema, checks every field
reference, infers a type for every node, and selects an evaluation
backend. The result is a CompiledExpr that the evaluator can run on
any pa.Table matching the schema.

F1 always picks the "arrow" backend (pyarrow.compute path). Future
phases will switch to "python" when the expression contains $meta or
UDFs that arrow_backend can't handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from litevecdb.schema.types import CollectionSchema, DataType, FieldSchema
from litevecdb.search.filter.ast import (
    And,
    BoolLit,
    CmpOp,
    Expr,
    FieldRef,
    FloatLit,
    InOp,
    IntLit,
    ListLit,
    Literal,
    Not,
    Or,
    StringLit,
)
from litevecdb.search.filter.exceptions import (
    FilterFieldError,
    FilterTypeError,
)


# ── Semantic types ──────────────────────────────────────────────────────────

# A small internal type lattice for filter operands. We don't reuse
# DataType because we don't care about INT8 vs INT64 here — int promotion
# rules collapse all integers to a single "int" semantic type.
SEM_INT = "int"
SEM_FLOAT = "float"
SEM_STRING = "string"
SEM_BOOL = "bool"

_SEM_TYPES = {SEM_INT, SEM_FLOAT, SEM_STRING, SEM_BOOL}

# Reserved field names that must not be referenced from filter expressions.
_RESERVED_FIELDS = frozenset({"_seq", "_partition", "$meta"})


def _datatype_to_sem(dtype: DataType) -> Optional[str]:
    """Map a schema DataType to the filter semantic type, or None if
    the field type is not allowed in scalar filters (e.g. FLOAT_VECTOR)."""
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return SEM_INT
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return SEM_FLOAT
    if dtype == DataType.VARCHAR:
        return SEM_STRING
    if dtype == DataType.BOOL:
        return SEM_BOOL
    if dtype == DataType.JSON:
        return SEM_STRING  # JSON column is stored as string in Phase F1
    if dtype == DataType.FLOAT_VECTOR:
        return None  # not allowed in scalar filter
    return None


def _types_compatible(left: str, right: str) -> bool:
    """Comparison-compatible: same type, or int↔float promotion."""
    if left == right:
        return True
    if {left, right} == {SEM_INT, SEM_FLOAT}:
        return True
    return False


def _common_type(left: str, right: str) -> Optional[str]:
    """Common type when comparing — float wins over int."""
    if left == right:
        return left
    if {left, right} == {SEM_INT, SEM_FLOAT}:
        return SEM_FLOAT
    return None


@dataclass(frozen=True)
class FieldInfo:
    """Schema-bound field descriptor passed to evaluator backends."""
    name: str
    dtype: DataType
    sem_type: str
    nullable: bool


@dataclass(frozen=True)
class CompiledExpr:
    """A type-checked, schema-bound, evaluation-ready expression.

    The same Expr tree is reused; this wrapper carries the metadata
    that the evaluator backends need.
    """
    ast: Expr
    fields: Dict[str, FieldInfo]
    backend: str   # "arrow" | "python"
    source: str    # original expression string (for error messages)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def compile_expr(
    expr: Expr,
    schema: CollectionSchema,
    source: str = "",
) -> CompiledExpr:
    """Bind field references, check types, choose backend.

    Args:
        expr: AST from parse_expr
        schema: target CollectionSchema
        source: original expression string (used in error messages — pass
            it through from your call site if you parsed via parse_expr)

    Raises:
        FilterFieldError: unknown field reference
        FilterTypeError:  type mismatch in operand or non-bool top-level
    """
    # Build a map of all schema field names → FieldSchema for fast lookup.
    schema_fields: Dict[str, FieldSchema] = {f.name: f for f in schema.fields}
    field_names = [f.name for f in schema.fields]

    fields_used: Dict[str, FieldInfo] = {}

    # Walk the AST: type-check + collect referenced fields.
    top_type = _check_node(expr, schema_fields, field_names, fields_used, source)

    # Top-level expression must be boolean.
    if top_type != SEM_BOOL:
        raise FilterTypeError(
            f"top-level filter expression must evaluate to bool, got {top_type}",
            source, getattr(expr, "pos", 0),
        )

    # Phase F1 always uses arrow_backend. Future phases (F2b, F3) will
    # inspect the AST for $meta refs / UDF calls and pick "python" when
    # arrow_backend can't handle the expression.
    backend = "arrow"

    return CompiledExpr(
        ast=expr,
        fields=fields_used,
        backend=backend,
        source=source,
    )


# ---------------------------------------------------------------------------
# Internal: recursive type checker
# ---------------------------------------------------------------------------

def _check_node(
    node: Expr,
    schema_fields: Dict[str, FieldSchema],
    field_names: List[str],
    fields_used: Dict[str, FieldInfo],
    source: str,
) -> str:
    """Recursively check a node and return its semantic type."""

    # ── Literals ────────────────────────────────────────────────
    if isinstance(node, IntLit):
        return SEM_INT
    if isinstance(node, FloatLit):
        return SEM_FLOAT
    if isinstance(node, StringLit):
        return SEM_STRING
    if isinstance(node, BoolLit):
        return SEM_BOOL

    # ── ListLit (only used inside InOp; pure-form is rare) ──────
    if isinstance(node, ListLit):
        # Returns the common element type, or "empty" sentinel.
        # Empty list is allowed but defers type-check to InOp.
        if not node.elements:
            return "empty"
        elem_types = [_check_node(e, schema_fields, field_names, fields_used, source)
                      for e in node.elements]
        first = elem_types[0]
        for t in elem_types[1:]:
            if not _types_compatible(first, t):
                raise FilterTypeError(
                    f"list elements must be of compatible types, "
                    f"got {first} and {t}",
                    source, node.pos,
                )
        return _common_type(first, first) or first

    # ── FieldRef ────────────────────────────────────────────────
    if isinstance(node, FieldRef):
        # Reserved fields are rejected outright.
        if node.name in _RESERVED_FIELDS:
            raise FilterFieldError(
                f"reserved field {node.name!r} cannot be used in filter expressions",
                source, node.pos, node.name, available_fields=field_names,
            )
        if node.name not in schema_fields:
            raise FilterFieldError(
                f"unknown field {node.name!r}",
                source, node.pos, node.name, available_fields=field_names,
            )
        field = schema_fields[node.name]
        sem = _datatype_to_sem(field.dtype)
        if sem is None:
            raise FilterTypeError(
                f"field {node.name!r} of type {field.dtype.value} cannot be used "
                f"in scalar filter expressions",
                source, node.pos, span=len(node.name),
            )
        fields_used[node.name] = FieldInfo(
            name=node.name,
            dtype=field.dtype,
            sem_type=sem,
            nullable=field.nullable,
        )
        return sem

    # ── CmpOp ───────────────────────────────────────────────────
    if isinstance(node, CmpOp):
        left_type = _check_node(node.left, schema_fields, field_names, fields_used, source)
        right_type = _check_node(node.right, schema_fields, field_names, fields_used, source)
        if not _types_compatible(left_type, right_type):
            left_desc = _describe_operand(node.left, left_type)
            right_desc = _describe_operand(node.right, right_type)
            raise FilterTypeError(
                f"comparison '{node.op}' between incompatible types",
                source, node.pos, span=len(node.op),
                left_desc=left_desc, right_desc=right_desc,
            )
        return SEM_BOOL

    # ── InOp ────────────────────────────────────────────────────
    if isinstance(node, InOp):
        field_type = _check_node(node.field, schema_fields, field_names, fields_used, source)
        if node.values.elements:
            list_type = _check_node(node.values, schema_fields, field_names, fields_used, source)
            if list_type != "empty" and not _types_compatible(field_type, list_type):
                raise FilterTypeError(
                    f"'in' list elements ({list_type}) incompatible with "
                    f"field {node.field.name!r} ({field_type})",
                    source, node.pos, span=2,
                )
        return SEM_BOOL

    # ── And / Or ────────────────────────────────────────────────
    if isinstance(node, And) or isinstance(node, Or):
        for op in node.operands:
            t = _check_node(op, schema_fields, field_names, fields_used, source)
            if t != SEM_BOOL:
                op_name = "and" if isinstance(node, And) else "or"
                raise FilterTypeError(
                    f"operands of '{op_name}' must be boolean, got {t}",
                    source, getattr(op, "pos", node.pos),
                )
        return SEM_BOOL

    # ── Not ─────────────────────────────────────────────────────
    if isinstance(node, Not):
        t = _check_node(node.operand, schema_fields, field_names, fields_used, source)
        if t != SEM_BOOL:
            raise FilterTypeError(
                f"operand of 'not' must be boolean, got {t}",
                source, node.pos,
            )
        return SEM_BOOL

    raise TypeError(f"unknown AST node type: {type(node).__name__}")


def _describe_operand(node: Expr, sem_type: str) -> str:
    """Build a human-readable description of an operand for type errors.

    Examples:
        '(int)'                          → "int"
        FieldRef('age', int)             → "int (field 'age')"
        IntLit(18)                       → "int"
        StringLit('hi')                  → "string"
    """
    if isinstance(node, FieldRef):
        return f"{sem_type} (field {node.name!r})"
    return sem_type
