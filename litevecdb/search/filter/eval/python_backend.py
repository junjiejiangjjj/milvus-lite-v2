"""Row-wise Python backend for filter evaluation.

Slow but flexible — used as the differential-test baseline in F1 and
as the actual fallback in F2b ($meta dynamic field) and F3 (UDFs).

NULL semantics: Kleene three-valued logic. The internal _eval_row
returns Python `bool`, `True`, `False`, or `None` (= unknown). The
top-level entry collapses None → False so the caller gets a clean
BooleanArray with no null entries.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Optional, Union

import pyarrow as pa

from litevecdb.search.filter.ast import (
    And,
    BoolLit,
    CmpOp,
    FieldRef,
    FloatLit,
    InOp,
    IntLit,
    ListLit,
    Not,
    Or,
    StringLit,
)

if TYPE_CHECKING:
    from litevecdb.search.filter.semantic import CompiledExpr


_CMP_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
}


def evaluate_python(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Row-wise interpreter. Returns BooleanArray of length data.num_rows.

    None (Kleene unknown) at the top level becomes False (no row matches).
    """
    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])

    rows = data.to_pylist()
    out = [False] * len(rows)
    for i, row in enumerate(rows):
        result = _eval_row(compiled.ast, row)
        if result is None:
            out[i] = False
        else:
            out[i] = bool(result)
    return pa.array(out, type=pa.bool_())


def _eval_row(node, row: dict) -> Any:
    """Evaluate a single AST node against a single row dict.

    Returns:
        - Python int / float / str / bool for literals and field refs
        - True / False for boolean operations
        - None for Kleene "unknown" (e.g., comparing a NULL field)
    """
    if isinstance(node, IntLit):
        return node.value
    if isinstance(node, FloatLit):
        return node.value
    if isinstance(node, StringLit):
        return node.value
    if isinstance(node, BoolLit):
        return node.value

    if isinstance(node, FieldRef):
        return row.get(node.name)

    if isinstance(node, CmpOp):
        left = _eval_row(node.left, row)
        right = _eval_row(node.right, row)
        if left is None or right is None:
            return None  # NULL propagation (Kleene)
        try:
            return _CMP_OPS[node.op](left, right)
        except TypeError:
            # Cross-type comparison at runtime — fall back to None.
            # (Should not happen if semantic.py did its job, but robust.)
            return None

    if isinstance(node, InOp):
        val = _eval_row(node.field, row)
        if val is None:
            return False if node.negate else False
        members = {el.value for el in node.values.elements}
        result = val in members
        return (not result) if node.negate else result

    if isinstance(node, And):
        # Kleene AND:
        #   any False → False
        #   else any None → None
        #   else True
        seen_null = False
        for op in node.operands:
            r = _eval_row(op, row)
            if r is False:
                return False
            if r is None:
                seen_null = True
        return None if seen_null else True

    if isinstance(node, Or):
        # Kleene OR:
        #   any True  → True
        #   else any None → None
        #   else False
        seen_null = False
        for op in node.operands:
            r = _eval_row(op, row)
            if r is True:
                return True
            if r is None:
                seen_null = True
        return None if seen_null else False

    if isinstance(node, Not):
        r = _eval_row(node.operand, row)
        if r is None:
            return None
        return not r

    raise TypeError(f"unknown AST node: {type(node).__name__}")
