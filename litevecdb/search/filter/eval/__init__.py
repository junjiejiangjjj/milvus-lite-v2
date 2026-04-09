"""Backend dispatcher for filter evaluation.

evaluate(compiled, table) → BooleanArray

The backend choice was made statically at compile_expr time and stored
in compiled.backend. Phase F1 always picks "arrow"; F2b will start
choosing "python" when an expression contains $meta dynamic-field access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pyarrow as pa

from litevecdb.search.filter.eval.arrow_backend import evaluate_arrow
from litevecdb.search.filter.eval.python_backend import evaluate_python

if TYPE_CHECKING:
    from litevecdb.search.filter.semantic import CompiledExpr


def evaluate(
    compiled: "CompiledExpr",
    data: Union[pa.Table, pa.RecordBatch],
) -> pa.BooleanArray:
    """Dispatch to the backend recorded on the compiled expression."""
    if compiled.backend == "arrow":
        return evaluate_arrow(compiled, data)
    if compiled.backend == "python":
        return evaluate_python(compiled, data)
    raise ValueError(f"unknown filter backend: {compiled.backend!r}")


__all__ = ["evaluate", "evaluate_arrow", "evaluate_python"]
