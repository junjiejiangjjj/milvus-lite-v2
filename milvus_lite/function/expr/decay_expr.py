"""DecayExpr — numeric field to decay factor (rerank stage).

Three decay curves (aligned with Milvus and existing DecayReranker):
- gauss:  exp(-0.5 * (dist / scale)^2 / (-1 / (2 * ln(decay) / scale^2)))
  simplified: exp(dist^2 * ln(decay) / scale^2)
- exp:    exp(ln(decay) * dist / scale)
- linear: max(0, 1 - (1-decay)/scale * dist)

where dist = max(0, |val - origin| - offset).

Corresponds to Milvus: internal/util/function/chain/expr/decay_expr.go
"""

from __future__ import annotations

import math
from typing import FrozenSet, List

from milvus_lite.function.types import STAGE_RERANK, FuncContext, FunctionExpr


class DecayExpr(FunctionExpr):
    """numeric column -> decay factor [0, 1]."""

    name = "decay"
    supported_stages: FrozenSet[str] = frozenset({STAGE_RERANK})

    def __init__(
        self,
        function: str,
        origin: float,
        scale: float,
        offset: float = 0.0,
        decay: float = 0.5,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"DecayExpr: scale must be > 0, got {scale}")
        if not (0 < decay < 1):
            raise ValueError(f"DecayExpr: decay must be 0 < decay < 1, got {decay}")
        self._function = function
        self._origin = float(origin)
        self._scale = float(scale)
        self._offset = float(offset)
        self._decay = float(decay)
        # Pre-compute constants (same as DecayReranker)
        self._ln_decay = math.log(decay)
        if function == "gauss":
            self._sigma_sq = scale * scale / self._ln_decay
        elif function == "linear":
            self._slope = (1.0 - decay) / scale

    def execute(self, ctx: FuncContext, inputs: List[list]) -> List[list]:
        values = inputs[0]
        factors: list = []
        for val in values:
            if val is None:
                factors.append(0.0)
                continue
            d = max(0.0, abs(float(val) - self._origin) - self._offset)
            if d == 0.0:
                factors.append(1.0)
                continue
            if self._function == "gauss":
                factors.append(math.exp(d * d / self._sigma_sq))
            elif self._function == "exp":
                factors.append(
                    math.exp(self._ln_decay * d / self._scale)
                )
            elif self._function == "linear":
                factors.append(max(0.0, 1.0 - self._slope * d))
            else:
                factors.append(0.0)
        return [factors]
