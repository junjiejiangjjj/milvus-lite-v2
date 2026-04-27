"""Boost Ranker support.

Boost Ranker is a request-level RERANK function used by Milvus search.
It adjusts candidate scores with metadata-driven rules before the final
top-k is selected.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from milvus_lite.exceptions import SchemaValidationError


def decode_kv_pairs(kv_pairs) -> Dict[str, Any]:
    """Decode proto KeyValuePair values using JSON when possible."""
    out: Dict[str, Any] = {}
    for kv in kv_pairs:
        try:
            out[kv.key] = json.loads(kv.value)
        except (json.JSONDecodeError, TypeError, ValueError):
            out[kv.key] = kv.value
    return out


def decode_boost_function_score(function_score) -> Optional[dict]:
    """Decode a SearchRequest.function_score if it contains Boost Rankers."""
    functions = []
    for fn in getattr(function_score, "functions", []):
        params = decode_kv_pairs(fn.params)
        reranker = str(params.get("reranker", "")).lower()
        if reranker != "boost":
            raise SchemaValidationError(
                "search ranker only supports Boost Ranker functions "
                f"(got reranker={params.get('reranker')!r})"
            )
        if list(getattr(fn, "input_field_names", [])):
            raise SchemaValidationError(
                f"Boost Ranker function {fn.name!r} requires empty input_field_names"
            )
        functions.append({
            "name": fn.name,
            "params": _validate_boost_params(fn.name, params),
        })

    if not functions:
        return None

    return {
        "functions": functions,
        "params": _normalize_function_score_params(
            decode_kv_pairs(getattr(function_score, "params", []))
        ),
    }


def apply_boost_ranker(
    results: List[List[dict]],
    ranker: dict,
    *,
    metric_type: str,
    pk_name: str,
    compile_filter,
    row_matches_filter,
) -> List[List[dict]]:
    """Apply Boost Ranker to candidate hits and sort by adjusted score."""
    if not ranker:
        return results

    functions = ranker.get("functions") or []
    params = ranker.get("params") or {}
    boost_mode = params.get("boost_mode", "multiply")
    function_mode = params.get("function_mode", "multiply")

    compiled_filters: Dict[str, Any] = {}
    for fn in functions:
        filt = fn["params"].get("filter")
        if filt:
            compiled_filters[filt] = compile_filter(filt)

    boosted: List[List[dict]] = []
    for hits in results:
        adjusted_hits = []
        for hit in hits:
            values = []
            for fn in functions:
                fn_params = fn["params"]
                filt = fn_params.get("filter")
                if filt:
                    row = dict(hit.get("entity") or {})
                    row[pk_name] = hit.get("id")
                    if not row_matches_filter(row, compiled_filters[filt]):
                        continue

                value = float(fn_params["weight"])
                random_score = fn_params.get("random_score")
                if random_score is not None:
                    value *= _stable_random_score(hit, random_score, pk_name)
                values.append(value)

            if not values:
                adjusted_hits.append(hit)
                continue

            combined = _combine(values, function_mode)
            new_hit = dict(hit)
            new_hit["distance"] = _apply_boost_to_distance(
                float(hit["distance"]), combined, boost_mode, metric_type
            )
            adjusted_hits.append(new_hit)

        adjusted_hits.sort(key=lambda h: h["distance"])
        boosted.append(adjusted_hits)

    return boosted


def _validate_boost_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if params.get("reranker", "").lower() != "boost":
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} requires params.reranker='boost'"
        )

    if "weight" not in params:
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} requires params.weight"
        )
    try:
        params["weight"] = float(params["weight"])
    except (TypeError, ValueError):
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} params.weight must be a number"
        )

    filt = params.get("filter")
    if filt is not None and not isinstance(filt, str):
        raise SchemaValidationError(
            f"Boost Ranker function {name!r} params.filter must be a string"
        )

    random_score = params.get("random_score")
    if random_score is not None:
        if isinstance(random_score, str):
            try:
                random_score = json.loads(random_score)
            except (json.JSONDecodeError, TypeError, ValueError):
                raise SchemaValidationError(
                    f"Boost Ranker function {name!r} params.random_score "
                    "must be an object"
                )
        if not isinstance(random_score, dict):
            raise SchemaValidationError(
                f"Boost Ranker function {name!r} params.random_score must be an object"
            )
        params["random_score"] = random_score

    return params


def _normalize_function_score_params(params: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for key, default in (("boost_mode", "multiply"), ("function_mode", "multiply")):
        value = str(params.get(key, default)).lower()
        if value not in ("multiply", "sum"):
            raise SchemaValidationError(
                f"FunctionScore params.{key} must be 'Multiply' or 'Sum'"
            )
        out[key] = value
    return out


def _combine(values: List[float], mode: str) -> float:
    if mode == "sum":
        return sum(values)
    product = 1.0
    for value in values:
        product *= value
    return product


def _apply_boost_to_distance(
    distance: float,
    value: float,
    boost_mode: str,
    metric_type: str,
) -> float:
    if boost_mode == "multiply":
        return distance * value

    # Sum mode operates on the metric's natural score.  IP and BM25 use
    # higher-is-better scores internally represented as negative distances.
    if metric_type in ("IP", "BM25"):
        return distance - value
    return distance + value


def _stable_random_score(hit: dict, random_score: dict, pk_name: str) -> float:
    seed = random_score.get("seed", 0)
    field = random_score.get("field")
    if field:
        entity = hit.get("entity") or {}
        if field == pk_name:
            value = hit.get("id")
        else:
            value = entity.get(field)
    else:
        value = hit.get("id")

    payload = f"{seed}:{value!r}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
