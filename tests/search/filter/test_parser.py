"""Tests for search/filter/parser.py — Pratt parser correctness."""

import pytest

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
from litevecdb.search.filter.exceptions import FilterParseError
from litevecdb.search.filter.parser import parse_expr


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

def test_int_literal():
    e = parse_expr("42")
    assert isinstance(e, IntLit)
    assert e.value == 42


def test_negative_int_literal_folded():
    """Unary minus on int literal is constant-folded into a single IntLit."""
    e = parse_expr("-7")
    assert isinstance(e, IntLit)
    assert e.value == -7


def test_float_literal():
    e = parse_expr("3.14")
    assert isinstance(e, FloatLit)
    assert e.value == pytest.approx(3.14)


def test_negative_float_folded():
    e = parse_expr("-1.5")
    assert isinstance(e, FloatLit)
    assert e.value == pytest.approx(-1.5)


def test_string_literal():
    e = parse_expr("'hello'")
    assert isinstance(e, StringLit)
    assert e.value == "hello"


def test_bool_literal_true():
    e = parse_expr("true")
    assert isinstance(e, BoolLit)
    assert e.value is True


# ---------------------------------------------------------------------------
# Field references
# ---------------------------------------------------------------------------

def test_field_ref():
    e = parse_expr("age")
    assert isinstance(e, FieldRef)
    assert e.name == "age"


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def test_cmp_simple():
    e = parse_expr("age > 18")
    assert isinstance(e, CmpOp)
    assert e.op == ">"
    assert isinstance(e.left, FieldRef)
    assert e.left.name == "age"
    assert isinstance(e.right, IntLit)
    assert e.right.value == 18


@pytest.mark.parametrize("op_text,op_value", [
    ("==", "=="), ("!=", "!="),
    ("<", "<"), ("<=", "<="),
    (">", ">"), (">=", ">="),
])
def test_all_cmp_operators(op_text, op_value):
    e = parse_expr(f"a {op_text} 1")
    assert isinstance(e, CmpOp)
    assert e.op == op_value


def test_cmp_reversed_lhs_literal():
    """Milvus accepts `18 < age` (literal on the left)."""
    e = parse_expr("18 < age")
    assert isinstance(e, CmpOp)
    assert e.op == "<"
    assert isinstance(e.left, IntLit)
    assert isinstance(e.right, FieldRef)


def test_chained_comparison_parses():
    """`a == b == c` is accepted at parse time. semantic.py rejects it
    later (the LHS of the second == becomes bool, which can't compare
    to int)."""
    e = parse_expr("a == b == 1")
    # Outer node should be the second ==
    assert isinstance(e, CmpOp)
    assert e.op == "=="
    assert isinstance(e.left, CmpOp)
    assert e.left.op == "=="


# ---------------------------------------------------------------------------
# Logical AND / OR / NOT
# ---------------------------------------------------------------------------

def test_and_two_operands():
    e = parse_expr("a > 1 and b < 2")
    assert isinstance(e, And)
    assert len(e.operands) == 2


def test_and_chain_flattened():
    """`a and b and c and d` flattens into a single And with 4 operands."""
    e = parse_expr("a and b and c and d")
    assert isinstance(e, And)
    assert len(e.operands) == 4


def test_or_chain_flattened():
    e = parse_expr("a or b or c")
    assert isinstance(e, Or)
    assert len(e.operands) == 3


def test_not_prefix():
    e = parse_expr("not (a > 1)")
    assert isinstance(e, Not)
    assert isinstance(e.operand, CmpOp)


def test_not_double():
    e = parse_expr("not not a")
    assert isinstance(e, Not)
    assert isinstance(e.operand, Not)


def test_bang_alias():
    e = parse_expr("!(a > 1)")
    assert isinstance(e, Not)


def test_logical_symbol_aliases():
    e1 = parse_expr("a > 1 && b < 2")
    e2 = parse_expr("a > 1 and b < 2")
    assert isinstance(e1, And)
    assert isinstance(e2, And)


# ---------------------------------------------------------------------------
# Operator precedence
# ---------------------------------------------------------------------------

def test_and_binds_tighter_than_or():
    """`a or b and c` parses as `a or (b and c)`."""
    e = parse_expr("a or b and c")
    assert isinstance(e, Or)
    assert len(e.operands) == 2
    # Second operand is the And
    assert isinstance(e.operands[1], And)


def test_not_binds_tighter_than_and():
    """`not a and b` parses as `(not a) and b`."""
    e = parse_expr("not a and b")
    assert isinstance(e, And)
    assert isinstance(e.operands[0], Not)


def test_cmp_binds_tighter_than_and():
    """`a > 1 and b < 2` — comparisons evaluated first."""
    e = parse_expr("a > 1 and b < 2")
    assert isinstance(e, And)
    assert all(isinstance(op, CmpOp) for op in e.operands)


def test_parens_override_precedence():
    """`(a or b) and c` — parens force or to be deeper than and."""
    e = parse_expr("(a or b) and c")
    assert isinstance(e, And)
    assert isinstance(e.operands[0], Or)


# ---------------------------------------------------------------------------
# IN / NOT IN
# ---------------------------------------------------------------------------

def test_in_simple():
    e = parse_expr("age in [10, 20, 30]")
    assert isinstance(e, InOp)
    assert e.field.name == "age"
    assert e.negate is False
    assert len(e.values.elements) == 3
    assert all(isinstance(el, IntLit) for el in e.values.elements)


def test_not_in():
    e = parse_expr("category not in ['a', 'b']")
    assert isinstance(e, InOp)
    assert e.negate is True
    assert e.field.name == "category"


def test_in_string_list():
    e = parse_expr("category in ['tech', 'news']")
    assert isinstance(e, InOp)
    assert [el.value for el in e.values.elements] == ["tech", "news"]


def test_in_empty_list():
    e = parse_expr("age in []")
    assert isinstance(e, InOp)
    assert e.values.elements == ()


def test_in_trailing_comma():
    e = parse_expr("age in [1, 2, 3,]")
    assert isinstance(e, InOp)
    assert len(e.values.elements) == 3


def test_in_negative_literal():
    e = parse_expr("temp in [-5, 0, 10]")
    assert [el.value for el in e.values.elements] == [-5, 0, 10]


def test_in_lhs_must_be_field():
    """`'a' in [...]` is rejected — Milvus alignment."""
    with pytest.raises(FilterParseError, match="must be a field"):
        parse_expr("'a' in [1, 2]")


def test_in_rhs_must_be_list():
    with pytest.raises(FilterParseError, match="expected '\\['"):
        parse_expr("age in 5")


# ---------------------------------------------------------------------------
# Realistic combined expressions
# ---------------------------------------------------------------------------

def test_complex_expression():
    e = parse_expr("age > 18 and category in ['tech', 'news'] or score >= 0.5")
    # Should parse as: (age > 18 and category in [...]) or score >= 0.5
    assert isinstance(e, Or)
    assert len(e.operands) == 2
    assert isinstance(e.operands[0], And)
    assert isinstance(e.operands[1], CmpOp)


def test_negated_subexpression():
    e = parse_expr("not (age > 18) and category == 'tech'")
    assert isinstance(e, And)
    assert isinstance(e.operands[0], Not)
    assert isinstance(e.operands[1], CmpOp)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_empty_input():
    with pytest.raises(FilterParseError, match="end of expression"):
        parse_expr("")


def test_unclosed_paren():
    with pytest.raises(FilterParseError, match="expected '\\)'"):
        parse_expr("(a > 1")


def test_dangling_operator():
    with pytest.raises(FilterParseError):
        parse_expr("a > ")


def test_function_call_rejected():
    with pytest.raises(FilterParseError, match="Phase F1"):
        parse_expr("json_contains(meta, 'x')")


def test_unary_minus_on_field_rejected():
    """Phase F1 doesn't have arithmetic, so -age is meaningless."""
    with pytest.raises(FilterParseError, match="numeric literals"):
        parse_expr("-age > 0")


def test_extra_token_after_expression():
    with pytest.raises(FilterParseError, match="expected end"):
        parse_expr("a > 1 b")


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

def test_field_ref_pos():
    e = parse_expr("    age > 1")
    assert isinstance(e, CmpOp)
    assert e.left.pos == 4


def test_error_pos_in_message():
    with pytest.raises(FilterParseError) as exc:
        parse_expr("age > > 1")
    # Caret should land on the second '>'
    msg = str(exc.value)
    assert "column 7" in msg
