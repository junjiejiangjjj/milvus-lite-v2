"""Tests for search/filter/tokens.py — tokenizer correctness."""

import pytest

from litevecdb.search.filter.exceptions import FilterParseError
from litevecdb.search.filter.tokens import Token, TokenKind, tokenize


def kinds(source: str) -> list[TokenKind]:
    return [t.kind for t in tokenize(source)]


def values(source: str) -> list:
    return [t.value for t in tokenize(source)]


# ---------------------------------------------------------------------------
# Empty / whitespace
# ---------------------------------------------------------------------------

def test_empty_source():
    assert kinds("") == [TokenKind.EOF]


def test_whitespace_only():
    assert kinds("   \t\n  ") == [TokenKind.EOF]


# ---------------------------------------------------------------------------
# Integer literals
# ---------------------------------------------------------------------------

def test_integer():
    tokens = tokenize("42")
    assert tokens[0].kind == TokenKind.INT
    assert tokens[0].value == 42


def test_zero():
    assert tokenize("0")[0].value == 0


def test_multi_digit():
    assert tokenize("123456789")[0].value == 123456789


def test_negative_integer_is_two_tokens():
    """Negative numbers are SUB followed by INT — parser builds Unary."""
    tokens = tokenize("-7")
    assert tokens[0].kind == TokenKind.SUB
    assert tokens[1].kind == TokenKind.INT
    assert tokens[1].value == 7


# ---------------------------------------------------------------------------
# Float literals
# ---------------------------------------------------------------------------

def test_float_simple():
    tok = tokenize("3.14")[0]
    assert tok.kind == TokenKind.FLOAT
    assert tok.value == pytest.approx(3.14)


def test_float_scientific_lowercase():
    assert tokenize("1e3")[0].value == pytest.approx(1000.0)


def test_float_scientific_uppercase():
    assert tokenize("1.5E2")[0].value == pytest.approx(150.0)


def test_float_negative_exponent():
    assert tokenize("1.5e-2")[0].value == pytest.approx(0.015)


def test_float_positive_exponent():
    assert tokenize("2e+3")[0].value == pytest.approx(2000.0)


def test_float_malformed_dot_no_digits():
    with pytest.raises(FilterParseError, match="digit expected after"):
        tokenize("5.foo")


def test_float_exponent_no_digits():
    with pytest.raises(FilterParseError, match="exponent"):
        tokenize("1eX")


# ---------------------------------------------------------------------------
# String literals
# ---------------------------------------------------------------------------

def test_string_double_quoted():
    tok = tokenize('"hello"')[0]
    assert tok.kind == TokenKind.STRING
    assert tok.value == "hello"


def test_string_single_quoted():
    tok = tokenize("'world'")[0]
    assert tok.value == "world"


def test_string_empty():
    assert tokenize('""')[0].value == ""


def test_string_with_escapes():
    tok = tokenize(r'"a\"b\\c\n"')[0]
    assert tok.value == 'a"b\\c\n'


def test_string_single_quote_escaped():
    assert tokenize(r"'a\'b'")[0].value == "a'b"


def test_string_unterminated():
    with pytest.raises(FilterParseError, match="unterminated"):
        tokenize('"hello')


def test_string_newline_inside():
    with pytest.raises(FilterParseError, match="unterminated"):
        tokenize('"hello\nworld"')


def test_string_unknown_escape():
    with pytest.raises(FilterParseError, match="unknown escape"):
        tokenize(r'"a\zb"')


# ---------------------------------------------------------------------------
# Boolean literals (6 forms only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["true", "True", "TRUE"])
def test_bool_true_forms(text):
    tok = tokenize(text)[0]
    assert tok.kind == TokenKind.BOOL
    assert tok.value is True


@pytest.mark.parametrize("text", ["false", "False", "FALSE"])
def test_bool_false_forms(text):
    tok = tokenize(text)[0]
    assert tok.kind == TokenKind.BOOL
    assert tok.value is False


@pytest.mark.parametrize("text", ["tRuE", "TrUe", "fAlSe"])
def test_bool_mixed_case_rejected(text):
    with pytest.raises(FilterParseError, match="did you mean"):
        tokenize(text)


# ---------------------------------------------------------------------------
# Identifiers (case-sensitive)
# ---------------------------------------------------------------------------

def test_identifier_simple():
    tok = tokenize("age")[0]
    assert tok.kind == TokenKind.IDENT
    assert tok.text == "age"


def test_identifier_with_underscore():
    assert tokenize("user_id")[0].text == "user_id"


def test_identifier_with_digits():
    assert tokenize("doc_42")[0].text == "doc_42"


def test_identifier_starts_with_underscore():
    assert tokenize("_internal")[0].text == "_internal"


def test_identifier_case_sensitive():
    """Age and age are different identifiers."""
    assert tokenize("Age")[0].text == "Age"
    assert tokenize("age")[0].text == "age"


# ---------------------------------------------------------------------------
# Keywords (case-insensitive)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_kind", [
    ("and", TokenKind.AND), ("AND", TokenKind.AND),
    ("or", TokenKind.OR), ("OR", TokenKind.OR),
    ("not", TokenKind.NOT), ("NOT", TokenKind.NOT),
    ("in", TokenKind.IN), ("IN", TokenKind.IN),
])
def test_keyword_forms(text, expected_kind):
    assert tokenize(text)[0].kind == expected_kind


def test_keyword_mixed_case_treated_as_identifier():
    """And / Or / Not / In are NOT recognized as keywords (just identifiers).
    This matches Milvus Plan.g4 — only lowercase and uppercase forms are
    keywords, not mixed-case variants."""
    assert tokenize("And")[0].kind == TokenKind.IDENT
    assert tokenize("Or")[0].kind == TokenKind.IDENT


# ---------------------------------------------------------------------------
# Symbol operators
# ---------------------------------------------------------------------------

def test_eq():
    assert tokenize("==")[0].kind == TokenKind.EQ


def test_ne():
    assert tokenize("!=")[0].kind == TokenKind.NE


def test_lt_le_gt_ge():
    assert tokenize("<")[0].kind == TokenKind.LT
    assert tokenize("<=")[0].kind == TokenKind.LE
    assert tokenize(">")[0].kind == TokenKind.GT
    assert tokenize(">=")[0].kind == TokenKind.GE


def test_logical_symbol_aliases():
    assert tokenize("&&")[0].kind == TokenKind.AND
    assert tokenize("||")[0].kind == TokenKind.OR
    assert tokenize("!")[0].kind == TokenKind.NOT


def test_punctuation():
    assert tokenize("()[],")[:5] == [
        Token(TokenKind.LPAREN, "(", 0),
        Token(TokenKind.RPAREN, ")", 1),
        Token(TokenKind.LBRACKET, "[", 2),
        Token(TokenKind.RBRACKET, "]", 3),
        Token(TokenKind.COMMA, ",", 4),
    ]


# ---------------------------------------------------------------------------
# Error cases — bad characters / hints
# ---------------------------------------------------------------------------

def test_bare_equals_rejected():
    with pytest.raises(FilterParseError, match="did you mean '=='"):
        tokenize("a = 1")


def test_bare_ampersand_rejected():
    with pytest.raises(FilterParseError, match="did you mean '&&'"):
        tokenize("a & b")


def test_bare_pipe_rejected():
    with pytest.raises(FilterParseError, match="did you mean '\\|\\|'"):
        tokenize("a | b")


def test_lt_gt_combo_rejected():
    with pytest.raises(FilterParseError, match="did you mean '!='"):
        tokenize("a <> b")


def test_unknown_character():
    with pytest.raises(FilterParseError, match="unexpected character"):
        tokenize("a @ b")


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

def test_token_positions():
    tokens = tokenize("age > 18")
    assert tokens[0].pos == 0       # age
    assert tokens[1].pos == 4       # >
    assert tokens[2].pos == 6       # 18
    assert tokens[3].pos == 8       # EOF


def test_error_position():
    with pytest.raises(FilterParseError) as exc:
        tokenize("age @ 18")
    assert exc.value.pos == 4
    assert "column 5" in str(exc.value)  # 1-indexed in message


# ---------------------------------------------------------------------------
# Realistic expressions
# ---------------------------------------------------------------------------

def test_full_expression():
    tokens = tokenize("age > 18 and category == 'tech'")
    assert kinds("age > 18 and category == 'tech'") == [
        TokenKind.IDENT, TokenKind.GT, TokenKind.INT,
        TokenKind.AND,
        TokenKind.IDENT, TokenKind.EQ, TokenKind.STRING,
        TokenKind.EOF,
    ]


def test_in_expression():
    assert kinds("age in [10, 20, 30]") == [
        TokenKind.IDENT, TokenKind.IN,
        TokenKind.LBRACKET,
        TokenKind.INT, TokenKind.COMMA,
        TokenKind.INT, TokenKind.COMMA,
        TokenKind.INT,
        TokenKind.RBRACKET,
        TokenKind.EOF,
    ]


def test_not_in_expression():
    assert kinds("category not in ['a', 'b']") == [
        TokenKind.IDENT, TokenKind.NOT, TokenKind.IN,
        TokenKind.LBRACKET,
        TokenKind.STRING, TokenKind.COMMA, TokenKind.STRING,
        TokenKind.RBRACKET,
        TokenKind.EOF,
    ]


def test_complex_with_parens_and_negation():
    assert kinds("not (age > 18) and !(score < 0.5)") == [
        TokenKind.NOT, TokenKind.LPAREN,
        TokenKind.IDENT, TokenKind.GT, TokenKind.INT,
        TokenKind.RPAREN,
        TokenKind.AND,
        TokenKind.NOT, TokenKind.LPAREN,
        TokenKind.IDENT, TokenKind.LT, TokenKind.FLOAT,
        TokenKind.RPAREN,
        TokenKind.EOF,
    ]
