"""
Unit tests for MechanicsDSL parser module.

Tests the tokenizer, AST nodes, and parser functionality.
"""

import pytest

from mechanics_dsl.parser import (
    BinaryOpExpr,
    ConstraintDef,
    DerivativeVarExpr,
    GreekLetterExpr,
    IdentExpr,
    MechanicsParser,
    NumberExpr,
    SystemDef,
    Token,
    UnaryOpExpr,
    VarDef,
    VectorExpr,
    tokenize,
)


class TestTokenize:
    """Tests for the tokenize function."""

    def test_tokenize_number(self):
        """Test tokenizing a number."""
        tokens = tokenize("42")
        assert len(tokens) >= 1
        assert any(t.type == "NUMBER" for t in tokens)

    def test_tokenize_float(self):
        """Test tokenizing a floating point number."""
        tokens = tokenize("3.14159")
        assert len(tokens) >= 1
        assert any(t.type == "NUMBER" for t in tokens)

    def test_tokenize_scientific_notation(self):
        """Test tokenizing scientific notation."""
        tokens = tokenize("1.5e-10")
        assert len(tokens) >= 1
        assert any(t.type == "NUMBER" for t in tokens)

    def test_tokenize_identifier(self):
        """Test tokenizing an identifier."""
        tokens = tokenize("mass")
        assert len(tokens) >= 1
        assert any(t.type == "IDENT" for t in tokens)

    def test_tokenize_greek_letter(self):
        """Test tokenizing Greek letters."""
        tokens = tokenize(r"\theta")
        assert len(tokens) >= 1
        assert any(t.type == "GREEK_LETTER" for t in tokens)

    def test_tokenize_system_command(self):
        """Test tokenizing system command."""
        tokens = tokenize(r"\system{Test}")
        assert any(t.type == "SYSTEM" for t in tokens)

    def test_tokenize_defvar_command(self):
        """Test tokenizing defvar command."""
        tokens = tokenize(r"\defvar{m}{1.0}")
        assert any(t.type == "DEFVAR" for t in tokens)

    def test_tokenize_command(self):
        """Test tokenizing general command."""
        tokens = tokenize(r"\coordinate{x}")
        # coordinate is parsed as a COMMAND
        assert any(t.type == "COMMAND" for t in tokens)

    def test_tokenize_lagrangian_command(self):
        """Test tokenizing lagrangian command."""
        tokens = tokenize(r"\lagrangian{L}")
        assert any(t.type == "LAGRANGIAN" for t in tokens)

    def test_tokenize_operators(self):
        """Test tokenizing operators."""
        tokens = tokenize("+ - * / ^")
        types = [t.type for t in tokens]
        assert "PLUS" in types
        assert "MINUS" in types
        assert "MULTIPLY" in types
        assert "DIVIDE" in types
        assert "POWER" in types

    def test_tokenize_braces(self):
        """Test tokenizing braces."""
        tokens = tokenize("{ } [ ] ( )")
        types = [t.type for t in tokens]
        assert "LBRACE" in types
        assert "RBRACE" in types
        assert "LBRACKET" in types
        assert "RBRACKET" in types
        assert "LPAREN" in types
        assert "RPAREN" in types

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        tokens = tokenize("")
        assert len(tokens) == 0

    def test_tokenize_whitespace_ignored(self):
        """Test that whitespace is filtered out."""
        tokens = tokenize("   ")
        # Whitespace should be filtered
        assert all(t.type != "WHITESPACE" for t in tokens)

    def test_tokenize_comment_ignored(self):
        """Test that comments are filtered out."""
        tokens = tokenize("x % this is a comment")
        # Comments should be filtered
        assert all(t.type != "COMMENT" for t in tokens)

    def test_tokenize_dot_notation(self):
        """Test tokenizing dot notation."""
        tokens = tokenize(r"\dot{x}")
        assert any(t.type == "DOT_NOTATION" for t in tokens)

    def test_tokenize_ddot_notation(self):
        """Test tokenizing double dot notation."""
        tokens = tokenize(r"\ddot{x}")
        assert any(t.type == "DOT_NOTATION" for t in tokens)


class TestToken:
    """Tests for Token class."""

    def test_token_creation(self):
        """Test creating a token."""
        token = Token(type="NUMBER", value="42", position=0)
        assert token.type == "NUMBER"
        assert token.value == "42"

    def test_token_repr(self):
        """Test token string representation."""
        token = Token(type="IDENT", value="x", position=0)
        repr_str = repr(token)
        assert "IDENT" in repr_str
        assert "x" in repr_str

    def test_token_position_tracking(self):
        """Test token position tracking."""
        token = Token(type="NUMBER", value="1", position=10, line=2, column=5)
        assert token.position == 10
        assert token.line == 2
        assert token.column == 5


class TestNumberExpr:
    """Tests for NumberExpr AST node."""

    def test_number_expr_creation(self):
        """Test creating NumberExpr."""
        expr = NumberExpr(value=3.14)
        assert expr.value == 3.14

    def test_number_expr_repr(self):
        """Test NumberExpr string representation."""
        expr = NumberExpr(value=42.0)
        repr_str = repr(expr)
        assert "42" in repr_str


class TestIdentExpr:
    """Tests for IdentExpr AST node."""

    def test_ident_expr_creation(self):
        """Test creating IdentExpr."""
        expr = IdentExpr(name="mass")
        assert expr.name == "mass"

    def test_ident_expr_repr(self):
        """Test IdentExpr string representation."""
        expr = IdentExpr(name="velocity")
        repr_str = repr(expr)
        assert "velocity" in repr_str


class TestGreekLetterExpr:
    """Tests for GreekLetterExpr AST node."""

    def test_greek_letter_creation(self):
        """Test creating GreekLetterExpr."""
        expr = GreekLetterExpr(letter="theta")
        assert expr.letter == "theta"

    def test_greek_letter_repr(self):
        """Test GreekLetterExpr string representation."""
        expr = GreekLetterExpr(letter="omega")
        repr_str = repr(expr)
        assert "omega" in repr_str


class TestDerivativeVarExpr:
    """Tests for DerivativeVarExpr AST node."""

    def test_first_derivative_creation(self):
        """Test creating first derivative."""
        expr = DerivativeVarExpr(var="x", order=1)
        assert expr.var == "x"
        assert expr.order == 1

    def test_second_derivative_creation(self):
        """Test creating second derivative."""
        expr = DerivativeVarExpr(var="theta", order=2)
        assert expr.var == "theta"
        assert expr.order == 2

    def test_derivative_repr(self):
        """Test DerivativeVarExpr string representation."""
        expr = DerivativeVarExpr(var="x", order=1)
        repr_str = repr(expr)
        assert "x" in repr_str


class TestBinaryOpExpr:
    """Tests for BinaryOpExpr AST node."""

    def test_addition_creation(self):
        """Test creating addition expression."""
        left = NumberExpr(value=1.0)
        right = NumberExpr(value=2.0)
        expr = BinaryOpExpr(left=left, operator="+", right=right)
        assert expr.operator == "+"

    def test_subtraction_creation(self):
        """Test creating subtraction expression."""
        left = NumberExpr(value=5.0)
        right = NumberExpr(value=3.0)
        expr = BinaryOpExpr(left=left, operator="-", right=right)
        assert expr.operator == "-"

    def test_multiplication_creation(self):
        """Test creating multiplication expression."""
        left = NumberExpr(value=2.0)
        right = NumberExpr(value=3.0)
        expr = BinaryOpExpr(left=left, operator="*", right=right)
        assert expr.operator == "*"

    def test_division_creation(self):
        """Test creating division expression."""
        left = NumberExpr(value=10.0)
        right = NumberExpr(value=2.0)
        expr = BinaryOpExpr(left=left, operator="/", right=right)
        assert expr.operator == "/"

    def test_power_creation(self):
        """Test creating power expression."""
        left = NumberExpr(value=2.0)
        right = NumberExpr(value=3.0)
        expr = BinaryOpExpr(left=left, operator="^", right=right)
        assert expr.operator == "^"


class TestUnaryOpExpr:
    """Tests for UnaryOpExpr AST node."""

    def test_negation_creation(self):
        """Test creating negation expression."""
        operand = NumberExpr(value=5.0)
        expr = UnaryOpExpr(operator="-", operand=operand)
        assert expr.operator == "-"

    def test_positive_creation(self):
        """Test creating positive expression."""
        operand = NumberExpr(value=5.0)
        expr = UnaryOpExpr(operator="+", operand=operand)
        assert expr.operator == "+"


class TestVectorExpr:
    """Tests for VectorExpr AST node."""

    def test_vector_creation(self):
        """Test creating vector expression."""
        components = [NumberExpr(1.0), NumberExpr(2.0), NumberExpr(3.0)]
        expr = VectorExpr(components=components)
        assert len(expr.components) == 3

    def test_empty_vector(self):
        """Test creating empty vector."""
        expr = VectorExpr(components=[])
        assert len(expr.components) == 0


class TestParse:
    """Tests for the parse function."""

    def test_parse_simple_system(self):
        """Test parsing simple system definition."""
        dsl = r"\system{Test}"
        tokens = tokenize(dsl)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert ast is not None

    def test_parse_with_lagrangian(self):
        """Test parsing with Lagrangian."""
        dsl = r"\system{Oscillator}\lagrangian{m}"
        tokens = tokenize(dsl)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert ast is not None

    def test_parse_with_initial_conditions(self):
        """Test parsing with initial conditions."""
        dsl = r"\system{Test}\initial{x=1.0}"
        tokens = tokenize(dsl)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert ast is not None

    def test_parse_returns_list(self):
        """Test that parse returns a list of statements."""
        dsl = r"\system{Test}"
        tokens = tokenize(dsl)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert isinstance(ast, list)
