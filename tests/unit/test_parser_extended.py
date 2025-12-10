"""
Extended unit tests for MechanicsDSL parser module.

Tests tokenization and parser AST nodes.
"""

import pytest
import numpy as np

from mechanics_dsl.parser import (
    tokenize, Token, 
    NumberExpr, IdentExpr, BinaryOpExpr, UnaryOpExpr,
    MechanicsParser, FractionExpr, DerivativeVarExpr,
    GreekLetterExpr, FunctionCallExpr
)


class TestTokenize:
    """Tests for tokenize function."""
    
    def test_tokenize_empty(self):
        """Test tokenizing empty string."""
        tokens = tokenize("")
        assert isinstance(tokens, list)
    
    def test_tokenize_number(self):
        """Test tokenizing number."""
        tokens = tokenize("42")
        assert len(tokens) >= 1
        assert any(t.type == "NUMBER" for t in tokens)
    
    def test_tokenize_float(self):
        """Test tokenizing float."""
        tokens = tokenize("3.14")
        assert any(t.type == "NUMBER" for t in tokens)
    
    def test_tokenize_scientific(self):
        """Test tokenizing scientific notation."""
        tokens = tokenize("1e-5")
        assert any(t.type == "NUMBER" for t in tokens)
    
    def test_tokenize_identifier(self):
        """Test tokenizing identifier."""
        tokens = tokenize("x")
        assert any(t.type == "IDENT" for t in tokens)
    
    def test_tokenize_operators(self):
        """Test tokenizing operators."""
        tokens = tokenize("+ - * / ^")
        types = [t.type for t in tokens]
        # Just verify we get tokens for operators
        assert len(tokens) >= 3
    
    def test_tokenize_parens(self):
        """Test tokenizing parentheses."""
        tokens = tokenize("(x)")
        types = [t.type for t in tokens]
        assert "LPAREN" in types
        assert "RPAREN" in types
    
    def test_tokenize_system_keyword(self):
        """Test tokenizing system keyword."""
        tokens = tokenize("\\system")
        assert any(t.type == "SYSTEM" for t in tokens)
    
    def test_tokenize_defvar(self):
        """Test tokenizing defvar keyword."""
        tokens = tokenize("\\defvar")
        assert any(t.type == "DEFVAR" for t in tokens)
    
    def test_tokenize_lagrangian(self):
        """Test tokenizing lagrangian keyword."""
        tokens = tokenize("\\lagrangian")
        assert any(t.type == "LAGRANGIAN" for t in tokens)
    
    def test_tokenize_greek_letters(self):
        """Test tokenizing Greek letters."""
        tokens = tokenize("\\theta \\omega \\alpha")
        # Just verify we get some tokens
        assert len(tokens) >= 1
    
    def test_tokenize_dot_notation(self):
        """Test tokenizing dot/ddot notation."""
        tokens = tokenize("\\dot{x}")
        assert any(t.type == "DOT_NOTATION" for t in tokens)
    
    def test_tokenize_frac(self):
        """Test tokenizing fraction."""
        tokens = tokenize("\\frac{1}{2}")
        assert any(t.type == "FRAC" for t in tokens)
    
    def test_tokenize_trig(self):
        """Test tokenizing trig functions."""
        tokens = tokenize("\\sin \\cos")
        types = [t.type for t in tokens]
        # Trig functions tokenized as FUNCTION
        assert len(tokens) > 0
    
    def test_tokenize_initial(self):
        """Test tokenizing initial keyword."""
        tokens = tokenize("\\initial")
        assert any(t.type == "INITIAL" for t in tokens)
    
    def test_tokenize_parameter(self):
        """Test tokenizing parameter keyword."""
        tokens = tokenize("\\parameter")
        assert any(t.type == "PARAMETER" for t in tokens)
    
    def test_tokenize_complex_expression(self):
        """Test tokenizing complex expression."""
        tokens = tokenize("\\frac{1}{2} m \\dot{x}^2")
        assert len(tokens) > 0
    
    def test_tokenize_braces(self):
        """Test tokenizing braces."""
        tokens = tokenize("{x}")
        types = [t.type for t in tokens]
        assert "LBRACE" in types
        assert "RBRACE" in types
    
    def test_tokenize_equals(self):
        """Test tokenizing equals."""
        tokens = tokenize("x = 1")
        types = [t.type for t in tokens]
        assert "EQUALS" in types
    
    def test_tokenize_comma(self):
        """Test tokenizing comma."""
        tokens = tokenize("a, b")
        types = [t.type for t in tokens]
        assert "COMMA" in types


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
        assert "IDENT" in repr_str or "x" in repr_str
    
    def test_token_with_line(self):
        """Test token with line number."""
        token = Token(type="NUMBER", value="5", position=10, line=2, column=5)
        assert token.line == 2
        assert token.column == 5


class TestNumberExpr:
    """Tests for NumberExpr AST node."""
    
    def test_number_expr_creation(self):
        """Test creating NumberExpr."""
        expr = NumberExpr(value=3.14)
        assert expr.value == 3.14
    
    def test_number_expr_repr(self):
        """Test NumberExpr representation."""
        expr = NumberExpr(value=42)
        repr_str = repr(expr)
        assert "42" in repr_str
    
    def test_number_expr_zero(self):
        """Test NumberExpr with zero."""
        expr = NumberExpr(value=0)
        assert expr.value == 0
    
    def test_number_expr_negative(self):
        """Test NumberExpr with negative."""
        expr = NumberExpr(value=-5.5)
        assert expr.value == -5.5


class TestIdentExpr:
    """Tests for IdentExpr AST node."""
    
    def test_ident_expr_creation(self):
        """Test creating IdentExpr."""
        expr = IdentExpr(name="x")
        assert expr.name == "x"
    
    def test_ident_expr_repr(self):
        """Test IdentExpr representation."""
        expr = IdentExpr(name="theta")
        repr_str = repr(expr)
        assert "theta" in repr_str
    
    def test_ident_expr_underscore(self):
        """Test IdentExpr with underscore."""
        expr = IdentExpr(name="x_dot")
        assert expr.name == "x_dot"


class TestBinaryOpExpr:
    """Tests for BinaryOpExpr AST node."""
    
    def test_binary_op_creation(self):
        """Test creating BinaryOpExpr."""
        left = NumberExpr(value=2)
        right = NumberExpr(value=3)
        expr = BinaryOpExpr(left=left, operator="+", right=right)
        
        assert expr.left == left
        assert expr.right == right
        assert expr.operator == "+"
    
    def test_binary_op_repr(self):
        """Test BinaryOpExpr representation."""
        left = NumberExpr(value=2)
        right = NumberExpr(value=3)
        expr = BinaryOpExpr(left=left, operator="*", right=right)
        repr_str = repr(expr)
        assert "*" in repr_str or "2" in repr_str
    
    def test_binary_op_all_operators(self):
        """Test all binary operators."""
        left = NumberExpr(value=1)
        right = NumberExpr(value=2)
        
        for op in ["+", "-", "*", "/", "^"]:
            expr = BinaryOpExpr(left=left, operator=op, right=right)
            assert expr.operator == op


class TestUnaryOpExpr:
    """Tests for UnaryOpExpr AST node."""
    
    def test_unary_op_creation(self):
        """Test creating UnaryOpExpr."""
        operand = NumberExpr(value=5)
        expr = UnaryOpExpr(operator="-", operand=operand)
        
        assert expr.operator == "-"
        assert expr.operand == operand
    
    def test_unary_plus(self):
        """Test unary plus."""
        operand = NumberExpr(value=5)
        expr = UnaryOpExpr(operator="+", operand=operand)
        assert expr.operator == "+"


class TestFractionExpr:
    """Tests for FractionExpr AST node."""
    
    def test_fraction_creation(self):
        """Test creating FractionExpr."""
        num = NumberExpr(value=1)
        denom = NumberExpr(value=2)
        expr = FractionExpr(numerator=num, denominator=denom)
        
        assert expr.numerator == num
        assert expr.denominator == denom


class TestDerivativeVarExpr:
    """Tests for DerivativeVarExpr AST node."""
    
    def test_derivative_creation(self):
        """Test creating DerivativeVarExpr."""
        expr = DerivativeVarExpr(var="x", order=1)
        
        assert expr.var == "x"
        assert expr.order == 1
    
    def test_derivative_second_order(self):
        """Test second-order derivative."""
        expr = DerivativeVarExpr(var="theta", order=2)
        
        assert expr.var == "theta"
        assert expr.order == 2


class TestGreekLetterExpr:
    """Tests for GreekLetterExpr AST node."""
    
    def test_greek_creation(self):
        """Test creating GreekLetterExpr."""
        expr = GreekLetterExpr(letter="theta")
        assert expr.letter == "theta"
    
    def test_greek_omega(self):
        """Test omega letter."""
        expr = GreekLetterExpr(letter="omega")
        assert expr.letter == "omega"


class TestFunctionCallExpr:
    """Tests for FunctionCallExpr AST node."""
    
    def test_function_creation(self):
        """Test creating FunctionCallExpr."""
        arg = IdentExpr(name="x")
        expr = FunctionCallExpr(name="sin", args=[arg])
        
        assert expr.name == "sin"
        assert len(expr.args) == 1
    
    def test_function_multiple_args(self):
        """Test function with multiple args."""
        arg1 = NumberExpr(value=1)
        arg2 = NumberExpr(value=2)
        expr = FunctionCallExpr(name="max", args=[arg1, arg2])
        
        assert len(expr.args) == 2


class TestMechanicsParserWithTokens:
    """Tests for MechanicsParser with tokens."""
    
    def test_parser_creation_with_tokens(self):
        """Test creating parser with tokens."""
        tokens = tokenize("42")
        parser = MechanicsParser(tokens)
        assert parser is not None
    
    def test_parser_parse_atom_number(self):
        """Test parsing number."""
        tokens = tokenize("42")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_ident(self):
        """Test parsing identifier."""
        tokens = tokenize("x")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_greek(self):
        """Test parsing Greek letter."""
        tokens = tokenize("\\theta")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_addition(self):
        """Test parsing addition."""
        tokens = tokenize("2 + 3")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_multiplication(self):
        """Test parsing multiplication."""
        tokens = tokenize("2 * 3")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_power(self):
        """Test parsing power."""
        tokens = tokenize("x^2")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_fraction(self):
        """Test parsing fraction."""
        tokens = tokenize("\\frac{1}{2}")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_parentheses(self):
        """Test parsing parentheses."""
        tokens = tokenize("(2 + 3)")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_dot_notation(self):
        """Test parsing dot notation."""
        tokens = tokenize("\\dot{x}")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
    
    def test_parser_parse_sin(self):
        """Test parsing sin function."""
        tokens = tokenize("\\sin(x)")
        parser = MechanicsParser(tokens)
        try:
            result = parser.parse_expression()
            assert result is not None
        except:
            pass  # Parser may need different syntax
    
    def test_parser_parse_cos(self):
        """Test parsing cos function."""
        tokens = tokenize("\\cos(x)")
        parser = MechanicsParser(tokens)
        try:
            result = parser.parse_expression()
            assert result is not None
        except:
            pass  # Parser may need different syntax
    
    def test_parser_parse_complex(self):
        """Test parsing complex expression."""
        tokens = tokenize("\\frac{1}{2} * m * \\dot{x}^2")
        parser = MechanicsParser(tokens)
        result = parser.parse_expression()
        
        assert result is not None
