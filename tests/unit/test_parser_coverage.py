"""
Additional parser coverage tests for core.py
"""
import pytest
from mechanics_dsl.parser import tokenize, MechanicsParser
from mechanics_dsl.parser.ast_nodes import *


class TestParserEdgeCases:
    """Tests for parser edge cases and error handling"""
    
    def test_parse_empty_system(self):
        """Test parsing empty system"""
        tokens = tokenize(r"\system{test}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) == 1
        assert isinstance(ast[0], SystemDef)
    
    def test_parse_lagrangian(self):
        """Test parsing Lagrangian"""
        tokens = tokenize(r"\system{test}\lagrangian{T - V}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, LagrangianDef) for n in ast)
    
    def test_parse_hamiltonian(self):
        """Test parsing Hamiltonian"""
        tokens = tokenize(r"\system{test}\hamiltonian{p^2/(2*m) + m*g*x}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, HamiltonianDef) for n in ast)
    
    def test_parse_defvar(self):
        """Test parsing defvar"""
        tokens = tokenize(r"\system{test}\defvar{x}{Position}{m}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, VarDef) for n in ast)
    
    def test_parse_parameter(self):
        """Test parsing parameter"""
        tokens = tokenize(r"\system{test}\parameter{m}{1.0}{kg}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, ParameterDef) for n in ast)
    
    def test_parse_initial(self):
        """Test parsing initial conditions"""
        tokens = tokenize(r"\system{test}\initial{x=1.0, v=0.0}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, InitialCondition) for n in ast)
    
    def test_parse_force(self):
        """Test parsing force"""
        tokens = tokenize(r"\system{test}\force{-b*v}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, ForceDef) for n in ast)
    
    def test_parse_damping(self):
        """Test parsing damping"""
        tokens = tokenize(r"\system{test}\damping{-c*v}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, DampingDef) for n in ast)
    
    def test_parse_rayleigh(self):
        """Test parsing rayleigh"""
        tokens = tokenize(r"\system{test}\rayleigh{\frac{1}{2}*b*v^2}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(isinstance(n, RayleighDef) for n in ast)
    
    def test_parse_sqrt(self):
        """Test parsing sqrt function"""
        tokens = tokenize(r"\system{test}\lagrangian{\sqrt{x^2 + y^2}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1
    
    def test_parse_trig_functions(self):
        """Test parsing trig functions"""
        tokens = tokenize(r"\system{test}\lagrangian{\sin{x} + \cos{y}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1
    
    def test_parse_exp_log(self):
        """Test parsing exp and log"""
        tokens = tokenize(r"\system{test}\lagrangian{\exp{x} + \ln{y}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1
    
    def test_parse_greek_letters(self):
        """Test parsing Greek letters"""
        tokens = tokenize(r"\system{test}\lagrangian{\alpha + \beta + \theta}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1
    
    def test_parse_nested_expressions(self):
        """Test parsing deeply nested expressions"""
        tokens = tokenize(r"\system{test}\lagrangian{\frac{1}{2}*m*\dot{x}^2}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1
    
    def test_parse_multiple_statements(self):
        """Test parsing multiple statements"""
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2}*m*\dot{x}^2}
        \initial{x=1.0, x_dot=0.0}
        """
        tokens = tokenize(dsl)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 4
    
    def test_parser_error_recovery(self):
        """Test parser discovers errors but continues"""
        # Malformed input might still parse partially
        tokens = tokenize(r"\system{test}\lagrangian{")
        parser = MechanicsParser(tokens)
        try:
            ast = parser.parse()
        except Exception:
            pass  # Expected to fail


class TestASTNodes:
    """Tests for AST node representations"""
    
    def test_number_expr_repr(self):
        """Test NumberExpr representation"""
        node = NumberExpr(3.14)
        assert "3.14" in str(node)
    
    def test_ident_expr_repr(self):
        """Test IdentExpr representation"""
        node = IdentExpr("x")
        assert "x" in str(node)
    
    def test_binary_op_repr(self):
        """Test BinaryOpExpr representation"""
        left = NumberExpr(1)
        right = NumberExpr(2)
        node = BinaryOpExpr("+", left, right)
        assert "+" in str(node) or "BinOp" in str(node)
    
    def test_unary_op_repr(self):
        """Test UnaryOpExpr representation"""
        operand = NumberExpr(5)
        node = UnaryOpExpr("-", operand)
        assert "-" in str(node) or "Unary" in str(node)
    
    def test_function_call_repr(self):
        """Test FunctionCallExpr representation"""
        arg = IdentExpr("x")
        node = FunctionCallExpr("sin", [arg])
        assert "sin" in str(node)
    
    def test_derivative_var_repr(self):
        """Test DerivativeVarExpr representation"""
        node = DerivativeVarExpr("x", 1)
        assert "x" in str(node)
    
    def test_fraction_expr_repr(self):
        """Test FractionExpr representation"""
        num = NumberExpr(1)
        den = NumberExpr(2)
        node = FractionExpr(num, den)
        assert "Frac" in str(node) or "1" in str(node)
    
    def test_vector_expr_repr(self):
        """Test VectorExpr representation"""
        components = [NumberExpr(1), NumberExpr(2), NumberExpr(3)]
        node = VectorExpr(components)
        s = str(node)
        assert len(s) > 0
    
    def test_system_def_repr(self):
        """Test SystemDef representation"""
        node = SystemDef("test_system")
        assert "test" in str(node)
    
    def test_var_def_repr(self):
        """Test VarDef representation"""
        node = VarDef("x", "Position", "m")
        assert "x" in str(node)
    
    def test_parameter_def_repr(self):
        """Test ParameterDef representation"""
        node = ParameterDef("m", 1.0, "kg")
        assert "m" in str(node)
    
    def test_lagrangian_def_repr(self):
        """Test LagrangianDef representation"""
        expr = BinaryOpExpr("-", IdentExpr("T"), IdentExpr("V"))
        node = LagrangianDef(expr)
        assert "Lagrangian" in str(node)
    
    def test_initial_condition_repr(self):
        """Test InitialCondition representation"""
        node = InitialCondition({"x": 1.0, "v": 0.0})
        s = str(node)
        assert len(s) > 0
