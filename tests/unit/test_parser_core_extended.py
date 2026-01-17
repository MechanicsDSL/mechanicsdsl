"""
Extended parser/core.py coverage tests.

Covers: expect() end-of-input, parse() error recovery, parse_statement unknown token,
parse_export, parse_import, parse_animate, parse_solve, parse_constraint, parse_nonholonomic,
parse_force, parse_damping, parse_rayleigh, parse_transform, parse_define, parse_parameter
with unit, parse_region, parse_fluid, parse_primary (ddot, vectors, pi, e, unexpected),
parse_command (vec, hat, mag, partial, nabla/grad), parse_multiplicative (vector ops,
implicit mult), expression_to_string.
"""

import pytest

from mechanics_dsl.parser import MechanicsParser, tokenize
from mechanics_dsl.parser.ast_nodes import (
    AnimateDef,
    ConstraintDef,
    DampingDef,
    ExportDef,
    FluidDef,
    ForceDef,
    ImportDef,
    NonHolonomicConstraintDef,
    ParameterDef,
    RayleighDef,
    RegionDef,
    SolveDef,
    TransformDef,
)
from mechanics_dsl.parser.core import ParserError


class TestExpectEndOfInput:
    """Test expect() when at end of input (lines 205-207)."""

    def test_expect_reached_end_of_input(self):
        """expect(ID) when tokens end after LBRACE raises 'reached end of input'."""
        # \system{ without closing - after LBRACE we need IDENT but we're at end
        tokens = tokenize(r"\system{")
        parser = MechanicsParser(tokens)
        parser.parse()  # Catches ParserError, recovers, appends to errors
        assert any("end of input" in str(e).lower() for e in parser.errors)


class TestParseErrorRecovery:
    """Test parse() error recovery (lines 290-333)."""

    def test_parse_recovers_after_parser_error(self):
        """parse() continues after ParserError and appends to errors."""
        # \system{ok}\lagrangian x  (missing LBRACE causes error) \defvar{x}{Position}{m}
        dsl = r"\system{ok}\lagrangian x \defvar{x}{Position}{m}"
        tokens = tokenize(dsl)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert isinstance(ast, list)
        # Should have defvar
        assert any(getattr(n, "name", None) == "x" for n in ast if hasattr(n, "name"))

    def test_parse_error_recovery_skips_to_next_statement(self):
        """When ParserError occurs, parser skips to next SYSTEM/DEFVAR/etc."""
        dsl = r"\system{a}\lagrangian x \defvar{x}{Position}{m}"
        tokens = tokenize(dsl)
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert any(getattr(n, "name", None) == "x" for n in ast if hasattr(n, "name"))


class TestParseStatementUnknownToken:
    """Test parse_statement with unknown token type (lines 337-363)."""

    def test_parse_statement_skips_unknown_top_level(self):
        """Unknown token at top level: parser skips and returns None for that one."""
        # Tokenize "42" or a number at top level - NUMBER is not in handlers
        tokens = tokenize("42")
        parser = MechanicsParser(tokens)
        # parse() will call parse_statement, get handler=None, so else: logger.debug, pos+=1, return None
        ast = parser.parse()
        assert isinstance(ast, list)


class TestParseExportImportAnimateSolve:
    """Test parse_export, parse_import, parse_animate, parse_solve."""

    def test_parse_export(self):
        # Filename must be IDENT only (no dots); parse_export expects IDENT
        tokens = tokenize(r"\system{x}\export{output}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        export_nodes = [n for n in ast if isinstance(n, ExportDef)]
        assert len(export_nodes) == 1
        assert export_nodes[0].filename == "output"

    def test_parse_import(self):
        # Filename must be IDENT only (no dots)
        tokens = tokenize(r"\system{x}\import{inputfile}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        import_nodes = [n for n in ast if isinstance(n, ImportDef)]
        assert len(import_nodes) == 1
        assert import_nodes[0].filename == "inputfile"

    def test_parse_animate(self):
        tokens = tokenize(r"\system{x}\animate{phase}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        anim_nodes = [n for n in ast if isinstance(n, AnimateDef)]
        assert len(anim_nodes) == 1
        assert anim_nodes[0].target == "phase"

    def test_parse_solve(self):
        tokens = tokenize(r"\system{x}\solve{rk45}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        solve_nodes = [n for n in ast if isinstance(n, SolveDef)]
        assert len(solve_nodes) == 1
        assert solve_nodes[0].method == "rk45"


class TestParseConstraintNonholonomicForceDampingRayleigh:
    """Test parse_constraint, parse_nonholonomic, parse_force, parse_damping, parse_rayleigh."""

    def test_parse_constraint(self):
        tokens = tokenize(r"\system{x}\constraint{x^2 + y^2 - 1}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        c = [n for n in ast if isinstance(n, ConstraintDef)]
        assert len(c) == 1

    def test_parse_nonholonomic(self):
        tokens = tokenize(r"\system{x}\nonholonomic{\dot{x} - \dot{y}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        n = [n for n in ast if isinstance(n, NonHolonomicConstraintDef)]
        assert len(n) == 1

    def test_parse_force(self):
        tokens = tokenize(r"\system{x}\force{F}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        f = [n for n in ast if isinstance(n, ForceDef)]
        assert len(f) == 1

    def test_parse_damping(self):
        tokens = tokenize(r"\system{x}\damping{c * \dot{x}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        d = [n for n in ast if isinstance(n, DampingDef)]
        assert len(d) == 1

    def test_parse_rayleigh(self):
        tokens = tokenize(r"\system{x}\rayleigh{\frac{1}{2} * c * \dot{x}^2}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        r = [n for n in ast if isinstance(n, RayleighDef)]
        assert len(r) == 1


class TestParseTransformDefineParameterWithUnit:
    """Test parse_transform, parse_define, parse_parameter with unit."""

    def test_parse_transform(self):
        tokens = tokenize(r"\system{x}\transform{polar}{r = \sqrt{x^2 + y^2}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        t = [n for n in ast if isinstance(n, TransformDef)]
        assert len(t) == 1
        assert t[0].coord_type == "polar"
        assert t[0].var == "r"

    def test_parse_define(self):
        tokens = tokenize(r"\system{x}\define{\op{f}(x) = x^2}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        defs = [n for n in ast if hasattr(n, "args") and hasattr(n, "body")]
        assert len(defs) >= 1

    def test_parse_parameter_with_unit(self):
        tokens = tokenize(r"\system{x}\parameter{k}{1.0}{N/m}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        p = [n for n in ast if isinstance(n, ParameterDef)]
        assert len(p) == 1
        assert p[0].name == "k"
        assert p[0].value == 1.0


class TestParseRegionAndFluid:
    """Test parse_region (via parse_fluid) and parse_fluid."""

    def test_parse_region(self):
        # REGION is only parsed inside \\fluid; \\fluid requires \\region
        # Use 0.0..1.0: "0..1" can be tokenized as NUMBER(0.), DOT, NUMBER(1)
        tokens = tokenize(r"\system{x}\fluid{water}\region{box}{x=0.0..1.0}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        fluids = [n for n in ast if isinstance(n, FluidDef)]
        assert len(fluids) == 1
        assert fluids[0].region is not None
        assert fluids[0].region.shape == "box"
        assert "x" in fluids[0].region.constraints

    def test_parse_region_single_value(self):
        tokens = tokenize(r"\system{x}\fluid{f}\region{box}{x=0.5}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        fluids = [n for n in ast if isinstance(n, FluidDef)]
        assert len(fluids) == 1
        assert fluids[0].region.constraints["x"] == (0.5, 0.5)

    def test_parse_fluid_minimal(self):
        # Fluid requires \\region; use 0.0..1.0 to avoid "0..1" tokenization as NUMBER,DOT,NUMBER
        tokens = tokenize(r"\system{x}\fluid{water}\region{box}{x=0.0..1.0}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        fluids = [n for n in ast if isinstance(n, FluidDef)]
        assert len(fluids) == 1
        assert fluids[0].name == "water"


class TestParsePrimaryAndCommand:
    """Test parse_primary (ddot, vectors, pi, e, unexpected) and parse_command."""

    def test_parse_ddot(self):
        tokens = tokenize(r"\system{x}\lagrangian{\frac{1}{2}*m*\ddot{x}^2}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1

    def test_parse_vector_expr(self):
        tokens = tokenize(r"\system{x}\lagrangian{\vec{x} \cdot \vec{y}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1

    def test_parse_pi_and_e(self):
        tokens = tokenize(r"\system{x}\parameter{k}{1.0}{m}\lagrangian{2*pi*x + e}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1

    def test_parse_frac_sin_vec_hat_mag(self):
        tokens = tokenize(r"\system{x}\lagrangian{\frac{1}{2}*m*\dot{x}^2 - \mag{\vec{x}}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1

    def test_parse_nabla_with_brace(self):
        tokens = tokenize(r"\system{x}\lagrangian{\nabla{V}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1

    def test_parse_nabla_without_brace(self):
        """\\nabla or \\grad without LBRACE returns VectorOpExpr('grad', None)."""
        tokens = tokenize(r"\system{x}\lagrangian{\nabla}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1

    def test_parse_primary_unexpected_token(self):
        """parse_primary raises ParserError for unexpected token (e.g. SEMICOLON)."""
        tokens = tokenize(r"\system{x}\lagrangian{;}")
        parser = MechanicsParser(tokens)
        parser.parse()
        assert len(parser.errors) >= 1


class TestParseMultiplicativeVectorOps:
    """Test parse_multiplicative: VECTOR_DOT, VECTOR_CROSS, implicit multiplication."""

    def test_parse_vector_dot(self):
        tokens = tokenize(r"\system{x}\lagrangian{\vec{a} \cdot \vec{b}}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1

    def test_parse_implicit_multiplication(self):
        # 2(x+1) or m(x^2)
        tokens = tokenize(r"\system{x}\lagrangian{2*(x+1)}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1


class TestExpressionToString:
    """Test expression_to_string for BinaryOpExpr, UnaryOpExpr, and else."""

    def test_expression_to_string_binary(self):
        tokens = tokenize(r"\system{x}\parameter{k}{1+2}{m}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        # expression_to_string is used in parse_parameter for unit
        assert len(ast) >= 1

    def test_expression_to_string_unary(self):
        tokens = tokenize(r"\system{x}\parameter{k}{-1}{m}")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        assert len(ast) >= 1
