"""
MechanicsParser - Recursive Descent Parser for MechanicsDSL.

This module contains the main parser class that converts a token stream
into an Abstract Syntax Tree (AST) for compilation.

The parser uses recursive descent with operator precedence for expressions:
- Additive: + -
- Multiplicative: * / dot cross
- Power: ^ (right associative)
- Unary: + -
- Postfix: function calls, subscripts
- Primary: numbers, identifiers, parentheses, vectors, commands

Example:
    >>> from mechanics_dsl.parser import tokenize, MechanicsParser
    >>> tokens = tokenize(r'''
    ...     \\system{simple_pendulum}
    ...     \\defvar{theta}{Angle}{rad}
    ...     \\lagrangian{\\frac{1}{2}*m*l^2*\\dot{theta}^2 + m*g*l*cos(theta)}
    ... ''')
    >>> parser = MechanicsParser(tokens)
    >>> ast = parser.parse()
    >>> print(len(ast))  # Number of AST nodes
    3
"""

from typing import List, Optional

import numpy as np

from ..utils import config, logger, profile_function
from .ast_nodes import (  # Base classes; Basic expressions; Operations; Vectors; Calculus; Functions; Statements; SPH
    AnimateDef,
    ASTNode,
    BinaryOpExpr,
    BoundaryDef,
    ConstraintDef,
    DampingDef,
    DefineDef,
    DerivativeExpr,
    DerivativeVarExpr,
    ExportDef,
    Expression,
    FluidDef,
    ForceDef,
    FractionExpr,
    FunctionCallExpr,
    GreekLetterExpr,
    HamiltonianDef,
    IdentExpr,
    ImportDef,
    InitialCondition,
    IntegralExpr,
    LagrangianDef,
    NonHolonomicConstraintDef,
    NumberExpr,
    ParameterDef,
    RayleighDef,
    RegionDef,
    SolveDef,
    SystemDef,
    TransformDef,
    UnaryOpExpr,
    VarDef,
    VectorExpr,
    VectorOpExpr,
)
from .tokens import Token

# ============================================================================
# PARSER ERROR
# ============================================================================


class ParserError(Exception):
    """
    Custom exception for parser errors with position tracking.

    Provides detailed error messages including line and column information
    when available from the token.

    Attributes:
        message: Error description.
        token: Optional token where the error occurred.

    Example:
        >>> try:
        ...     parser.expect("RBRACE")
        ... except ParserError as e:
        ...     print(e)
        Expected RBRACE but got EOF at line 5, column 10
    """

    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format error message with position information."""
        if self.token:
            return f"{self.message} at line {self.token.line}, column {self.token.column}"
        return self.message


# ============================================================================
# MAIN PARSER CLASS
# ============================================================================


class MechanicsParser:
    """
    Recursive descent parser for MechanicsDSL.

    Converts a list of tokens into an Abstract Syntax Tree (AST)
    that can be compiled into equations of motion.

    Attributes:
        tokens: List of Token objects to parse.
        pos: Current position in token stream.
        current_system: Name of current system being parsed.
        errors: List of parsing errors encountered.
        max_errors: Maximum errors before giving up (from config).

    Methods:
        parse(): Parse complete token stream into AST.
        peek(offset): Look ahead at tokens.
        match(*types): Match and consume token if type matches.
        expect(type): Require specific token type.

    Example:
        >>> tokens = tokenize(r"\\lagrangian{T - V}")
        >>> parser = MechanicsParser(tokens)
        >>> ast = parser.parse()
        >>> print(ast[0])
        Lagrangian(...)
    """

    def __init__(self, tokens: List[Token]):
        """
        Initialize parser with token stream.

        Args:
            tokens: List of Token objects from tokenize().
        """
        self.tokens = tokens
        self.pos = 0
        self.current_system = None
        self.errors: List[str] = []
        self.max_errors = config.max_parser_errors

    def peek(self, offset: int = 0) -> Optional[Token]:
        """
        Look ahead at token without consuming it.

        Args:
            offset: Number of tokens to look ahead (default 0).

        Returns:
            Token at position pos+offset, or None if past end.
        """
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def match(self, *expected_types: str) -> Optional[Token]:
        """
        Match and consume token if type matches.

        Args:
            *expected_types: One or more acceptable token types.

        Returns:
            Matched token if successful, None otherwise.
        """
        token = self.peek()
        if token and token.type in expected_types:
            self.pos += 1
            return token
        return None

    def expect(self, expected_type: str) -> Token:
        """
        Expect a specific token type, raise error if not found.

        Args:
            expected_type: Required token type.

        Returns:
            The matched token.

        Raises:
            ParserError: If token doesn't match expected type.
        """
        token = self.match(expected_type)
        if not token:
            current = self.peek()
            if current:
                error_msg = f"Expected {expected_type} but got {current.type} '{current.value}'"
                self.errors.append(error_msg)
                raise ParserError(error_msg, current)
            else:
                error_msg = f"Expected {expected_type} but reached end of input"
                self.errors.append(error_msg)
                raise ParserError(error_msg)
        return token

    @profile_function
    def parse(self) -> List[ASTNode]:
        """
        Parse the complete DSL with comprehensive error recovery.

        Returns:
            List of ASTNode objects representing the parsed program.

        Note:
            The parser attempts error recovery to parse as much as
            possible even when errors are encountered.
        """
        nodes = []
        error_count = 0

        while self.pos < len(self.tokens) and error_count < self.max_errors:
            try:
                node = self.parse_statement()
                if node:
                    nodes.append(node)
                    logger.debug(f"Parsed node: {type(node).__name__}")
            except ParserError as e:
                self.errors.append(str(e))
                error_count += 1
                logger.error(f"Parser error: {e}")

                # Error recovery: skip to next statement
                while self.pos < len(self.tokens):
                    token = self.peek()
                    if token and token.type in [
                        "SYSTEM",
                        "DEFVAR",
                        "DEFINE",
                        "LAGRANGIAN",
                        "HAMILTONIAN",
                        "CONSTRAINT",
                        "INITIAL",
                        "SOLVE",
                    ]:
                        break
                    self.pos += 1

        if self.errors:
            logger.warning(f"Parser encountered {len(self.errors)} errors")

        logger.info(f"Successfully parsed {len(nodes)} AST nodes")
        return nodes

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a top-level statement."""
        token = self.peek()
        if not token:
            return None

        handlers = {
            "SYSTEM": self.parse_system,
            "DEFVAR": self.parse_defvar,
            "PARAMETER": self.parse_parameter,
            "DEFINE": self.parse_define,
            "LAGRANGIAN": self.parse_lagrangian,
            "HAMILTONIAN": self.parse_hamiltonian,
            "TRANSFORM": self.parse_transform,
            "CONSTRAINT": self.parse_constraint,
            "NONHOLONOMIC": self.parse_nonholonomic,
            "FORCE": self.parse_force,
            "DAMPING": self.parse_damping,
            "RAYLEIGH": self.parse_rayleigh,
            "INITIAL": self.parse_initial,
            "SOLVE": self.parse_solve,
            "ANIMATE": self.parse_animate,
            "EXPORT": self.parse_export,
            "IMPORT": self.parse_import,
            "FLUID": self.parse_fluid,
            "BOUNDARY": self.parse_boundary,
        }

        handler = handlers.get(token.type)
        if handler:
            return handler()
        else:
            logger.debug(f"Skipping unknown token: {token}")
            self.pos += 1
            return None

    # ========================================================================
    # STATEMENT PARSERS
    # ========================================================================

    def parse_region(self) -> RegionDef:
        """Parse \\region{shape}{constraints}."""
        self.expect("REGION")
        self.expect("LBRACE")
        shape = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")

        constraints = {}

        while True:
            var = self.expect("IDENT").value
            self.expect("EQUALS")

            # Parse Start
            start_sign = -1.0 if self.match("MINUS") else 1.0
            start_token = self.expect("NUMBER")
            start = start_sign * float(start_token.value)

            # Check for Range ".."
            if self.match("RANGE_OP"):
                # Parse End
                end_sign = -1.0 if self.match("MINUS") else 1.0
                end_token = self.expect("NUMBER")
                end = end_sign * float(end_token.value)
            else:
                # Single value (e.g. x=0.5) -> range is [0.5, 0.5]
                end = start

            constraints[var] = (start, end)

            if not self.match("COMMA"):
                break

        self.expect("RBRACE")
        return RegionDef(shape, constraints)

    def parse_fluid(self) -> FluidDef:
        """Parse \\fluid{name} with properties."""
        self.expect("FLUID")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")

        # Defaults
        region = None
        mass = 1.0
        eos = "tait"

        # Parse fluid properties
        while self.peek() and self.peek().type in ["REGION", "PARTICLE_MASS", "EOS"]:
            if self.peek().type == "REGION":
                region = self.parse_region()
            elif self.match("PARTICLE_MASS"):
                self.expect("LBRACE")
                mass = float(self.expect("NUMBER").value)
                self.expect("RBRACE")
            elif self.match("EOS"):
                self.expect("LBRACE")
                eos = self.expect("IDENT").value
                self.expect("RBRACE")

        if not region:
            raise ParserError("Fluid must have a region definition")

        return FluidDef(name, region, mass, eos)

    def parse_boundary(self) -> BoundaryDef:
        """Parse \\boundary{name} \\region{...}."""
        self.expect("BOUNDARY")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        region = self.parse_region()
        return BoundaryDef(name, region)

    def parse_system(self) -> SystemDef:
        """Parse \\system{name}."""
        self.expect("SYSTEM")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.current_system = name
        return SystemDef(name)

    def parse_defvar(self) -> VarDef:
        """Parse \\defvar{name}{type}{unit}."""
        self.expect("DEFVAR")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")

        vartype_parts = []
        while True:
            tok = self.peek()
            if not tok or tok.type == "RBRACE":
                break
            self.pos += 1
            vartype_parts.append(tok.value)
        vartype = " ".join(vartype_parts).strip()
        self.expect("RBRACE")

        self.expect("LBRACE")
        unit_expr = self.parse_expression()
        unit = self.expression_to_string(unit_expr)
        self.expect("RBRACE")

        is_vector = vartype in [
            "Vector",
            "Vector3",
            "Position",
            "Velocity",
            "Force",
            "Momentum",
            "Acceleration",
        ]

        return VarDef(name, vartype, unit, is_vector)

    def parse_parameter(self) -> ParameterDef:
        """Parse \\parameter{name}{value}{unit}."""
        self.expect("PARAMETER")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        value = float(self.expect("NUMBER").value)
        self.expect("RBRACE")
        self.expect("LBRACE")
        unit_expr = self.parse_expression()
        unit = self.expression_to_string(unit_expr)
        self.expect("RBRACE")
        return ParameterDef(name, value, unit)

    def parse_define(self) -> DefineDef:
        """Parse \\define{\\op{name}(args) = expression}."""
        self.expect("DEFINE")
        self.expect("LBRACE")

        self.expect("COMMAND")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")

        self.expect("LPAREN")
        args = []
        if self.peek() and self.peek().type == "IDENT":
            args.append(self.expect("IDENT").value)
            while self.match("COMMA"):
                args.append(self.expect("IDENT").value)
        self.expect("RPAREN")

        self.expect("EQUALS")
        body = self.parse_expression()
        self.expect("RBRACE")

        return DefineDef(name, args, body)

    def parse_lagrangian(self) -> LagrangianDef:
        """Parse \\lagrangian{expression}."""
        self.expect("LAGRANGIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return LagrangianDef(expr)

    def parse_hamiltonian(self) -> HamiltonianDef:
        """Parse \\hamiltonian{expression}."""
        self.expect("HAMILTONIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return HamiltonianDef(expr)

    def parse_transform(self) -> TransformDef:
        """Parse \\transform{type}{var = expr}."""
        self.expect("TRANSFORM")
        self.expect("LBRACE")
        coord_type = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return TransformDef(coord_type, var, expr)

    def parse_constraint(self) -> ConstraintDef:
        """Parse \\constraint{expression}."""
        self.expect("CONSTRAINT")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return ConstraintDef(expr)

    def parse_nonholonomic(self) -> NonHolonomicConstraintDef:
        """Parse \\nonholonomic{expression}."""
        self.expect("NONHOLONOMIC")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return NonHolonomicConstraintDef(expr)

    def parse_force(self) -> ForceDef:
        """Parse \\force{expression}."""
        self.expect("FORCE")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return ForceDef(expr)

    def parse_damping(self) -> DampingDef:
        """Parse \\damping{expression}."""
        self.expect("DAMPING")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return DampingDef(expr)

    def parse_rayleigh(self):
        """
        Parse \\rayleigh{expression}.

        The Rayleigh dissipation function F represents velocity-dependent
        dissipative forces. The generalized dissipative forces are:
        Q_i = -∂F/∂q̇_i

        For linear damping: F = ½ Σ bᵢⱼ q̇ᵢ q̇ⱼ

        Example:
            \\rayleigh{\\frac{1}{2} * b * \\dot{x}^2}
        """
        from .ast_nodes import RayleighDef

        self.expect("RAYLEIGH")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return RayleighDef(expr)

    def parse_initial(self) -> InitialCondition:
        """Parse \\initial{var1=val1, var2=val2, ...}."""
        self.expect("INITIAL")
        self.expect("LBRACE")

        conditions = {}
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        val = float(self.expect("NUMBER").value)
        conditions[var] = val

        while self.match("COMMA"):
            var = self.expect("IDENT").value
            self.expect("EQUALS")
            val = float(self.expect("NUMBER").value)
            conditions[var] = val

        self.expect("RBRACE")
        return InitialCondition(conditions)

    def parse_solve(self) -> SolveDef:
        """Parse \\solve{method}."""
        self.expect("SOLVE")
        self.expect("LBRACE")
        method = self.expect("IDENT").value
        self.expect("RBRACE")
        return SolveDef(method)

    def parse_animate(self) -> AnimateDef:
        """Parse \\animate{target}."""
        self.expect("ANIMATE")
        self.expect("LBRACE")
        target = self.expect("IDENT").value
        self.expect("RBRACE")
        return AnimateDef(target)

    def parse_export(self) -> ExportDef:
        """Parse \\export{filename}."""
        self.expect("EXPORT")
        self.expect("LBRACE")
        filename = self.expect("IDENT").value
        self.expect("RBRACE")
        return ExportDef(filename)

    def parse_import(self) -> ImportDef:
        """Parse \\import{filename}."""
        self.expect("IMPORT")
        self.expect("LBRACE")
        filename = self.expect("IDENT").value
        self.expect("RBRACE")
        return ImportDef(filename)

    # ========================================================================
    # EXPRESSION PARSERS (Operator Precedence)
    # ========================================================================

    def parse_expression(self) -> Expression:
        """Parse expressions with full operator precedence."""
        return self.parse_additive()

    def parse_additive(self) -> Expression:
        """Addition and subtraction (lowest precedence)."""
        left = self.parse_multiplicative()

        while True:
            if self.match("PLUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "+", right)
            elif self.match("MINUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "-", right)
            else:
                break

        return left

    def parse_multiplicative(self) -> Expression:
        """Multiplication, division, and vector operations."""
        left = self.parse_power()

        while True:
            if self.match("MULTIPLY"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "*", right)
            elif self.match("DIVIDE"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "/", right)
            elif self.match("VECTOR_DOT"):
                right = self.parse_power()
                left = VectorOpExpr("dot", left, right)
            elif self.match("VECTOR_CROSS"):
                right = self.parse_power()
                left = VectorOpExpr("cross", left, right)
            else:
                # Improved implicit multiplication - only for safe cases
                next_token = self.peek()
                if (
                    next_token
                    and next_token.type == "LPAREN"
                    and isinstance(left, (NumberExpr, IdentExpr, GreekLetterExpr))
                    and not self.at_end_of_expression()
                ):
                    # Safe implicit multiplication: 2(x+y), m(v^2), etc.
                    right = self.parse_power()
                    left = BinaryOpExpr(left, "*", right)
                else:
                    break

        return left

    def parse_power(self) -> Expression:
        """Exponentiation (right associative)."""
        left = self.parse_unary()

        if self.match("POWER"):
            right = self.parse_power()
            return BinaryOpExpr(left, "^", right)

        return left

    def parse_unary(self) -> Expression:
        """Unary operators (+, -)."""
        if self.match("MINUS"):
            operand = self.parse_unary()
            return UnaryOpExpr("-", operand)
        elif self.match("PLUS"):
            return self.parse_unary()

        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Function calls, subscripts, etc."""
        expr = self.parse_primary()

        while True:
            if self.match("LPAREN"):
                # Function call
                args = []
                if self.peek() and self.peek().type != "RPAREN":
                    args.append(self.parse_expression())
                    while self.match("COMMA"):
                        args.append(self.parse_expression())
                self.expect("RPAREN")

                if isinstance(expr, IdentExpr):
                    expr = FunctionCallExpr(expr.name, args)
                elif isinstance(expr, GreekLetterExpr):
                    expr = FunctionCallExpr(expr.letter, args)
                else:
                    raise ParserError("Invalid function call syntax")
            else:
                break

        return expr

    def parse_primary(self) -> Expression:
        """Primary expressions: literals, identifiers, parentheses, vectors, commands."""

        # Numbers
        if self.match("NUMBER"):
            return NumberExpr(float(self.tokens[self.pos - 1].value))

        # Time derivatives: \dot{x} and \ddot{x}
        token = self.peek()
        if token and token.type == "DOT_NOTATION":
            self.pos += 1
            order = 2 if token.value == r"\ddot" else 1
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeVarExpr(var, order)

        # Identifiers
        if self.match("IDENT"):
            return IdentExpr(self.tokens[self.pos - 1].value)

        # Greek letters
        if self.match("GREEK_LETTER"):
            letter = self.tokens[self.pos - 1].value[1:]
            return GreekLetterExpr(letter)

        # Parentheses
        if self.match("LPAREN"):
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr

        # Vectors [x, y, z]
        if self.match("LBRACKET"):
            components = []
            components.append(self.parse_expression())
            while self.match("COMMA"):
                components.append(self.parse_expression())
            self.expect("RBRACKET")
            return VectorExpr(components)

        # Commands (LaTeX-style functions)
        token = self.peek()
        if token and token.type in {"COMMAND", "FRAC"}:
            self.pos += 1
            return self.parse_command(token.value)

        # Mathematical constants
        if token and token.value in ["pi", "e"]:
            self.pos += 1
            if token.value == "pi":
                return NumberExpr(np.pi)
            elif token.value == "e":
                return NumberExpr(np.e)

        current = self.peek()
        if current:
            raise ParserError(f"Unexpected token {current.type} '{current.value}'", current)
        else:
            raise ParserError("Unexpected end of input")

    def parse_command(self, cmd: str) -> Expression:
        """Parse LaTeX-style commands."""

        if cmd == r"\frac":
            self.expect("LBRACE")
            num = self.parse_expression()
            self.expect("RBRACE")
            self.expect("LBRACE")
            denom = self.parse_expression()
            self.expect("RBRACE")
            return FractionExpr(num, denom)

        elif cmd == r"\vec":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("vec", expr)

        elif cmd == r"\hat":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("unit", expr)

        elif cmd in [r"\mag", r"\norm"]:
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("magnitude", expr)

        elif cmd == r"\partial":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeExpr(expr, var, 1, True)

        elif cmd in [
            r"\sin",
            r"\cos",
            r"\tan",
            r"\exp",
            r"\log",
            r"\ln",
            r"\sqrt",
            r"\sinh",
            r"\cosh",
            r"\tanh",
            r"\arcsin",
            r"\arccos",
            r"\arctan",
        ]:
            func_name = cmd[1:]
            self.expect("LBRACE")
            arg = self.parse_expression()
            self.expect("RBRACE")
            return FunctionCallExpr(func_name, [arg])

        elif cmd in [r"\nabla", r"\grad"]:
            if self.peek() and self.peek().type == "LBRACE":
                self.expect("LBRACE")
                expr = self.parse_expression()
                self.expect("RBRACE")
                return VectorOpExpr("grad", expr)
            return VectorOpExpr("grad", None)

        else:
            # Unknown command - treat as identifier
            return IdentExpr(cmd[1:])

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def at_end_of_expression(self) -> bool:
        """Check if we're at the end of an expression."""
        token = self.peek()
        return not token or token.type in [
            "RBRACE",
            "RPAREN",
            "RBRACKET",
            "COMMA",
            "SEMICOLON",
            "EQUALS",
        ]

    def expression_to_string(self, expr: Expression) -> str:
        """Convert expression back to string for unit parsing."""
        if isinstance(expr, NumberExpr):
            return str(expr.value)
        elif isinstance(expr, IdentExpr):
            return expr.name
        elif isinstance(expr, BinaryOpExpr):
            left = self.expression_to_string(expr.left)
            right = self.expression_to_string(expr.right)
            return f"({left} {expr.operator} {right})"
        elif isinstance(expr, UnaryOpExpr):
            operand = self.expression_to_string(expr.operand)
            return f"{expr.operator}{operand}"
        else:
            return str(expr)


__all__ = ["MechanicsParser", "ParserError"]
