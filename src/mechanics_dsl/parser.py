"""
Parser for MechanicsDSL with tokenization and AST generation
"""
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field

from .utils import logger, config, profile_function

# ============================================================================
# TOKEN SYSTEM
# ============================================================================

TOKEN_TYPES = [
    # Physics specific commands (order matters!)
    ("DOT_NOTATION", r"\\ddot|\\dot"),
    ("SYSTEM", r"\\system"),
    ("DEFVAR", r"\\defvar"),
    ("DEFINE", r"\\define"),
    ("LAGRANGIAN", r"\\lagrangian"),
    ("HAMILTONIAN", r"\\hamiltonian"),
    ("TRANSFORM", r"\\transform"),
    ("CONSTRAINT", r"\\constraint"),
    ("NONHOLONOMIC", r"\\nonholonomic"),
    ("FORCE", r"\\force"),
    ("DAMPING", r"\\damping"),
    ("INITIAL", r"\\initial"),
    ("SOLVE", r"\\solve"),
    ("ANIMATE", r"\\animate"),
    ("PLOT", r"\\plot"),
    ("PARAMETER", r"\\parameter"),
    ("EXPORT", r"\\export"),
    ("IMPORT", r"\\import"),
    ("EULER_ANGLES", r"\\euler"),
    ("QUATERNION", r"\\quaternion"),
    
    # Vector operations
    ("VEC", r"\\vec"),
    ("HAT", r"\\hat"),
    ("MAGNITUDE", r"\\mag|\\norm"),
    
    # Advanced math operators
    ("VECTOR_DOT", r"\\cdot"),
    ("VECTOR_CROSS", r"\\times|\\cross"),
    ("GRADIENT", r"\\nabla|\\grad"),
    ("DIVERGENCE", r"\\div"),
    ("CURL", r"\\curl"),
    ("LAPLACIAN", r"\\laplacian|\\Delta"),
    
    # Calculus
    ("PARTIAL", r"\\partial"),
    ("INTEGRAL", r"\\int"),
    ("OINT", r"\\oint"),
    ("SUM", r"\\sum"),
    ("LIMIT", r"\\lim"),
    ("FRAC", r"\\frac"),
    
    # Greek letters (comprehensive)
    ("GREEK_LETTER", r"\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\varepsilon|\\zeta|\\eta|\\theta|\\vartheta|\\iota|\\kappa|\\lambda|\\mu|\\nu|\\xi|\\omicron|\\pi|\\varpi|\\rho|\\varrho|\\sigma|\\varsigma|\\tau|\\upsilon|\\phi|\\varphi|\\chi|\\psi|\\omega"),

    ("FLUID", r"\\fluid"),
    ("BOUNDARY", r"\\boundary"),
    ("REGION", r"\\region"),
    ("PARTICLE_MASS", r"\\particle_mass"),
    ("EOS", r"\\equation_of_state"),
    ("RANGE_OP", r"\.\."),
    
    # General commands
    ("COMMAND", r"\\[a-zA-Z_][a-zA-Z0-9_]*"),
    
    # Brackets and grouping
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    
    # Mathematical operators
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("POWER", r"\^"),
    ("EQUALS", r"="),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("COLON", r":"),
    ("DOT", r"\."),
    ("UNDERSCORE", r"_"),
    ("PIPE", r"\|"),
    
    # Basic tokens
    ("NUMBER", r"\d+\.?\d*([eE][+-]?\d+)?"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("WHITESPACE", r"\s+"),
    ("NEWLINE", r"\n"),
    ("COMMENT", r"%.*"),
]

token_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES)
token_pattern = re.compile(token_regex)

@dataclass
class Token:
    """Token with position tracking for better error messages"""
    type: str
    value: str
    position: int = 0
    line: int = 1
    column: int = 1

    def __repr__(self) -> str:
        return f"{self.type}:{self.value}@{self.line}:{self.column}"

def tokenize(source: str) -> List[Token]:
    """
    Tokenizer with position tracking and comprehensive error reporting
    
    Args:
        source: DSL source code
        
    Returns:
        List of tokens (excluding whitespace and comments)
    """
    tokens = []
    line = 1
    line_start = 0
    
    for match in token_pattern.finditer(source):
        kind = match.lastgroup
        value = match.group()
        position = match.start()
        
        # Update line tracking
        while line_start < position and '\n' in source[line_start:position]:
            newline_pos = source.find('\n', line_start)
            if newline_pos != -1 and newline_pos < position:
                line += 1
                line_start = newline_pos + 1
            else:
                break
                
        column = position - line_start + 1
        
        if kind not in ["WHITESPACE", "COMMENT"]:
            tokens.append(Token(kind, value, position, line, column))
    
    logger.debug(f"Tokenized {len(tokens)} tokens from {line} lines")
    return tokens

# ============================================================================
# AST SYSTEM
# ============================================================================

class ASTNode:
    """Base class for all AST nodes"""
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Expression(ASTNode):
    """Base class for all expressions"""
    pass

# Basic expressions
@dataclass
class NumberExpr(Expression):
    value: float
    def __repr__(self) -> str:
        return f"Num({self.value})"

@dataclass
class IdentExpr(Expression):
    name: str
    def __repr__(self) -> str:
        return f"Id({self.name})"

@dataclass
class GreekLetterExpr(Expression):
    letter: str
    def __repr__(self) -> str:
        return f"Greek({self.letter})"

@dataclass
class DerivativeVarExpr(Expression):
    """Represents \\dot{x} or \\ddot{x} notation"""
    var: str
    order: int = 1
    def __repr__(self) -> str:
        return f"DerivativeVar({self.var}, order={self.order})"

# Binary operations with type safety
@dataclass
class BinaryOpExpr(Expression):
    left: Expression
    operator: Literal["+", "-", "*", "/", "^"]
    right: Expression
    def __repr__(self) -> str:
        return f"BinOp({self.left} {self.operator} {self.right})"

@dataclass
class UnaryOpExpr(Expression):
    operator: Literal["+", "-"]
    operand: Expression
    def __repr__(self) -> str:
        return f"UnaryOp({self.operator}{self.operand})"

# Vector expressions
@dataclass
class VectorExpr(Expression):
    components: List[Expression]
    def __repr__(self) -> str:
        return f"Vector({self.components})"

@dataclass
class VectorOpExpr(Expression):
    operation: str
    left: Expression
    right: Optional[Expression] = None
    def __repr__(self) -> str:
        if self.right:
            return f"VectorOp({self.operation}, {self.left}, {self.right})"
        return f"VectorOp({self.operation}, {self.left})"

# Calculus expressions
@dataclass
class DerivativeExpr(Expression):
    expr: Expression
    var: str
    order: int = 1
    partial: bool = False
    def __repr__(self) -> str:
        type_str = "Partial" if self.partial else "Total"
        return f"{type_str}Deriv({self.expr}, {self.var}, order={self.order})"

@dataclass
class IntegralExpr(Expression):
    expr: Expression
    var: str
    lower: Optional[Expression] = None
    upper: Optional[Expression] = None
    line_integral: bool = False
    def __repr__(self) -> str:
        return f"Integral({self.expr}, {self.var}, {self.lower}, {self.upper})"

# Function calls
@dataclass
class FunctionCallExpr(Expression):
    name: str
    args: List[Expression]
    def __repr__(self) -> str:
        return f"Call({self.name}, {self.args})"

@dataclass
class FractionExpr(Expression):
    numerator: Expression
    denominator: Expression
    def __repr__(self) -> str:
        return f"Frac({self.numerator}/{self.denominator})"

# Physics-specific AST nodes
@dataclass
class SystemDef(ASTNode):
    name: str
    def __repr__(self) -> str:
        return f"System({self.name})"

@dataclass
class VarDef(ASTNode):
    name: str
    vartype: str
    unit: str
    vector: bool = False
    def __repr__(self) -> str:
        vec_str = " [Vector]" if self.vector else ""
        return f"VarDef({self.name}: {self.vartype}[{self.unit}]{vec_str})"

@dataclass
class ParameterDef(ASTNode):
    name: str
    value: float
    unit: str
    def __repr__(self) -> str:
        return f"Parameter({self.name} = {self.value} [{self.unit}])"

@dataclass
class DefineDef(ASTNode):
    name: str
    args: List[str]
    body: Expression
    def __repr__(self) -> str:
        return f"Define({self.name}({', '.join(self.args)}) = {self.body})"

@dataclass
class LagrangianDef(ASTNode):
    expr: Expression
    def __repr__(self) -> str:
        return f"Lagrangian({self.expr})"

@dataclass
class HamiltonianDef(ASTNode):
    expr: Expression
    def __repr__(self) -> str:
        return f"Hamiltonian({self.expr})"

@dataclass
class TransformDef(ASTNode):
    coord_type: str
    var: str
    expr: Expression
    def __repr__(self) -> str:
        return f"Transform({self.coord_type}: {self.var} = {self.expr})"

@dataclass
class ConstraintDef(ASTNode):
    expr: Expression
    constraint_type: str = "holonomic"
    def __repr__(self) -> str:
        return f"Constraint({self.expr}, type={self.constraint_type})"

@dataclass
class NonHolonomicConstraintDef(ASTNode):
    """Non-holonomic constraint (velocity-dependent)"""
    expr: Expression
    def __repr__(self) -> str:
        return f"NonHolonomicConstraint({self.expr})"

@dataclass
class ForceDef(ASTNode):
    """Non-conservative force definition"""
    expr: Expression
    force_type: str = "general"  # "friction", "damping", "drag", "general"
    def __repr__(self) -> str:
        return f"Force({self.expr}, type={self.force_type})"

@dataclass
class DampingDef(ASTNode):
    """Damping force definition"""
    expr: Expression
    damping_coefficient: Optional[float] = None
    def __repr__(self) -> str:
        return f"Damping({self.expr}, coeff={self.damping_coefficient})"

@dataclass
class InitialCondition(ASTNode):
    conditions: Dict[str, float]
    def __repr__(self) -> str:
        return f"Initial({self.conditions})"

@dataclass
class SolveDef(ASTNode):
    method: str
    options: Dict[str, Any] = field(default_factory=dict)
    def __repr__(self) -> str:
        return f"Solve({self.method}, {self.options})"

@dataclass
class AnimateDef(ASTNode):
    target: str
    options: Dict[str, Any] = field(default_factory=dict)
    def __repr__(self) -> str:
        return f"Animate({self.target}, {self.options})"

@dataclass
class ExportDef(ASTNode):
    filename: str
    format: str = "json"
    def __repr__(self) -> str:
        return f"Export({self.filename}, {self.format})"

@dataclass
class ImportDef(ASTNode):
    filename: str
    def __repr__(self) -> str:
        return f"Import({self.filename})"
        
@dataclass
class RegionDef(ASTNode):
    shape: str  # "rectangle", "circle", "line"
    constraints: Dict[str, Tuple[float, float]] # {'x': (0.0, 1.0), 'y': ...}
    def __repr__(self) -> str:
        return f"Region({self.shape}, {self.constraints})"

@dataclass
class FluidDef(ASTNode):
    name: str
    region: RegionDef
    mass: float
    eos: str
    def __repr__(self) -> str:
        return f"Fluid({self.name}, {self.eos}, mass={self.mass})"

@dataclass
class BoundaryDef(ASTNode):
    name: str
    region: RegionDef
    def __repr__(self) -> str:
        return f"Boundary({self.name})"

# ============================================================================
# PARSER
# ============================================================================

class ParserError(Exception):
    """Custom exception for parser errors"""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        if self.token:
            return f"{self.message} at line {self.token.line}, column {self.token.column}"
        return self.message

class MechanicsParser:
    """Parser with improved error handling and feature completeness"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_system = None
        self.errors: List[str] = []
        self.max_errors = config.max_parser_errors

    def peek(self, offset: int = 0) -> Optional[Token]:
        """Look ahead at token without consuming it"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def match(self, *expected_types: str) -> Optional[Token]:
        """Match and consume token if type matches"""
        token = self.peek()
        if token and token.type in expected_types:
            self.pos += 1
            return token
        return None

    def expect(self, expected_type: str) -> Token:
        """Expect a specific token type, raise error if not found"""
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
        """Parse the complete DSL with comprehensive error recovery"""
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
                    if token and token.type in ["SYSTEM", "DEFVAR", "DEFINE", 
                                                "LAGRANGIAN", "HAMILTONIAN", 
                                                "CONSTRAINT", "INITIAL", "SOLVE"]:
                        break
                    self.pos += 1
        
        if self.errors:
            logger.warning(f"Parser encountered {len(self.errors)} errors")
            
        logger.info(f"Successfully parsed {len(nodes)} AST nodes")
        return nodes

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a top-level statement"""
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

    def parse_region(self) -> RegionDef:
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
        """Parse \fluid{name} ... properties ..."""
        self.expect("FLUID")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        
        # Defaults
        region = None
        mass = 1.0
        eos = "tait"
        
        # Parse fluid properties line by line until hit new command
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
        """Parse \boundary{name} \region{...}"""
        self.expect("BOUNDARY")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        
        region = self.parse_region()
        return BoundaryDef(name, region)

    def parse_system(self) -> SystemDef:
        """Parse \\system{name}"""
        self.expect("SYSTEM")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.current_system = name
        return SystemDef(name)

    def parse_defvar(self) -> VarDef:
        """Parse \\defvar{name}{type}{unit}"""
        self.expect("DEFVAR")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        
        vartype_parts = []
        while True:
            tok = self.peek()
            if not tok or tok.type == 'RBRACE':
                break
            self.pos += 1
            vartype_parts.append(tok.value)
        vartype = ' '.join(vartype_parts).strip()
        self.expect("RBRACE")
        
        self.expect("LBRACE")
        unit_expr = self.parse_expression()
        unit = self.expression_to_string(unit_expr)
        self.expect("RBRACE")
        
        is_vector = vartype in ["Vector", "Vector3", "Position", "Velocity", 
                               "Force", "Momentum", "Acceleration"]
        
        return VarDef(name, vartype, unit, is_vector)
    
    def parse_parameter(self) -> ParameterDef:
        """Parse \\parameter{name}{value}{unit}"""
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
        """Parse \\define{\\op{name}(args) = expression}"""
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
        """Parse \\lagrangian{expression}"""
        self.expect("LAGRANGIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return LagrangianDef(expr)

    def parse_hamiltonian(self) -> HamiltonianDef:
        """Parse \\hamiltonian{expression}"""
        self.expect("HAMILTONIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return HamiltonianDef(expr)

    def parse_transform(self) -> TransformDef:
        """Parse \\transform{type}{var = expr}"""
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
        """Parse \\constraint{expression}"""
        self.expect("CONSTRAINT")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return ConstraintDef(expr)

    def parse_nonholonomic(self) -> NonHolonomicConstraintDef:
        """Parse \\nonholonomic{expression}"""
        self.expect("NONHOLONOMIC")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return NonHolonomicConstraintDef(expr)

    def parse_force(self) -> ForceDef:
        """Parse \\force{expression}"""
        self.expect("FORCE")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return ForceDef(expr)

    def parse_damping(self) -> DampingDef:
        """Parse \\damping{expression}"""
        self.expect("DAMPING")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return DampingDef(expr)

    def parse_initial(self) -> InitialCondition:
        """Parse \\initial{var1=val1, var2=val2, ...}"""
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
        """Parse \\solve{method}"""
        self.expect("SOLVE")
        self.expect("LBRACE")
        method = self.expect("IDENT").value
        self.expect("RBRACE")
        return SolveDef(method)

    def parse_animate(self) -> AnimateDef:
        """Parse \\animate{target}"""
        self.expect("ANIMATE")
        self.expect("LBRACE")
        target = self.expect("IDENT").value
        self.expect("RBRACE")
        return AnimateDef(target)

    def parse_export(self) -> ExportDef:
        """Parse \\export{filename}"""
        self.expect("EXPORT")
        self.expect("LBRACE")
        filename = self.expect("IDENT").value
        self.expect("RBRACE")
        return ExportDef(filename)

    def parse_import(self) -> ImportDef:
        """Parse \\import{filename}"""
        self.expect("IMPORT")
        self.expect("LBRACE")
        filename = self.expect("IDENT").value
        self.expect("RBRACE")
        return ImportDef(filename)

    def parse_expression(self) -> Expression:
        """Parse expressions with full operator precedence"""
        return self.parse_additive()

    def parse_additive(self) -> Expression:
        """Addition and subtraction"""
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
        """Multiplication, division, and explicit products only"""
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
                if (next_token and 
                    next_token.type == "LPAREN" and
                    isinstance(left, (NumberExpr, IdentExpr, GreekLetterExpr)) and
                    not self.at_end_of_expression()):
                    # Safe implicit multiplication: 2(x+y), m(v^2), etc.
                    right = self.parse_power()
                    left = BinaryOpExpr(left, "*", right)
                else:
                    break
                    
        return left

    def parse_power(self) -> Expression:
        """Exponentiation (right associative)"""
        left = self.parse_unary()
        
        if self.match("POWER"):
            right = self.parse_power()
            return BinaryOpExpr(left, "^", right)
            
        return left

    def parse_unary(self) -> Expression:
        """Unary operators"""
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
        """Primary expressions: literals, identifiers, parentheses, vectors, commands"""

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
        """Parse LaTeX-style commands"""
        
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
            
        elif cmd in [r"\sin", r"\cos", r"\tan", r"\exp", r"\log", r"\ln", r"\sqrt", 
                     r"\sinh", r"\cosh", r"\tanh", r"\arcsin", r"\arccos", r"\arctan"]:
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

    def at_end_of_expression(self) -> bool:
        """Check if we're at the end of an expression"""
        token = self.peek()
        return (not token or 
                token.type in ["RBRACE", "RPAREN", "RBRACKET", "COMMA", 
                              "SEMICOLON", "EQUALS"])

    def expression_to_string(self, expr: Expression) -> str:
        """Convert expression back to string for unit parsing"""
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
