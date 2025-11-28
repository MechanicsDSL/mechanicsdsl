import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from .utils import logger, config

TOKEN_TYPES = [
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
    ("PARAMETER", r"\\parameter"),
    ("COMMAND", r"\\[a-zA-Z_][a-zA-Z0-9_]*"),
    ("LBRACE", r"\{"), ("RBRACE", r"\}"),
    ("LPAREN", r"\("), ("RPAREN", r"\)"),
    ("PLUS", r"\+"), ("MINUS", r"-"),
    ("MULTIPLY", r"\*"), ("DIVIDE", r"/"), ("POWER", r"\^"),
    ("EQUALS", r"="), ("COMMA", r","),
    ("NUMBER", r"\d+\.?\d*([eE][+-]?\d+)?"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("WHITESPACE", r"\s+"), ("NEWLINE", r"\n"), ("COMMENT", r"%.*"),
]

token_pattern = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES))

@dataclass
class Token:
    type: str
    value: str
    line: int = 1
    column: int = 1

def tokenize(source: str) -> List[Token]:
    tokens = []
    line = 1
    line_start = 0
    for match in token_pattern.finditer(source):
        kind = match.lastgroup
        value = match.group()
        if kind not in ["WHITESPACE", "COMMENT"]:
            tokens.append(Token(kind, value, line=line))
    return tokens

@dataclass
class ASTNode: pass

@dataclass
class SystemDef(ASTNode): name: str
@dataclass
class VarDef(ASTNode): name: str; vartype: str; unit: str
@dataclass
class ParameterDef(ASTNode): name: str; value: float; unit: str
@dataclass
class LagrangianDef(ASTNode): expr: Any
@dataclass
class InitialCondition(ASTNode): conditions: Dict[str, float]
@dataclass
class SolveDef(ASTNode): method: str
@dataclass
class AnimateDef(ASTNode): target: str

# Expression Nodes
@dataclass
class Expression(ASTNode): pass
@dataclass
class NumberExpr(Expression): value: float
@dataclass
class IdentExpr(Expression): name: str
@dataclass
class BinaryOpExpr(Expression): left: Expression; operator: str; right: Expression
@dataclass
class UnaryOpExpr(Expression): operator: str; operand: Expression
@dataclass
class FunctionCallExpr(Expression): name: str; args: List[Expression]
@dataclass
class DerivativeVarExpr(Expression): var: str; order: int

class ParserError(Exception): pass

class MechanicsParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors = []

    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    
    def match(self, type_):
        tok = self.peek()
        if tok and tok.type == type_:
            self.pos += 1
            return tok
        return None

    def expect(self, type_):
        tok = self.match(type_)
        if not tok: raise ParserError(f"Expected {type_}")
        return tok

    def parse(self) -> List[ASTNode]:
        nodes = []
        while self.pos < len(self.tokens):
            try:
                if self.peek().type == "SYSTEM": nodes.append(self.parse_system())
                elif self.peek().type == "DEFVAR": nodes.append(self.parse_defvar())
                elif self.peek().type == "PARAMETER": nodes.append(self.parse_parameter())
                elif self.peek().type == "LAGRANGIAN": nodes.append(self.parse_lagrangian())
                elif self.peek().type == "INITIAL": nodes.append(self.parse_initial())
                elif self.peek().type == "SOLVE": nodes.append(self.parse_solve())
                elif self.peek().type == "ANIMATE": nodes.append(self.parse_animate())
                else: self.pos += 1 
            except ParserError as e:
                self.errors.append(str(e))
                self.pos += 1
        return nodes

    # --- Parser Methods (Simplified for brevity, assuming full logic is copied) ---
    def parse_system(self): 
        self.expect("SYSTEM"); self.expect("LBRACE"); name = self.expect("IDENT").value; self.expect("RBRACE")
        return SystemDef(name)
    
    def parse_defvar(self):
        self.expect("DEFVAR"); self.expect("LBRACE"); name = self.expect("IDENT").value; self.expect("RBRACE")
        self.expect("LBRACE"); type_ = self.expect("IDENT").value; self.expect("RBRACE")
        self.expect("LBRACE"); unit = self.expect("IDENT").value; self.expect("RBRACE")
        return VarDef(name, type_, unit)

    def parse_parameter(self):
        self.expect("PARAMETER"); self.expect("LBRACE"); name = self.expect("IDENT").value; self.expect("RBRACE")
        self.expect("LBRACE"); val = float(self.expect("NUMBER").value); self.expect("RBRACE")
        self.expect("LBRACE"); unit = self.expect("IDENT").value; self.expect("RBRACE")
        return ParameterDef(name, val, unit)

    def parse_lagrangian(self):
        self.expect("LAGRANGIAN"); self.expect("LBRACE"); expr = self.parse_expression(); self.expect("RBRACE")
        return LagrangianDef(expr)

    def parse_initial(self):
        self.expect("INITIAL"); self.expect("LBRACE")
        conds = {}
        while True:
            name = self.expect("IDENT").value; self.expect("EQUALS"); val = float(self.expect("NUMBER").value)
            conds[name] = val
            if not self.match("COMMA"): break
        self.expect("RBRACE")
        return InitialCondition(conds)

    def parse_solve(self):
        self.expect("SOLVE"); self.expect("LBRACE"); method = self.expect("IDENT").value; self.expect("RBRACE")
        return SolveDef(method)

    def parse_animate(self):
        self.expect("ANIMATE"); self.expect("LBRACE"); target = self.expect("IDENT").value; self.expect("RBRACE")
        return AnimateDef(target)

    def parse_expression(self): return self.parse_additive()
    
    def parse_additive(self):
        left = self.parse_multiplicative()
        while True:
            if self.match("PLUS"): left = BinaryOpExpr(left, "+", self.parse_multiplicative())
            elif self.match("MINUS"): left = BinaryOpExpr(left, "-", self.parse_multiplicative())
            else: break
        return left

    def parse_multiplicative(self):
        left = self.parse_primary()
        while True:
            if self.match("MULTIPLY"): left = BinaryOpExpr(left, "*", self.parse_primary())
            elif self.match("DIVIDE"): left = BinaryOpExpr(left, "/", self.parse_primary())
            else: break
        return left

    def parse_primary(self):
        if self.match("LPAREN"):
            expr = self.parse_expression(); self.expect("RPAREN"); return expr
        if self.match("NUMBER"): return NumberExpr(float(self.tokens[self.pos-1].value))
        if self.match("IDENT"): return IdentExpr(self.tokens[self.pos-1].value)
        tok = self.peek()
        if tok and tok.type == "DOT_NOTATION":
            self.pos += 1; order = 2 if tok.value == r"\ddot" else 1
            self.expect("LBRACE"); var = self.expect("IDENT").value; self.expect("RBRACE")
            return DerivativeVarExpr(var, order)
        if tok and tok.type == "COMMAND":
            self.pos += 1; func = tok.value[1:]
            self.expect("LBRACE"); arg = self.parse_expression(); self.expect("RBRACE")
            return FunctionCallExpr(func, [arg])
        raise ParserError(f"Unexpected token {tok}")
