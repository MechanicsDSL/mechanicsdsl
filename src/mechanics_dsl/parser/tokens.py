"""
Token system for MechanicsDSL parser.

This module provides the tokenization layer that converts DSL source code
into a stream of tokens for parsing.

Classes:
    Token: Represents a token with position tracking for error messages.

Functions:
    tokenize: Convert source code string to list of tokens.

Example:
    >>> from mechanics_dsl.parser.tokens import tokenize
    >>> tokens = tokenize(r"\\system{pendulum}")
    >>> print(tokens[0])
    SYSTEM:\\system@1:1
"""
import re
from typing import List
from dataclasses import dataclass

from ..utils import logger

# ============================================================================
# TOKEN TYPE DEFINITIONS
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

# Compile token regex pattern
token_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES)
token_pattern = re.compile(token_regex)


# ============================================================================
# TOKEN CLASS
# ============================================================================

@dataclass
class Token:
    """
    Token with position tracking for better error messages.
    
    Attributes:
        type: The token type (e.g., 'IDENT', 'NUMBER', 'LAGRANGIAN').
        value: The raw string value matched from source.
        position: Character position in source (0-indexed).
        line: Line number (1-indexed).
        column: Column number (1-indexed).
    
    Example:
        >>> token = Token('IDENT', 'theta', position=10, line=2, column=5)
        >>> print(token)
        IDENT:theta@2:5
    """
    type: str
    value: str
    position: int = 0
    line: int = 1
    column: int = 1

    def __repr__(self) -> str:
        return f"{self.type}:{self.value}@{self.line}:{self.column}"


# ============================================================================
# TOKENIZER FUNCTION
# ============================================================================

def tokenize(source: str) -> List[Token]:
    """
    Tokenize DSL source code with position tracking.
    
    Converts a string of MechanicsDSL code into a list of tokens,
    excluding whitespace and comments.
    
    Args:
        source: DSL source code string.
        
    Returns:
        List of Token objects (excluding whitespace and comments).
        
    Raises:
        No explicit exceptions, but malformed input may produce
        unexpected token sequences.
        
    Example:
        >>> tokens = tokenize(r"\\lagrangian{T - V}")
        >>> [t.type for t in tokens]
        ['LAGRANGIAN', 'LBRACE', 'IDENT', 'MINUS', 'IDENT', 'RBRACE']
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


__all__ = [
    'TOKEN_TYPES',
    'token_pattern',
    'Token',
    'tokenize',
]
