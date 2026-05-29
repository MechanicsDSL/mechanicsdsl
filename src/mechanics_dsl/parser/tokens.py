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
from dataclasses import dataclass
from typing import List, Tuple

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
    ("RAYLEIGH", r"\\rayleigh"),
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
    (
        "GREEK_LETTER",
        r"\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\varepsilon|\\zeta|\\eta|\\theta|\\vartheta|\\iota|\\kappa|\\lambda|\\mu|\\nu|\\xi|\\omicron|\\pi|\\varpi|\\rho|\\varrho|\\sigma|\\varsigma|\\tau|\\upsilon|\\phi|\\varphi|\\chi|\\psi|\\omega",  # noqa: E501
    ),
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
    excluding whitespace and comments. Unrecognized characters are reported
    as a single error rather than silently dropped.

    Args:
        source: DSL source code string.

    Returns:
        List of Token objects (excluding whitespace and comments).

    Raises:
        ValueError: If the source contains characters that do not match any
            token pattern (e.g. ``@``, ``$``, ``&``).

    Example:
        >>> tokens = tokenize(r"\\lagrangian{T - V}")
        >>> [t.type for t in tokens]
        ['LAGRANGIAN', 'LBRACE', 'IDENT', 'MINUS', 'IDENT', 'RBRACE']
    """
    tokens: List[Token] = []
    line = 1
    line_start = 0
    pos = 0
    unknown: List[Tuple[int, int, str]] = []

    def _account_for_text(text: str, base_pos: int) -> None:
        """Advance line/line_start across `text` starting at `base_pos`."""
        nonlocal line, line_start
        last_nl = text.rfind("\n")
        if last_nl != -1:
            line += text.count("\n")
            line_start = base_pos + last_nl + 1

    for match in token_pattern.finditer(source):
        start = match.start()

        # Anything between the last match and this one is unmatched. Track
        # line numbers across that gap and record any non-whitespace chars.
        if start > pos:
            gap = source[pos:start]
            for offset, ch in enumerate(gap):
                if ch == "\n":
                    line += 1
                    line_start = pos + offset + 1
                elif not ch.isspace():
                    unknown.append((line, (pos + offset) - line_start + 1, ch))

        kind = match.lastgroup
        value = match.group()
        column = start - line_start + 1

        if kind not in ("WHITESPACE", "COMMENT"):
            tokens.append(Token(kind, value, start, line, column))

        # Update line tracking for newlines inside the matched span too.
        if "\n" in value:
            _account_for_text(value, start)
        pos = match.end()

    # Anything left after the final match.
    if pos < len(source):
        tail = source[pos:]
        for offset, ch in enumerate(tail):
            if ch == "\n":
                line += 1
                line_start = pos + offset + 1
            elif not ch.isspace():
                unknown.append((line, (pos + offset) - line_start + 1, ch))

    if unknown:
        head = unknown[:5]
        details = ", ".join(f"{ch!r} at line {ln}, col {col}" for ln, col, ch in head)
        more = "" if len(unknown) <= len(head) else f" (+{len(unknown) - len(head)} more)"
        raise ValueError(f"Unrecognized character(s): {details}{more}")

    logger.debug(f"Tokenized {len(tokens)} tokens from {line} lines")
    return tokens


__all__ = [
    "TOKEN_TYPES",
    "token_pattern",
    "Token",
    "tokenize",
]
