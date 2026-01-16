"""
MechanicsDSL Language Server Protocol (LSP) Implementation

Provides IDE features for .mdsl files:
- Syntax error diagnostics
- Autocomplete for commands and variables
- Hover documentation
- Go to definition

Usage:
    python -m mechanics_dsl.lsp

With VS Code:
    Install extension and configure:
    "mechanicsdsl.lsp.path": "python -m mechanics_dsl.lsp"
"""

from .server import (
    MechanicsDSLLanguageServer,
    start_io_server,
    start_tcp_server,
)

__all__ = [
    "MechanicsDSLLanguageServer",
    "start_tcp_server",
    "start_io_server",
]
