"""
MechanicsDSL Language Server.

Implements the Language Server Protocol for MechanicsDSL files.
"""
import re
from typing import List, Optional, Dict, Any
import logging

try:
    from pygls.server import LanguageServer
    from pygls.lsp.types import (
        TEXT_DOCUMENT_DID_OPEN,
        TEXT_DOCUMENT_DID_CHANGE,
        TEXT_DOCUMENT_COMPLETION,
        TEXT_DOCUMENT_HOVER,
        TEXT_DOCUMENT_DEFINITION,
        CompletionList,
        CompletionItem,
        CompletionItemKind,
        Diagnostic,
        DiagnosticSeverity,
        Hover,
        MarkupContent,
        MarkupKind,
        Position,
        Range,
        Location,
        TextDocumentPositionParams,
        DidOpenTextDocumentParams,
        DidChangeTextDocumentParams,
    )
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False
    LanguageServer = object


logger = logging.getLogger("MechanicsDSL-LSP")


# Command documentation
COMMAND_DOCS = {
    r'\system': {
        'signature': r'\system{name}',
        'description': 'Define the system name.',
        'example': r'\system{double_pendulum}',
    },
    r'\defvar': {
        'signature': r'\defvar{name}{type}{unit}',
        'description': 'Define a variable with type and unit.',
        'example': r'\defvar{theta}{Angle}{rad}',
    },
    r'\parameter': {
        'signature': r'\parameter{name}{value}{unit}',
        'description': 'Define a constant parameter.',
        'example': r'\parameter{m}{1.0}{kg}',
    },
    r'\lagrangian': {
        'signature': r'\lagrangian{expression}',
        'description': 'Define the Lagrangian (T - V).',
        'example': r'\lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}',
    },
    r'\hamiltonian': {
        'signature': r'\hamiltonian{expression}',
        'description': 'Define the Hamiltonian (T + V).',
        'example': r'\hamiltonian{\frac{p^2}{2*m} + \frac{1}{2}*k*x^2}',
    },
    r'\force': {
        'signature': r'\force{expression}',
        'description': 'Define a non-conservative force.',
        'example': r'\force{-b*x_dot}',
    },
    r'\initial': {
        'signature': r'\initial{var=value, var_dot=value}',
        'description': 'Set initial conditions.',
        'example': r'\initial{theta=1.5, theta_dot=0.0}',
    },
    r'\constraint': {
        'signature': r'\constraint{expression = 0}',
        'description': 'Define a holonomic constraint.',
        'example': r'\constraint{x^2 + y^2 - l^2}',
    },
    r'\frac': {
        'signature': r'\frac{num}{denom}',
        'description': 'Fraction (division).',
        'example': r'\frac{1}{2}',
    },
    r'\dot': {
        'signature': r'\dot{var}',
        'description': 'Time derivative (velocity).',
        'example': r'\dot{theta}',
    },
    r'\ddot': {
        'signature': r'\ddot{var}',
        'description': 'Second time derivative (acceleration).',
        'example': r'\ddot{theta}',
    },
    r'\cos': {
        'signature': r'\cos{expr}',
        'description': 'Cosine function.',
        'example': r'\cos{theta}',
    },
    r'\sin': {
        'signature': r'\sin{expr}',
        'description': 'Sine function.',
        'example': r'\sin{theta}',
    },
    r'\sqrt': {
        'signature': r'\sqrt{expr}',
        'description': 'Square root.',
        'example': r'\sqrt{x^2 + y^2}',
    },
}

# Common variable types
VARIABLE_TYPES = [
    'Angle', 'Position', 'Velocity', 'Mass', 'Length',
    'Spring Constant', 'Damping Coeff', 'Acceleration',
    'Force', 'Energy', 'Momentum', 'Moment of Inertia',
]

# Common units
UNITS = [
    'rad', 'm', 'kg', 's', 'N', 'J', 'W',
    'm/s', 'm/s^2', 'rad/s', 'N/m', 'N*s/m',
    'kg*m^2', 'kg*m/s',
]

# Greek letters
GREEK_LETTERS = [
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
    'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu',
    'nu', 'xi', 'pi', 'rho', 'sigma', 'tau',
    'upsilon', 'phi', 'chi', 'psi', 'omega',
]


class MechanicsDSLLanguageServer(LanguageServer if PYGLS_AVAILABLE else object):
    """
    Language Server for MechanicsDSL.
    
    Provides diagnostics, completion, hover, and go-to-definition.
    """
    
    def __init__(self):
        if not PYGLS_AVAILABLE:
            raise ImportError("pygls not installed. Install with: pip install pygls")
        
        super().__init__('MechanicsDSL-LSP', 'v1.0.0')
        
        # Document state
        self.documents: Dict[str, str] = {}
        self.parsed: Dict[str, Dict] = {}
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register LSP event handlers."""
        
        @self.feature(TEXT_DOCUMENT_DID_OPEN)
        def did_open(params: DidOpenTextDocumentParams):
            uri = params.text_document.uri
            text = params.text_document.text
            self.documents[uri] = text
            self._validate_document(uri)
        
        @self.feature(TEXT_DOCUMENT_DID_CHANGE)
        def did_change(params: DidChangeTextDocumentParams):
            uri = params.text_document.uri
            for change in params.content_changes:
                self.documents[uri] = change.text
            self._validate_document(uri)
        
        @self.feature(TEXT_DOCUMENT_COMPLETION)
        def completion(params: TextDocumentPositionParams) -> CompletionList:
            return self._get_completions(params)
        
        @self.feature(TEXT_DOCUMENT_HOVER)
        def hover(params: TextDocumentPositionParams) -> Optional[Hover]:
            return self._get_hover(params)
    
    def _validate_document(self, uri: str):
        """Parse document and publish diagnostics."""
        text = self.documents.get(uri, "")
        diagnostics = []
        
        lines = text.split('\n')
        
        # Track defined variables
        defined_vars = set()
        
        for line_num, line in enumerate(lines):
            # Check for unclosed braces
            open_braces = line.count('{')
            close_braces = line.count('}')
            if open_braces != close_braces:
                diagnostics.append(Diagnostic(
                    range=Range(
                        start=Position(line=line_num, character=0),
                        end=Position(line=line_num, character=len(line))
                    ),
                    message=f"Mismatched braces: {open_braces} open, {close_braces} close",
                    severity=DiagnosticSeverity.Error,
                    source="mechanicsdsl"
                ))
            
            # Extract defined variables
            defvar_match = re.search(r'\\defvar\{(\w+)\}', line)
            if defvar_match:
                defined_vars.add(defvar_match.group(1))
            
            param_match = re.search(r'\\parameter\{(\w+)\}', line)
            if param_match:
                defined_vars.add(param_match.group(1))
            
            # Check for unknown commands
            for match in re.finditer(r'\\([a-zA-Z]+)', line):
                cmd = '\\' + match.group(1)
                if cmd not in COMMAND_DOCS and match.group(1) not in GREEK_LETTERS:
                    diagnostics.append(Diagnostic(
                        range=Range(
                            start=Position(line=line_num, character=match.start()),
                            end=Position(line=line_num, character=match.end())
                        ),
                        message=f"Unknown command: {cmd}",
                        severity=DiagnosticSeverity.Warning,
                        source="mechanicsdsl"
                    ))
        
        self.publish_diagnostics(uri, diagnostics)
    
    def _get_completions(self, params: TextDocumentPositionParams) -> CompletionList:
        """Get completion items at position."""
        uri = params.text_document.uri
        pos = params.position
        text = self.documents.get(uri, "")
        
        lines = text.split('\n')
        if pos.line >= len(lines):
            return CompletionList(is_incomplete=False, items=[])
        
        line = lines[pos.line]
        prefix = line[:pos.character]
        
        items = []
        
        # Command completions
        if '\\' in prefix:
            last_backslash = prefix.rfind('\\')
            partial = prefix[last_backslash+1:]
            
            for cmd, doc in COMMAND_DOCS.items():
                cmd_name = cmd[1:]  # Remove backslash
                if cmd_name.startswith(partial):
                    items.append(CompletionItem(
                        label=cmd_name,
                        kind=CompletionItemKind.Function,
                        detail=doc['signature'],
                        documentation=doc['description'],
                        insert_text=cmd_name + '{$1}',
                    ))
            
            # Greek letters
            for letter in GREEK_LETTERS:
                if letter.startswith(partial):
                    items.append(CompletionItem(
                        label=letter,
                        kind=CompletionItemKind.Constant,
                        detail=f'Greek letter: {letter}',
                    ))
        
        # Type completions inside defvar
        if re.search(r'\\defvar\{[^}]+\}\{$', prefix):
            for vtype in VARIABLE_TYPES:
                items.append(CompletionItem(
                    label=vtype,
                    kind=CompletionItemKind.TypeParameter,
                ))
        
        # Unit completions
        if re.search(r'\\(defvar|parameter)\{[^}]+\}\{[^}]+\}\{$', prefix):
            for unit in UNITS:
                items.append(CompletionItem(
                    label=unit,
                    kind=CompletionItemKind.Unit,
                ))
        
        return CompletionList(is_incomplete=False, items=items)
    
    def _get_hover(self, params: TextDocumentPositionParams) -> Optional[Hover]:
        """Get hover information at position."""
        uri = params.text_document.uri
        pos = params.position
        text = self.documents.get(uri, "")
        
        lines = text.split('\n')
        if pos.line >= len(lines):
            return None
        
        line = lines[pos.line]
        
        # Find command at position
        for match in re.finditer(r'\\([a-zA-Z]+)', line):
            if match.start() <= pos.character <= match.end():
                cmd = '\\' + match.group(1)
                if cmd in COMMAND_DOCS:
                    doc = COMMAND_DOCS[cmd]
                    content = f"**{doc['signature']}**\n\n{doc['description']}\n\n```mdsl\n{doc['example']}\n```"
                    return Hover(
                        contents=MarkupContent(
                            kind=MarkupKind.Markdown,
                            value=content
                        )
                    )
        
        return None


def start_tcp_server(host: str = "127.0.0.1", port: int = 2087):
    """Start LSP server on TCP."""
    if not PYGLS_AVAILABLE:
        print("pygls not installed. Install with: pip install pygls")
        return
    
    server = MechanicsDSLLanguageServer()
    server.start_tcp(host, port)


def start_io_server():
    """Start LSP server on stdio."""
    if not PYGLS_AVAILABLE:
        print("pygls not installed. Install with: pip install pygls")
        return
    
    server = MechanicsDSLLanguageServer()
    server.start_io()


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="MechanicsDSL Language Server")
    parser.add_argument("--tcp", action="store_true", help="Use TCP instead of stdio")
    parser.add_argument("--port", type=int, default=2087, help="TCP port")
    args = parser.parse_args()
    
    if args.tcp:
        print(f"Starting TCP server on port {args.port}")
        start_tcp_server(port=args.port)
    else:
        start_io_server()


if __name__ == "__main__":
    main()


__all__ = [
    'MechanicsDSLLanguageServer',
    'start_tcp_server',
    'start_io_server',
]
