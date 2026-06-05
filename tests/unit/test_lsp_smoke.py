"""
Smoke tests for the LSP server.

Skipped entirely if pygls + lsprotocol aren't installed (they're in the
``[lsp]`` extra). When they are installed, this pins the modern pygls
2.x / lsprotocol API the v2.1.1 migration moved to. The pre-1.0 pygls
imports silently fell back to ``PYGLS_AVAILABLE = False``, which made
the LSP look "uninstalled" even when pygls was in the environment.
"""

import pytest

pytest.importorskip("pygls")
pytest.importorskip("lsprotocol")


def test_lsp_module_imports_with_modern_pygls():
    """The server module loads without falling back to PYGLS_AVAILABLE=False."""
    from mechanics_dsl.lsp.server import PYGLS_AVAILABLE

    assert PYGLS_AVAILABLE is True, (
        "pygls/lsprotocol are installed but the LSP module fell back to "
        "PYGLS_AVAILABLE=False - the imports are still on the pre-1.0 API."
    )


def test_lsp_server_instantiates():
    """The server constructor must succeed and expose the modern transport
    entry points pygls 2.x provides."""
    from mechanics_dsl.lsp.server import MechanicsDSLLanguageServer

    server = MechanicsDSLLanguageServer()
    assert hasattr(server, "start_tcp")
    assert hasattr(server, "start_io")
    # Document state is initialised so request handlers can read it.
    assert isinstance(server.documents, dict)


def test_lsp_diagnostics_detect_unknown_command():
    """A nonsense \\widget command must be flagged as a Warning diagnostic."""
    from mechanics_dsl.lsp.server import MechanicsDSLLanguageServer

    server = MechanicsDSLLanguageServer()
    # Reach into the validation code directly - we don't want to spin up a
    # real LSP transport here; the validation function is the bit users
    # actually rely on for diagnostics.
    server.documents["file:///fake.mdsl"] = (
        r"\system{x}" + "\n" + r"\widget{nope}"
    )
    # We can't easily intercept publish_diagnostics without a running
    # transport. Patch it.
    captured = []
    server.publish_diagnostics = lambda uri, diags: captured.append((uri, diags))
    server._validate_document("file:///fake.mdsl")
    assert captured, "no diagnostics published"
    _uri, diags = captured[0]
    messages = [d.message for d in diags]
    assert any("\\widget" in m for m in messages), messages
