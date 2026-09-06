"""Central policy for symbolic simplification.

Every derivation path routes its simplification through `maybe_simplify` so
that the decision is made in exactly one place. Before this existed, seven
call sites each repeated

    if config.simplification_timeout > 0:
        with timeout(config.simplification_timeout):
            expr = sp.simplify(expr)
    else:
        expr = sp.simplify(expr)

which has two defects. The `else` branch makes a timeout of 0 mean "simplify
with no deadline" -- the slowest possible path -- although 0 is the value a
user reaches for to turn simplification off. And the watchdog cannot interrupt
SymPy inside a C-level routine, so on the expensive calls the deadline passes
without effect and the full cost is paid anyway.

Simplification is cosmetic: it rewrites an expression into an equivalent one,
changing its size and never its value. So skipping it is always safe for
correctness and only ever costs legibility of the derived equations.
"""

from __future__ import annotations

from typing import Any

from .config import config
from .logging import logger
from .profiling import TimeoutError, timeout

__all__ = ["maybe_simplify"]


def maybe_simplify(expr: Any, what: str = "expression") -> Any:
    """Simplify `expr` if configured to, and never fail because of it.

    Returns the simplified expression, or the original one when
    simplification is disabled, times out, or raises. `what` names the
    quantity in log messages.
    """
    if not config.enable_simplification:
        return expr

    import sympy as sp

    try:
        if config.simplification_timeout > 0:
            with timeout(config.simplification_timeout):
                return sp.simplify(expr)
        return sp.simplify(expr)
    except TimeoutError:
        logger.warning(
            f"Simplification of {what} exceeded "
            f"{config.simplification_timeout}s; using the unsimplified form. "
            "Note the deadline is advisory and cannot interrupt SymPy inside "
            "a C-level routine, so the cost may already have been paid. Set "
            "config.enable_simplification = False to skip simplification."
        )
        return expr
    except (ValueError, TypeError, AttributeError, RecursionError) as e:
        logger.warning(
            f"Simplification of {what} failed ({type(e).__name__}: {e}); "
            "using the unsimplified form."
        )
        return expr
