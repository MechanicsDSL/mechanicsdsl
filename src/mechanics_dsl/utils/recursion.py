"""Deep-recursion support for deriving large systems.

WHY THIS EXISTS
---------------
Deriving the equations of motion of a strongly coupled system walks expression
trees whose depth grows with the number of coordinates. Past roughly thirty
coordinates the planar chain exceeds CPython's default recursion limit of
1000 and derivation fails with

    ValueError: Equation derivation failed: maximum recursion depth exceeded

That is an interpreter default refusing the problem, not the engine or the
mathematics finding it intractable. Measured on the planar chain, with
simplification disabled, raising the limit converts an instant refusal into a
completed derivation: N=32 refuses at the default and completes in 145s at a
limit of 60000; N=60 completes in 916s, with accelerations agreeing with an
independent closed-form reference to 4.1e-13.

WHY A THREAD
------------
`sys.setrecursionlimit` alone is not safe. The limit guards against exhausting
the C stack; raising it without also enlarging the stack replaces a catchable
RecursionError with a hard interpreter crash, which is strictly worse. Python
cannot resize the main thread's stack, but `threading.stack_size()` sets the
stack of threads created afterwards. Running the derivation on such a thread is
therefore the only way to raise the limit safely.

The cost is that the derivation no longer runs on the main thread. Anything
requiring the main thread must tolerate that -- notably `utils.timeout`, whose
SIGALRM path is main-thread only and which now degrades to an advisory deadline
off it. That guard is in profiling.py and is required for this module to be
safe to use.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from .logging import logger

__all__ = ["elevated_recursion", "run_with_deep_recursion", "DEFAULT_DEEP_LIMIT",
           "DEFAULT_STACK_MB"]

# Enough for the planar chain to N=60, measured. Deeper systems may need more;
# the limit is cheap, the stack is what actually costs.
DEFAULT_DEEP_LIMIT = 200_000

# Windows rejected 256MB outright in testing; 128MB was accepted and sufficient
# for N=60. Kept below the point where allocation starts failing.
DEFAULT_STACK_MB = 128


@contextmanager
def elevated_recursion(limit: int) -> Iterator[None]:
    """Raise the recursion limit for the duration of the block, then restore.

    Use only on a thread whose stack has been enlarged to match; on the main
    thread prefer `run_with_deep_recursion`, which arranges both.
    """
    previous = sys.getrecursionlimit()
    if limit > previous:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


def run_with_deep_recursion(
    fn: Callable[[], Any],
    limit: int = DEFAULT_DEEP_LIMIT,
    stack_mb: int = DEFAULT_STACK_MB,
) -> Any:
    """Run `fn()` on a worker thread with an enlarged stack and recursion limit.

    Returns fn()'s value, or re-raises its exception on the calling thread so
    the caller sees ordinary control flow. If the requested stack size is
    refused by the platform, falls back to progressively smaller ones and
    finally to running inline, rather than failing outright.
    """
    box: dict = {}

    def target() -> None:
        try:
            with elevated_recursion(limit):
                box["value"] = fn()
        except BaseException as e:  # re-raised on the caller's thread below
            box["error"] = e

    previous_stack = threading.stack_size()
    started = False
    for mb in (stack_mb, 64, 32, 16):
        if mb is None or mb <= 0:
            continue
        try:
            threading.stack_size(mb * 1024 * 1024)
        except (ValueError, RuntimeError):
            continue
        try:
            worker = threading.Thread(target=target, name="mdsl-derivation")
            worker.start()
            started = True
        except (RuntimeError, MemoryError) as e:
            logger.debug(f"Could not start derivation thread at {mb}MB stack: {e}")
            continue
        finally:
            try:
                threading.stack_size(previous_stack)
            except (ValueError, RuntimeError):
                pass
        if mb != stack_mb:
            logger.info(f"Derivation thread started with a {mb}MB stack "
                        f"({stack_mb}MB was refused)")
        worker.join()
        break

    if not started:
        logger.warning(
            "Could not start a derivation thread with an enlarged stack; "
            "running inline at the default recursion limit. Very large systems "
            "may raise RecursionError."
        )
        return fn()

    if "error" in box:
        raise box["error"]
    return box.get("value")
