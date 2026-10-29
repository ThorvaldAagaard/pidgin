"""The double dummy solver, one solve at a time for the whole process.

DDS keeps its search state in process-wide memory, so two solves from different Python
threads at once corrupt each other. The server's gunicorn worker died that way (SIGSEGV,
SIGABRT, DDS's "Sum 2 is not four") when the Brill API's card search ran while a table
solved its board, each under its own desk's lock. Import the solver from here, never
from endplay.dds directly, and add a wrapper below for any other endplay solve. Some
are hidden: endplay's `Deal.legal_moves()` runs SolveBoard, so use `legal_moves` here.

`LOCK` is the innermost lock: a solve takes nothing else while it holds it, so callers
may hold their own locks around these calls. ImportError without endplay, as before.

Importing this module first also keeps endplay's package __init__ from running: it imports
endplay's dealer, which imports matplotlib, ~45 MB resident that no solve needs.
`endplay.dds` and `endplay.types` import as usual; `import endplay` then has only the
submodules imported so far, so import those by name.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.util
import sys
import threading


def _bare_package(name):
    """Register package `name` without running its __init__, so its submodules import alone."""
    if name in sys.modules:
        return
    spec = importlib.util.find_spec(name)
    if spec is None or spec.submodule_search_locations is None:
        return                                   # not installed: the import below says so
    sys.modules[name] = importlib.util.module_from_spec(spec)


_bare_package("endplay")
from endplay import _dds as _lib   # noqa: E402
from endplay import dds as _dds    # noqa: E402

LOCK = threading.Lock()


def _one_at_a_time(fn):
    @functools.wraps(fn)
    def solve(*args, **kwargs):
        with LOCK:
            return fn(*args, **kwargs)
    return solve


analyse_start = _one_at_a_time(_dds.analyse_start)
calc_all_tables = _one_at_a_time(_dds.calc_all_tables)
calc_dd_table = _one_at_a_time(_dds.calc_dd_table)
solve_all_boards = _one_at_a_time(_dds.solve_all_boards)
solve_board = _one_at_a_time(_dds.solve_board)


def legal_moves(deal):
    """`deal.legal_moves()`, which endplay answers with a SolveBoard call."""
    with LOCK:
        return deal.legal_moves()


def set_memory(max_mb=30, threads=1):
    """Size DDS's transposition tables for `threads` threads in about `max_mb` MB.

    At import DDS sizes them from the free memory it sees, which on Linux it reads off the
    swap line of `free`: the 1 GB droplet with 2 GB swap gets the 160 MB table, and it fills
    as the Brill API plays. Below 123 MB a thread gets the 30 MB table instead. DDS is exact,
    so the answers stay the same; a hard solve can take a little longer. Below 23 MB DDS
    would start no thread at all. Linux only: macOS's DDS loses its threads when this is
    set a second time. True when set."""
    if not sys.platform.startswith("linux"):
        return False
    with LOCK:
        _lib._dll.SetResources(ctypes.c_int(max(int(max_mb), 24)), ctypes.c_int(threads))
    return True
