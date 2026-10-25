"""The double dummy solver, one solve at a time for the whole process.

DDS keeps its search state in process-wide memory, so two solves from different Python
threads at once corrupt each other. The server's gunicorn worker died that way (SIGSEGV,
SIGABRT, DDS's "Sum 2 is not four") when the Brill API's card search ran while a table
solved its board, each under its own desk's lock. Import the solver from here, never
from endplay.dds directly, and add a wrapper below for any other endplay solve. Some
are hidden: endplay's `Deal.legal_moves()` runs SolveBoard, so use `legal_moves` here.

`LOCK` is the innermost lock: a solve takes nothing else while it holds it, so callers
may hold their own locks around these calls. ImportError without endplay, as before.
"""

from __future__ import annotations

import functools
import threading

from endplay import dds as _dds

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
