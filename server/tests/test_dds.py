"""Every double dummy solve goes through training.bridge.dds, one at a time.

Two solves at once from different threads crashed the gunicorn worker (SIGSEGV).

    python3 -m pytest -q tests/test_dds.py
"""

import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from endplay.types import Deal  # noqa: E402

from training.bridge import dds  # noqa: E402

DEAL = "N:AKQJ.T98.765.432 T98.765.432.AKQJ 765.432.AKQJ.T98 432.AKQJ.T98.765"
# endplay.dds directly, or Deal.legal_moves (it runs SolveBoard), bypass the lock.
UNLOCKED = re.compile(r"from endplay\.dds import|import endplay\.dds|endplay\.dds\.|\.legal_moves\(\)")


def test_server_code_solves_only_through_the_lock():
    files = [*(ROOT / "emergent").rglob("*.py"), *(ROOT.parent / "training").rglob("*.py")]
    offenders = [f"{p.relative_to(ROOT.parent)}:{n}"
                 for p in files if p.resolve() != Path(dds.__file__).resolve()
                 for n, line in enumerate(p.read_text().splitlines(), 1)
                 if UNLOCKED.search(line.split("#")[0])]
    assert offenders == []


def test_a_solve_waits_for_the_lock():
    done = threading.Event()

    def solve():
        dds.calc_dd_table(Deal(DEAL))
        done.set()

    with dds.LOCK:
        worker = threading.Thread(target=solve)
        worker.start()
        assert not done.wait(0.3)
    worker.join(30)
    assert done.is_set()


def test_legal_moves_waits_for_the_lock():
    deal = Deal(DEAL)
    out = []
    with dds.LOCK:
        worker = threading.Thread(target=lambda: out.append(dds.legal_moves(deal)))
        worker.start()
        worker.join(0.3)
        assert out == []
    worker.join(30)
    assert len(out[0]) > 0
