"""The DMC card player (emergent/playdmc.py): greedy, the trainer's input width, hints from E48.

    python3 -m pytest -q tests/test_playdmc.py

Its play was checked against the trainer's own net on 200 evaluation boards, each with the DMC
net declaring and defending against the Pidgin Q-net: every board took the same tricks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import bidserver, engine, playdesk, playdmc, teams  # noqa: E402

FILE = ROOT / "models" / "play_run6_s36-38k.pt"
pytestmark = pytest.mark.skipif(not FILE.exists(), reason="models/play_run6_s36-38k.pt not synced")


@pytest.fixture(scope="module")
def team():
    bidserver.create_app()
    return teams.get("PidginV3.1")


def board(seed):
    """A fresh 1NT-by-North game dict."""
    owners = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), 13)).astype(np.int64)
    return {"owners": owners, "calls": [4, 35, 35, 35], "dealer": 0, "trump": 4, "declarer": 0,
            "level": 1, "doubled": 0, "vul_ns": False, "vul_ew": True, "played": [], "bench_dd": None}


def test_input_width():
    assert playdmc.FEAT_DIM == 2658
    g = board(1)
    c = playdesk.contracts_of(g)
    assert playdmc.features(c, playdesk.batch_of(g, c)).shape == (1, 2658)


def test_plays_a_board_greedy_whatever_the_search_toggle(team):
    bot = team.play
    assert bot.family == "dmc" and team.id == "PidginV3.1"
    assert all(teams.get(t).play is bot for t in ("PidginV2", "BRL"))      # one net, loaded once
    g = board(7)
    c = playdesk.contracts_of(g)
    while len(g["played"]) < 52:
        b = playdesk.batch_of(g, c)
        card, top = engine.choose_card(bot, c, b, search=True, seed=3)
        assert (card, top) == engine.choose_card(bot, c, b, search=False)
        assert top[0] == (card, 0.0) and all(v <= 0 for _, v in top)
        assert card == int(bot.values(c, b)[0].argmax())
        assert playdesk.play_card(g, card, b)
    assert 0 <= int(playdesk.batch_of(g, c).declarer_tricks()[0]) <= 13


def test_hints_quote_e48_loaded_on_first_hint(team):
    engine.PLAY_MODELS.clear()
    playdesk.load_models()
    assert not engine.PLAY_MODELS.loaded()
    hint = teams.hint_bot(team.play)
    assert hint is engine.PLAY_MODELS["E48_wideleagueH"] and hasattr(hint, "view")
    plain = engine.PLAY_MODELS["E48_leagueE"]
    assert teams.hint_bot(plain) is plain                                    # a policy net quotes itself


def test_slim_keeps_one_core_in_float32():
    sd = torch.load(FILE, map_location="cpu", weights_only=True)
    assert set(sd) >= {"net", "width", "depth", "feat_dim", "step"}
    assert all(v.dtype == torch.float32 for v in sd["net"].values())
    assert sd["net"]["inp.weight"].shape == (sd["width"], 2658) and sd["net"]["role_emb"].shape[0] == 3
