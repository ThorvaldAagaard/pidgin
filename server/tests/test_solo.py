"""The shadow Solo rating: start, results, forfeits, repeats, uncertainty, replay.

    python3 -m pytest -q tests/test_solo.py
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import players, solo  # noqa: E402

A, B = "a" * 32, "b" * 32


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(players, "DATA_DIR", tmp_path)
    monkeypatch.setattr(players._LOCAL, "conn", None, raising=False)
    yield


def play(pid, deal, imps):
    assert players.board_started(pid, deal)
    return players.board_finished(pid, deal, imps)


def sr(pid):
    return solo.rating(players.db(), pid)


def test_a_new_player_starts_level_with_the_rated_team():
    r = sr(A)
    assert r == {"mu": 1000, "sigma": pytest.approx(100 * 5 / math.sqrt(8)), "boards": 0}
    assert players.me(A)["rating"] == players.START_RATING    # the Elo is untouched


def test_plus_and_minus_imps_move_mu_up_and_down():
    play(A, "d1", 5)
    assert sr(A)["mu"] == pytest.approx(1000 + 100 * 5 / 9)
    play(B, "d1", -5)
    assert sr(B)["mu"] == pytest.approx(1000 - 100 * 5 / 9)
    play(A, "d2", 0)                                          # a push pulls a lucky start back
    assert sr(A)["mu"] == pytest.approx(1000 + 100 * 5 / 10)


def test_a_repeated_submission_counts_once():
    play(A, "d1", 7)
    before = sr(A)
    assert players.board_finished(A, "d1", 7) is None
    assert not players.board_started(A, "d1")
    # even called straight, the same (player, deal) does not count again
    assert solo.rate(players.db(), A, "d1", 7) is None
    assert sr(A) == before and sr(A)["boards"] == 1
    assert len(solo.history(players.db(), A)) == 1


def test_a_board_left_unfinished_is_a_forfeit():
    players.board_started(A, "d1")
    players.forfeit_open(A)
    r = sr(A)
    assert r["boards"] == 1 and r["mu"] == pytest.approx(1000 + 100 * solo.FORFEIT_IMPS / 9)
    h = solo.history(players.db(), A)
    assert h[0]["forfeit"] == 1 and h[0]["imps"] == solo.FORFEIT_IMPS
    assert players.board_finished(A, "d1", 10) is None        # too late to finish it
    assert sr(A) == r


def test_uncertainty_shrinks_with_every_board_whatever_the_result():
    sigmas = [sr(A)["sigma"]]
    for k, y in enumerate([12, -12, 0, 3, -24, 24, 1, -1]):
        play(A, f"d{k}", y)
        sigmas.append(sr(A)["sigma"])
    assert all(b < a for a, b in zip(sigmas, sigmas[1:]))
    assert sigmas[-1] == pytest.approx(100 * 5 / math.sqrt(8 + 8))


def test_a_steady_player_settles_near_their_skill():
    # +1 IMP a board for 500 boards: μ goes to 1100, and later boards move it little
    for k in range(500):
        play(A, f"d{k}", 1)
    r = sr(A)
    assert r["mu"] == pytest.approx(1000 + 100 * 500 / 508)
    assert r["sigma"] < 25
    play(A, "big", -24)                                       # one disaster costs under 5 points
    assert r["mu"] - sr(A)["mu"] < 5


def test_replay_rebuilds_the_same_ratings_from_the_boards():
    play(A, "d1", 4)
    play(B, "d1", -3)
    play(A, "d2", -9)
    players.board_started(B, "d2")
    players.forfeit_open(B)
    conn = players.db()
    want = {p: sr(p) for p in (A, B)}
    hist = {p: solo.history(conn, p) for p in (A, B)}
    conn.execute("UPDATE solo SET mu = 0")                    # damage it
    assert solo.replay(conn) == 4
    assert {p: sr(p) for p in (A, B)} == want
    assert {p: solo.history(conn, p) for p in (A, B)} == hist


def test_an_existing_database_is_backfilled_on_first_start(monkeypatch):
    play(A, "d1", 6)
    play(A, "d2", -2)
    conn = players.db()
    conn.execute("DROP TABLE solo")
    conn.execute("DROP TABLE solo_history")                   # as before this code existed
    monkeypatch.setattr(players._LOCAL, "conn", None, raising=False)
    r = sr(A)
    assert r["boards"] == 2 and r["mu"] == pytest.approx(1000 + 100 * 4 / 10)


def test_the_leaderboard_orders_by_mu_minus_two_sigma():
    play(A, "d1", 20)                                         # one lucky board: high μ, wide σ
    for k in range(200):
        play(B, f"d{k}", 1)                                    # steady and well known
    players.set_name(A, "Lucky")
    players.set_name(B, "Steady")
    assert sr(A)["mu"] > sr(B)["mu"]
    assert [r["name"] for r in solo.leaderboard(players.db())] == ["Steady", "Lucky"]
