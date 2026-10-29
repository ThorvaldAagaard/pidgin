"""/table practice: rotating dealer and vulnerability, show all hands, skip the play,
every hand face up for a user who is dummy; bidding practice (/?mode=bid); /card.

    python3 -m pytest -q tests/test_table_practice.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import bidserver, tabledesk  # noqa: E402

PASS = 35
HANDS = {"n": "AQJ32.Q8.T32.K83", "e": "T54.KT63.875.AJ9",
         "s": "K987.AJ.AKQJ.QT4", "w": "6.97542.964.7652"}


@pytest.fixture(scope="module")
def client():
    bidserver.create_app()
    return bidserver.app.test_client()


def test_board_cycle():
    assert [tabledesk.board_dealer_vul(n)[0] for n in (1, 2, 3, 4, 5, 17)] == [0, 1, 2, 3, 0, 0]
    v = {n: tabledesk.board_dealer_vul(n)[1] for n in (1, 2, 3, 4, 8, 13, 16, 17)}
    assert v == {1: (False, False), 2: (True, False), 3: (False, True), 4: (True, True),
                 8: (False, False), 13: (True, True), 16: (False, True), 17: (False, False)}


def test_new_board_rotates(client):
    h = {"X-Game": "prac-rot"}
    d = client.get("/api/table/state", headers=h).json
    assert d["board_no"] == 1 and d["dealer"] == 0 and d["vul"] == "none"
    d = client.post("/api/table/new_board", json={}, headers=h).json
    assert d["board_no"] == 2 and d["dealer"] == 1 and d["vul"] == "ns"
    d = client.post("/api/table/new_board", json={}, headers=h).json
    assert d["board_no"] == 3 and d["dealer"] == 2 and d["vul"] == "ew"


def test_reveal_and_skip_the_play(client):
    h = {"X-Game": "prac-rev"}
    d = client.post("/api/table/new_board", json={}, headers=h).json
    assert sum(s["hidden"] for s in d["seats"]) == 3
    d = client.post("/api/table/reveal", json={"reveal": True}, headers=h).json
    assert d["reveal"] and not any(s["hidden"] for s in d["seats"])
    d = client.post("/api/table/reveal", json={"reveal": False}, headers=h).json
    assert not d["reveal"] and sum(s["hidden"] for s in d["seats"]) == 3
    # Bid to a contract: the user passes, the nets bid; a passed-out board is redealt.
    for _ in range(20):
        while d["phase"] == "auction":
            d = client.post("/api/table/call", json={"call": PASS}, headers=h).json
        if d["phase"] == "play":
            break
        d = client.post("/api/table/new_board", json={}, headers=h).json
    assert d["phase"] == "play"
    d = client.post("/api/table/finish", json={"all": True}, headers=h).json
    assert d["phase"] == "over" and d["result"] is not None


def test_dummy_sees_every_hand(client):
    # South opens 1NT, all pass: North (the user) is dummy.
    h = {"X-Game": "prac-dummy"}
    d = client.post("/api/table/debug", json={**HANDS, "dealer": "S", "auction": "1NT-P-P-P",
                                              "play": "", "seat": "N"}, headers=h).json
    g = tabledesk.GAMES["prac-dummy"]
    g["debug"] = False                         # the same position, as on /
    d = client.get("/api/table/state", headers=h).json
    assert d["role"] == "dummy" and sum(s["hidden"] for s in d["seats"]) == 3
    d = client.post("/api/table/advance", json={}, headers=h).json
    assert not any(s["hidden"] for s in d["seats"])


# ------------------------------------------------------------ bidding practice (/?mode=bid)

def bid_to_contract(client, h):
    """Pass every turn; a passed-out board is dealt again."""
    d = client.post("/api/table/new_board", json={}, headers=h).json
    for _ in range(20):
        while d["phase"] == "auction":
            d = client.post("/api/table/call", json={"call": PASS}, headers=h).json
        if d["contract"]:
            return d
        d = client.post("/api/table/new_board", json={}, headers=h).json
    raise AssertionError("twenty boards passed out")


def test_bid_only_ends_with_the_auction(client, monkeypatch):
    def never(*a, **k):
        raise AssertionError("bidding practice played a card, searched or solved")
    # No card play, no bidding search, and no solve: the deal comes with its table.
    monkeypatch.setattr(tabledesk.playdesk, "net_card", never)
    monkeypatch.setattr(tabledesk.engine, "search_call", never)
    monkeypatch.setattr(tabledesk.playdesk, "dd_table", never)
    h = {"X-Game": "prac-bid", "X-Mode": "bid"}
    d = bid_to_contract(client, h)
    assert d["bid_only"] and d["phase"] == "over" and not d["tricks"]
    assert not any(s["hidden"] for s in d["seats"])
    assert "no card play" in d["meta"]
    assert d["team"] == tabledesk.default_team()      # the site's default team (Pidgin V3.1)
    g, r, ct = tabledesk.GAMES["prac-bid"], d["result"], d["contract"]
    assert r["bid_only"] and r["tricks"] == r["dd_tricks"] == g["tricks"][ct["declarer"]][ct["trump"]]
    assert r["par"] is not None and r["imps"] is not None
    # Finish does nothing more; bid it again keeps the cards and their table.
    d = client.post("/api/table/finish", json={"all": True}, headers=h).json
    assert d["phase"] == "over" and not d["tricks"]
    table = g["tricks"].copy()
    d = client.post("/api/table/restart", json={}, headers=h).json
    assert d["phase"] == "auction" and d["bid_only"] and (g["tricks"] == table).all()
    r = client.post("/api/table/challenge/job", json={}, headers=h)
    assert r.status_code == 400


def test_bid_only_keeps_the_default_team(client):
    h = {"X-Game": "prac-bid-team", "X-Mode": "bid"}
    others = [t for t in tabledesk.table_teams() if t != tabledesk.default_team()]
    if not others:
        pytest.skip("one team on disk")
    d = client.post("/api/table/new_board", json={"team": others[0]}, headers=h).json
    assert d["bid_only"] and d["team"] == tabledesk.default_team()


def test_bid_only_deals_are_solved(client):
    owners, tricks = tabledesk.practice_deal(np.random.default_rng(1))
    assert sorted(np.bincount(owners, minlength=4)) == [13] * 4
    assert (tricks == tabledesk.playdesk.dd_table(owners)).all()


def test_the_same_game_without_the_mode_plays_on(client):
    h = {"X-Game": "prac-bid-2", "X-Mode": "bid"}
    d = bid_to_contract(client, h)
    assert d["phase"] == "over"
    d = client.get("/api/table/state", headers={"X-Game": "prac-bid-2"}).json
    assert not d["bid_only"] and d["phase"] == "play"


def test_card_page(client, monkeypatch, tmp_path):
    monkeypatch.setattr(tabledesk, "STATIC_DIR", str(tmp_path))
    assert client.get("/card").status_code == 404
    (tmp_path / "card.html").write_text("<p>card</p>")
    r = client.get("/card")
    assert r.status_code == 200 and b"card" in r.data
