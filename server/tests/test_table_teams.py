"""The table's bots are one deployed team (models/teams.json), picked in Settings.

    python3 -m pytest -q tests/test_table_teams.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import bidserver, engine, teams  # noqa: E402
from tests.test_table_challenge import finish_board  # noqa: E402

td = bidserver.tabledesk
IDS = ["PidginV3.1", "PidginV2", "PidginV1", "BRL"]          # as Settings lists them


@pytest.fixture(scope="module")
def client():
    bidserver.create_app()
    return bidserver.app.test_client()


def post(client, gid, path, **body):
    return client.post(path, json=body, headers={"X-Game": gid})


def test_a_fresh_tab_plays_pidgin_v3_1(client):
    d = client.get("/api/table/state", headers={"X-Game": "team-fresh"}).json
    assert d["team"] == "PidginV3.1" and d["team_name"] == "Pidgin"          # the id stays in links
    assert d["meta"] == "Bots: Pidgin · Bidding: V3.1 (VieNNa system) · Card play: V2"
    assert [t["id"] for t in d["teams"]] == IDS
    assert [t["name"] for t in d["teams"]] == ["Pidgin", "Pidgin V2", "Pidgin V1", "BRL"]
    assert d["teams"][0]["label"] == "Bidding: V3.1 (VieNNa system) · Card play: V2"
    assert d["code"]["t"] == "PidginV3.1" and d["code"]["m"] == "" and d["code"]["pm"] == ""
    page = client.get("/").data.decode()
    assert 'id="teams"' in page and 'if(c.t) q.set("t", c.t);' in page


@pytest.mark.parametrize("team", IDS)
def test_the_team_bids_and_plays(client, monkeypatch, team):
    gid = f"team-{team}"
    d = post(client, gid, "/api/table/new_board", team=team, seat=2).json
    assert d["team"] == team and d["meta"].startswith(f"Bots: {td.team_name(team)}")
    t = teams.TEAMS[team]
    calls, cards, searched = [], [], []
    real_call, real_search, real_card = engine.choose_call, engine.search_call, engine.choose_card
    monkeypatch.setattr(engine, "choose_call", lambda bot, *a, **k: calls.append(bot) or real_call(bot, *a, **k))
    monkeypatch.setattr(engine, "search_call", lambda bot, *a, **k: searched.append(bot) or real_search(bot, *a, **k))
    monkeypatch.setattr(engine, "choose_card",
                        lambda bot, c, b, search=None, seed=None: cards.append((bot, search)) or real_card(bot, c, b, search))
    # Open at 1NT so there is always card play: the bots answer, then the board runs out.
    d = post(client, gid, "/api/table/new_board", seat=0).json      # North deals: the user opens
    assert d["team"] == team                                         # a new board keeps the team
    post(client, gid, "/api/table/hints", search=False)              # quick; search on: below
    post(client, gid, "/api/table/call", call=4)                     # 1NT
    d = finish_board(client, {"X-Game": gid})
    assert d["phase"] == "over"
    bots_called = searched if t.bid_search else calls
    assert bots_called and all(b is t.bid for b in bots_called)
    assert bool(searched) == t.bid_search                            # only Pidgin V2 bids with search
    if d["contract"] is not None:
        assert cards and all(b is t.play for b, _ in cards) and all(s is False for _, s in cards)
    if team != "PidginV1":
        assert t.play.family == "dmc"                                # the DMC value net


def test_the_search_toggle_reaches_the_card_player(client, monkeypatch):
    """The toggle is handed over; the DMC net plays greedy either way (tests/test_playdmc.py)."""
    gid = "team-search"
    post(client, gid, "/api/table/new_board", team="PidginV3.1", seat=0)
    post(client, gid, "/api/table/hints", search=True)
    seen = []
    real = teams.TEAMS["PidginV3.1"].play.choose_card
    monkeypatch.setattr(teams.TEAMS["PidginV3.1"].play, "choose_card",
                        lambda c, b, search=None, seed=None: seen.append(search) or real(c, b, False))
    post(client, gid, "/api/table/call", call=4)
    post(client, gid, "/api/table/advance")
    d = client.get("/api/table/state", headers={"X-Game": gid}).json
    if d["phase"] == "play" and not d["your_turn"]:
        post(client, gid, "/api/table/advance")
    if d["contract"] is not None:
        assert seen and all(s is True for s in seen)


def test_links(client):
    d = post(client, "team-link", "/api/table/new_board", team="BRL").json
    k = d["code"]
    body = {"deal": k["d"], "seat": k["s"], "team": k["t"]}
    f = post(client, "team-link-2", "/api/table/load", **body).json
    assert f["team"] == "BRL"
    # An old link names its nets with m/pm and opens exactly those, with no team.
    old = post(client, "team-link-3", "/api/table/load", deal=k["d"], seat="S",
               model="D_cw_s75k", play_model="E48_leagueE").json
    assert old["team"] is None and (old["model"], old["play_model"]) == ("D_cw_s75k", "E48_leagueE")
    assert old["code"]["m"] == "D_cw_s75k" and old["code"]["t"] == ""
    assert old["meta"].startswith("bidding: Pidgin V1 bidding")
    r = post(client, "team-link-4", "/api/table/load", deal=k["d"], seat="S", team="nope")
    assert r.status_code == 400 and "not on this server" in r.json["error"]
    # An unknown team in Settings keeps the tab's own.
    assert post(client, "team-link", "/api/table/new_board", team="nope").json["team"] == "BRL"


def test_picking_a_net_leaves_the_team(client):
    post(client, "team-menu", "/api/table/new_board", team="PidginV2")
    d = post(client, "team-menu", "/api/table/models", play_model="E48_leagueE").json
    assert d["team"] is None and d["play_model"] == "E48_leagueE"
    assert d["model"] in engine.BID_MODELS


def test_debug_has_no_team(client):
    d = client.post("/api/table/debug", json={}, headers={"X-Game": "team-dbg"}).json
    assert d["debug"] and d["team"] is None and d["model"] == engine.default_bid_model()


def test_only_the_rated_team_is_rated(client):
    assert td.rated({"team": "PidginV3.1"})
    assert not td.rated({"team": "PidginV2"}) and not td.rated({"team": None})
    h = {"X-Game": "team-chal"}
    post(client, "team-chal", "/api/table/new_board", team="BRL")
    post(client, "team-chal", "/api/table/hints", search=False)
    d = client.post("/api/table/challenge", json={"boards": 1}, headers=h).json
    ch = d["challenge"]
    assert d["team"] == "BRL" and not ch["rated"] and ch["team_name"] == "BRL"
    assert ch["code"]["t"] == "BRL" and ch["code"]["m"] == ""
    d = finish_board(client, h)
    k = d["challenge"]["code"]
    f = client.post("/api/table/challenge/load", headers={"X-Game": "team-chal-2"},
                    json={"deals": k["c"], "auctions": k["ca"], "played": k["cp"], "team": k["t"],
                          "search": k["r"]}).json
    assert f["team"] == "BRL" and f["challenge"]["code"] == k
