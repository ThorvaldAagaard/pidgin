"""The Brill API's teams (models/teams.json): PidginV1, BRL, PidginV2, PidginV3.1 by `model=<id>`.

    python3 -m pytest -q tests/test_teams.py
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import bidserver, teams  # noqa: E402
from tests.test_apis import deal, hand_text, play_board  # noqa: E402

IDS = ["PidginV1", "BRL", "PidginV2", "PidginV3.1"]


class WithModel:
    """A test client that adds `model=<id>` to every Brill request."""

    def __init__(self, client, model):
        self.client, self.model = client, model

    def get(self, path, query_string):
        return self.client.get(path, query_string={**query_string, "model": self.model})


@pytest.fixture(scope="module")
def client():
    bidserver.create_app()
    return bidserver.app.test_client()


def test_teams_load_on_first_use(client):
    q = {"seat": "N", "dealer": "N", "vul": "None", "ctx": "", "hand": hand_text(deal(3)[0])}
    teams.TEAMS.clear()
    assert client.get("/apis/brill/bid", query_string={**q, "model": "BRL"}).json["model"] == "BRL"
    assert list(teams.TEAMS) == ["BRL"]
    assert client.get("/apis/brill/bid", query_string=q).json["model"] == teams.DEFAULT_TEAM   # no model: default
    assert list(teams.TEAMS) == ["BRL", teams.DEFAULT_TEAM]


def test_bbo_uses_the_default_team(client):
    """robot.php names no model: the default team bids and plays, card search and all."""
    h = deal(3)
    q = {"pov": "N", "d": "n", "v": "-", "n": hand_text(h[0]), "h": ""}
    r = client.get("/apis/bbo.php", query_string=q).data.decode()
    assert f'model="{teams.DEFAULT_TEAM}"' in r and '<r type="bid"' in r
    q = {**q, "h": "1c-p-p-p", "pov": "E", "e": hand_text(h[1])}   # North declares 1C; East leads
    r = client.get("/apis/bbo.php", query_string=q).data.decode()
    assert f'model="{teams.DEFAULT_TEAM}"' in r and '<r type="play"' in r


def test_root_lists_the_teams(client):
    assert client.get("/apis/brill/").json["models"] == IDS


@pytest.mark.parametrize("model", IDS)
def test_full_board(client, model):
    t0 = time.time()
    ctx, played = play_board(WithModel(client, model), deal(7), 0, "None")
    print(f"{model}: {ctx} {time.time() - t0:.1f}s")
    assert played is None or len(played) == 52


def test_model_id_works_too(client):
    q = {"seat": "N", "dealer": "N", "vul": "None", "ctx": "", "hand": hand_text(deal(3)[0])}
    for m in IDS:
        assert client.get("/apis/brill/bid", query_string={**q, "model_id": m}).json["model"] == m
    assert client.get("/apis/brill/bid", query_string={**q, "model_id": "nope"}).status_code == 400


def test_team_bid_search_is_deterministic(client):
    q = {"seat": "N", "dealer": "N", "vul": "None", "ctx": "", "hand": hand_text(deal(11)[0]), "model": "PidginV2"}
    assert teams.TEAMS["PidginV2"].bid_search
    assert client.get("/apis/brill/bid", query_string=q).json == client.get("/apis/brill/bid", query_string=q).json
