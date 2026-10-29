"""The /articles/ pages: the write-up is served, nothing else in the folder tree is."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emergent import bidserver  # noqa: E402


@pytest.fixture(scope="module")
def client():
    bidserver.create_app()
    return bidserver.app.test_client()


def test_article_is_served(client):
    r = client.get("/articles/bridgewinners.html")
    assert r.status_code == 200 and b"VieNNa" in r.data


@pytest.mark.parametrize("url", ["/articles/nothing.html", "/articles/../table.html", "/articles/%2e%2e/table.html"])
def test_other_names_are_404(client, url):
    assert client.get(url).status_code == 404
