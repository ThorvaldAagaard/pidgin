# Pidgin server

This Flask app serves the playable bridge table, a debug table, and bot APIs. The public site is [bridge.localgeek.jp](https://bridge.localgeek.jp/).

| Path | What it does |
|---|---|
| `/` or `/table` | Play one seat against three bots, with optional hints, challenges, and scoring. |
| `/?mode=bid` | Bidding only: bid one seat against Pidgin V3.1, then see the contract scored double dummy, with no card play. |
| `/card` | The Pidgin V3.1 convention card (`emergent/bidserver_static/card.html`). |
| `/debug` | Inspect a deal with all hands visible; choose seats, dealer, vulnerability, and models. |
| `/bench` | Learn how to compare bidding bots on fixed deals; running a comparison requires the benchmark file. |
| `/apis/brill/` | Discover the stateless Brill Seat Robot API and its available teams. |

The former `/play` page redirects to `/`. The older `/api/*` and `/api/play/*` endpoints remain available for clients, but new integrations should use `/apis/brill/`.

## Run from the repository root

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install -r server/requirements.txt
scripts/get_models.sh
scripts/serve.sh
```

Open <http://127.0.0.1:8787/>. `scripts/get_models.sh` downloads released weights from Hugging Face into `server/models/`; it requires network access. Set `PORT=8080` before `scripts/serve.sh` to use another port. The model classes live in `training/`; the server uses that code when it loads the weights.

For a container, run these commands from the repository root after downloading the weights:

```bash
docker build -f server/Dockerfile -t pidgin-brill .
docker run --rm -p 8080:8080 pidgin-brill
curl http://localhost:8080/apis/brill/
```

## Play and share a board

Open `/` and choose a chair. You bid and play that chair; if you declare, you also play dummy's cards. The other seats use bots: one of the public teams below, Pidgin V3.1 unless you pick another in Settings. The header line names the team. Only challenges against Pidgin V3.1 are rated. The page shows the score against double-dummy par when the board ends. Hints can be switched on during the game. **Copy link** shares the current position, including the auction and cards already played. The link encodes all four hands, so anyone with it can recover the full deal.

Dealer and vulnerability follow the board number as at a duplicate club (board 1: North, nobody; board 2: East, North-South; and so on over 16 boards). Challenges stay at North, nobody. Outside a challenge, **Show all hands** turns the other hands face up and **Skip the play** lets the bots play every remaining card, yours too. When you are dummy you see all four hands once the opening lead is out.

**Bidding only** (`/?mode=bid`) ends the board with the auction: all four hands face up, the contract scored at its double-dummy tricks, and par. The bots are always Pidgin V3.1 with no bidding search. The deals are the first 90,000 rows of `models/bench_100k.npz`, which carry their double-dummy tables, so no board needs a solve. The page sends `X-Mode: bid` with each request. Challenges are not offered there. The header links to the convention card (`/card`) once the server has it.

Use `/debug` to examine a position. It shows all hands and lets you change the dealer, vulnerability, chair, and models. With no chair selected you may act for all four seats; **Net: this call/card** advances one bot move and **Nets play to the end** finishes the board. A short example deals random hands with East as dealer and North-South vulnerable:

```text
http://127.0.0.1:8787/debug#dealer=E&vul=ns&seat=S
```

You can also specify hands as `spades.hearts.diamonds.clubs`, for example `n=AKQJ.T98.765.432`. A full deal must assign 13 distinct cards to each seat. `auction` uses calls such as `1S-P-2C-P`; `play` uses cards such as `H7-HQ-HK-HA`. The server checks that calls and cards are legal. A shared debug link resumes at the stated position when you press **Continue from here**.

The table's search switch controls card-play search. On `/debug`, it also controls bidding search for bot calls. Bidding search considers alternative calls using sampled unseen hands and double-dummy scoring. Its default settings come from the environment variables in `emergent/engine.py`; the most useful are `PLAY_SEARCH=0` to disable card search, `PLAY_SEARCH_SAMPLES=20`, `BID_SEARCH_SAMPLES=32`, and `BID_SEARCH_K=1` to disable bidding search. These are read at server start.

## Use a bot through the Brill API

The API is stateless: each request includes the acting seat's hand, dealer, vulnerability, and auction. Card-play requests also include the played cards and dummy's hand once dummy is visible. A successful bid request returns JSON with `bid`, `candidates`, `explanation`, and `model`; a play request returns `card`, `candidates`, and `model`. Invalid positions return HTTP 400 with an `error` field.

The `model` parameter selects a complete team for `/bid`, `/lead`, and `/play`:

| Public team ID | Bidding | Card play |
|---|---|---|
| `PidginV1` | Pidgin V1 bidding model | Pidgin V1 card-play model with search |
| `PidginV2` | Pidgin V2 bidding model with search | Pidgin card play V2, no search |
| `PidginV3.1` | Pidgin V3.1 bidding model, no bidding search | Pidgin card play V2, no search |
| `BRL` | External BRL bidding baseline | Pidgin card play V2, no search |

The table shows the `PidginV3.1` team as **Pidgin**, with "Bidding: V3.1 (VieNNa system) · Card play: V2"
as its details (`name` and `label` in `models/teams.json`); links and the APIs keep the ID.

These IDs are listed at `/apis/brill/`. The internal weight filenames are recorded in `models/teams.json` and described in [Serving](../docs/serving.md). A Brill or BBO request without `model` is answered by the `PidginV3.2` team, bidding and card play (`BRILL_DEFAULT_MODEL` picks another team). The BBO endpoint `/apis/bbo.php` takes `model=<team>` too. A checkpoint `model`, or a `play_model` without `model`, still selects single nets. Specify `model` when comparing teams so the result is unambiguous.

For example, ask Pidgin V1 to bid from North on a new auction:

```bash
curl -G 'http://127.0.0.1:8787/apis/brill/bid' \
  --data-urlencode 'model=PidginV1' \
  --data-urlencode 'seat=N' --data-urlencode 'dealer=N' \
  --data-urlencode 'vul=None' --data-urlencode 'ctx=' \
  --data-urlencode 'hand=AKQJ.T98.765.432'
```

`ctx` contains two-character calls from the dealer (`1S--2H` means 1♠, pass, 2♥); the API also accepts dash-separated calls such as `1S-P-2H`. Card-play requests use `/apis/brill/lead` before the opening lead and `/apis/brill/play` afterward; `played` is a string of suit-rank pairs such as `H7HQ`. See the [Brill Seat Robot API specification](https://brill.aalborgdata.dk/seat-api.html) for the full request format. The BBO-compatible endpoint is `/apis/bbo.php`; see `emergent/apis.py` for its accepted parameters.

## Benchmark a bidding bot

When `server/models/bench_100k.npz` is present, `/bench` serves a fixed deal set and reports results from two auctions per deal: your bot at North-South and at East-West against the same opponent. Card play is scored with a double-dummy solver. The server exposes the full protocol at `/apis/bench/agent.md` and `/apis/bench/openapi.json`.

```bash
curl 'http://127.0.0.1:8787/apis/bench/deals?offset=0&limit=10'
python server/examples/bench_client.py --url http://127.0.0.1:8787 \
  --bot D_cw_s75k --opp brl_fsp --boards 1000
```

The example client currently accepts checkpoint IDs, so `D_cw_s75k` is the file-backed ID for the Pidgin V1 bidding model. It downloads the deals, runs both auctions with local models, then submits the report. Replace its `bid()` function to use your own bidding bot. Install the benchmark file before requesting deals; that endpoint needs it to respond.

## Models and deployment

`models/teams.json` maps public team IDs to weights and search choices. `models/models.json` and `models/play_models.json` supply the debug table's model menus; entries with missing files are skipped. The first available entry is the default for its menu. Weight filenames and checkpoint IDs stay as they are because the manifests and APIs use them; use public team names when describing behavior to readers.

`server/deploy.sh` ships the app to a configured Linux host and restarts its systemd service. Set `DEPLOY_HOST=user@host` or put it in the ignored `server/.deploy.env`, then run `server/deploy.sh` from the repository root. It requires every manifest weight and the bidding belief model. The host's nginx and systemd configuration lives outside this repository.

For implementation details, start with `emergent/tabledesk.py` (table), `emergent/apis.py` (bot APIs), `emergent/bench.py` (benchmark), and `emergent/engine.py` (model choices and search).
