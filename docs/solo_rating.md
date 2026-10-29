# Solo rating (shadow mode)

A second rating for rated challenges, computed next to the Elo in `server/emergent/players.py`.
It is stored but not shown: no page, leaderboard or API reads it yet. Code: `server/emergent/solo.py`.

## Which boards count

The same boards as the Elo: boards of a challenge the server dealt against the rated team
(Pidgin V3.1: V3.1 bidding, card play V2), one result per player and deal. A board that was
started and left unfinished counts as a forfeit when that player next starts a challenge.
Challenge links, other teams and practice tables are not rated.

## The model

For board `i` the result is `y_i`, the IMPs of the user's table against the other table on the
same deal (the rated team plays all four seats there), from -24 to +24. A forfeit is `y = -7`.

The model: a player has a fixed skill `θ`, in IMPs per board against the rated team, and

    y_i = θ + e_i,      e_i ~ Normal(0, s²)
    θ ~ Normal(0, s² / λ)          (the prior: a new player is as good as the rated team)

After `n` boards with sum `Y = y_1 + … + y_n` the posterior is

    θ | y ~ Normal( Y / (n + λ),  s² / (n + λ) )

Rating points, with 1000 = the rated team and 100 points = 1 IMP per board:

    μ = 1000 + 100 · Y / (n + λ)
    σ = 100 · s / √(n + λ)

One board, from `n` to `n + 1` boards (the same thing, step by step):

    μ_new = μ_old + 100 · (y − (μ_old − 1000) / 100) / (n + 1 + λ)

| parameter | value | meaning |
|---|---|---|
| `s` | 5.0 IMPs | spread of one board's result around the player's skill |
| `λ` | 8 | the prior is worth 8 boards played level with the rated team |
| forfeit | -7 IMPs | the mean of a lost board |
| start | μ 1000, σ 177 | |
| scale | 100 points per IMP per board | display only |

σ after 20 / 100 / 500 boards: 95 / 48 / 22. σ depends only on the number of boards.
The first board moves μ by `100 · y / 9`: +56 for +5 IMPs, -78 for a forfeit.

## Where the numbers come from

The offline experiment is `lab/experiments/ranking/` (README there). It played 26 bots
(the rated team itself, and copies with noise added to their bidding or card play) through
30,000 bench deals each, one bot in one seat and the rated team in the other three, and scored
each board in IMPs against the rated team's own table. Rows 0-20,000 give each bot's true
strength; seasons of 12-board sessions from rows 20,000-28,000 test how well each rating
system finds that order. Team A in that experiment is the site's rated team.

- Model: the margin model, result "IMPs vs bot" (`run_margin`, kind `bot` in `ratings.py`).
  Against the rated team each player's posterior is independent of the others, so the formula
  above is that model exactly.
- `s`: the experiment estimates `s²` from all residuals at each step. The server fixes it. The
  pooled residual SD over 24 bots × 20,000 boards is 4.91 IMPs (team A) and 5.45 (team B, a
  different card net in the rated team). `s = 5.0`.
- `λ`: tuned on future-board accuracy, team A picked 0.5 and team B 8. On team A, heavy activity,
  test seeds 10-59 (50 seasons), Spearman against true strength at 12 / 96 / 960 boards per
  player: λ 0.5 gives 0.738 / 0.889 / 0.970, λ 8 gives 0.737 / 0.889 / 0.970 (±0.003 to
  ±0.028, 95%). Leaderboard churn is the same. λ 8 makes the first boards move μ far less.
  The spread of true strengths over the bots is 1.48 IMPs, close to `s / √λ` = 1.77.
  Log: `lab/runs/ranking/lam_check_teamA.log` (`lam_check.py`).
- Forfeit: in the same 20,000 boards, 31% of boards are lost against the rated team; a lost board
  is -6.95 IMPs on average (median -5, 10th percentile -10). The Elo counts a forfeit as one
  loss; -7 is one average loss here.

Bots on this scale (team A, 20,000 boards each, 1000 + 100 × IMPs per board):

| bot | IMPs per board | points |
|---|---|---|
| the rated team | 0 | 1000 |
| card temperature 0.03 | -0.16 ±0.04 | 984 |
| card temperature 0.07 | -0.56 ±0.05 | 944 |
| card net E48 wideleagueH | -0.84 ±0.05 | 917 |
| card temperature 0.15 | -1.30 ±0.06 | 870 |
| card temperature 0.3 | -2.21 ±0.07 | 779 |
| bidding temperature 8 | -2.23 ±0.09 | 777 |
| card temperature 1 | -3.18 ±0.07 | 682 |
| bidding 8, card 1 | -5.39 ±0.10 | 461 |

## Assumptions

- Skill does not change. Every board counts the same, old or new; there is no decay.
- Board results are independent and normal around the skill. Real IMPs have heavy tails.
- The scale belongs to the rated team. With another card net in the rated team (team B) every
  bot measured 0.72 IMPs per board higher. Changing the rated team changes what the numbers mean.
- The experiment's players were bots with random noise, not people.

Not yet checked offline: players who get better or worse during a season, and human-like errors.

## Storage and replay

`players.db` (the Elo's file):

- `solo`: per player `n`, `sum_imps`, `mu`, `sigma`.
- `solo_history`: one row per rated result, key `(player, deal)` (the same key as the Elo's
  `boards`): the IMPs used, forfeit or not, μ and σ before and after, `n` after, time.

A board is rated in the same transaction as its Elo. A result already in `solo_history` is never
counted again. The Elo's `boards` table holds every rated result, so the Solo rating can be
rebuilt from it at any time:

    cd server && python -m emergent.solo replay      # rebuild solo and solo_history from boards
    cd server && python -m emergent.solo report      # every player: n, μ, σ
    cd server && python -m emergent.solo history ID  # one player's boards, oldest first

The first start with an empty `solo_history` replays existing boards by itself.

## Ordering

`solo.leaderboard()` sorts by `μ − 2σ`, so a player with few boards does not lead on a few lucky
ones, and shows μ. No route uses it yet.
