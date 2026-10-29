# Card play

The released teams use three kinds of card-play model. A **policy model** (Pidgin V1
card play) picks a card directly and learns through self-play. **Pidgin card play V2**
estimates the value of each legal card and learns it through self-play too, by deep Monte Carlo
(inspired by DouZero), with no solver in training; it plays its best value without search.
**Pidgin card play V3** is the same kind of network, trained further with the same recipe.
The earlier **Q-net** estimates each legal card's value from double-dummy solver results.
Pidgin V1 card play and the Q-net use search on the site: they sample possible hidden deals,
solve them, and pick the card with the best average result. See `training/play/search.py`
and `server/emergent/engine.py`.

| Public name | Released filename | Used by |
|---|---|---|
| Pidgin card play V3 | `play_run18_s4000.pt` | Pidgin V3.2 (team `PidginV3.2`, the default team). |
| Pidgin card play V2 | `play_run6_s36-38k.pt` | Pidgin V3.1, Pidgin V2 and BRL. |
| Pidgin V1 card play | `play_E48_wideleagueH.pt` | Pidgin V1; the table's play hints for card play V2 and V3. |
| Compact self-play card-play model | `play_E48_leagueE.pt` | Alternative in the site's debug view. |
| Pidgin Q-net card play | `play_B2g_s540k.pt` | No team since card play V2; the reference card play V2 is measured against. |

Card play V2 and V3 answer with one forward pass: its best legal value, nothing searched.
For Pidgin V1 card play with search enabled for the current move, card selection follows
this path:

```mermaid
flowchart TD
    view["Visible cards, auction, contract and cards played"] --> model["Policy model or Q-net: rank legal cards"]
    view --> sampler["Card-play belief head: sample possible hidden deals"]
    sampler --> solver["Double-dummy solver: value legal cards on each deal"]
    solver --> choice["Choose the card with the highest average trick value"]
    model -->|Break ties| choice
```

The Q-net used the Pidgin V1 card-play model's belief head to sample deals.
This is separate from the [bidding belief model](belief.md). When search is off
or does not apply to a move, the card-play model chooses directly. Search works
with guessed hidden deals, not the opponents' actual hands.

## What the models see

Both models describe the position from the player making the decision. They see
their own cards, dummy after the opening lead, the cards already played, the
contract, vulnerability, and auction. Declarer chooses cards for dummy too.
Neither model receives the hidden hands when it makes a move.

| Model | Position encoding and network | Outputs |
|---|---|---|
| Pidgin V1 policy model | 740 numbers: card ownership and play history, contract and score state, calls made by each seat, and a 96-number reading of the ordered auction. A three-layer fully connected network uses width 1,024 in the released V1 model (512 in the compact model). The auction reader embeds each call and its caller, then runs a GRU once per deal. | A probability for each of 52 cards, with illegal cards masked out; and 52 × 4 scores for which seat holds each card. |
| Pidgin card play V2 | 2,658 numbers, seats counted from declarer: calls made by each seat, the first 24 calls in order with who made each, dealer, contract and vulnerability; then the hand on turn, the other hand it sees, cards played, led and in the current trick by each seat, trick number and position, tricks won by each side, and for every card how many higher cards of its suit are in each place (own hand, seen hand, unseen, current trick) and whether it beats the card now winning the trick. One residual network, width 1,024, six blocks, serves declarer (with dummy) and both defenders; a learned role vector is added after its first layer. | One value per card, illegal cards masked: the acting side's expected final tricks, relative to the deal's double-dummy result. |
| Pidgin Q-net | 899 numbers: visible and played cards, trick position, contract and score state, calls made by each seat, the last six calls, and dealer position. The released checkpoint has a width-1,024, three-block residual network. | One predicted trick-value difference for each of 52 cards. Auxiliary heads output 52 × 4 card-owner scores, 4 × 4 × 14 suit-length scores, and 4 × 38 high-card-point scores; serving uses its card values. |

The policy trainer also has a **critic** with a separate 384-number input that
includes all four hands. It estimates tricks the side on turn will still win and
reduces noise in the policy update. The critic is never used when playing a board.
The Q-net has no critic. Its optional `--belief` mode appends 171 features from a
separate bidding belief model; the released Q-net has the standard 899 inputs and
does not use that mode. See
[`training/play/model.py`](../training/play/model.py),
[`qnet/train_play.py`](../qnet/train_play.py), and
[`server/emergent/playq.py`](../server/emergent/playq.py).

```mermaid
flowchart TD
    policy_in["Policy input: 740 values, including the encoded auction"] --> policy["Three dense layers, width 1,024"]
    policy --> card["52 card probabilities"]
    policy --> hidden["52 × 4 card-holder scores"]
    q_in["Q-net input: 899 values"] --> q["Projection and three residual blocks, width 1,024"]
    q --> values["52 card values"]
    q --> auxiliary["Card-holder, suit-length and point predictions"]
```

## How the policy model learns

The policy trainer plays complete deals against itself or frozen earlier
snapshots. After each deal, it counts how many tricks the acting side won from
each decision onward. That return trains the card policy. The known dealt hands
provide labels for the hidden-card head; the double-dummy table is used for
evaluation only.

For each learner decision, the loss in
[`training/play/train.py`](../training/play/train.py) is:

```text
policy = -mean(normalized_advantage × log_probability(chosen_card))
critic = mean((predicted_remaining_tricks - actual_remaining_tricks)²)
belief = cross_entropy(hidden_card_owner, over unseen and unplayed cards)
total = policy + 0.5 × critic + belief_weight × belief - 0.01 × policy_entropy
```

The advantage is actual remaining tricks minus the critic's prediction, then
standardized across learner decisions in the batch. Only decisions made by the
learning partnership enter these losses when an older snapshot plays the other
side. The default belief weight is 1.0; `--belief-final` can lower it linearly
over the first half of the run (the V1 league example below ends at 0.1).
With `--group K`, the trainer instead plays each deal K times and compares each
result with the mean of its other copies; the critic loss is then zero.

```text
load deals, auctions, contracts, and held-out deals
for each training step:
    sample deals; optionally choose a frozen opponent
    play all 52 cards, sampling legal cards from each side's model
    for each learner move, compute remaining tricks and hidden-card labels
    estimate advantage with the critic (or other copies of the same deal)
    update the policy, critic, and hidden-card head with the total loss
    periodically evaluate on held-out deals and save a checkpoint
```

## How card play V2 learns

Card play V2 learns by deep Monte Carlo self-play. Pidgin V3.1 bids the deals (all
four seats, greedy); the net then plays 2,048 boards at a time against itself, with 2% random
legal cards. On 30% of boards one side is a past snapshot (a champion snapshot on 15% of those
rounds). After each board, every decision's target is the acting side's result: declarer's
tricks minus the deal's double-dummy trick count (a fixed number per board, read from the
dataset, that only centres the target; declarer's side takes it, defenders its negative),
blended with the net's own later estimates by TD(λ = 0.8). The loss is the squared error
of the chosen card's value, plus 0.1 × an auxiliary "this card wins the trick" head and
0.05 × a hidden-card-holder head; neither head is served. No solver runs in training,
and the net never sees hidden hands.

```text
for each step:
    take 2,048 deals with their Pidgin V3.1 auctions
    play them out: the net (or a past snapshot for one side) picks the best value, 2% random cards
    target per decision: TD(0.8) toward tricks - double-dummy tricks, from the acting side
    two shuffled passes of squared error on the chosen cards, plus the auxiliary heads
    keep an exponential moving average of the weights (0.995); it acts and is saved
```

The released `play_run6_s36-38k.pt` is the weight average of five snapshots of one run,
steps 36,000 to 38,000 (about 74 million boards). The trainer's checkpoint holds three copies
of the shared network, one per role; the serving file keeps one, in float32 (62 MB). Float16
changed the tricks on 2 of 400 check boards, so it is not used. `server/emergent/playdmc.py`
is a copy of the trainer's input code; on 200 evaluation boards, each played twice against
the Q-net (card play V2 declaring, then defending), the server and the trainer took the same
tricks on every board.

## Card play V3

Card play V3 (`play_run18_s4000.pt`) has the same input, network and serving format as card
play V2: one shared network, width 1,024, six blocks, 2,658 inputs, float32 (62 MB).
`server/emergent/playdmc.py` loads it unchanged.

It was trained with the same trainer, `cardplay/`, in five stages. The stage names are the
run names in `cardplay/train.sh`.

- **run6, continued.** The card play V2 run, trained on to step 46,000.
- **run11.** Three separate role nets from random weights, trained with 208 extra "oracle"
  input columns: the holder of every card. The nets see them at first, then less and less, then
  never (steps 8,000 to 13,000).
- **run12.** An open-hand critic: a copy of run11 at step 8,000 that always sees every hand. It
  learns the value of the play it watches. The players (run11 at step 13,500) learn toward the
  critic's value of the true deal, not only toward the board's result. 3,000 steps.
- **run14.** run6 at step 46,000 as the players (the oracle columns added with zero weights, so it
  plays as before) with the run12 critic. 3,000 steps.
- **run18.** run14 continued from step 3,000 to 4,000 with true-deal branching at leads. In each
  round of 2,048 boards the trainer picks one lead to a trick (the opening lead in 25% of rounds,
  otherwise the lead to a random later trick). There it plays every legal card on the true deal
  and lets the current nets play out the rest. The net then learns a value for every legal lead
  from these results, not only for the card that was played.

The players never see hidden hands when they play; only the critic and run11's early actors do.
`cardplay/to_server.py` cuts the 208 oracle columns from run18's checkpoint at step 4,000 and
writes the serving file. The server and the trainer took the same tricks on all 2,000 check boards
(1,000 boards, each chair, against the Q-net).

```mermaid
flowchart TD
    v2["run6 (card play V2 run), step 46,000"] --> r14["run14: players"]
    r11["run11: separate role nets, oracle columns faded out"] -->|step 8,000| r12["run12: open-hand critic"]
    r11 -->|step 13,500, players| r12
    r12 -->|critic| r14
    r14 -->|step 3,000, branching at leads| r18["run18, step 4,000"]
    r18 -->|oracle columns cut| v3["Pidgin card play V3"]
```

Results, card play only: Pidgin V3.1 auctions fixed in advance (dataset contract rows 1,000
to 9,989, 8,990 boards), greedy, no search, IMPs per board with a 95% interval:

| Match | IMPs/board |
|---|---|
| Card play V3 vs the Q-net, both chairs | +0.834 ± 0.051 |
| Card play V2 vs the Q-net, both chairs (same boards) | +0.688 ± 0.052 |
| Card play V3 vs card play V2, duplicate | +0.294 ± 0.090 |
| of which declarer play | +0.217 ± 0.080 |
| of which defence | +0.075 ± 0.078 |

## How the Q-net learns

The Q-net's data generator plays out auctions with the Pidgin V1 policy model.
At every card position, a double-dummy solver evaluates **every legal card**.
Training positions include some random legal moves, so the model sees positions
outside the policy model's usual path. For a legal card, the target is its solver
trick count for the side on turn **minus the best legal card's count**: 0 is best,
and a negative value is tricks lost. Illegal cards are excluded from the loss.

The trainer's default loss is the mean squared error of these targets over legal
cards. With `--bhead`, it also predicts each hidden card's owner, each hidden
hand's original suit lengths, and high-card points. Its actual combined loss is:

```text
Q = mean((predicted_value - solver_value_relative_to_best)², legal cards only)
card = cross_entropy(hidden_card_owner, hidden unplayed cards only)
summary = mean_suit_length_cross_entropy + high_card_point_cross_entropy
total = Q + 0.1 × (card + 0.25 × summary)
```

The owner head excludes seats the player can already see. The summary losses
cover hidden seats only. The released Q-net was trained with this auxiliary head,
but serving uses the Pidgin V1 policy model's belief head to sample hidden deals
for search. This keeps the Q-net's value predictions and the deal sampler as
separate roles.

```text
generate auctions and play each deal with the policy model
at each of 52 positions, solve every legal card and save its trick value
for each training step:
    sample a saved position and build the player's visible 899-number input
    set legal-card targets relative to the best solver value
    update the residual Q-net on legal-card error and hidden-hand labels
    periodically measure card choices on held-out positions and save a checkpoint
```

The released `play_B2g_s540k.pt` records step 540,000, input size 899, width
1,024, and depth 3. The commands below train the same kind of model.

## Results

Opening leads on 5,000 benchmark contracts, every legal lead solved double dummy
([all results](results.md#opening-leads)). DDOLAR: share of leads that give up no trick;
ADDOLAR: the same without deals where every lead gives the same result.

| Lead | DDOLAR | ADDOLAR |
|---|---|---|
| policy net (Pidgin V1 team) | 76.5 ± 0.6% | 68.2 ± 0.8% |
| Q-net (Pidgin V2 and BRL teams) | 81.3 ± 0.6% | 74.7 ± 0.7% |
| top human experts (Hammond) | about 81% | about 74.7% |

Card play V2 against the Q-net, both without search, 8,990 fresh boards bid by Pidgin V3.1,
each played with V2 declaring and with V2 defending: +0.69 ± 0.05 IMPs per board. With
sampled-deal search (20 deals) on top, V2 gained nothing (−0.07 ± 0.15), so it plays without.

Declarer play, 49,927 benchmark boards, IMPs per board: policy net vs random cards
+6.30 ± 0.03, vs the earlier policy net +0.44 ± 0.02. Q-net vs policy net, plain nets:
+0.93 ± 0.07 (5,990 boards); with PIMC search on both sides: +0.05 ± 0.06.

## Training history

<p align="center"><img src="figures/card_play/below_double_dummy.svg" alt="Policy net shortfall against double dummy" width="720"></p>

<p align="center"><img src="figures/qnet/tricks_lost.svg" alt="Q-net tricks lost against double dummy per decision" width="720"></p>

The policy net learns most of its declarer and defence play in the first 10,000
self-play steps. The Q-net beats the policy net's double-dummy accuracy at every
position type. See [training curves](training_curves.md).

Download the released models with
`scripts/get_models.sh`. To compare the Pidgin V1 policy model with random legal
play on the published benchmark (without search):

```bash
python -m pip install -e .
python -m pip install -r server/requirements.txt
scripts/get_models.sh
python -m training.play.match server/models/bench_100k.npz \
  --challenger server/models/play_E48_wideleagueH.pt \
  --reference random --deals 100
```

This command evaluates direct policy decisions. The site also runs search,
which can change the result.

## Train the policy model

The trainer is `training/play/train.py`. It needs auctions paired with hands and
double-dummy trick tables. Download `data/dds_results_100M.npy` as shown in the
[README](../README.md#training-data), then generate sampled Pidgin V1 auctions:

```bash
python -m pip install -r server/requirements.txt
mkdir -p data/play
python scripts/generate_auctions.py --model PidginV1 --n 1000000 \
  --temperature 1,3,5 --data data/dds_results_100M.npy --start 0 \
  --out data/play/auctions_1M.npz
```

The policy trainer supports two example model widths. The compact model uses
the first pair; Pidgin V1 card play uses the second. Each second run starts from
the matching first run and plays against a pool of past snapshots:

```bash
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/compact_base --width 512 --batch 512 --steps 25000
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/compact_league --init runs/play/compact_base/last.pt \
  --width 512 --pool-size 5 --steps 20000
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/v1_base --width 1024 --batch 512 --steps 25000
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/v1_league --init runs/play/v1_base/last.pt \
  --width 1024 --pool-size 5 --belief-final 0.1 --steps 20000
```

Both examples train a base policy for 25,000 steps, then fine-tune it against
earlier checkpoints for 20,000 steps. Use validation matches to decide whether
a longer run helps.

## Train card play V2 and V3

The trainer is `cardplay/`; its [README](../cardplay/README.md) lists every stage's options. It
needs the DDS dataset (see the [README](../README.md#training-data), or set `BRIDGE_DATA`), the
Pidgin V3.1 bidder that bids the deals, and the Q-net as the evaluation opponent. From the
repository root:

```bash
python -m pip install -e . endplay
python -m pip install -r server/requirements.txt
scripts/get_models.sh
cardplay/train.sh smoke
DEVICE=cuda cardplay/train.sh v2
DEVICE=cuda cardplay/train.sh v3
```

`smoke` runs both stage sequences with tiny nets for a few steps on the CPU. `v2` writes
`runs/cardplay/play_run6_s36-38k.pt`; `v3` continues from the V2 run and writes
`runs/cardplay/play_run18_s4000.pt`. `DEVICE` is `cuda`, `mps` or `cpu`. The first run makes
the 10,000 evaluation deals `data/fresh_deals_10k.npz` with `endplay`; the evaluation also picks
the league's champion snapshots, so it is part of the recipe.

## Train the Q-net

The Q-net learns solver values for every legal card. Its data pipeline first
generates Pidgin V1 auctions, then plays each deal with the policy model and
solves each legal card. It requires the full DDS dataset, the released Pidgin V1
bidder and card-play model, and the `endplay` package for double-dummy solving.
From the repository root:

```bash
python -m pip install endplay
scripts/get_models.sh
bash qnet/make_auc.sh
bash qnet/worker.sh
python qnet/train_play.py --run values --steps 10000
python qnet/train_play.py --run belief --bhead \
  --init runs/play_q/values/last.pt --steps 10000
python qnet/train_play.py --run refined --bhead \
  --init runs/play_q/belief/last.pt --device cuda --steps 10000
```

`make_auc.sh` writes auction files under `qnet/auc/`; `worker.sh` writes solved
positions under `qnet/pos/`. The worker can be run in several processes and
exits when all current auctions are processed. `--device cuda` requires a CUDA
GPU; choose `--device cpu` when one is unavailable. The 10,000-step values above
are an example, not the released training duration. The published Q-net is a
checkpoint from the final stage at step 540,000.
