# Card-play trainer

Trains the card-play networks served as card play V2 (`play_run6_s36-38k.pt`) and card play V3
(`play_run18_s4000.pt`). The method is deep Monte Carlo self-play, inspired by DouZero: the nets play
whole boards against themselves and past snapshots and learn each card's value from the result. No
solver runs in training; the double-dummy trick count of a deal only centres the target. How the models
learn and how they score: [docs/card_play.md](../docs/card_play.md).

| File | What |
|---|---|
| `train.py` | The trainer (`python -m cardplay.train --help`). |
| `env.py` | Contracts (bid by Pidgin V3.1 at all four seats, cached in blocks), scoring, input features. |
| `model.py` | The residual MLP, the shared-network role views, the table players. |
| `selfsearch.py` | True-deal branching at leads (`--search-pos`, used by run 18). |
| `to_server.py` | Trainer checkpoint(s) to the file `server/emergent/playdmc.py` loads. |
| `train.sh` | The V2 and V3 command sequences below. |

## Requirements

- The DD dataset `data/dds_results_100M.npy` (README "Training data"), or set `BRIDGE_DATA`. It is
  memory-mapped. Training uses rows 3,033,000 to 99,990,000 in one-million-deal blocks.
- `scripts/get_models.sh`: the bidder `server/models/pidginv3_1_bid_s20000.pt` (Pidgin V3.1) and the
  evaluation opponent `server/models/play_B2g_s540k.pt` (the solver-trained Q-net).
- Evaluation deals: the recipes evaluate on 10,000 newly generated deals,
  `python -m training.bridge.deals --out data/fresh_deals_10k.npz --deals 10000 --seed 20261020`
  (needs `pip install endplay`; `train.sh` runs it when the file is missing). Without `--eval-data` the
  trainer evaluates on the dataset's validation rows 3,028,000 to 3,033,000. The evaluation also picks
  the league champions (`--league-best`), so it is part of the recipe.

## Commands

```bash
DEVICE=cuda cardplay/train.sh v2        # card play V2 -> runs/cardplay/play_run6_s36-38k.pt
DEVICE=cuda cardplay/train.sh v3        # card play V3 -> runs/cardplay/play_run18_s4000.pt (after v2)
cardplay/train.sh smoke                 # both sequences, tiny nets, a few steps, CPU
python -m cardplay.train --smoke --out runs/cardplay/smoke --shared-mlp --aux-belief 0.05   # one stage
```

`DEVICE` is `cuda`, `mps` or `cpu`. Every stage uses these settings (the script's `RUN` and `RECIPE`):

```text
--boards 2048 --minibatch 2048 --block-size 1000000 --width 1024 --depth 6 --eps 0.02
--return tricks --passes 2 --td-lambda 0.8 --league 0.3 --aux-trick 0.1 --ema 0.995
--eval-data data/fresh_deals_10k.npz --eval-start 0 --eval-deals 10000
--eval-every 100 --eval-random 2000 --eval-qnet 1000 --ckpt-every 500 --log-every 5 --resume
```

Card play V2, one shared network (`--shared-mlp --aux-belief 0.05`) from random weights:

| Stage | Steps | Options |
|---|---|---|
| run6 | 0 to 20,000 | `--lr 3e-4` |
| run6 | to 23,000 | `--reset-lr --lr 1.5e-4` |
| run6 | to 40,000 | `--lr 1.5e-4 --lr-halve-every 5000 --lr-halve-from 20000 --league-best 0.15` |
| serve | | `to_server.py` on the average of `ckpt_step` 36,000, 36,500, 37,000, 37,500, 38,000 |

Card play V3:

| Stage | Steps | Options |
|---|---|---|
| run6 | to 46,000 | `--shared-mlp --aux-belief 0.05 --reset-lr --lr 1.875e-5 --league-best 0.15` |
| run11 | 0 to 5,500 | three separate role nets, `--lr 3e-4 --oracle-guide --oracle-hold 1000000 --oracle-until 1000001 --league-best 0.15` |
| run11 | to 13,500 | `--reset-lr --lr 1e-4 --oracle-guide --oracle-hold 8000 --oracle-until 13000 --league-best 0.15` |
| run12 | 0 to 3,000 | `--lr 1e-4 --init run11 step 13,500 --critic-init run11 step 8,000 --critic-ramp 1000 --critic-lambda 0.8 --critic-lr 1e-4 --league-best 0.15` |
| run14 | 0 to 3,000 | `--shared-mlp --aux-belief 0.05 --lr 5e-5 --init run6 step 46,000 --critic-init run12/last.pt --critic-ramp 500 --critic-lambda 0.8 --critic-lr 1e-4 --league-best 0.15` |
| run18 | 3,000 to 4,000 | run14's `last.pt` and snapshots, then run14's options plus `--ckpt-every 250 --search-pos 2048 --search-leads --search-open-p 0.25 --search-w 1.0 --search-steps 8 --search-buffer 32768` |
| serve | | `to_server.py` on run18 `ckpt_step0004000.pt` |

Oracle columns: run11 and every stage with `--critic-init` give the nets 208 more inputs, the holder
of every card. Run11's actors see them with a probability that falls to zero (`--oracle-guide`); the
critic always sees them; players with a critic never do. `--init` from a net without them inserts zero
weights, so it plays as before. `to_server.py` cuts them.

`to_server.py` checks the written file: the server's player and the trainer net must take the same
tricks on `--check` evaluation boards against the Q-net in both chairs.

```bash
python -m cardplay.to_server runs/cardplay/run18/ckpt_step0004000.pt runs/cardplay/play_run18_s4000.pt \
  --eval-data data/fresh_deals_10k.npz --eval-start 0 --eval-deals 10000
```
