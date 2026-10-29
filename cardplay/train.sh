#!/usr/bin/env bash
# Card play V2 and V3 training recipes (see cardplay/README.md).
#   cardplay/train.sh v2 [RUNS]       card play V2: run6 to step 40,000, then the average of steps 36,000-38,000
#   cardplay/train.sh v3 [RUNS]       card play V3: run6 to 46,000, run11, run12, run14, run18 (needs run6 from v2)
#   cardplay/train.sh smoke [RUNS]    both sequences with tiny nets and a few steps on data/smoke_128.npz
# Variables: DEVICE (cuda, mps or cpu; default cuda), THREADS, BID_THREADS, PYTHON, BRIDGE_DATA (DD dataset).
# Every stage passes --resume, so rerunning the script continues an interrupted stage.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:?usage: cardplay/train.sh v2|v3|smoke [RUNS]}
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-8}
BID_THREADS=${BID_THREADS:-8}
FRESH=data/fresh_deals_10k.npz

RECIPE=(--return tricks --passes 2 --td-lambda 0.8 --league 0.3 --aux-trick 0.1 --ema 0.995 --eps 0.02)
if [[ $MODE == smoke ]]; then
  R=${2:-runs/cardplay_smoke}
  RUN=(--smoke --device cpu --threads 2 --bid-threads 1 --cache "$R/contracts" --ckpt-every 1 "${RECIPE[@]}")
  CKPT18=1
  S6A=2 S6B=3 S6C=6 AVG=(4 5 6) S6D=7 HALVE=(--lr-halve-every 2 --lr-halve-from 2)
  S11A=2 S11B=4 CRIT11=2 HOLD11=2 UNTIL11=4 S12=2 S14=2 S18=3 RAMP12=1 RAMP14=1
  CHECK=(--check 8 --first 0 --cache "$R/contracts" --eval-data data/smoke_128.npz --eval-start 0 --eval-deals 64)
else
  R=${2:-runs/cardplay}
  if [[ ! -e $FRESH ]]; then   # the evaluation deals: 10,000 new random deals with DD tables (needs endplay)
    "$PY" -m training.bridge.deals --out "$FRESH" --deals 10000 --seed 20261020
  fi
  RUN=(--device "$DEVICE" --threads "$THREADS" --bid-threads "$BID_THREADS" --cache "$R/contracts"
       --boards 2048 --minibatch 2048 --block-size 1000000 --width 1024 --depth 6
       --eval-data "$FRESH" --eval-start 0 --eval-deals 10000 --eval-every 100 --eval-random 2000 --eval-qnet 1000
       --ckpt-every 500 --log-every 5 "${RECIPE[@]}")
  CKPT18=250
  S6A=20000 S6B=23000 S6C=40000 AVG=(36000 36500 37000 37500 38000) S6D=46000
  HALVE=(--lr-halve-every 5000 --lr-halve-from 20000)
  S11A=5500 S11B=13500 CRIT11=8000 HOLD11=8000 UNTIL11=13000 S12=3000 S14=3000 S18=4000 RAMP12=1000 RAMP14=500
  CHECK=(--check 1000 --first 1000 --cache "$R/contracts" --eval-data "$FRESH" --eval-start 0 --eval-deals 10000)
fi
train() { "$PY" -u -m cardplay.train "${RUN[@]}" --resume "$@"; }
ck() { printf "%s/ckpt_step%07d.pt" "$1" "$2"; }
SHARED=(--shared-mlp --aux-belief 0.05)

v2() {
  echo "== card play V2: run6 (one shared net), from random weights"
  train --out "$R/run6" "${SHARED[@]}" --lr 3e-4 --steps "$S6A"
  train --out "$R/run6" "${SHARED[@]}" --reset-lr --lr 1.5e-4 --steps "$S6B"
  train --out "$R/run6" "${SHARED[@]}" --lr 1.5e-4 "${HALVE[@]}" --league-best 0.15 --steps "$S6C"
  local files=()
  for s in "${AVG[@]}"; do files+=("$(ck "$R/run6" "$s")"); done
  "$PY" -m cardplay.to_server "${files[@]}" "$R/play_run6_s36-38k.pt" "${CHECK[@]}"
}

v3() {
  echo "== run6 continued (players of run14), lr kept at its last value"
  train --out "$R/run6" "${SHARED[@]}" --reset-lr --lr 1.875e-5 --league-best 0.15 --steps "$S6D"
  echo "== run11: separate role nets, oracle guiding (its step $CRIT11, all hands shown, is the critic)"
  train --out "$R/run11" --lr 3e-4 --oracle-guide --oracle-from 1.0 --oracle-hold 1000000 --oracle-until 1000001 \
    --league-best 0.15 --steps "$S11A"
  train --out "$R/run11" --reset-lr --lr 1e-4 --oracle-guide --oracle-from 1.0 --oracle-hold "$HOLD11" \
    --oracle-until "$UNTIL11" --league-best 0.15 --steps "$S11B"
  echo "== run12: the open-hand critic adapts to hidden-hand players"
  train --out "$R/run12" --league-best 0.15 --lr 1e-4 --init "$(ck "$R/run11" "$S11B")" \
    --critic-init "$(ck "$R/run11" "$CRIT11")" --critic-w 1.0 --critic-ramp "$RAMP12" --critic-lambda 0.8 \
    --critic-lr 1e-4 --steps "$S12"
  echo "== run14: run6 players (oracle columns padded with zero weights) learn from run12's critic"
  train --out "$R/run14" "${SHARED[@]}" --league-best 0.15 --lr 5e-5 --init "$(ck "$R/run6" "$S6D")" \
    --critic-init "$R/run12/last.pt" --critic-w 1.0 --critic-ramp "$RAMP14" --critic-lambda 0.8 \
    --critic-lr 1e-4 --steps "$S14"
  echo "== run18: run14 continued with true-deal branching at leads"
  if [[ ! -e $R/run18/last.pt ]]; then
    mkdir -p "$R/run18"
    cp "$R/run14/last.pt" "$R/run18/last.pt"
    for f in "$R"/run14/ckpt_step*.pt; do ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$R/run18/"; done
  fi
  train --out "$R/run18" "${SHARED[@]}" --league-best 0.15 --lr 5e-5 --ckpt-every "$CKPT18" \
    --critic-init "$R/run12/last.pt" --critic-w 1.0 --critic-ramp "$RAMP14" --critic-lambda 0.8 --critic-lr 1e-4 \
    --search-pos 2048 --search-leads --search-open-p 0.25 --search-w 1.0 --search-steps 8 --search-buffer 32768 \
    --steps "$S18"
  "$PY" -m cardplay.to_server "$(ck "$R/run18" "$S18")" "$R/play_run18_s4000.pt" "${CHECK[@]}"
}

case $MODE in
  v2) v2 ;;
  v3) v3 ;;
  smoke) v2; v3 ;;
  *) echo "usage: cardplay/train.sh v2|v3|smoke [RUNS]" >&2; exit 2 ;;
esac
