"""Bid every board of a PBN file with one model in all four seats; write the PBN with auctions.

    python scripts/bid_pbn.py boards.pbn bid.pbn --model PidginV2
    python scripts/bid_pbn.py boards.pbn bid.pbn --model server/models/D_cw_s75k.pt

Each board keeps its own [Deal], [Dealer] and [Vulnerable]. All boards are bid together,
one network pass per round of calls, greedily and without bidding search (the same calls
as scripts/generate_auctions.py). The output is the input with each record's [Auction]
replaced; [Contract], [Declarer], [Result] and [Play] are dropped, since they belonged to
the old auction. Records without a [Deal] are copied unchanged.
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import torch

from generate_auctions import bid_batch
from pidgin import RANKS, SEATS, call_name, load_bidder

VULS = {"none": (0, 0), "-": (0, 0), "love": (0, 0), "ns": (1, 0), "ew": (0, 1),
        "all": (1, 1), "both": (1, 1)}
DROP = {"Auction", "Contract", "Declarer", "Result", "Play"}   # tags tied to the old auction


def owners(deal: str) -> np.ndarray:
    """A PBN deal string 'N:AKQ.JT9.876.5432 ...' -> (52,) seat of each card."""
    first, hands = deal.split(":")
    out = np.full(52, 255, dtype=np.uint8)
    for k, hand in enumerate(hands.split()):
        seat = (SEATS.index(first.strip().upper()) + k) % 4
        for s, holding in enumerate(hand.split(".")):
            for ch in holding:
                out[s * 13 + RANKS.index(ch.upper())] = seat
    if (out == 255).any():
        raise ValueError(f"not a full deal: {deal}")
    return out


def strip_old_auction(rec: str) -> list[str]:
    """The record's lines without the tags in DROP and the data lines after them."""
    keep, skipping = [], False
    for line in rec.splitlines():
        tag = re.match(r"\s*\[(\w+)\s", line)
        if tag:
            skipping = tag.group(1) in DROP
        if not skipping:
            keep.append(line)
    return keep


def auction_lines(dealer: int, calls: list[int]) -> list[str]:
    names = ["Pass" if call_name(c) == "P" else call_name(c) for c in calls]
    return [f'[Auction "{SEATS[dealer]}"]'] + [" ".join(names[i:i + 4]) for i in range(0, len(names), 4)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pbn_in")
    ap.add_argument("pbn_out")
    ap.add_argument("--model", default="PidginV2", help="team id or checkpoint file")
    ap.add_argument("--batch", type=int, default=4096, help="boards bid together")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    records = re.split(r"\n\s*\n", open(args.pbn_in, encoding="utf-8-sig").read().strip())
    boards = []                                    # (record index, owners, dealer, vul)
    for r, rec in enumerate(records):
        tags = dict(re.findall(r'\[(\w+) "([^"]*)"\]', rec))
        if "Deal" in tags:
            boards.append((r, owners(tags["Deal"]), SEATS.index(tags.get("Dealer", "N").strip().upper()),
                           VULS[tags.get("Vulnerable", "None").strip().lower()]))
    if not boards:
        raise SystemExit(f"no [Deal] in {args.pbn_in}")

    bot = load_bidder(args.model)
    own = np.stack([b[1] for b in boards])
    dealer = np.array([b[2] for b in boards])
    vul = np.array([b[3] for b in boards], dtype=np.uint8)
    auctions, t0 = [], time.time()
    for s in range(0, len(boards), args.batch):
        e = min(s + args.batch, len(boards))
        auctions += bid_batch(bot, own[s:e], dealer[s:e], vul[s:e])
        print(f"{e}/{len(boards)} boards, {time.time() - t0:.1f}s", flush=True)

    out = {r: strip_old_auction(records[r]) + auction_lines(d, a)
           for (r, _, d, _), a in zip(boards, auctions)}
    with open(args.pbn_out, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join("\n".join(out[r]) if r in out else rec for r, rec in enumerate(records)) + "\n")
    print(f"wrote {args.pbn_out}: {len(boards)} boards bid by {args.model}")


if __name__ == "__main__":
    main()
