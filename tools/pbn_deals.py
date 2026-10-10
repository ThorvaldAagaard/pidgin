"""Turn the deals of a PBN file into a deal file ``tools/match.py`` can play.

Needs each deal's double-dummy table, which a PBN usually does not carry: pass the CSV that
Brill's ``Brill.GameValidation.Cli <file.pbn> --dd-table dd.csv`` writes (one row per board,
columns ``N_S .. W_NT``). The output ``.npz`` holds ``owners``/``tricks`` like the other deal
files, plus each board's own ``dealer`` and ``vul``, which ``match.py --as-dealt`` plays
instead of the 4 dealers x 4 vulnerabilities rotation. One deal per [Board] number; a file
with two rooms per board keeps the first record.

    python tools/pbn_deals.py match.pbn dd.csv data/match_deals.npz
"""

from __future__ import annotations

import csv
import re
import sys

import numpy as np

SEATS, SUITS, RANKS = "NESW", "SHDC", "AKQJT98765432"
TABLE_STRAINS = ("S", "H", "D", "C", "NT")          # the tricks table's strain order
VULS = {"none": 0, "-": 0, "love": 0, "ns": 1, "ew": 2, "all": 3, "both": 3}


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    pbn, dd_csv, out = sys.argv[1:]
    boards: dict[int, tuple[str, str, str]] = {}
    text = open(pbn, encoding="utf-8").read()
    for rec in re.split(r"\n\s*\n", text):
        tags = dict(re.findall(r'\[(\w+) "([^"]*)"\]', rec))
        if "Board" in tags and "Deal" in tags:
            boards.setdefault(int(tags["Board"]), (tags["Deal"], tags.get("Dealer", "N"),
                                                   tags.get("Vulnerable", "None")))
    dd = {int(r["board"]): r for r in csv.DictReader(open(dd_csv, encoding="utf-8"))}

    order = sorted(b for b in boards if b in dd)
    missing = sorted(set(boards) - set(dd))
    if missing:
        print(f"no DD row for {len(missing)} board(s), skipped: {missing[:10]}")
    owners = np.zeros((len(order), 52), dtype=np.uint8)
    tricks = np.zeros((len(order), 4, 5), dtype=np.uint8)
    dealer = np.zeros(len(order), dtype=np.int64)
    vul = np.zeros(len(order), dtype=np.int64)
    for i, b in enumerate(order):
        deal, dl, v = boards[b]
        first, hands = deal.split(":")
        for k, hand in enumerate(hands.split()):
            seat = (SEATS.index(first.upper()) + k) % 4
            for s, holding in enumerate(hand.split(".")):
                for ch in holding:
                    owners[i, s * 13 + RANKS.index(ch.upper())] = seat
        for seat, sn in enumerate(SEATS):
            for t, strain in enumerate(TABLE_STRAINS):
                tricks[i, seat, t] = int(dd[b][f"{sn}_{strain}"])
        dealer[i] = SEATS.index(dl.upper())
        vul[i] = VULS[v.strip().lower()]
    np.savez_compressed(out, owners=owners, tricks=tricks, dealer=dealer, vul=vul,
                        board=np.asarray(order, dtype=np.int64))
    print(f"{len(order)} deals -> {out}")


if __name__ == "__main__":
    main()
