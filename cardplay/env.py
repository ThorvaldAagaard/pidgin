"""Contracts, scoring and input features for the card-play trainer.

Contracts come from one bidder at all four seats (default Pidgin V3.1, greedy), cached to disk
in blocks. Play runs on `training.bridge.play.PlayBatch` (batched torch).

Seats in features are relative to declarer: 0 declarer, 1 LHO, 2 dummy, 3 RHO.
Roles: 0 declarer side (declarer and dummy), 1 LHO defender, 2 RHO defender.
Card index = suit*13 + rank, suits S,H,D,C, rank 0 = ace. Trump 0..3 suit, 4 NT (card order).

Input = [static | dynamic]. The static part (auction, contract, vulnerability) is fixed for a
board. Nets trained with oracle columns (`--oracle-guide`, `--critic-init`) have 208 more static
columns at the END of the static part: the holder of every card relative to declarer. They are
zero whenever a net plays; only the trainer's actor and the open-hand critic fill them.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.bridge.calls import DOUBLE, PASS, REDOUBLE, STRAIN_PERM
from training.bridge.deals import load_dataset
from training.bridge.scoring import IMP_LOWER_BOUNDS, contract_score

ROOT = Path(__file__).resolve().parents[1]
DATA = os.environ.get("BRIDGE_DATA", str(ROOT / "data" / "dds_results_100M.npy"))
BIDDER = str(ROOT / "server/models/pidginv3_1_bid_s20000.pt")
QNET = str(ROOT / "server/models/play_B2g_s540k.pt")
TRAIN_POOL = (3_033_000, 99_990_000)
EVAL_START, EVAL_DEALS = 3_028_000, 5_000     # dataset validation rows
N_CALLS = 38          # 0..34 bids, 35 P, 36 X, 37 XX; 38 = padding
SEQ = 24              # auction calls encoded in order (longer auctions: rest only in the bag)

# SCORE[vul, doubled, level-1, bid_strain(C,D,H,S,NT), tricks] = declarer score
SCORE = torch.tensor([[[[[contract_score(lv + 1, s, k, x, bool(v)) for k in range(14)]
                         for s in range(5)] for lv in range(7)] for x in range(3)] for v in range(2)],
                     dtype=torch.float32)
IMP_BOUNDS = torch.tensor(IMP_LOWER_BOUNDS, dtype=torch.float32)


def imps_t(points: torch.Tensor) -> torch.Tensor:
    return torch.sign(points) * torch.searchsorted(IMP_BOUNDS, points.abs(), right=True).float()


@dataclass
class Contracts:
    """One row per played (not passed-out) board. Field names match what the Q-net's `position` reads."""
    owner: torch.Tensor      # (n,52) long
    trump: torch.Tensor      # (n,) card-order strain 0..4
    strain: torch.Tensor     # (n,) bid-order strain C,D,H,S,NT
    declarer: torch.Tensor   # (n,) absolute seat
    level: torch.Tensor      # (n,) 1..7
    doubled: torch.Tensor    # (n,) 0/1/2
    vul_ns: torch.Tensor     # (n,) bool
    vul_ew: torch.Tensor
    calls: torch.Tensor      # (n,W) long, padding 38
    n_calls: torch.Tensor    # (n,)
    dealer: torch.Tensor     # (n,)
    dd_tricks: torch.Tensor  # (n,) DD tricks for this declarer/strain
    deal_index: torch.Tensor  # (n,) row in the source file

    def __len__(self):
        return len(self.owner)

    def subset(self, idx) -> "Contracts":
        return Contracts(**{k: v[idx] for k, v in self.__dict__.items()})

    def to(self, device) -> "Contracts":
        return Contracts(**{k: v.to(device) for k, v in self.__dict__.items()})   # blocking copies (MPS)

    def decl_vul(self) -> torch.Tensor:
        return torch.where(self.declarer % 2 == 0, self.vul_ns, self.vul_ew)

    def score(self, tricks: torch.Tensor) -> torch.Tensor:
        """Declarer score for `tricks` declarer tricks."""
        return SCORE[self.decl_vul().long(), self.doubled, self.level - 1, self.strain, tricks.long()]

    def imps_vs_dd(self, tricks: torch.Tensor) -> torch.Tensor:
        """Declarer-side IMPs: score(actual) - score(DD) for the same contract."""
        return imps_t(self.score(tricks) - self.score(self.dd_tricks))


def contracts_from_auctions(hist: np.ndarray, dealer: np.ndarray, vul_ns, vul_ew, owners: np.ndarray,
                            tricks: np.ndarray, deal_index: np.ndarray) -> Contracts:
    """hist (n,W) calls, -1 = none. Declarer = first of the winning side to name the strain."""
    n = len(hist)
    keep, trump, strain, decl, level, dbl = [], [], [], [], [], []
    for r in range(n):
        seq = hist[r][hist[r] >= 0]
        bids = np.nonzero(seq < PASS)[0]
        if len(bids) == 0:
            continue
        li = int(bids[-1]); last = int(seq[li]); s = last % 5
        side = (dealer[r] + li) % 2
        first = next(int(i) for i in bids if seq[i] % 5 == s and (dealer[r] + i) % 2 == side)
        tail = seq[li + 1:]
        keep.append(r); strain.append(s); trump.append(STRAIN_PERM[s])
        decl.append((dealer[r] + first) % 4); level.append(last // 5 + 1)
        dbl.append(2 if (tail == REDOUBLE).any() else (1 if (tail == DOUBLE).any() else 0))
    keep = np.asarray(keep, dtype=np.int64)
    h = hist[keep]
    width = max(1, int((h >= 0).sum(1).max())) if len(h) else 1
    calls = np.where(h[:, :width] >= 0, h[:, :width], N_CALLS).astype(np.int64)
    decl = np.asarray(decl, dtype=np.int64); trump = np.asarray(trump, dtype=np.int64)
    return Contracts(
        owner=torch.from_numpy(owners[keep].astype(np.int64)),
        trump=torch.from_numpy(trump), strain=torch.tensor(strain, dtype=torch.long),
        declarer=torch.from_numpy(decl), level=torch.tensor(level, dtype=torch.long), doubled=torch.tensor(dbl, dtype=torch.long),
        vul_ns=torch.from_numpy(np.asarray(vul_ns)[keep].astype(bool)),
        vul_ew=torch.from_numpy(np.asarray(vul_ew)[keep].astype(bool)),
        calls=torch.from_numpy(calls), n_calls=torch.from_numpy((h >= 0).sum(1).astype(np.int64)),
        dealer=torch.from_numpy(dealer[keep].astype(np.int64)),
        dd_tricks=torch.from_numpy(tricks[keep, decl, trump].astype(np.int64)),
        deal_index=torch.from_numpy(deal_index[keep].astype(np.int64)))


def save_contracts(c: Contracts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")   # np.savez adds .npz only when missing
    # small ints as uint8 (a 1M block: ~0.7 GB -> ~0.1 GB); load_contracts widens them back to int64
    np.savez(tmp, **{k: v.numpy().astype(np.uint8) if v.dtype == torch.int64 and k != "deal_index" else v.numpy()
                     for k, v in c.__dict__.items()})
    tmp.rename(path)


def load_contracts(path: Path) -> Contracts:
    z = np.load(path)
    return Contracts(**{k: torch.from_numpy(z[k].astype(np.int64) if z[k].dtype == np.uint8 else z[k])
                        for k in Contracts.__dataclass_fields__})


# ------------------------------------------------------------------ bidding

def bid_range(data: str, start: int, count: int, seed: int, chunk: int = 8000,
              bidder: str = BIDDER, log=print) -> Contracts:
    """Greedy auctions by one bidder at all four seats for deals [start, start+count); seeded dealer/vul."""
    from tools import match as M
    from training.contract.data import load_range
    player = M.FourSeatPlayer(bidder)
    rng = np.random.default_rng(seed)
    dealer_all = rng.integers(0, 4, count)
    vul_all = rng.integers(0, 4, count)
    owners_mm, tricks_mm = load_dataset(data)
    parts = []
    t0 = time.time()
    for i in range(0, count, chunk):
        m = min(chunk, count - i)
        deals = load_range(data, start + i, m)
        dl = torch.from_numpy(dealer_all[i:i + m]); v = torch.from_numpy(vul_all[i:i + m])
        vns, vew = (v == 1) | (v == 3), (v == 2) | (v == 3)
        with torch.no_grad():
            tab = M.play([player], deals, torch.arange(m), dl, vns, vew, torch.zeros(m, 2, dtype=torch.long))
        hist = tab.history.numpy()
        owners = np.asarray(owners_mm[start + i:start + i + m])
        tricks = np.asarray(tricks_mm[start + i:start + i + m])
        c = contracts_from_auctions(hist, dl.numpy(), vns.numpy(), vew.numpy(), owners, tricks,
                                    np.arange(start + i, start + i + m))
        _, last, d = tab.ns_scores()                       # cross-check declarer with the match table
        played = last >= 0
        assert np.array_equal(d[played], c.declarer.numpy()), "declarer mismatch vs match table"
        parts.append(c)
        if log and (i // chunk) % 25 == 0:
            log(f"  bid {i + m}/{count} deals ({time.time() - t0:.0f}s)")
    return concat(parts)


def concat(parts) -> Contracts:
    width = max(p.calls.shape[1] for p in parts)
    out = {}
    for k in Contracts.__dataclass_fields__:
        vs = [getattr(p, k) for p in parts]
        if k == "calls":
            vs = [F.pad(v, (0, width - v.shape[1]), value=N_CALLS) for v in vs]
        out[k] = torch.cat(vs)
    return Contracts(**out)


def block_start(block: int, block_size: int, seed: int = 0, pool=TRAIN_POOL) -> int:
    n_blocks = (pool[1] - pool[0]) // block_size
    order = np.random.default_rng(seed).permutation(n_blocks)
    return pool[0] + int(order[block % n_blocks]) * block_size


def block_path(block: int, cache: Path, block_size: int, seed: int = 0, pool=TRAIN_POOL) -> Path:
    return Path(cache) / f"block_bs{block_size}_s{block_start(block, block_size, seed, pool)}.npz"


def check_cache_bidder(cache: Path, bidder: str) -> None:
    """One bidder per cache dir, recorded in cache/BIDDER.txt (by file name); a mismatch is an error."""
    f = Path(cache) / "BIDDER.txt"
    name = Path(bidder).name
    if f.exists():
        assert Path(f.read_text().strip()).name == name, f"{cache} holds {f.read_text().strip()} blocks, not {bidder}"
    else:
        assert not any(Path(cache).glob("*.npz")), f"{cache} has blocks from an unknown bidder"
        Path(cache).mkdir(parents=True, exist_ok=True)
        f.write_text(bidder + "\n")


def train_block(block: int, cache: Path, block_size: int, data: str = DATA, seed: int = 0, log=print,
                bidder: str = BIDDER, pool=TRAIN_POOL) -> Contracts:
    """Block k of the training pool, bidding cached at cache/block_*.npz. Blocks are taken in a seeded order."""
    start = block_start(block, block_size, seed, pool)
    path = block_path(block, cache, block_size, seed, pool)
    if path.exists():
        return load_contracts(path)
    log(f"bidding training block {block} start={start} n={block_size} -> {path}")
    check_cache_bidder(cache, bidder)
    c = bid_range(data, start, block_size, seed=start, log=log, bidder=bidder)
    save_contracts(c, path)
    return c


def eval_contracts(cache: Path, data: str = DATA, start: int = EVAL_START, n: int = EVAL_DEALS, seed: int = 12345,
                   log=print, bidder: str = BIDDER) -> Contracts:
    """Deals [start, start+n) of `data` bid once and cached. Seed 12345 sets dealer and vulnerability."""
    path = Path(cache) / f"eval_{Path(data).stem}_s{start}_n{n}_seed{seed}.npz"
    if path.exists():
        return load_contracts(path)
    log(f"bidding eval contracts: {data} rows {start}..{start + n}")
    check_cache_bidder(cache, bidder)
    c = bid_range(data, start, n, seed=seed, log=log, bidder=bidder)
    save_contracts(c, path)
    return c


# ----------------------------------------------------------------- features

STATIC_DIM = 4 * N_CALLS + SEQ * (N_CALLS + 1) + 4 + 7 + 5 + 3 + 2 + SEQ * 4   # auction + contract + vul + callers
ORACLE_DIM = 52 * 4                                                                 # holder of every card
REL_DIM = 4 * 3 * 52 + 52
DYN_DIM = 52 + 52 + 4 + 3 * 4 * 52 + 13 + 4 + 14 + 14 + REL_DIM
FEAT_DIM = STATIC_DIM + DYN_DIM                                                     # 2658, the served input


def static_dim(oracle: bool = False) -> int:
    return STATIC_DIM + (ORACLE_DIM if oracle else 0)


def feat_dim(oracle: bool = False) -> int:
    return static_dim(oracle) + DYN_DIM


def oracle_columns(c: Contracts) -> torch.Tensor:
    """(n, ORACLE_DIM) uint8: one-hot holder of every card relative to declarer (open hands)."""
    return F.one_hot((c.owner - c.declarer[:, None]) % 4, 4).reshape(len(c), -1).to(torch.uint8)


def static_features(c: Contracts, oracle: bool = False) -> torch.Tensor:
    """(n, static_dim(oracle)) uint8: auction (bag by seat rel. to declarer + first SEQ calls in order,
    dealer rel. to declarer), contract level/strain(card order)/doubled, vulnerability of each side,
    who made each of the first SEQ calls; then (oracle) ORACLE_DIM zeros."""
    n = len(c)
    decl = c.declarer
    W = c.calls.shape[1]
    pos = torch.arange(W, device=decl.device)
    live = pos[None] < c.n_calls[:, None]
    rel_seat = (c.dealer[:, None] + pos[None] - decl[:, None]) % 4
    bag = torch.zeros(n, 4, N_CALLS + 1, dtype=torch.uint8, device=decl.device)
    idx = rel_seat * (N_CALLS + 1) + c.calls
    bag.view(n, -1).scatter_(1, torch.where(live, idx, N_CALLS), 1)   # pad lands in seat 0's pad slot
    bag = bag[:, :, :N_CALLS].reshape(n, -1)
    seq = c.calls[:, :SEQ]
    if seq.shape[1] < SEQ:
        seq = F.pad(seq, (0, SEQ - seq.shape[1]), value=N_CALLS)
    seq = F.one_hot(seq, N_CALLS + 1).reshape(n, -1).to(torch.uint8)
    dv = c.decl_vul()
    ov = torch.where(decl % 2 == 0, c.vul_ew, c.vul_ns)
    rs = F.pad(rel_seat, (0, max(0, SEQ - W)))[:, :SEQ]
    lv = F.pad(live, (0, max(0, SEQ - W)))[:, :SEQ]
    parts = [bag, seq, F.one_hot((c.dealer - decl) % 4, 4), F.one_hot(c.level - 1, 7),
             F.one_hot(c.trump, 5), F.one_hot(c.doubled, 3), dv[:, None], ov[:, None],
             (F.one_hot(rs, 4) * lv[..., None]).reshape(n, -1)]
    if oracle:
        parts.append(torch.zeros(n, ORACLE_DIM, dtype=torch.uint8, device=decl.device))
    return torch.cat([p.to(torch.uint8) for p in parts], 1)


def dynamic_features(b, decl: torch.Tensor) -> torch.Tensor:
    """(n, DYN_DIM) uint8 for the seat on turn. Only cards that seat may see:
    the hand it plays from, the dummy (declarer's hand when dummy is on turn) once the lead is
    made, and the public play history. Hidden cards never enter."""
    n, t = b.n, b.t
    seat = b.to_play()
    rel_turn = (seat - decl) % 4
    dummy = (decl + 2) % 4
    played = ~b.unplayed
    own = b.hand(seat)
    other_seat = torch.where(rel_turn == 2, decl, dummy)
    other = b.hand(other_seat) & (t > 0)
    rel_owner = (b.owner - decl[:, None]) % 4                         # (n,52)
    oh = F.one_hot(rel_owner, 4).bool()                              # (n,52,4)
    played_by = oh & played[..., None]
    dev = decl.device
    lead = torch.zeros(n, 52, dtype=torch.bool, device=dev)
    if t:
        lead.scatter_(1, b.history[:, 0:t:4], True)
    led_by = oh & lead[..., None]
    cur = torch.zeros(n, 52, dtype=torch.bool, device=dev)
    start = (t // 4) * 4
    if t > start:
        cur.scatter_(1, b.history[:, start:t], True)
    cur_by = oh & cur[..., None]
    side_won = b.tricks_won                                           # (n,2) by absolute side
    dw = side_won.gather(1, (decl % 2)[:, None]).squeeze(1)
    fw = side_won.gather(1, (1 - decl % 2)[:, None]).squeeze(1)
    parts = [own, other, F.one_hot(rel_turn, 4),
             played_by.transpose(1, 2).reshape(n, -1), led_by.transpose(1, 2).reshape(n, -1),
             cur_by.transpose(1, 2).reshape(n, -1),
             F.one_hot(torch.full((n,), t // 4, device=dev), 13), F.one_hot(torch.full((n,), t % 4, device=dev), 4),
             F.one_hot(dw, 14), F.one_hot(fw, 14), rel_features(b, own, other, cur)]
    return torch.cat([p.to(torch.uint8) for p in parts], 1)


def rel_features(b, own, other, cur) -> torch.Tensor:
    """(n, REL_DIM) bool, same-suit rank relations. Per card c (all 52, wherever it is): the number of
    HIGHER cards of c's suit in each place (own hand, the other hand the seat sees, unseen, current
    trick) as a thermometer >=1, >=2, >=3 (4 x 3 x 52, place-major); then 52 bits: c would beat the
    card now winning the trick (0 on the lead)."""
    n, t = b.n, b.t
    unseen = b.unplayed & ~own & ~other
    out = []
    for m in (own, other, unseen, cur):
        v = m.view(n, 4, 13).long()
        hi = (v.cumsum(2) - v).view(n, 52)                                # rank 0 = ace: lower index = higher
        out += [hi >= 1, hi >= 2, hi >= 3]
    beats = torch.zeros(n, 52, dtype=torch.bool, device=own.device)
    if t % 4:
        tc = b.trick_cards()                                              # (n,pos)
        suit, rank = b.suit_of, b.rank_of
        trump = b.trump[:, None]
        led = suit[tc[:, :1]]
        sc = torch.where(suit[tc] == trump, 200 - rank[tc], torch.where(suit[tc] == led, 100 - rank[tc], 0))
        w = tc.gather(1, sc.argmax(1, keepdim=True))                     # card now winning (n,1)
        ws, wr = suit[w], rank[w]
        cs, cr = suit[None], rank[None]
        beats = ((cs == ws) & (cr < wr)) | ((cs == trump) & (ws != trump))
    return torch.cat(out + [beats], 1)


def roles(b, decl: torch.Tensor) -> torch.Tensor:
    """(n,) role of the seat on turn: 0 declarer side, 1 LHO, 2 RHO."""
    rel = (b.to_play() - decl) % 4
    return torch.tensor([0, 1, 0, 2], device=rel.device)[rel]


def lambda_returns(v_decl: torch.Tensor, g: torch.Tensor, lam: float) -> torch.Tensor:
    """TD(lambda) targets, declarer side. v_decl (T,n): V_decl(s_t) of the state before decision t;
    g (n,): final return. Q predicts the final return, so there are no per-step rewards:
    G_{T-1} = g, G_t = (1 - lam) V_decl(s_{t+1}) + lam G_{t+1}. lam = 1 gives g at every step."""
    out = torch.empty_like(v_decl)
    out[-1] = g
    for t in range(v_decl.shape[0] - 2, -1, -1):
        out[t] = (1 - lam) * v_decl[t + 1] + lam * out[t + 1]
    return out
