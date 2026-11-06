"""Card play V4: a belief net + PIMC, with card play V3 breaking ties.

For each card, the belief net (52M-parameter GLU MLP) gives every hidden card's owner probabilities
from the seat's view: auction, play so far, and both pairs' bidding systems. Layouts are drawn card
by card from those probabilities (respecting known voids), importance-weighted until the effective
sample size reaches MIN_ESS, resampled to K, and the distinct layouts are solved double dummy,
most-drawn first, until BUDGET_S runs out (at least MIN_SOLVED of them). The
card with the best mean DD tricks wins; cards within TIE_EPS tricks of it count as tied, and the
tie goes to card play V3's choice (run 18, Pidgin V3.2) when V3 picks one of them, else to a seeded
random tied card. Measured: +0.81 IMPs a board vs Lia, +1.15 vs Brill over V3.1 cards (docs/pidgin_v4.md).

`PimcPlayBot.choose_card` has the signature of `engine.choose_card`. The bidding systems come from
`SYSTEMS` (a context variable the Brill API sets per request); the site's own tables leave the
default, VieNNa at both sides.
"""
from __future__ import annotations

import contextvars
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from emergent.beliefbatch import E, IN_DIM, W, build

# belief-net system slots (the trainer's systems.json order; slot 3 = WBridge5's data)
VIENNA, SAYC, TWO_OVER_ONE, WBRIDGE5 = 0, 1, 2, 3
# (our side's system, the opponents' system), relative to the seat on turn
SYSTEMS: contextvars.ContextVar[tuple[int, int]] = contextvars.ContextVar("systems", default=(VIENNA, VIENNA))

K, N_RAW, MAX_RAW, MIN_ESS, TIE_EPS = 64, 256, 4096, 16.0, 0.05
# DD solving stops after this many seconds once MIN_SOLVED layouts are in (most-drawn first):
# the opening tricks would otherwise take ~2 s a card on the 1-vCPU server.
BUDGET_S, MIN_SOLVED = float(os.environ.get("PIMC_BUDGET_S", "0.6")), 16
TRUMP_OF_STRAIN = [3, 2, 1, 0, 4]       # bid strain C D H S N -> card suit (S H D C), 4 = notrump
SUIT = np.arange(52) // 13
RANKS = "AKQJT98765432"
AUC_DIM = 35 * 4 + 35 + 35 + 4 * 39
STATE_DIM = IN_DIM + AUC_DIM + 8


# ------------------------------------------------------------------ net

class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln, self.up, self.down = nn.LayerNorm(d), nn.Linear(d, 2 * h), nn.Linear(h, d)

    def forward(self, x):
        a, g = self.up(self.ln(x)).chunk(2, -1)
        return self.down(F.silu(g) * a)


class GLUMLP(nn.Module):
    def __init__(self, d_in, d, blocks, h):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList(SwiGLU(d, h) for _ in range(blocks))
        self.ln = nn.LayerNorm(d)

    def forward(self, s):
        h = self.inp(s)
        for blk in self.blocks:
            h = h + blk(h)
        return self.ln(h)


class BeliefNet(nn.Module):
    def __init__(self, d: int, blocks: int, h: int):
        super().__init__()
        self.core = GLUMLP(STATE_DIM, d, blocks, h)
        self.belief, self.shape = nn.Linear(d, 208), nn.Linear(d, 15)

    def forward(self, s):
        h = self.core(s)
        return self.belief(h).view(-1, 52, 4), self.shape(h)


def aux_feats(t, b):
    """Per (row = 4*deal + viewer, step): auction grid (AUC_DIM) and system one-hots."""
    calls, nc, dealer = t["calls"], t["n_calls"], t["dealer"]
    B = len(calls)
    k = torch.arange(E)[None]
    j = torch.minimum(k, nc[:, None])                                       # calls made before step k
    i = torch.arange(W)[None]
    seat = (dealer[:, None] + i) % 4                                        # abs seat of call i
    live = i < nc[:, None]
    c = calls.clamp(min=0)
    is_bid = live & (calls >= 0) & (calls < 35)
    g = torch.zeros(B, W, 35, 4)
    g[is_bid] = F.one_hot(c[is_bid] * 4 + seat[is_bid], 140).float().view(-1, 35, 4)
    gp = torch.cat([torch.zeros(B, 1, 35, 4), g.cumsum(1)], 1)
    pos = torch.where(is_bid, i, -1).cummax(1).values                       # last bid index at or before i
    lastbid = c.gather(1, pos.clamp(min=0))
    xd = torch.zeros(B, W, 35, 2)
    for v, col in ((36, 0), (37, 1)):
        m = live & (calls == v) & (pos >= 0)
        xd[..., col][m] = F.one_hot(lastbid[m], 35).float()
    xp = torch.cat([torch.zeros(B, 1, 35, 2), xd.cumsum(1)], 1)
    jj = j[..., None, None]
    bid_by = gp.gather(1, jj.expand(B, E, 35, 4).clamp(max=W))
    dx = xp.gather(1, jj.expand(B, E, 35, 2).clamp(max=W)).clamp(max=1)
    last = []
    for back in range(1, 5):
        q = j - back
        cc = torch.where(q >= 0, calls.gather(1, q.clamp(min=0)).long(), 38)
        last.append(F.one_hot(cc.clamp(max=38), 39).float())
    last = torch.cat(last, -1)
    v = torch.arange(4)
    relseat = (v[:, None] + torch.arange(4)[None]) % 4
    bid_rel = bid_by[:, None].expand(B, 4, E, 35, 4).gather(4, relseat[None, :, None, None, :].expand(B, 4, E, 35, 4))
    auc = torch.cat([bid_rel.reshape(B, 4, E, 140), dx[:, None].expand(B, 4, E, 35, 2).reshape(B, 4, E, 70),
                     last[:, None].expand(B, 4, E, 156)], -1).reshape(B * 4, E, AUC_DIM)
    sys_ = b["sys"]
    so = torch.cat([F.one_hot(sys_[:, 0].clamp(0, 3), 4), F.one_hot(sys_[:, 1].clamp(0, 3), 4)], -1).float()
    return {"auc": auc, "sysoh": so}


def masked_belief(logits, lead_out, dummy_rel):
    """Owner logits without the impossible classes: the viewer itself, and dummy once face up."""
    cls = torch.arange(4)
    dummy_up = lead_out[..., None] & (cls == dummy_rel[:, None, None]) & (dummy_rel != 0)[:, None, None]
    bad = (cls == 0)[None, None] | dummy_up
    return logits.masked_fill(bad[:, :, None, :], -1e9)


# ------------------------------------------------------------------ position

class Board:
    """One deal as the belief net reads it. `owner` only needs to be right for the cards the seat
    on turn can see (its hand, dummy once faced, the cards played); the rest is never read."""

    def __init__(self, owner, calls, dealer, declarer, level, strain, doubled, vul_ns, vul_ew, sys_ns_ew):
        self.owner = np.asarray(owner, dtype=np.int64)
        self.nc = len(calls)
        self.calls = np.full(W, -1, dtype=np.int64)
        self.calls[:self.nc] = calls
        self.dealer, self.decl, self.level, self.strain, self.dbl = dealer, declarer, level, strain, doubled
        self.trump = TRUMP_OF_STRAIN[strain]
        self.vns, self.vew = vul_ns, vul_ew
        self.sys = sys_ns_ew

    def tensors(self, hist: list[int], nxt: int) -> dict:
        cards = torch.full((1, 52), -1, dtype=torch.long)
        cards[0, :len(hist)] = torch.tensor(hist, dtype=torch.long)
        cards[0, len(hist)] = nxt                          # any card of the actor: fixes seat and controller
        tt = lambda v, dt=torch.long: torch.tensor([v], dtype=dt)
        return dict(owner=torch.as_tensor(self.owner)[None], dealer=tt(self.dealer), vul_ns=tt(self.vns, torch.bool),
                    vul_ew=tt(self.vew, torch.bool), calls=torch.as_tensor(self.calls)[None],
                    n_calls=tt(self.nc), cards=cards, declarer=tt(self.decl), level=tt(self.level),
                    strain=tt(self.strain), doubled=tt(self.dbl), score_ns=torch.zeros(1), tricks=tt(-1),
                    sys_ns=tt(self.sys[0]), sys_ew=tt(self.sys[1]))


@torch.no_grad()
def belief(net: BeliefNet, bd: Board, hist: list[int], nxt: int, ctrl: int) -> torch.Tensor:
    """(52, 4) owner probabilities, seats counted from `ctrl`."""
    tt = bd.tensors(hist, nxt)
    b = build(tt)
    b.update(aux_feats(tt, b))
    k = bd.nc + len(hist)
    s = torch.cat([b["x"][ctrl, k].float(), b["auc"][ctrl, k], b["sysoh"][ctrl]], -1)[None]
    logits, _ = net(s)
    m = masked_belief(logits.view(1, 1, 52, 4), b["lead_out"][ctrl:ctrl + 1, k:k + 1], b["dummy_rel"][ctrl:ctrl + 1])
    return m[0, 0].softmax(-1)


def position(owner, hist, decl, ctrl):
    """Hidden seats, their unseen cards and counts, shown-out suits, the current trick."""
    t = len(hist)
    start = t - t % 4
    seen = {ctrl} | ({(decl + 2) % 4} if t > 0 else set())
    hidden = [s for s in range(4) if s not in seen]
    played = set(hist)
    pool = [c for c in range(52) if owner[c] in hidden and c not in played]
    void = {s: set() for s in range(4)}
    for i in range(t):
        st = i - i % 4
        if i % 4 and SUIT[hist[i]] != SUIT[hist[st]]:
            void[int(owner[hist[i]])].add(int(SUIT[hist[st]]))
    return dict(hidden=hidden, pool=pool, void=void, need={s: sum(1 for c in pool if owner[c] == s) for s in hidden},
                trick=hist[start:], start=start)


def draw(owner, p, bel_abs, n, rng):
    """(n, 52) layouts and their log importance weights (target = product of belief probabilities)."""
    pool = np.array(p["pool"], dtype=np.int64)
    lays = np.repeat(owner[None], n, 0)
    allow = np.zeros((52, 4), bool)
    for s in p["hidden"]:
        allow[:, s] = True
        for u in p["void"].get(s, ()):
            allow[SUIT == u, s] = False
    room = np.zeros((n, 4))
    for s in p["hidden"]:
        room[:, s] = p["need"][s]
    lw = np.zeros(n)
    ok = np.ones(n, bool)
    order = rng.permuted(np.tile(pool, (n, 1)), axis=1)
    rows = np.arange(n)
    bel = np.clip(bel_abs, 1e-6, None)
    for j in range(len(pool)):
        cd = order[:, j]
        pr = bel[cd] * allow[cd] * (room > 0)
        dead = pr.sum(1) <= 0
        if dead.any():
            ok &= ~dead
            pr[dead] = (room[dead] > 0).astype(float)
        pr /= pr.sum(1, keepdims=True)
        st = (pr.cumsum(1) < rng.random(n)[:, None]).sum(1).clip(max=3)
        lw += np.log(bel[cd, st]) - np.log(pr[rows, st])
        lays[rows, cd] = st
        room[rows, st] -= 1
    lw[~ok] = -np.inf
    return lays, lw


def ess(lw):
    if not len(lw) or not np.isfinite(lw).any():
        return 0.0
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w ** 2).sum())


def hand_pbn(cards) -> str:
    return ".".join("".join(RANKS[c % 13] for c in sorted(cards) if c // 13 == su) for su in range(4))


def solve_many(jobs, budget_s=float("inf"), min_solved=0):
    """DD tricks per legal card for each (hands, leader, trick so far, trump); one DDS call per deal.
    Stops early, after at least `min_solved` deals, once `budget_s` seconds have passed."""
    from endplay.dds import solve_board
    from endplay.types import Card, Deal, Denom, Player
    out = []
    t0 = time.perf_counter()
    for rem, leader, trick, trump in jobs:
        if len(out) >= min_solved and time.perf_counter() - t0 > budget_s:
            break
        d = Deal("N:" + " ".join(hand_pbn(h) for h in rem))
        d.trump = Denom(trump)
        d.first = Player(leader)
        for c in trick:
            d.play(Card("SHDC"[c // 13] + RANKS[c % 13]))
        out.append({int(card.suit) * 13 + 15 - int(card.rank).bit_length(): t for card, t in solve_board(d)})
    return out


def pimc_values(bd: Board, hist, leader, ctrl, bel_rel, rng):
    """(cards, mean DD tricks per card) over K belief-weighted layouts. Seats are counted from
    `ctrl` throughout, so the same view turned round the table draws the same layouts."""
    owner = (bd.owner - ctrl) % 4
    p = position(owner, hist, (bd.decl - ctrl) % 4, 0)
    leader = (leader - ctrl) % 4
    bel = bel_rel.numpy().astype(np.float64)
    Ls, lws = [], []
    while True:
        L, lw = draw(owner, p, bel, N_RAW, rng)
        Ls.append(L)
        lws.append(lw)
        lwc = np.concatenate(lws)
        if ess(lwc) >= MIN_ESS or len(lwc) >= MAX_RAW:
            break
    L = np.concatenate(Ls)
    w = np.exp(lwc - lwc.max())
    idx = rng.choice(len(L), K, replace=True, p=w / w.sum())
    u, cnt = np.unique(idx, return_counts=True)
    first = np.argsort(-cnt, kind="stable")
    u, cnt = u[first], cnt[first]
    gone = set(hist[:p["start"]])
    jobs = [([[c for c in range(52) if lay[c] == s and c not in gone] for s in range(4)], leader, p["trick"], bd.trump)
            for lay in L[u]]
    sols = solve_many(jobs, BUDGET_S, MIN_SOLVED)
    cnt = cnt[:len(sols)]
    cards = sorted(sols[0])
    tab = np.array([[s[c] for c in cards] for s in sols], dtype=np.float64)
    return cards, (cnt[:, None] * tab).sum(0) / cnt.sum()


# ------------------------------------------------------------------ bot

class PimcPlayBot:
    """Card play V4. `tie` is the card player that breaks PIMC ties (card play V3)."""

    def __init__(self, path: str, tie):
        st = torch.load(path, map_location="cpu", weights_only=True)
        c = st["cfg"]
        self.net = BeliefNet(c["width"], c["blocks"], c["hidden"]).eval()
        self.net.load_state_dict(st["net"])
        self.tie = tie
        self.lock = threading.Lock()

    def choose_card(self, contracts, batch, search=None, seed=None):
        c = contracts.subset([0]) if len(contracts) > 1 else contracts
        hist = [int(x) for x in batch.history[0, :batch.t]]
        actor = int(batch.to_play()[0])
        decl = int(c.declarer[0])
        ctrl = decl if actor == (decl + 2) % 4 else actor
        legal = [int(x) for x in torch.nonzero(batch.legal()[0]).flatten()]
        if len(legal) == 1:
            return legal[0], [(legal[0], 1.0)]
        us, them = SYSTEMS.get()
        sys_ns_ew = (us, them) if ctrl % 2 == 0 else (them, us)
        trump = int(c.trump[0])
        bd = Board(c.owner[0].numpy(), [int(x) for x in c.calls[0, :int(c.n_calls[0])]], int(c.dealer[0]), decl,
                   int(c.level[0]), TRUMP_OF_STRAIN.index(trump), int(c.doubled[0]),
                   bool(c.vul_ns[0]), bool(c.vul_ew[0]), sys_ns_ew)
        leader = int(batch.leader[0])
        rng = np.random.default_rng(seed)
        with self.lock, torch.inference_mode():
            bel = belief(self.net, bd, hist, legal[0], ctrl)
            cards, mean = pimc_values(bd, hist, leader, ctrl, bel, rng)
        tied = [cards[i] for i in np.flatnonzero(mean >= mean.max() - TIE_EPS)]
        card = int(rng.choice(tied))
        if len(tied) > 1:
            v3 = int(self.tie.choose_card(contracts, batch, search, seed)[0])
            if v3 in tied:
                card = v3
        order = np.argsort(-mean, kind="stable")
        return card, [(cards[i], float(mean[i])) for i in order[:4]]
