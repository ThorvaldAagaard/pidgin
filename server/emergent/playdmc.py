"""The DMC card player (run 6): one value per legal card, learned by self-play, played greedy.

Trained DouZero-style (deep Monte Carlo): the net plays whole boards against itself and a league of
its past snapshots, and learns each card's value from the final result. No double dummy solver is
used in training. One residual MLP serves all three roles (declarer with dummy, LHO, RHO); a learned
role vector is added after its first layer. See docs/card_play.md.

`features` is a port of the trainer's `env.static_features` + `env.dynamic_features` with its
`--rel-feats` and `--auction-seats` columns (lab experiments/douzero_play/env.py), so play-time
inputs match training exactly. Seats are counted from declarer: 0 declarer, 1 LHO, 2 dummy, 3 RHO.

It plays its best value and never searches: PIMC on top of it measured no better (-0.07 IMPs a
board). `DMCPlayBot.choose_card` has the signature of `engine.choose_card`, which hands over to it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from training.play.data import N_CALLS

SEQ = 24                                   # auction calls one-hot in order; the rest only in the bag
STATIC_DIM = 4 * N_CALLS + SEQ * (N_CALLS + 1) + 4 + 7 + 5 + 3 + 2 + SEQ * 4
REL_DIM = 4 * 3 * 52 + 52
DYN_DIM = 52 + 52 + 4 + 3 * 4 * 52 + 13 + 4 + 14 + 14 + REL_DIM
FEAT_DIM = STATIC_DIM + DYN_DIM            # 2658
ROLE = torch.tensor([0, 1, 0, 2])          # seat from declarer -> role: declarer side, LHO, RHO


def static_features(c):
    """(n, STATIC_DIM): the auction (calls by seat from declarer, the first SEQ calls in order with
    who made them, dealer), contract level / strain / doubling, each side's vulnerability."""
    n, decl = len(c.owner), c.declarer
    w = c.calls.shape[1]
    pos = torch.arange(w)
    live = pos[None] < c.n_calls[:, None]
    rel_seat = (c.dealer[:, None] + pos[None] - decl[:, None]) % 4
    bag = torch.zeros(n, 4, N_CALLS + 1, dtype=torch.uint8)
    bag.view(n, -1).scatter_(1, torch.where(live, rel_seat * (N_CALLS + 1) + c.calls, N_CALLS), 1)
    seq = F.pad(c.calls[:, :SEQ], (0, max(0, SEQ - w)), value=N_CALLS)
    ns = decl % 2 == 0
    mine = torch.where(ns, c.vul_ns, c.vul_ew)
    theirs = torch.where(ns, c.vul_ew, c.vul_ns)
    rs = F.pad(rel_seat, (0, max(0, SEQ - w)))[:, :SEQ]
    lv = F.pad(live, (0, max(0, SEQ - w)))[:, :SEQ]
    parts = [bag[:, :, :N_CALLS].reshape(n, -1), F.one_hot(seq, N_CALLS + 1).reshape(n, -1),
             F.one_hot((c.dealer - decl) % 4, 4), F.one_hot(c.level - 1, 7), F.one_hot(c.trump, 5),
             F.one_hot(c.doubled, 3), mine[:, None], theirs[:, None],
             (F.one_hot(rs, 4) * lv[..., None]).reshape(n, -1)]
    return torch.cat([p.to(torch.uint8) for p in parts], 1)


def dynamic_features(b, decl):
    """(n, DYN_DIM) for the seat on turn, from the cards it may see: the hand it plays from, the
    other hand face up once the lead is made (dummy, or declarer's when dummy is on turn), and the
    public play. Hidden cards never enter."""
    n, t = b.n, b.t
    rel_turn = (b.to_play() - decl) % 4
    own = b.hand(b.to_play())
    other = b.hand(torch.where(rel_turn == 2, decl, (decl + 2) % 4)) & (t > 0)
    oh = F.one_hot((b.owner - decl[:, None]) % 4, 4).bool()       # (n, 52, 4) holder from declarer
    lead = torch.zeros(n, 52, dtype=torch.bool)
    if t:
        lead.scatter_(1, b.history[:, 0:t:4], True)
    cur = torch.zeros(n, 52, dtype=torch.bool)
    start = (t // 4) * 4
    if t > start:
        cur.scatter_(1, b.history[:, start:t], True)
    won = b.tricks_won
    dw = won.gather(1, (decl % 2)[:, None]).squeeze(1)
    fw = won.gather(1, (1 - decl % 2)[:, None]).squeeze(1)
    by = [(oh & m[..., None]).transpose(1, 2).reshape(n, -1) for m in (~b.unplayed, lead, cur)]
    parts = [own, other, F.one_hot(rel_turn, 4), *by,
             F.one_hot(torch.full((n,), t // 4), 13), F.one_hot(torch.full((n,), t % 4), 4),
             F.one_hot(dw, 14), F.one_hot(fw, 14), rel_features(b, own, other, cur)]
    return torch.cat([p.to(torch.uint8) for p in parts], 1)


def rel_features(b, own, other, cur):
    """(n, REL_DIM): per card, how many higher cards of its suit are in each place (own hand, the
    other hand seen, unseen, the current trick) as >=1, >=2, >=3; then whether it would beat the
    card now winning the trick (all 0 on the lead)."""
    n, t = b.n, b.t
    unseen = b.unplayed & ~own & ~other
    out = []
    for m in (own, other, unseen, cur):
        v = m.view(n, 4, 13).long()
        hi = (v.cumsum(2) - v).view(n, 52)                          # rank 0 = ace
        out += [hi >= 1, hi >= 2, hi >= 3]
    beats = torch.zeros(n, 52, dtype=torch.bool)
    if t % 4:
        tc = b.trick_cards()
        suit, rank, trump = b.suit_of, b.rank_of, b.trump[:, None]
        led = suit[tc[:, :1]]
        sc = torch.where(suit[tc] == trump, 200 - rank[tc], torch.where(suit[tc] == led, 100 - rank[tc], 0))
        win = tc.gather(1, sc.argmax(1, keepdim=True))
        ws, wr = suit[win], rank[win]
        beats = ((suit[None] == ws) & (rank[None] < wr)) | ((suit[None] == trump) & (ws != trump))
    return torch.cat(out + [beats], 1)


def features(contracts, batch):
    """(n, FEAT_DIM) float input for the seat on turn at the batch's current step."""
    x = torch.cat([static_features(contracts), dynamic_features(batch, contracts.declarer)], 1)
    return x.float()


class DMCNet(nn.Module):
    """The trainer's shared-MLP QNet: input layer, residual blocks, 52 values; `role_emb[r]` is added
    to the first layer's output for role r."""

    def __init__(self, d_in=FEAT_DIM, width=1024, depth=6):
        super().__init__()
        self.role_emb = nn.Parameter(torch.zeros(3, width))
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                                  nn.Linear(width, width)) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 52))

    def forward(self, x, role):
        h = self.inp(x) + self.role_emb[role]
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def slim(ck):
    """The serving file's dict from a trainer checkpoint: its three role state dicts are three
    copies of one shared core, so one copy is kept. Float32: float16 weights changed the result
    of 2 boards in 400 against the trainer's own net."""
    a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    assert ck.get("arch") == "mlp" and a["shared_mlp"] and a["rel_feats"] and a["auction_seats"], \
        "this port reads the shared MLP with --rel-feats and --auction-seats only"
    assert not a.get("oracle_guide") and ck["feat_dim"] == FEAT_DIM
    nets = ck["nets"]
    assert all(torch.equal(nets[0][k], s[k]) for s in nets[1:] for k in nets[0]), "role nets differ"
    core = {k.removeprefix("core."): v.float() for k, v in nets[0].items()}
    return {"net": core, "width": a["width"], "depth": a["depth"], "feat_dim": ck["feat_dim"],
            "step": ck["step"], "averaged_from": ck.get("averaged_from")}


class DMCPlayBot:
    family = "dmc"

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        with torch.device("meta"):                 # no second copy of the 62 MB: take the loaded tensors
            self.net = DMCNet(ck["feat_dim"], ck["width"], ck["depth"])
        self.net.load_state_dict(ck["net"], assign=True)
        self.net.eval()
        self.step = ck["step"]
        self.hint_file = None     # teams.json "hints": the policy net the table's hints quote

    @torch.no_grad()
    def values(self, contracts, batch):
        """(n, 52) values for the seat on turn, -1e9 where illegal."""
        role = ROLE[(batch.to_play() - contracts.declarer) % 4]
        return self.net(features(contracts, batch), role).masked_fill(~batch.legal(), -1e9)

    @torch.no_grad()
    def choose_card(self, contracts, batch, search=None, seed=None):
        """(card, top) as `engine.choose_card`: the best value, never a search (`search` and `seed`
        are accepted and ignored); top is the best four legal cards as [(card, value)], values in
        tricks below the best card (0 = best)."""
        q = self.values(contracts, batch)[0]
        legal = batch.legal()[0]
        card = int(q.argmax())                       # as in training: the first of tied values
        order = [int(c) for c in q.argsort(descending=True) if legal[int(c)]]
        return card, [(c, float(q[c] - q[card])) for c in order[:4]]
