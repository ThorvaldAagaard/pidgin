"""Q-nets and the players that sit at the table.

Model: residual MLP, input = binary features (env), output = 52 Q-values (expected final result for
the acting side), illegal cards masked. A 52-way head over a flat history encoding needs one forward
per decision instead of one per legal card, which is what makes the batched table fast.
"""
from __future__ import annotations

import importlib.util

import torch
from torch import nn

from training.bridge.play import PlayBatch

from . import env
from .env import STATIC_DIM, dynamic_features, roles, static_features

ROLE_NAMES = ("decl", "lho", "rho")


class QNet(nn.Module):
    def __init__(self, d_in: int = env.FEAT_DIM, width: int = 512, depth: int = 3, static_dim: int = STATIC_DIM):
        super().__init__()
        self.static_dim = static_dim
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                                  nn.Linear(width, width)) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 52))

    def forward(self, x):
        return self.trunk(self.inp(x))

    def trunk(self, h):
        return self.out(self.body(h))

    def body(self, h):
        """Residual blocks: first-layer output -> last hidden layer (what the aux heads read)."""
        for blk in self.blocks:
            h = h + blk(h)
        return h

    # Acting split of the first layer: features are [static | dynamic] and the static part is
    # fixed for a whole board, so its projection is computed once per board (train.actor_round).
    def static_proj(self, s):
        return s @ self.inp.weight[:, :self.static_dim].T + self.inp.bias

    def dyn_proj(self, d):
        return d @ self.inp.weight[:, self.static_dim:].T


class SharedCore(QNet):
    """--shared-mlp: one QNet for all three roles; the role enters as a learned vector added to the
    first-layer output (same as a role one-hot input)."""

    def __init__(self, d_in: int = env.FEAT_DIM, width: int = 512, depth: int = 3, static_dim: int = STATIC_DIM):
        super().__init__(d_in, width, depth, static_dim)
        self.role_emb = nn.Parameter(torch.zeros(3, width))


class RoleView(nn.Module):
    """The shared core seen as role r's net: the same interface as QNet (inp/body/out/trunk/
    static_proj/dyn_proj), so actors, players and evaluators need no change. The three views share
    every parameter; their state_dicts are three copies of the core."""

    def __init__(self, core: SharedCore, r: int):
        super().__init__()
        self.core, self.r = core, r

    @property
    def static_dim(self):
        return self.core.static_dim

    def inp(self, x):
        return self.core.inp(x) + self.core.role_emb[self.r]

    def forward(self, x):
        return self.core.trunk(self.inp(x))

    def trunk(self, h):
        return self.core.trunk(h)

    def body(self, h):
        return self.core.body(h)

    def out(self, h):
        return self.core.out(h)

    def static_proj(self, s):
        return self.core.static_proj(s) + self.core.role_emb[self.r]

    def dyn_proj(self, d):
        return self.core.dyn_proj(d)


def make_nets(width=512, depth=3, seed=0, shared=False, oracle=False):
    """Three role nets (or three views of one shared core). oracle: + the ORACLE_DIM static columns."""
    torch.manual_seed(seed)
    d_in, s_dim = env.feat_dim(oracle), env.static_dim(oracle)
    if shared:
        core = SharedCore(d_in, width, depth, s_dim)
        return [RoleView(core, r) for r in range(3)]
    return [QNet(d_in, width, depth, s_dim) for _ in range(3)]


def pad_oracle(sd: dict) -> dict:
    """A state dict trained without oracle columns, with zero first-layer weights inserted for them
    (the net then computes exactly what it did; it can learn to use the columns)."""
    s0 = STATIC_DIM
    return {k: torch.cat([v[:, :s0], v.new_zeros(v.shape[0], env.ORACLE_DIM), v[:, s0:]], 1)
            if k.endswith("inp.weight") else v for k, v in sd.items()}


def make_belief_head(width=512):
    """--aux-belief: per card, logits over its holder (seat relative to declarer), read off the last
    hidden layer; trained only on cards the acting side cannot see."""
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 52 * 4))


def make_aux_heads(width=512):
    """--aux-trick: per role, logits of P(acting side wins the current trick | state, card), read off
    the last hidden layer. Kept outside QNet (own optimiser, own ckpt key) so the Q-nets' parameters,
    optimiser state and checkpoints stay exactly as without it."""
    return [nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 52)) for _ in range(3)]


class Table:
    """One batch of boards being played: contracts, engine, cached static features.
    `static` never holds oracle columns; `oracle` (n, ORACLE_DIM) is what nets with oracle columns
    read there: zeros, or every holder when `open_hand` (the open-hand eval).
    `device` puts the engine and features there (static features are built on CPU, then copied)."""

    def __init__(self, c, device=None, open_hand=False):
        self.static = static_features(c)
        self.oracle = env.oracle_columns(c) if open_hand else torch.zeros(len(c), env.ORACLE_DIM, dtype=torch.uint8)
        if device is not None:
            c = c.to(device)
            self.static = self.static.to(device)
            self.oracle = self.oracle.to(device)
        self.c = c
        self.b = PlayBatch(c.owner, c.trump, c.declarer)

    def features(self, oracle=False) -> torch.Tensor:
        parts = [self.static, self.oracle] if oracle else [self.static]
        return torch.cat(parts + [dynamic_features(self.b, self.c.declarer)], 1)


class NetPlayer:
    """The three role nets, greedy with epsilon-random legal cards."""

    def __init__(self, nets, device="cpu", eps=0.0, generator=None):
        self.nets, self.device, self.eps, self.g = nets, device, eps, generator
        self.oracle = nets[0].static_dim > STATIC_DIM

    @torch.no_grad()
    def act(self, table: Table, rows: torch.Tensor) -> torch.Tensor:
        b = table.b
        x = table.features(self.oracle)
        legal = b.legal()
        role = roles(b, table.c.declarer)
        card = torch.empty(b.n, dtype=torch.long)
        for r in range(3):
            sel = rows & (role == r)
            if not sel.any():
                continue
            idx = sel.nonzero().squeeze(1)
            q = self.nets[r](x[idx].to(self.device).float()).cpu()   # blocking copies only (MPS)
            card[idx] = q.masked_fill(~legal[idx], -1e9).argmax(1)
        if self.eps > 0:
            flip = (torch.rand(b.n, generator=self.g) < self.eps) & rows
            if flip.any():
                rnd = torch.multinomial(legal[flip].float(), 1, generator=self.g).squeeze(1)
                card[flip] = rnd
        return card


class RandomPlayer:
    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)

    def act(self, table, rows):
        return torch.multinomial(table.b.legal().float(), 1, generator=self.g).squeeze(1)


def _server_module(name):
    """server/emergent/<name>.py, imported by file path (server/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"cardplay_{name}", env.ROOT / "server/emergent" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class QNetPlayer:
    """The released solver-trained play Q-net (play_B2g_s540k.pt), greedy, no search
    (server/emergent/playq.py). The trainer's evaluation opponent."""

    def __init__(self, path=env.QNET):
        self.mod = _server_module("playq")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = self.mod.QNet(ck["input_dim"], ck["width"], ck["depth"])
        self.net.load_state_dict(ck["net"])
        self.net.eval()

    @torch.no_grad()
    def act(self, table, rows):
        b = table.b
        idx = rows.nonzero().squeeze(1)
        card = torch.zeros(b.n, dtype=torch.long)
        if len(idx) == 0:
            return card
        d = self.mod.position(table.c, b, b.t)
        f = self.mod.features(d, idx, torch.full((len(idx),), b.t))
        q = self.net(f).masked_fill(~b.legal()[idx], -1e9)
        card[idx] = q.argmax(1)
        return card


def play_out(c, decl_player, def_player, open_hand=False) -> torch.Tensor:
    """Play every board; returns declarer tricks (n,). open_hand: the oracle columns hold every card's
    true holder, so nets trained with them see all hands; other players ignore them."""
    table = Table(c, open_hand=open_hand)
    b = table.b
    while not b.done:
        side = roles(b, c.declarer) == 0
        card = torch.empty(b.n, dtype=torch.long)
        for player, rows in ((decl_player, side), (def_player, ~side)):
            if rows.any():
                card[rows] = player.act(table, rows)[rows]
        b.play(card)
    return b.declarer_tricks()
