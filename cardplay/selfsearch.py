"""--search-*: true-deal branching for train.py (shared MLP only). No double dummy anywhere.

Each actor round: pick one card step t (the same for all boards of the round; the engine plays in
lockstep) and up to `n_pos` boards with more than one legal card at t. On each, play every legal card
on the TRUE deal and play the rest out greedily with the current nets in all four seats. Target for
each legal card = declarer tricks - DD tricks of the deal, sign flipped for defenders: the same units
as the Q target. Over many deals this is an unbiased sample of Q(information state, card) under the
current policy, for every legal card, not only the played one. The learner then fits Q(s, a) for
every legal card at these positions.

leads (--search-leads): t is the first card of a trick, t=0 (opening lead) with probability open_p,
else the lead to a random later trick.
"""
from __future__ import annotations

import torch

from training.bridge.play import PlayBatch

from . import env


class Rollout:
    """Greedy role nets for every seat of a rollout. Static first-layer projection cached per base board."""

    def __init__(self, nets, device):
        self.nets, self.device = nets, device

    def contracts(self, c):
        """`c` on self.device, cached (the rollouts index it every step)."""
        if getattr(self, "_c", (None,))[0] is not c:
            self._c = (c, c.to(self.device))
        return self._c[1]

    @torch.no_grad()
    def prepare(self, static):
        s = static.to(self.device).float()
        self.sp = [n.static_proj(s) for n in self.nets]

    @torch.no_grad()
    def act(self, b, decl, base):
        """Greedy card for every row of PlayBatch `b`; `base` = row of the prepared boards."""
        d = self.device
        x = env.dynamic_features(b, decl).to(d).float()
        role = env.roles(b, decl).to(d)
        base = base.to(d)
        q = torch.empty(b.n, 52, device=d)
        for r, net in enumerate(self.nets):
            idx = (role == r).nonzero().squeeze(1)
            if len(idx):
                q[idx] = net.trunk(self.sp[r][base[idx]] + net.dyn_proj(x[idx]))
        q = q.masked_fill(~b.legal().to(d), float("-inf"))
        return q.argmax(1).to(b.device)


@torch.no_grad()
def rollout_values(player, b, c, rows, owners, chunk):
    """(m,52) mean final declarer tricks per legal card (nan for illegal) over the k layouts `owners`
    (m,k,52): each legal card is played on each layout and the rest is played out by `player`."""
    m, k, _ = owners.shape
    legal = b.legal()[rows]
    pr, pc = legal.nonzero(as_tuple=True)
    e_row = pr.repeat_interleave(k)
    e_card = pc.repeat_interleave(k)
    e_samp = torch.arange(k, device=b.device).repeat(len(pr))
    vals = torch.empty(len(e_row), device=b.device)
    dev = player.device
    cd = player.contracts(c)
    for s in range(0, len(e_row), chunk):
        er, ec, es = e_row[s:s + chunk], e_card[s:s + chunk], e_samp[s:s + chunk]
        base = rows[er]
        bd = base.to(dev)
        pb = PlayBatch(owners[er, es].to(dev), cd.trump[bd], cd.declarer[bd])
        hist = b.history[base, :b.t].to(dev)
        for i in range(b.t):
            pb.play(hist[:, i])
        pb.play(ec.to(dev))
        while not pb.done:
            pb.play(player.act(pb, cd.declarer[bd], bd))
        vals[s:s + chunk] = pb.declarer_tricks().float().to(vals.device)
    out = torch.full((m, 52), float("nan"), device=b.device)
    out[pr, pc] = vals.view(len(pr), k).mean(1)
    return out


@torch.no_grad()
def search_targets(nets, c, cards, static, role, n_pos, gen, device, chunk, stats, leads=False, open_p=0.0):
    """Returns sample ids s = t*n + board (m,) into the round's store, and (m,52) per-card targets from the
    mover's side (nan = illegal). c: the round's contracts (CPU); cards/static/role: actor_round outputs."""
    n = len(c)
    if not leads:
        t = int(torch.randint(52, (1,), generator=gen))
    elif float(torch.rand(1, generator=gen)) < open_p:
        t = 0
    else:
        t = 4 * int(torch.randint(1, 13, (1,), generator=gen))
    b = PlayBatch(c.owner, c.trump, c.declarer)
    hist = cards.view(52, n)[:t].cpu()
    for i in range(t):
        b.play(hist[i])
    multi = (b.legal().sum(1) > 1).nonzero().squeeze(1)
    rows = multi[torch.randperm(len(multi), generator=gen)[:n_pos]].sort().values
    if len(rows) == 0:
        return None
    sid = (t * n + rows).to(device)
    mover = role[sid].cpu()
    owners = c.owner[rows][:, None, :].clone()                    # the true deal, one layout
    player = Rollout(nets, device)
    player.prepare(static)
    v = rollout_values(player, b, c, rows, owners, chunk) - c.dd_tricks[rows, None].float()
    v = torch.where((mover == 0)[:, None], v, -v)
    stats["positions"] += len(rows); stats["t"].append(t)
    return sid, v.to(device)


def search_learn(core, opt, x, rl, v, minibatch, w, clip):
    """One step: MSE of Q(s, a) to the search target for every legal card, on a minibatch of the buffer."""
    j = torch.randperm(len(x), device=x.device)[:minibatch]
    q = core.out(core.body(core.inp(x[j].float()) + core.role_emb[rl[j]]))
    legal = ~torch.isnan(v[j])
    loss = ((q - v[j].nan_to_num()) ** 2)[legal].mean()
    opt.zero_grad(set_to_none=True)
    (w * loss).backward()
    torch.nn.utils.clip_grad_norm_(core.parameters(), clip)
    opt.step()
    return float(loss.detach())
