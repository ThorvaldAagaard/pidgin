"""Card play from self-play by deep Monte Carlo (inspired by DouZero), from random weights or a checkpoint.

Each step: actors play `--boards` boards in lockstep with the current nets (epsilon-greedy, some boards
against a past snapshot); every decision is stored as (features, card, role). When the boards end each
decision gets a target for its side: declarer tricks minus the deal's double-dummy tricks (`--return
tricks`), negated for the defenders, optionally blended with the nets' later estimates by TD(lambda) or
replaced by an open-hand critic's value. The learner then makes `--passes` shuffled passes of squared
error on the played card's Q value. See cardplay/README.md for the released recipes.

    python -m cardplay.train --out runs/cardplay/smoke --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import env
from .env import STATIC_DIM, dynamic_features, roles
from .model import (ROLE_NAMES, NetPlayer, QNetPlayer, RandomPlayer, Table, make_aux_heads, make_belief_head,
                    make_nets, pad_oracle, play_out)

RUNS = env.ROOT / "runs" / "cardplay"
SMOKE_DATA = env.ROOT / "data" / "smoke_128.npz"


def oracle_keep(args, step):
    """--oracle-guide keep rate: --oracle-from until --oracle-hold, then linear to 0 at --oracle-until."""
    if not args.oracle_guide:
        return 0.0
    if step < args.oracle_hold:
        return args.oracle_from
    return max(0.0, args.oracle_from * (1 - (step - args.oracle_hold) / max(1, args.oracle_until - args.oracle_hold)))


def has_oracle(nets) -> bool:
    return nets[0].static_dim > STATIC_DIM


def actor_round(nets, c, device, eps, gen, want_v=False, league=None, oracle=0.0, critic=None):
    """Play contracts `c` with epsilon-greedy nets, all on `device` in lockstep (52 card steps).
    All three role nets score every row and the row's role picks its Q-values (no per-role gather,
    no host syncs). The static part of the first layer is computed once per board.
    `gen` is a torch.Generator on `device` (epsilon flips and random legal cards).
    Returns, all on `device`: static (n,static_dim) uint8, Xd (S,DYN_DIM) uint8, card (S,), role (S,),
    board (S,) row of `static` for each sample, declarer tricks (n,), and `extra` (dict).
    Sample s = step*n + board.

    want_v: extra["v_decl"] (52,n) = V_decl of the state before each decision, from the CURRENT nets'
      greedy Q (max over legal cards of the role to move, sign + for declarer side, - for defenders).
    league: (snap_nets, rows (k,) long board indices, snap_decl (k,) bool). On those boards the
      snapshot plays the declarer side (snap_decl) or the defenders; it plays greedily (no eps).
      extra["mine"] (S,) bool marks decisions made by the current nets (the only ones to train on).
      V_decl stays the current nets' value at snapshot turns too.
    oracle: --oracle-guide keep rate: each card's true holder (rel. declarer) is shown in the static
      oracle columns with this probability, per board and card, to every seat (snapshots too).
    critic: --critic role nets (oracle columns). They see every holder (keep 1, they never act).
      extra["static_full"] (n,static_dim) is that static; extra["qc"] (52,n) = critic Q(s_t, played card),
      sign + for declarer side, - for defenders (every decision, league turns too)."""
    table = Table(c, device)
    b, decl, n = table.b, table.c.declarer, table.b.n
    static = torch.cat([table.static, table.oracle], 1) if has_oracle(nets) else table.static
    if oracle > 0:
        rel = (table.c.owner - decl[:, None]) % 4
        keep = torch.rand(n, 52, device=device, generator=gen) < oracle
        static[:, -env.ORACLE_DIM:] = (F.one_hot(rel, 4) * keep[..., None]).reshape(n, -1).to(torch.uint8)
    extra = {}
    if critic is not None:
        static_full = static.clone()
        static_full[:, -env.ORACLE_DIM:] = F.one_hot((table.c.owner - decl[:, None]) % 4, 4).reshape(n, -1).to(torch.uint8)
        extra["static_full"] = static_full
    with torch.no_grad():
        sf = static.float()
        h0 = [net.static_proj(sf) for net in nets]
        Xd = torch.empty(52, n, env.DYN_DIM, dtype=torch.uint8, device=device)
        cards = torch.empty(52, n, dtype=torch.long, device=device)
        rs = torch.empty(52, n, dtype=torch.long, device=device)
        if want_v:
            vd = torch.empty(52, n, device=device)
        if critic is not None:
            hc = [net.static_proj(static_full.float()) for net in critic]
            qc_t = torch.empty(52, n, device=device)
        if league is not None:
            snets, lrows, ldecl = league
            sh0 = [net.static_proj(sf[lrows]) for net in snets]
            mine = torch.ones(52, n, dtype=torch.bool, device=device)
        while not b.done:
            t = b.t
            xd = dynamic_features(b, decl)
            Xd[t] = xd
            xf = xd.float()
            q = torch.stack([net.trunk(h + net.dyn_proj(xf)) for net, h in zip(nets, h0)])   # (3,n,52)
            role = roles(b, decl)
            q = q.gather(0, role[None, :, None].expand(1, n, 52)).squeeze(0)
            legal = b.legal()
            qm = q.masked_fill(~legal, -1e9)
            card = qm.argmax(1)
            if want_v:
                v = qm.max(1).values
                vd[t] = torch.where(role == 0, v, -v)
            if eps > 0:
                flip = torch.rand(n, device=device, generator=gen) < eps
                rnd = torch.rand(n, 52, device=device, generator=gen).masked_fill(~legal, -1.0).argmax(1)
                card = torch.where(flip, rnd, card)      # uniform random legal card
            if league is not None:
                xl = xf[lrows]
                lrole = role[lrows]
                sq = torch.stack([net.trunk(h + net.dyn_proj(xl)) for net, h in zip(snets, sh0)])
                sq = sq.gather(0, lrole[None, :, None].expand(1, len(lrows), 52)).squeeze(0)
                scard = sq.masked_fill(~legal[lrows], -1e9).argmax(1)
                snap_turn = (lrole == 0) == ldecl
                card[lrows] = torch.where(snap_turn, scard, card[lrows])
                mine[t, lrows] = ~snap_turn
            if critic is not None:
                qc = torch.stack([net.trunk(h + net.dyn_proj(xf)) for net, h in zip(critic, hc)])
                qc = qc.gather(0, role[None, :, None].expand(1, n, 52)).squeeze(0).gather(1, card[:, None]).squeeze(1)
                qc_t[t] = torch.where(role == 0, qc, -qc)
            cards[t] = card; rs[t] = role
            b.play(card)
    if want_v:
        extra["v_decl"] = vd
    if critic is not None:
        extra["qc"] = qc_t
    if league is not None:
        extra["mine"] = mine.reshape(-1)
    extra["trick_winner"] = b.trick_winner
    board = torch.arange(n, device=device).expand(52, n).reshape(-1)
    return static, Xd.reshape(-1, env.DYN_DIM), cards.reshape(-1), rs.reshape(-1), board, b.declarer_tricks(), extra


def trick_labels(trick_winner, decl, role, n):
    """(S,) float: 1 if the acting side won the trick that decision s belongs to (s = t*n + board)."""
    decl_won = ((trick_winner - decl[:, None]) % 2 == 0)                 # (n,13)
    won = decl_won.T.repeat_interleave(4, 0).reshape(-1)                  # (52*n,) row t*n+b -> trick t//4
    return (won == (role == 0)).float()


def learn(nets, opts, static, Xd, card, role, board, target, minibatch, clip, passes=1, mine=None, aux=None):
    """`passes` shuffled passes over the samples, per role (separate role nets); everything on the device.
    mine: (S,) bool, train only on these samples (league: decisions of the current nets).
    aux: (heads, aux_opts, labels (S,), weight): + weight * BCE of the trick head on the played card.
    Returns mean Q loss per role over all minibatches (+ "aux" mean BCE when aux is on)."""
    losses = {}
    aux_tot, aux_k = torch.zeros((), device=Xd.device), 0
    for r in range(3):
        sel = role == r if mine is None else (role == r) & mine
        idx_r = sel.nonzero().squeeze(1)
        tot, k = torch.zeros((), device=Xd.device), 0
        for _ in range(passes):
            idx = idx_r[torch.randperm(len(idx_r), device=Xd.device)]
            for s in range(0, len(idx), minibatch):
                j = idx[s:s + minibatch]
                if len(j) < minibatch // 4:
                    break
                x = torch.cat([static[board[j]], Xd[j]], 1).float()
                if aux is None:
                    q = nets[r](x).gather(1, card[j][:, None]).squeeze(1)
                    loss = F.mse_loss(q, target[j])
                else:
                    heads, aux_opts, labels, w = aux
                    h = nets[r].body(nets[r].inp(x))
                    q = nets[r].out(h).gather(1, card[j][:, None]).squeeze(1)
                    loss = F.mse_loss(q, target[j])
                    logit = heads[r](h).gather(1, card[j][:, None]).squeeze(1)
                    la = F.binary_cross_entropy_with_logits(logit, labels[j])
                    aux_opts[r].zero_grad(set_to_none=True)
                    aux_tot += la.detach(); aux_k += 1
                opts[r].zero_grad(set_to_none=True)
                (loss if aux is None else loss + w * la).backward()
                torch.nn.utils.clip_grad_norm_(nets[r].parameters(), clip)
                opts[r].step()
                if aux is not None:
                    torch.nn.utils.clip_grad_norm_(heads[r].parameters(), clip)
                    aux_opts[r].step()
                tot += loss.detach(); k += 1
        losses[ROLE_NAMES[r]] = float(tot) / max(k, 1)
    if aux is not None:
        losses["aux"] = float(aux_tot) / max(aux_k, 1)
    return losses


VIS = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.bool)   # role -> seats (rel. decl) seen before the lead


def learn_shared(nets, opt, static, Xd, card, role, board, target, minibatch, clip, passes=1, mine=None, aux=None,
                 belief=None):
    """learn() for --shared-mlp: `passes` shuffled passes over all samples, roles mixed in each minibatch,
    one optimiser (the three views share the core). aux as in learn() (each sample uses its role's trick head).
    belief: (head, opt, rel_owner (n,52) holder seat rel. declarer, weight): + weight * cross-entropy of the
    holder of every card the acting side cannot see (not played, not in its own or a visible hand; dummy is
    visible from the second card on). Returns mean Q loss per role (+ "aux", "belief")."""
    core = nets[0].core
    dev, n = Xd.device, static.shape[0]
    idx_all = torch.arange(len(card), device=dev) if mine is None else mine.nonzero().squeeze(1)
    tot, cnt = torch.zeros(3, device=dev), torch.zeros(3, device=dev)
    aux_tot, bel_tot, k = torch.zeros((), device=dev), torch.zeros((), device=dev), 0
    gn_tot = torch.zeros((), device=dev)
    if belief is not None:
        bhead, bopt, rel_owner, bw = belief
        oh = F.one_hot(card.view(52, n), 52)
        played = (oh.cumsum(0) - oh).bool().reshape(-1, 52)                 # cards played before sample s
        vis = VIS.to(dev)
    for _ in range(passes):
        idx = idx_all[torch.randperm(len(idx_all), device=dev)]
        for s in range(0, len(idx), minibatch):
            j = idx[s:s + minibatch]
            if len(j) < minibatch // 4:
                break
            x = torch.cat([static[board[j]], Xd[j]], 1).float()
            rj = role[j]
            h = core.body(core.inp(x) + core.role_emb[rj])
            q = core.out(h).gather(1, card[j][:, None]).squeeze(1)
            err = (q - target[j]) ** 2
            loss = err.mean()
            total = loss
            if aux is not None:
                heads, aux_opts, labels, w = aux
                logit = torch.stack([hd(h) for hd in heads]).gather(0, rj[None, :, None].expand(1, len(j), 52))
                logit = logit.squeeze(0).gather(1, card[j][:, None]).squeeze(1)
                la = F.binary_cross_entropy_with_logits(logit, labels[j])
                total = total + w * la
                aux_tot += la.detach()
                for o in aux_opts:
                    o.zero_grad(set_to_none=True)
            if belief is not None:
                ro = rel_owner[board[j]]                                       # (m,52)
                seats = vis[rj].clone()
                seats[:, 2] |= (j // n) > 0
                hidden = ~seats.gather(1, ro) & ~played[j]
                lb = F.cross_entropy(bhead(h).view(len(j), 52, 4)[hidden], ro[hidden])
                total = total + bw * lb
                bel_tot += lb.detach()
                bopt.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gn_tot += torch.nn.utils.clip_grad_norm_(core.parameters(), clip).detach()   # pre-clip norm
            opt.step()
            if aux is not None:
                for hd, o in zip(heads, aux_opts):
                    torch.nn.utils.clip_grad_norm_(hd.parameters(), clip)
                    o.step()
            if belief is not None:
                torch.nn.utils.clip_grad_norm_(bhead.parameters(), clip)
                bopt.step()
            oh_r = F.one_hot(rj, 3).float()
            tot += (err.detach()[:, None] * oh_r).sum(0); cnt += oh_r.sum(0)
            k += 1
    losses = {ROLE_NAMES[r]: float(tot[r] / cnt[r].clamp(min=1)) for r in range(3)}
    if aux is not None:
        losses["aux"] = float(aux_tot) / max(k, 1)
    if belief is not None:
        losses["belief"] = float(bel_tot) / max(k, 1)
    losses["gnorm"] = float(gn_tot) / max(k, 1)
    return losses


class League:
    """Frozen past snapshots (ckpt_step*.pt in the run dir), the `keep` most recent on the device, sampled
    uniformly. best > 0 (--league-best): that share of league rounds uses a champion instead: the snapshot
    with the best declaring score vs the Q-net declares (or, half the time, the best defending one defends).
    Score of a snapshot = mean of the trainer evals within +-200 steps of it (metrics.jsonl), so one lucky
    eval does not decide."""

    def __init__(self, out, device, make, keep=8, best=0.0):
        """make: () -> list of three role nets for one snapshot."""
        self.out, self.device, self.make, self.keep, self.best = out, device, make, keep, best
        self.snaps = {}                       # step -> [3 nets]
        self.champ = {}                       # "decl"/"def" -> (step, [3 nets])

    def _load(self, f):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        with torch.random.fork_rng(devices=[]):    # init draws must not move the training RNG
            nets = self.make()
        for net, sd in zip(nets, ck["nets"]):
            net.load_state_dict(sd)
        return [net.to(self.device).eval() for net in nets]

    def _champions(self, allf):
        m = self.out / "metrics.jsonl"
        if not m.exists():
            return
        ev = []
        for line in open(m):
            if '"eval"' in line:
                r = json.loads(line)
                q = r.get("qnet") or r.get("b2g")          # "b2g": metrics written by the lab trainer
                if r.get("kind") == "eval" and q:
                    ev.append((r["step"], q["decl_paired_imps"]["mean"], q["def_paired_imps"]["mean"]))
        if not ev:
            return
        ev = np.array(ev)
        files = {int(f.stem[len("ckpt_step"):]): f for f in allf}
        for side, col in (("decl", 1), ("def", 2)):
            best = None
            for st in files:
                near = np.abs(ev[:, 0] - st) <= 200
                if near.sum() >= 3:
                    sc = ev[near, col].mean()
                    if best is None or sc > best[0]:
                        best = (sc, st)
            if best is not None and self.champ.get(side, (None,))[0] != best[1]:
                self.champ[side] = (best[1], self._load(files[best[1]]))
                print(f"league champion {side}: step {best[1]} ({best[0]:+.3f} vs the Q-net)", flush=True)

    def refresh(self):
        allf = sorted(self.out.glob("ckpt_step*.pt"))
        files = allf[-self.keep:]
        if self.best > 0:
            self._champions(allf)
        steps = {int(f.stem[len("ckpt_step"):]): f for f in files}
        for s in list(self.snaps):
            if s not in steps:
                del self.snaps[s]
        for s, f in steps.items():
            if s not in self.snaps:
                self.snaps[s] = self._load(f)

    def __len__(self):
        return len(self.snaps)

    def sample(self, gen_cpu):
        """(step, nets, side): side None = the snapshot takes a random chair per board; "decl"/"def" = a
        champion, which plays only the chair it is best in."""
        if self.champ and float(torch.rand(1, generator=gen_cpu)) < self.best:
            sides = sorted(self.champ)
            side = sides[int(torch.randint(len(sides), (1,), generator=gen_cpu))]
            st, nets = self.champ[side]
            return st, nets, side
        steps = sorted(self.snaps)
        s = steps[int(torch.randint(len(steps), (1,), generator=gen_cpu))]
        return s, self.snaps[s], None


def mean_ci(v: torch.Tensor):
    v = v.double()
    n = len(v)
    se = float(v.std() / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "ci95": 1.96 * se, "n": n}


class Evaluator:
    """Fixed held-out contracts; greedy nets vs random and vs the Q-net (no search), both chairs."""

    def __init__(self, ec, n_random, n_qnet, seed=0, qnet_path=env.QNET, oracle=False):
        """oracle: the nets have oracle columns; + an open-hand leg ("oracle_qnet") vs the Q-net."""
        self.cr = ec.subset(torch.arange(min(n_random, len(ec))))
        self.cb = ec.subset(torch.arange(min(n_qnet, len(ec))))
        self.seed, self.oracle = seed, oracle
        self.qnet = QNetPlayer(qnet_path) if n_qnet > 0 else None
        self.base_rr = play_out(self.cr, RandomPlayer(seed), RandomPlayer(seed + 1))
        self.base_bb = play_out(self.cb, self.qnet, self.qnet) if self.qnet else None

    @torch.no_grad()
    def __call__(self, nets, device, oracle_nets=None):
        """oracle_nets (--critic): the open-hand leg plays these instead of `nets`."""
        for n in nets:
            n.eval()
        me = NetPlayer(nets, device, 0.0)
        out = {}
        for name, c, opp, base in (("random", self.cr, None, self.base_rr), ("qnet", self.cb, self.qnet, self.base_bb)):
            if base is None:
                continue
            o1 = RandomPlayer(self.seed) if opp is None else opp
            o2 = RandomPlayer(self.seed + 1) if opp is None else opp
            t_decl = play_out(c, me, o2)       # we declare, opponent defends
            t_def = play_out(c, o1, me)        # opponent declares, we defend
            s_base = c.score(base)
            out[name] = {
                "decl_imps_vs_dd": mean_ci(c.imps_vs_dd(t_decl)),
                "def_imps_vs_dd": mean_ci(-c.imps_vs_dd(t_def)),
                "base_decl_imps_vs_dd": mean_ci(c.imps_vs_dd(base)),
                "decl_paired_imps": mean_ci(env.imps_t(c.score(t_decl) - s_base)),
                "def_paired_imps": mean_ci(env.imps_t(s_base - c.score(t_def))),
                "decl_tricks_minus_dd": mean_ci((t_decl - c.dd_tricks).float()),
                "def_tricks_minus_dd": mean_ci((c.dd_tricks - t_def).float()),
            }
        if self.oracle and self.qnet is not None:   # open-hand skill: our nets see every hand, the Q-net does not
            c, s_base = self.cb, self.cb.score(self.base_bb)
            if oracle_nets is not None:
                for n in oracle_nets:
                    n.eval()
                me = NetPlayer(oracle_nets, device, 0.0)
            t_decl = play_out(c, me, self.qnet, open_hand=True)
            t_def = play_out(c, self.qnet, me, open_hand=True)
            out["oracle_qnet"] = {"decl_paired_imps": mean_ci(env.imps_t(c.score(t_decl) - s_base)),
                                  "def_paired_imps": mean_ci(env.imps_t(s_base - c.score(t_def)))}
        if oracle_nets is not None:
            for n in oracle_nets:
                n.train()
            me = NetPlayer(nets, device, 0.0)
        t_self = play_out(self.cr, me, me)
        out["self"] = {"decl_tricks_minus_dd": mean_ci((t_self - self.cr.dd_tricks).float()),
                       "decl_imps_vs_dd": mean_ci(self.cr.imps_vs_dd(t_self))}
        for n in nets:
            n.train()
        return out


def ckpt_args(args) -> dict:
    """Saved run settings. rel_feats / auction_seats are always on here (the served input layout);
    they are recorded for readers of the lab checkpoints' format (server/emergent/playdmc.slim)."""
    return {**vars(args), "rel_feats": True, "auction_seats": True}


def save_ckpt(path, nets, opts, step, boards, args, block_i=None, cursor=None, aux=None, ema=None, belief=None,
              critic=None):
    """`opts=None` writes the nets only (snapshots); last.pt keeps the optimiser and the pool
    position (block_i, cursor) for --resume. aux = (heads, aux_opts) adds the --aux-trick heads."""
    tmp = path.with_suffix(".tmp")
    rec = {"nets": [n.state_dict() for n in nets], "opts": [o.state_dict() for o in opts] if opts else None,
           "step": step, "boards": boards, "args": ckpt_args(args), "feat_dim": nets[0].core.inp.in_features
           if hasattr(nets[0], "core") else nets[0].inp.in_features,
           "block_i": block_i, "cursor": cursor, "arch": "mlp"}
    if ema is not None:
        rec["ema"] = [n.state_dict() for n in ema]
    if aux is not None and opts:
        rec["aux_heads"] = [h.state_dict() for h in aux[0]]
        rec["aux_opts"] = [o.state_dict() for o in aux[1]]
    if belief is not None and opts:
        rec["belief_head"] = belief[0].state_dict()
        rec["belief_opt"] = belief[1].state_dict()
    if critic is not None:   # (nets, opts, net args): last.pt keeps the optimisers
        rec["critic"] = [n.state_dict() for n in critic[0]]
        rec["critic_args"] = critic[2]
        if opts:
            rec["critic_opts"] = [o.state_dict() for o in critic[1]]
    torch.save(rec, tmp)
    os.replace(tmp, path)


def _bid_block(block, cache, block_size, data, seed, threads, bidder, pool):
    """Background process: bid one training block into the cache (train_block writes tmp + rename)."""
    torch.set_num_threads(threads)
    env.train_block(block, Path(cache), block_size, data, seed=seed,
                    log=lambda m: print(f"[prefetch] {m}", flush=True), bidder=bidder, pool=pool)


def start_prefetch(block, cache, args, log):
    """Bid `block` in a spawned process unless it is cached already. Returns the process or None."""
    pool = tuple(args.train_pool)
    if env.block_path(block, cache, args.block_size, args.seed, pool).exists():
        return None
    import multiprocessing as mp
    proc = mp.get_context("spawn").Process(target=_bid_block, daemon=True,
                                            args=(block, str(cache), args.block_size, args.data, args.seed,
                                                  args.bid_threads, args.bidder, pool))
    proc.start()
    log(f"prefetch: bidding block {block} in background pid {proc.pid}")
    return proc


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(RUNS / "run"))
    p.add_argument("--cache", default=str(RUNS / "contracts"), help="bid contracts, one dir per bidder")
    p.add_argument("--bidder", default=env.BIDDER, help="four-seat bidding checkpoint that bids every deal")
    p.add_argument("--data", default=env.DATA, help="DD dataset (default $BRIDGE_DATA or data/dds_results_100M.npy)")
    p.add_argument("--train-pool", type=int, nargs=2, default=list(env.TRAIN_POOL), metavar=("START", "END"),
                   help="training deals: rows [START, END) of --data, in --block-size blocks")
    p.add_argument("--eval-data", default=None, help="evaluation deals (default: --data)")
    p.add_argument("--eval-start", type=int, default=env.EVAL_START, help="first --eval-data row")
    p.add_argument("--eval-deals", type=int, default=env.EVAL_DEALS, help="evaluation deals bid and cached")
    p.add_argument("--qnet", default=env.QNET, help="evaluation opponent: the released play Q-net")
    p.add_argument("--block-size", type=int, default=1_000_000)
    p.add_argument("--boards", type=int, default=2048, help="boards per actor round")
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-halve-every", type=int, default=0,
                   help="step schedule: lr = --lr * 0.5 ** ((step - --lr-halve-from) // K) from --lr-halve-from on (0 = off)")
    p.add_argument("--lr-halve-from", type=int, default=0)
    p.add_argument("--reset-lr", action="store_true", help="on --resume, override the checkpoint optimiser lr with --lr")
    p.add_argument("--clip", type=float, default=40.0)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--return", dest="ret", choices=("imps", "tricks"), default="imps",
                   help="imps: imps(score - DD score); tricks: tricks - DD tricks (declarer side)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--bid-threads", type=int, default=4, help="torch threads of the background bidding process")
    p.add_argument("--passes", type=int, default=1, help="shuffled learner passes over each actor round's samples")
    p.add_argument("--td-lambda", type=float, default=None,
                   help="TD(lambda) targets from the actor's greedy V (default: Monte Carlo final return)")
    p.add_argument("--aux-trick", type=float, default=0.0,
                   help="weight of an aux BCE head: P(acting side wins the current trick | state, card)")
    p.add_argument("--league", type=float, default=0.0,
                   help="per-board probability that one side (50/50) is a past snapshot (ckpt_step*.pt)")
    p.add_argument("--league-keep", type=int, default=8, help="most recent snapshots kept on the device")
    p.add_argument("--league-best", type=float, default=0.0,
                   help="share of league rounds vs a champion: best-declaring snapshot declares / best-defending defends")
    p.add_argument("--init", default=None,
                   help="start (not --resume) from this ckpt's nets: same width/depth, fresh optimiser, step 0; "
                        "a net without oracle columns gets zero weights for them")
    p.add_argument("--steps", type=int, default=100_000, help="actor+learner iterations (total, counting resumed steps)")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-random", type=int, default=2000)
    p.add_argument("--eval-qnet", type=int, default=1000, help="boards per chair vs the Q-net at each eval")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay per step (0 = off). The EMA nets act, evaluate and are the ckpt_step* "
                        "snapshots; the online nets learn. last.pt keeps both.")
    p.add_argument("--shared-mlp", action="store_true",
                   help="ONE QNet for all three roles (+ a learned role vector); roles mixed in each minibatch")
    p.add_argument("--oracle-guide", action="store_true",
                   help="Suphx oracle guiding: actors also see hidden holders, keep rate decays to 0 (eval never sees them)")
    p.add_argument("--oracle-from", type=float, default=1.0, help="keep rate at step 0")
    p.add_argument("--oracle-hold", type=int, default=0, help="steps at the full keep rate before the decay starts")
    p.add_argument("--oracle-until", type=int, default=8000, help="step where the keep rate reaches 0 (linear)")
    p.add_argument("--critic-init", default=None,
                   help="PerfectDou-style open-hand critic (separate role nets): from this ckpt with oracle columns "
                        "(or its \"critic\" entry); the players learn toward its Q of the played card")
    p.add_argument("--critic-w", type=float, default=1.0,
                   help="player target = w * critic Q(true deal, played card) + (1 - w) * the usual (TD-lambda) target")
    p.add_argument("--critic-ramp", type=int, default=0, help="w rises linearly from 0 over the first K steps (critic adapts)")
    p.add_argument("--critic-lambda", type=float, default=0.8,
                   help="critic target: SARSA(lambda) return bootstrapped on its own Q of the played cards (1 = final return)")
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--search-pos", type=int, default=0,
                   help="true-deal branching (selfsearch.py, --shared-mlp): positions per actor round (0 = off)")
    p.add_argument("--search-leads", action="store_true", help="branch at leads to a trick only")
    p.add_argument("--search-open-p", type=float, default=0.25, help="--search-leads: share of rounds at the opening lead")
    p.add_argument("--search-w", type=float, default=1.0, help="weight of the branching-target loss")
    p.add_argument("--search-buffer", type=int, default=16384, help="most recent branched positions kept")
    p.add_argument("--search-steps", type=int, default=2, help="branching-loss minibatch steps per actor round")
    p.add_argument("--search-chunk", type=int, default=65536, help="rollouts per forward batch")
    p.add_argument("--aux-belief", type=float, default=0.0,
                   help="--shared-mlp: weight of an aux head that predicts the holder of every card the actor cannot see")
    p.add_argument("--smoke", action="store_true",
                   help="tiny pipeline check on data/smoke_128.npz: width 64, depth 1, 32 boards, 4 steps")
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--smoke" in argv:   # defaults only: flags given on the command line still win
        p.set_defaults(data=str(SMOKE_DATA), train_pool=[0, 128], block_size=64, eval_data=str(SMOKE_DATA),
                       eval_start=0, eval_deals=64, width=64, depth=1, boards=32, minibatch=64, steps=4,
                       eval_every=2, ckpt_every=2, log_every=1, eval_random=16,
                       eval_qnet=16 if Path(env.QNET).exists() else 0, search_buffer=256)
    args = p.parse_args(argv)
    if args.eval_data is None:
        args.eval_data = args.data
    if args.aux_belief > 0 and not args.shared_mlp:
        p.error("--aux-belief needs --shared-mlp")
    if args.critic_init and args.oracle_guide:
        p.error("--critic-init: without --oracle-guide (players never see the holders)")
    if args.search_pos > 0 and not args.shared_mlp:
        p.error("--search-pos needs --shared-mlp")
    return args


def main(argv=None):
    args = parse_args(argv)
    oracle = bool(args.oracle_guide or args.critic_init)     # nets carry the oracle columns
    fdim = env.feat_dim(oracle)
    pool = tuple(args.train_pool)
    torch.set_num_threads(args.threads)
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n"); log_f.flush()

    device = torch.device(args.device)
    nets = [n.to(device) for n in make_nets(args.width, args.depth, args.seed, shared=args.shared_mlp, oracle=oracle)]
    ema = [n.eval().requires_grad_(False) for n in copy.deepcopy(nets)] if args.ema else None   # one deepcopy keeps sharing
    if args.shared_mlp:
        opts = [torch.optim.Adam(nets[0].parameters(), lr=args.lr)] * 3
    else:
        opts = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in nets]
    aux = None
    if args.aux_trick > 0:
        with torch.random.fork_rng(devices=[]):
            heads = [h.to(device) for h in make_aux_heads(args.width)]
        aux = (heads, [torch.optim.Adam(h.parameters(), lr=args.lr) for h in heads])
    belief = None
    if args.aux_belief > 0:
        with torch.random.fork_rng(devices=[]):
            bh = make_belief_head(args.width).to(device)
        belief = (bh, torch.optim.Adam(bh.parameters(), lr=args.lr))
    critic = None
    if args.critic_init:   # (nets, opts, net args)
        src = out / "last.pt" if args.resume and (out / "last.pt").exists() else Path(args.critic_init)
        ck = torch.load(src, map_location="cpu", weights_only=False)
        if "critic" in ck:
            ca, sds = ck["critic_args"], ck["critic"]
        else:
            a_ = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
            if ck.get("feat_dim") != env.feat_dim(True) or a_.get("shared_mlp"):
                raise SystemExit(f"--critic-init: needs a split MLP ckpt with oracle columns (feat_dim {env.feat_dim(True)})")
            ca, sds = {"width": a_["width"], "depth": a_["depth"], "from": str(src), "step": ck.get("step")}, ck["nets"]
        with torch.random.fork_rng(devices=[]):
            cn = make_nets(ca["width"], ca["depth"], oracle=True)
        for n, sd in zip(cn, sds):
            n.load_state_dict(sd)
        cn = [n.to(device) for n in cn]
        copts = [torch.optim.Adam(n.parameters(), lr=args.critic_lr) for n in cn]
        if ck.get("critic_opts"):
            for o, sd in zip(copts, ck["critic_opts"]):
                o.load_state_dict(sd)
            if args.reset_lr:
                for o in copts:
                    for g_ in o.param_groups:
                        g_["lr"] = args.critic_lr
        critic = (cn, copts, ca)
        log(f"critic {ca['width']}x{ca['depth']} from {ca['from']} (step {ca['step']}), loaded from {src.name}; "
            f"w {args.critic_w} ramp {args.critic_ramp} lambda {args.critic_lambda} lr {args.critic_lr}")
        del ck
    step, boards_done = 0, 0
    pos = None
    if args.init and not (args.resume and (out / "last.pt").exists()):
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        sds = ck["nets"]
        if ck.get("feat_dim") == env.feat_dim(False) and oracle:   # a net without oracle columns: zero weights
            sds = [pad_oracle(sd) for sd in sds]
        elif ck.get("feat_dim") != fdim:
            raise SystemExit(f"--init: feat_dim {ck.get('feat_dim')} != {fdim}")
        for n, sd in zip(nets, sds):
            n.load_state_dict(sd)
        if ema is not None:
            for n, sd in zip(ema, sds):
                n.load_state_dict(sd)
        log(f"init from {args.init} (step {ck.get('step')})")
    if args.resume and (out / "last.pt").exists():
        # CPU load: Adam then keeps its "step" counters on CPU (on MPS each update syncs per param)
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if ck.get("arch", "mlp") != "mlp" or ck.get("feat_dim") != fdim:
            raise SystemExit(f"--resume: last.pt is {ck.get('arch', 'mlp')} with feat_dim {ck.get('feat_dim')}, "
                             f"this run needs an mlp with {fdim}")
        a_ = ck["args"]
        if not (a_.get("rel_feats") and a_.get("auction_seats")) or bool(a_.get("shared_mlp", False)) != args.shared_mlp:
            raise SystemExit("--resume: last.pt has a different input layout or --shared-mlp setting")
        for n, s in zip(nets, ck["nets"]):
            n.load_state_dict(s)
        if ema is not None:
            for n, s in zip(ema, ck.get("ema") or ck["nets"]):
                n.load_state_dict(s)
        for o, s in zip(opts, ck["opts"]):
            o.load_state_dict(s)
        if aux is not None and ck.get("aux_heads") is not None:
            for h, sd in zip(aux[0], ck["aux_heads"]):
                h.load_state_dict(sd)
            for o, sd in zip(aux[1], ck["aux_opts"]):
                o.load_state_dict(sd)
        if belief is not None and ck.get("belief_head") is not None:
            belief[0].load_state_dict(ck["belief_head"])
            belief[1].load_state_dict(ck["belief_opt"])
        if args.reset_lr:   # the loaded optimiser state carries the old lr; use --lr from now on
            for o in list(opts) + (list(aux[1]) if aux is not None else []) + ([belief[1]] if belief else []):
                for g in o.param_groups:
                    g["lr"] = args.lr
            log(f"lr set to {args.lr} (--reset-lr)")
        step, boards_done = ck["step"], ck["boards"]
        if ck.get("block_i") is not None:
            pos = (ck["block_i"], ck["cursor"])
        log(f"resumed step {step} boards {boards_done}" + (f" block {pos[0]} cursor {pos[1]}" if pos else
                                                           " (no pool position saved: boards // block_size)"))
    (out / "run.json").write_text(json.dumps({**ckpt_args(args), "feat_dim": fdim}, indent=2))
    metrics = open(out / "metrics.jsonl", "a")

    def emit(rec):
        metrics.write(json.dumps(rec) + "\n"); metrics.flush()

    env.check_cache_bidder(cache, args.bidder)
    ec = env.eval_contracts(cache, args.eval_data, args.eval_start, args.eval_deals, log=log, bidder=args.bidder)
    log(f"eval contracts: {len(ec)} ({args.eval_data} rows {args.eval_start}..); building evaluator")
    evaluator = Evaluator(ec, args.eval_random, args.eval_qnet, seed=args.seed, qnet_path=args.qnet, oracle=oracle)

    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * step)
    if pos is not None:
        block_i, cursor = pos
    else:
        block_i, cursor = boards_done // args.block_size, boards_done % args.block_size
    pool_c = env.train_block(block_i, cache, args.block_size, args.data, seed=args.seed, log=log,
                             bidder=args.bidder, pool=pool)
    prefetch = start_prefetch(block_i + 1, cache, args, log)
    league = None
    if args.league > 0:
        league = League(out, device, lambda: make_nets(args.width, args.depth, shared=args.shared_mlp, oracle=oracle),
                        args.league_keep, best=args.league_best)
        league.refresh()
        gen_league = torch.Generator().manual_seed(args.seed + 7 + 1000 * step)
        log(f"league: p={args.league}, {len(league)} snapshots loaded")

    def new_acc():
        return {"ret": [], "tdd": [], "loss": {k: [] for k in ROLE_NAMES}, "t_act": 0.0, "t_learn": 0.0, "boards": 0,
                "aux": [], "belief": [], "gnorm": [], "lg_boards": 0, "lg_ret": [], "lg_snaps": [],
                "search": [], "search_s": 0.0, "critic": [], "critic_w": 0.0, "qc_err": [], "s_stats": {"positions": 0, "t": []}}
    acc = new_acc()
    t_win = time.time()

    def do_eval():
        t0 = time.time()
        res = evaluator(ema or nets, device, oracle_nets=critic[0] if critic else None)
        rec = {"kind": "eval", "step": step, "boards": boards_done, "eval_seconds": round(time.time() - t0, 1), **res}
        emit(rec)
        r, b = res["random"], res.get("qnet", {})
        log(f"EVAL step {step} | vs random decl {r['decl_paired_imps']['mean']:+.2f} def {r['def_paired_imps']['mean']:+.2f}"
            + (f" | vs Q-net decl {b['decl_paired_imps']['mean']:+.2f}±{b['decl_paired_imps']['ci95']:.2f}"
               f" def {b['def_paired_imps']['mean']:+.2f}±{b['def_paired_imps']['ci95']:.2f}" if b else "")
            + (f" | {'critic' if critic else 'oracle'} vs Q-net decl {res['oracle_qnet']['decl_paired_imps']['mean']:+.2f}"
               f" def {res['oracle_qnet']['def_paired_imps']['mean']:+.2f}" if "oracle_qnet" in res else "")
            + f" | self t-dd {res['self']['decl_tricks_minus_dd']['mean']:+.2f} ({rec['eval_seconds']}s)")

    sbuf = None
    if args.search_pos > 0:
        from . import selfsearch
        gen_search = torch.Generator().manual_seed(args.seed + 11 + 1000 * step)
        sbuf = {"x": torch.empty(0, fdim, dtype=torch.uint8, device=device),
                "r": torch.empty(0, dtype=torch.long, device=device), "v": torch.empty(0, 52, device=device)}
        log(f"true-deal branching: {args.search_pos} positions/round, w {args.search_w}, buffer {args.search_buffer}, "
            f"{args.search_steps} steps/round" + (f", leads only (opening lead p={args.search_open_p})" if args.search_leads else ""))
    if step == 0:
        do_eval()
    all_opts = list({id(o): o for o in list(opts) + (list(aux[1]) if aux is not None else [])
                     + ([belief[1]] if belief else [])}.values())
    while step < args.steps:
        if args.lr_halve_every > 0 and step >= args.lr_halve_from:
            lr_now = args.lr * 0.5 ** ((step - args.lr_halve_from) // args.lr_halve_every)
            if all_opts[0].param_groups[0]["lr"] != lr_now:
                for o in all_opts:
                    for g_ in o.param_groups:
                        g_["lr"] = lr_now
                log(f"lr schedule: step {step} lr {lr_now:g}")
        if cursor + args.boards > len(pool_c):
            block_i += 1
            if prefetch is not None:
                tw = time.time()
                prefetch.join()
                log(f"prefetched block {block_i} (exit {prefetch.exitcode}, waited {time.time() - tw:.0f}s)")
            pool_c = env.train_block(block_i, cache, args.block_size, args.data, seed=args.seed, log=log,
                                     bidder=args.bidder, pool=pool)
            prefetch = start_prefetch(block_i + 1, cache, args, log)
            cursor = 0
        c = pool_c.subset(torch.arange(cursor, cursor + args.boards))
        cursor += args.boards
        t0 = time.time()
        for n in nets:
            n.eval()
        lg = None
        if league is not None and len(league) > 0:
            u = torch.rand(len(c), generator=gen_league)
            lrows = (u < args.league).nonzero().squeeze(1)
            if len(lrows):
                snap_step, snets, side = league.sample(gen_league)
                ldecl = torch.rand(len(lrows), generator=gen_league) < 0.5
                if side is not None:
                    ldecl = torch.full((len(lrows),), side == "decl")
                lg = (snets, lrows.to(device), ldecl.to(device))
        static, Xd, card, role, board, tricks, extra = actor_round(
            ema or nets, c, device, args.eps, gen, want_v=args.td_lambda is not None, league=lg,
            oracle=oracle_keep(args, step), critic=critic[0] if critic else None)
        for n in nets:
            n.train()
        if ema is not None:
            for n in ema:
                n.eval()
        tricks = tricks.cpu()                          # waits for the actor
        g = c.imps_vs_dd(tricks)                       # declarer-side return per board (logged)
        gt = (g if args.ret == "imps" else (tricks - c.dd_tricks).float()).to(device)
        if args.td_lambda is None:
            target = torch.where(role == 0, gt[board], -gt[board])
        else:
            gl = env.lambda_returns(extra["v_decl"], gt, args.td_lambda).reshape(-1)
            target = torch.where(role == 0, gl, -gl)
        if critic is not None:   # PerfectDou: players regress toward the open-hand critic's Q of the played card
            qc = extra["qc"]
            cw = args.critic_w * (min(1.0, step / args.critic_ramp) if args.critic_ramp > 0 else 1.0)
            qcs = qc.reshape(-1)
            target = cw * torch.where(role == 0, qcs, -qcs) + (1 - cw) * target
            cl = env.lambda_returns(qc, gt, args.critic_lambda).reshape(-1)
            c_target = torch.where(role == 0, cl, -cl)
            acc["critic_w"] = cw
            acc["qc_err"].append(float(((qc[0] - gt) ** 2).mean()))     # critic at the opening lead vs the final return
        aux_in = None
        if aux is not None:
            labels = trick_labels(extra["trick_winner"], c.declarer.to(device), role, len(c))
            aux_in = (aux[0], aux[1], labels, args.aux_trick)
        if lg is not None:   # current nets' return on league boards (their side's view, training return)
            lr_cpu = lrows
            my = torch.where(ldecl, -gt.cpu()[lr_cpu], gt.cpu()[lr_cpu])   # snapshot declares -> we defend
            acc["lg_boards"] += len(lr_cpu); acc["lg_ret"].append(my); acc["lg_snaps"].append(snap_step)
        t1 = time.time()
        if args.shared_mlp:
            bel_in = (belief[0], belief[1], (c.owner.to(device) - c.declarer.to(device)[:, None]) % 4,
                      args.aux_belief) if belief else None
            losses = learn_shared(nets, opts[0], static, Xd, card, role, board, target, args.minibatch, args.clip,
                                  args.passes, mine=extra.get("mine"), aux=aux_in, belief=bel_in)
            if sbuf is not None:
                ts = time.time()
                for n in nets:
                    n.eval()
                st = selfsearch.search_targets(ema or nets, c, card, static, role, args.search_pos, gen_search,
                                               device, args.search_chunk, acc["s_stats"],
                                               leads=args.search_leads, open_p=args.search_open_p)
                for n in nets:
                    n.train()
                if st is not None:
                    sid, sv = st
                    xs = torch.cat([static[board[sid]], Xd[sid]], 1)
                    for k_, new in (("x", xs), ("r", role[sid]), ("v", sv)):
                        sbuf[k_] = torch.cat([sbuf[k_], new])[-args.search_buffer:]
                for _ in range(args.search_steps if len(sbuf["x"]) else 0):
                    acc["search"].append(selfsearch.search_learn(nets[0].core, opts[0], sbuf["x"], sbuf["r"], sbuf["v"],
                                                                 args.minibatch, args.search_w, args.clip))
                acc["search_s"] += time.time() - ts
        else:
            losses = learn(nets, opts, static, Xd, card, role, board, target, args.minibatch, args.clip,
                           args.passes, mine=extra.get("mine"), aux=aux_in)
        if critic is not None:
            for n in critic[0]:
                n.train()
            cls = learn(critic[0], critic[1], extra["static_full"], Xd, card, role, board, c_target, args.minibatch,
                        args.clip, args.passes, mine=extra.get("mine"))
            for n in critic[0]:
                n.eval()
            acc["critic"].append(cls)
        del static, Xd, card, role, board, target, extra
        if ema is not None:
            with torch.no_grad():
                seen = set()
                for ne, n in zip(ema, nets):
                    for pe, pn in zip(ne.parameters(), n.parameters()):
                        if id(pe) not in seen:     # shared views: each core parameter once
                            seen.add(id(pe))
                            pe.lerp_(pn, 1.0 - args.ema)
        t2 = time.time()
        step += 1
        boards_done += args.boards
        acc["ret"].append(g); acc["tdd"].append((tricks - c.dd_tricks).float())
        for k, v in losses.items():
            (acc[k] if k in ("aux", "belief", "gnorm") else acc["loss"][k]).append(v)
        acc["t_act"] += t1 - t0; acc["t_learn"] += t2 - t1; acc["boards"] += args.boards
        if step % args.log_every == 0:
            el = time.time() - t_win
            ret = torch.cat(acc["ret"]); tdd = torch.cat(acc["tdd"])
            rec = {"kind": "train", "step": step, "boards": boards_done, "time": time.time(),
                   "eps": args.eps, "lr": opts[0].param_groups[0]["lr"],
                   **({"oracle_keep": oracle_keep(args, step)} if args.oracle_guide else {}),
                   "loss": {k: float(np.mean(v)) for k, v in acc["loss"].items()},
                   "return": {"decl": float(ret.mean()), "lho": float(-ret.mean()), "rho": float(-ret.mean())},
                   "tricks_minus_dd": float(tdd.mean()),
                   "boards_per_s": acc["boards"] / el,
                   "act_frac": acc["t_act"] / el, "learn_frac": acc["t_learn"] / el,
                   "block": block_i}
            if acc["aux"]:
                rec["aux_trick_bce"] = float(np.mean(acc["aux"]))
            if acc["belief"]:
                rec["aux_belief_ce"] = float(np.mean(acc["belief"]))
            if acc["gnorm"]:
                rec["grad_norm"] = float(np.mean(acc["gnorm"]))
            if acc["critic"]:
                rec["critic"] = {"loss": {k: float(np.mean([d[k] for d in acc["critic"]])) for k in ROLE_NAMES},
                                 "w": acc["critic_w"], "lead_mse_vs_return": float(np.mean(acc["qc_err"]))}
            if acc["search"]:
                rec["search"] = {"loss": float(np.mean(acc["search"])), "positions": acc["s_stats"]["positions"],
                                 "secs": acc["search_s"], "buffer": len(sbuf["x"])}
            if league is not None:
                lr_all = torch.cat(acc["lg_ret"]) if acc["lg_ret"] else torch.zeros(0)
                rec["league"] = {"share": acc["lg_boards"] / acc["boards"], "boards": acc["lg_boards"],
                                 "ret_vs_snap": float(lr_all.mean()) if len(lr_all) else None,
                                 "ret_vs_snap_ci95": mean_ci(lr_all)["ci95"] if len(lr_all) > 1 else None,
                                 "snap_steps": sorted(set(acc["lg_snaps"])), "n_snaps": len(league)}
            emit(rec)
            log(f"step {step} boards {boards_done} | loss " + " ".join(f"{k} {v:.2f}" for k, v in rec["loss"].items())
                + f" | decl ret {rec['return']['decl']:+.3f} t-dd {rec['tricks_minus_dd']:+.3f}"
                f" | {rec['boards_per_s']:.0f} boards/s (act {rec['act_frac']:.2f} learn {rec['learn_frac']:.2f})"
                + (f" | aux {rec['aux_trick_bce']:.3f}" if "aux_trick_bce" in rec else "")
                + (f" | belief {rec['aux_belief_ce']:.3f}" if "aux_belief_ce" in rec else "")
                + (" | critic loss " + " ".join(f"{v:.2f}" for v in rec["critic"]["loss"].values())
                   + f" w {rec['critic']['w']:.2f} lead mse {rec['critic']['lead_mse_vs_return']:.2f}" if "critic" in rec else "")
                + (f" | search {rec['search']['loss']:.3f} ({rec['search']['secs']:.0f}s)" if "search" in rec else "")
                + (f" | league {rec['league']['share']:.2f} ret {rec['league']['ret_vs_snap']:+.3f}"
                   if league is not None and rec["league"]["ret_vs_snap"] is not None else ""))
            acc = new_acc()
            t_win = time.time()
        if step % args.eval_every == 0:
            do_eval()
            t_win = time.time()
        if step % args.ckpt_every == 0 or step == args.steps:
            save_ckpt(out / "last.pt", nets, opts, step, boards_done, args, block_i, cursor, aux, ema, belief, critic)
            save_ckpt(out / f"ckpt_step{step:07d}.pt", ema or nets, None, step, boards_done, args)
            if league is not None:
                league.refresh()
    if step % args.eval_every:
        do_eval()
    if prefetch is not None and prefetch.is_alive():
        prefetch.terminate()
    log("done")


if __name__ == "__main__":
    main()
