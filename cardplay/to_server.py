"""Trainer checkpoint(s) -> the serving file read by server/emergent/playdmc.py.

Keeps one copy of the shared network (the three role entries are views of one core), in float32, with
the oracle input columns cut (they are zero whenever the net plays). Several checkpoints are averaged
weight by weight first (card play V2 is the average of steps 36,000 to 38,000).
Check: the served file and the trainer net play the same cards on `--check` evaluation boards,
against the Q-net in both chairs.

    python -m cardplay.to_server runs/cardplay/run18/ckpt_step0004000.pt server/models/play_run18_s4000.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import env
from .model import NetPlayer, QNetPlayer, _server_module, make_nets, play_out


def average(paths):
    """(state dict of nets[0], args, step, feat_dim) of one ckpt or the weight average of several."""
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    a = cks[0]["args"] if isinstance(cks[0]["args"], dict) else vars(cks[0]["args"])
    assert cks[0].get("arch", "mlp") == "mlp" and a.get("shared_mlp") and a.get("rel_feats") and a.get("auction_seats"), \
        "the serving format holds one shared MLP with --rel-feats and --auction-seats inputs"
    assert all(ck["feat_dim"] == cks[0]["feat_dim"] for ck in cks), "checkpoints differ in input width"
    sds = [ck["nets"][0] for ck in cks]
    if len(sds) == 1:
        sd = {k: v.float().clone() for k, v in sds[0].items()}
    else:
        sd = {k: torch.stack([s[k].float() for s in sds]).mean(0) for k in sds[0]}
    return sd, a, cks[0]["step"], cks[0]["feat_dim"]


def serving_dict(paths):
    sd, a, step, fd = average(paths)
    core = {k.removeprefix("core."): v for k, v in sd.items() if k.startswith("core.")}
    w = core["inp.weight"]
    if w.shape[1] == env.feat_dim(True):                  # oracle columns: the last static block
        keep = torch.ones(w.shape[1], dtype=torch.bool)
        keep[env.STATIC_DIM:env.static_dim(True)] = False
        core["inp.weight"] = w[:, keep].contiguous()
    assert core["inp.weight"].shape[1] == env.FEAT_DIM, core["inp.weight"].shape
    out = {"net": core, "width": a["width"], "depth": a["depth"], "feat_dim": env.FEAT_DIM, "step": step}
    if len(paths) > 1:
        out["averaged_from"] = [str(p) for p in paths]
    else:
        out["source"] = str(paths[0])
    return out, sd, a, fd


class Served:
    """The serving file as a table player."""

    def __init__(self, path):
        self.bot = _server_module("playdmc").DMCPlayBot(path)

    @torch.no_grad()
    def act(self, table, rows):
        return self.bot.values(table.c, table.b).argmax(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpts", nargs="+", help="one trainer checkpoint, or several to average")
    p.add_argument("out", help="serving file to write")
    p.add_argument("--check", type=int, default=1000, help="boards per chair vs the Q-net (0 = no check)")
    p.add_argument("--first", type=int, default=1000, help="first evaluation contract row of the check")
    p.add_argument("--cache", default=str(env.ROOT / "runs/cardplay/contracts"))
    p.add_argument("--bidder", default=env.BIDDER)
    p.add_argument("--eval-data", default=env.DATA)
    p.add_argument("--eval-start", type=int, default=env.EVAL_START)
    p.add_argument("--eval-deals", type=int, default=env.EVAL_DEALS)
    p.add_argument("--qnet", default=env.QNET)
    a = p.parse_args(argv)
    rec, sd, args, fd = serving_dict(a.ckpts)
    torch.save(rec, a.out)
    print(f"wrote {a.out}: feat_dim {rec['feat_dim']}, width {rec['width']}, depth {rec['depth']}, step {rec['step']}")
    if a.check <= 0:
        return
    c = env.eval_contracts(Path(a.cache), a.eval_data, a.eval_start, a.eval_deals, bidder=a.bidder)
    c = c.subset(torch.arange(a.first, min(a.first + a.check, len(c))))
    nets = make_nets(args["width"], args["depth"], shared=True, oracle=fd == env.feat_dim(True))
    nets[0].load_state_dict(sd)                           # the three views share this core
    mine, served, qnet = NetPlayer([n.eval() for n in nets]), Served(a.out), QNetPlayer(a.qnet)
    bad = 0
    with torch.no_grad():
        for d, f in ((mine, qnet), (qnet, mine)):
            t1 = play_out(c, d, f)
            t2 = play_out(c, served if d is mine else qnet, served if f is mine else qnet)
            bad += int((t1 != t2).sum())
    print(f"check: {bad} of {2 * len(c)} boards differ in tricks (served file vs trainer net, vs the Q-net both chairs)")


if __name__ == "__main__":
    main()
