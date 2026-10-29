"""Card-play trainer: true-deal branching targets, one learner step, and (with the released bidder and
Q-net present) a smoke run end to end through the serving file."""
from pathlib import Path

import numpy as np
import pytest
import torch

from cardplay import env, selfsearch, train
from cardplay.model import Table, make_belief_head, make_aux_heads, make_nets
from test_cardplay_env import random_contracts

ROOT = Path(__file__).resolve().parents[1]


def test_truth_branch_values_replay_by_hand():
    """Each legal card's value = declarer tricks of playing it on the true deal, then greedy nets everywhere."""
    c = random_contracts(10, seed=21)
    nets = make_nets(32, 1, seed=1, shared=True)
    static, Xd, card, role, board, tricks, extra = train.actor_round(nets, c, "cpu", 0.0, torch.Generator())
    stats = {"positions": 0, "t": []}
    sid, v = selfsearch.search_targets(nets, c, card, static, role, 4, torch.Generator().manual_seed(3), "cpu",
                                       4096, stats, leads=True, open_p=1.0)
    n = len(c)
    assert stats["t"] == [0] and torch.all(sid < n) and len(sid) == 4     # opening leads
    for i, s in enumerate(sid.tolist()):
        for k in range(52):
            legal = bool(Table(c.subset([s])).b.legal()[0, k])
            assert legal != bool(torch.isnan(v[i, k]))
            if not legal:
                continue
            t = Table(c.subset([s]))
            t.b.play(torch.tensor([k]))
            while not t.b.done:
                x = t.features().float()
                r = int(env.roles(t.b, t.c.declarer)[0])
                with torch.no_grad():
                    q = nets[r](x).masked_fill(~t.b.legal(), float("-inf"))
                t.b.play(q.argmax(1))
            got = float(t.b.declarer_tricks()[0] - c.dd_tricks[s])
            want = got if int(role[s]) == 0 else -got
            assert v[i, k].item() == pytest.approx(want)


def test_shared_learn_step_with_heads_and_branching():
    c = random_contracts(16, seed=22)
    nets = make_nets(32, 1, seed=1, shared=True)
    opt = torch.optim.Adam(nets[0].parameters(), 1e-3)
    heads, bh = make_aux_heads(32), make_belief_head(32)
    static, Xd, card, role, board, tricks, extra = train.actor_round(nets, c, "cpu", 0.1, torch.Generator().manual_seed(0),
                                                                     want_v=True)
    labels = train.trick_labels(extra["trick_winner"], c.declarer, role, len(c))
    before = [p.clone() for p in nets[0].parameters()]
    losses = train.learn_shared(nets, opt, static, Xd, card, role, board, torch.randn(len(card)), 64, 40.0, 2,
                                aux=(heads, [torch.optim.Adam(h.parameters(), 1e-3) for h in heads], labels, 0.1),
                                belief=(bh, torch.optim.Adam(bh.parameters(), 1e-3), (c.owner - c.declarer[:, None]) % 4, 0.05))
    assert {"decl", "lho", "rho", "aux", "belief", "gnorm"} <= set(losses)
    assert all(np.isfinite(v) for v in losses.values())
    assert any(not torch.equal(a, p) for a, p in zip(before, nets[0].parameters()))
    sid, v = selfsearch.search_targets(nets, c, card, static, role, 8, torch.Generator().manual_seed(1), "cpu", 4096,
                                       {"positions": 0, "t": []})
    x = torch.cat([static[board[sid]], Xd[sid]], 1)
    assert np.isfinite(selfsearch.search_learn(nets[0].core, opt, x, role[sid], v, 64, 1.0, 40.0))


def test_oracle_keep_schedule():
    a = train.parse_args(["--oracle-guide", "--oracle-from", "1.0", "--oracle-hold", "8000", "--oracle-until", "13000"])
    assert [train.oracle_keep(a, s) for s in (0, 7999, 8000, 10500, 13000, 20000)] == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]


NEED = [ROOT / "server/models/pidginv3_1_bid_s20000.pt", ROOT / "server/models/play_B2g_s540k.pt"]


@pytest.mark.skipif(not all(p.exists() for p in NEED), reason="released bidder / Q-net missing (scripts/get_models.sh)")
def test_smoke_run_resume_and_serving_file(tmp_path):
    from cardplay import to_server
    from cardplay.model import _server_module
    base = ["--smoke", "--cache", str(tmp_path / "contracts"), "--threads", "2", "--bid-threads", "1",
            "--return", "tricks", "--passes", "2", "--td-lambda", "0.8", "--league", "0.3", "--aux-trick", "0.1",
            "--ema", "0.995", "--shared-mlp", "--aux-belief", "0.05"]
    v2 = tmp_path / "v2"
    train.main(base + ["--out", str(v2)])
    assert (v2 / "ckpt_step0000004.pt").exists() and (v2 / "last.pt").exists()
    # V3-shaped: players from the V2-shaped run (oracle columns padded), an open-hand critic, branching at leads
    crit = tmp_path / "split"
    train.main(base[:-3] + ["--out", str(crit), "--oracle-guide", "--oracle-hold", "2", "--oracle-until", "4"])
    v3 = tmp_path / "v3"
    train.main(base + ["--out", str(v3), "--init", str(v2 / "ckpt_step0000004.pt"), "--critic-init",
                       str(crit / "ckpt_step0000004.pt"), "--critic-ramp", "2", "--steps", "2"])
    train.main(base + ["--out", str(v3), "--resume", "--critic-init", str(crit / "ckpt_step0000004.pt"),
                       "--search-pos", "8", "--search-leads", "--search-steps", "2"])
    ck = torch.load(v3 / "last.pt", weights_only=False)
    assert ck["step"] == 4 and ck["feat_dim"] == env.feat_dim(True) and "critic" in ck
    out = tmp_path / "served.pt"
    to_server.main([str(v3 / "ckpt_step0000004.pt"), str(out), "--check", "8", "--first", "0",
                    "--cache", str(tmp_path / "contracts"), "--eval-data", str(train.SMOKE_DATA),
                    "--eval-start", "0", "--eval-deals", "64"])
    bot = _server_module("playdmc").DMCPlayBot(str(out))
    c = env.eval_contracts(tmp_path / "contracts", str(train.SMOKE_DATA), 0, 64).subset(torch.arange(4))
    t = Table(c)
    nets = make_nets(64, 1, shared=True, oracle=True)
    nets[0].load_state_dict(torch.load(v3 / "ckpt_step0000004.pt", weights_only=False)["nets"][0])
    with torch.no_grad():
        for _ in range(9):
            role = env.roles(t.b, c.declarer)
            q = torch.stack([nets[r](t.features(True).float()) for r in range(3)]).gather(
                0, role[None, :, None].expand(1, len(c), 52)).squeeze(0).masked_fill(~t.b.legal(), -1e9)
            assert torch.allclose(bot.values(c, t.b), q, atol=1e-5)
            t.b.play(q.argmax(1))
    avg = tmp_path / "avg.pt"
    to_server.main([str(v2 / "ckpt_step0000002.pt"), str(v2 / "ckpt_step0000004.pt"), str(avg), "--check", "0"])
    assert len(torch.load(avg, weights_only=True)["averaged_from"]) == 2
