"""Card-play trainer environment: legality, trick winner, scoring and returns, auction to contract,
the feature layout, and that no input reads a hidden card."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cardplay import env, train
from cardplay.model import NetPlayer, RandomPlayer, Table, make_nets, pad_oracle, play_out
from training.bridge.play import PlayBatch
from training.bridge.scoring import contract_score, imps

ROOT = Path(__file__).resolve().parents[1]


def random_contracts(n, seed=0):
    g = np.random.default_rng(seed)
    owner = np.stack([g.permutation(np.repeat(np.arange(4), 13)) for _ in range(n)])
    tricks = g.integers(0, 14, (n, 4, 5))
    hist = np.full((n, 8), -1)
    dealer = g.integers(0, 4, n)
    for i in range(n):
        bid = int(g.integers(0, 35))
        tail = [35, 35, 35] if i % 3 == 0 else ([36, 35, 35, 35] if i % 3 == 1 else [36, 37, 35, 35, 35])
        seq = [bid] + tail
        hist[i, :len(seq)] = seq
    v = g.integers(0, 4, n)
    return env.contracts_from_auctions(hist, dealer, (v == 1) | (v == 3), (v == 2) | (v == 3), owner, tricks,
                                       np.arange(n))


def test_contract_from_auction_declarer_and_double():
    owner = np.repeat(np.arange(4), 13)[None]
    tricks = np.arange(20).reshape(1, 4, 5) % 14
    hist = np.array([[2, 35, 7, 36, 35, 35, 35, -1]])   # dealer E: 1H P 2H X P P P -> E declares 2H X
    c = env.contracts_from_auctions(hist, np.array([1]), np.array([False]), np.array([True]), owner, tricks, np.array([0]))
    assert int(c.declarer[0]) == 1 and int(c.level[0]) == 2 and int(c.doubled[0]) == 1
    assert int(c.strain[0]) == 2 and int(c.trump[0]) == 1          # hearts: bid order 2, card order 1
    assert int(c.dd_tricks[0]) == tricks[0, 1, 1]
    assert bool(c.decl_vul()[0])                                    # E is EW, EW vulnerable
    c2 = env.contracts_from_auctions(np.array([[35, 35, 35, 35]]), np.array([0]), np.array([False]),
                                     np.array([False]), owner, tricks, np.array([0]))
    assert len(c2) == 0                                             # passed out is dropped


def test_score_table_and_imps_match_reference():
    c = random_contracts(300)
    for k in (0, 5, 9, 13):
        s = c.score(torch.full((len(c),), k))
        for i in range(len(c)):
            ref = contract_score(int(c.level[i]), int(c.strain[i]), k, int(c.doubled[i]), bool(c.decl_vul()[i]))
            assert int(s[i]) == ref
    diffs = torch.tensor([0, 10, 20, -20, 49, 50, -2990, 3000, 5000], dtype=torch.float32)
    assert env.imps_t(diffs).tolist() == [imps(int(d)) for d in diffs]


def test_return_zero_at_dd_and_sign():
    c = random_contracts(200)
    assert torch.all(c.imps_vs_dd(c.dd_tricks) == 0)
    assert torch.all(c.imps_vs_dd((c.dd_tricks + 1).clamp(max=13)) >= 0)
    assert torch.all(c.imps_vs_dd((c.dd_tricks - 1).clamp(min=0)) <= 0)


def test_trick_winner_trump_and_follow_suit():
    owner = torch.arange(52)[None] // 13          # N spades, E hearts, S diamonds, W clubs; declarer N
    for trump, expect in ((4, 1), (0, 0), (3, 3)):
        b = PlayBatch(owner, torch.tensor([trump]), torch.tensor([0]))
        assert int(b.to_play()[0]) == 1
        for _ in range(4):
            legal = b.legal()[0]
            assert legal.sum() == 13 - b.trick_no
            b.play(legal.nonzero()[0])
        assert int(b.trick_winner[0, 0]) == expect


def test_must_follow_suit_and_random_games_consistent():
    c = random_contracts(64, seed=3)
    b = PlayBatch(c.owner, c.trump, c.declarer)
    g = torch.Generator().manual_seed(0)
    while not b.done:
        legal = b.legal()
        hand = b.hand(b.to_play())
        assert torch.all(legal <= hand) and torch.all(legal.any(1))
        if b.pos > 0:
            led = b.led_suit()
            has = (hand & (b.suit_of[None] == led[:, None])).any(1)
            off_suit = legal & (b.suit_of[None] != led[:, None])
            assert not (off_suit.any(1) & has).any()
        b.play(torch.multinomial(legal.float(), 1, generator=g).squeeze(1))
    assert torch.all(b.tricks_won.sum(1) == 13)
    assert torch.all(~b.unplayed)


def _redeal_hidden(b, c, seed_base=0):
    """Owners with the unplayed cards of the two hands the actor cannot see shuffled between them."""
    seat, decl = b.to_play(), c.declarer
    dummy = (decl + 2) % 4
    owner2 = c.owner.clone()
    for i in range(len(c)):
        visible = {int(seat[i])} | ({int(dummy[i])} if b.t > 0 else set())
        if int(seat[i]) == int(dummy[i]):
            visible.add(int(decl[i]))
        hidden = [s for s in range(4) if s not in visible]
        cards = [k for k in range(52) if b.unplayed[i, k] and int(c.owner[i, k]) in hidden]
        perm = np.random.default_rng(seed_base + i).permutation([int(c.owner[i, k]) for k in cards])
        for k, o in zip(cards, perm):
            owner2[i, k] = int(o)
    b2 = PlayBatch(owner2, c.trump, c.declarer)
    b2.unplayed, b2.history, b2.t = b.unplayed.clone(), b.history.clone(), b.t
    b2.tricks_won, b2.leader, b2.trick_winner = b.tricks_won.clone(), b.leader.clone(), b.trick_winner.clone()
    return owner2, b2


@pytest.mark.parametrize("steps", [0, 1, 9, 14, 30])
def test_features_never_see_hidden_cards(steps):
    c = random_contracts(32, seed=7)
    g = torch.Generator().manual_seed(1)
    b = PlayBatch(c.owner, c.trump, c.declarer)
    for _ in range(steps):
        b.play(torch.multinomial(b.legal().float(), 1, generator=g).squeeze(1))
    owner2, b2 = _redeal_hidden(b, c)
    assert not torch.equal(owner2, c.owner)
    x = torch.cat([env.static_features(c), env.dynamic_features(b, c.declarer)], 1)
    x2 = torch.cat([env.static_features(c), env.dynamic_features(b2, c.declarer)], 1)
    assert torch.equal(x, x2)


def test_feature_dims_roles_and_oracle_columns():
    c = random_contracts(16)
    t = Table(c)
    assert env.FEAT_DIM == 2658 and env.feat_dim(True) == 2658 + 208
    assert t.features().shape == (16, env.FEAT_DIM)
    x = t.features(oracle=True)
    assert x.shape == (16, env.feat_dim(True)) and not x[:, env.STATIC_DIM:env.static_dim(True)].any()
    assert torch.equal(env.static_features(c, oracle=True)[:, :env.STATIC_DIM], env.static_features(c))
    assert torch.all(env.roles(t.b, c.declarer) == 1)                # opening lead is LHO
    open_t = Table(c, open_hand=True)
    o = open_t.features(oracle=True)[:, env.STATIC_DIM:env.static_dim(True)].view(16, 52, 4).argmax(-1)
    assert torch.equal(o, (c.owner - c.declarer[:, None]) % 4)


def test_pad_oracle_keeps_q_values():
    c = random_contracts(12, seed=2)
    plain = make_nets(32, 2, seed=3, shared=True)
    wide = make_nets(32, 2, seed=4, shared=True, oracle=True)
    wide[0].load_state_dict(pad_oracle(plain[0].state_dict()))
    t = Table(c, open_hand=True)                                    # even with the holders shown
    with torch.no_grad():
        for r in range(3):
            assert torch.allclose(plain[r](t.features().float()), wide[r](t.features(True).float()), atol=1e-5)


def test_server_features_match_trainer():
    """server/emergent/playdmc.py reproduces the trainer's input at every step of a random play."""
    spec = importlib.util.spec_from_file_location("playdmc_t", ROOT / "server/emergent/playdmc.py")
    pd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd)
    assert pd.FEAT_DIM == env.FEAT_DIM and pd.STATIC_DIM == env.STATIC_DIM
    c = random_contracts(24, seed=8)
    t = Table(c)
    g = torch.Generator().manual_seed(4)
    while not t.b.done:
        assert torch.equal(pd.features(c, t.b), t.features().float())
        t.b.play(torch.multinomial(t.b.legal().float(), 1, generator=g).squeeze(1))


def test_play_out_runs_with_nets():
    c = random_contracts(16)
    tr = play_out(c, NetPlayer(make_nets(64, 1)), RandomPlayer(0))
    assert tr.shape == (16,) and tr.min() >= 0 and tr.max() <= 13
    tr = play_out(c, NetPlayer(make_nets(64, 1, oracle=True)), RandomPlayer(0), open_hand=True)
    assert tr.shape == (16,)


def _actor_batch(n=24, seed=5, league=None):
    c = random_contracts(n, seed=seed)
    nets = make_nets(32, 1, seed=seed)
    return c, train.actor_round(nets, c, "cpu", 0.0, torch.Generator().manual_seed(0), want_v=True, league=league)


def test_td_lambda_one_equals_monte_carlo():
    c, (static, Xd, card, role, board, tricks, extra) = _actor_batch()
    gt = c.imps_vs_dd(tricks)
    mc = torch.where(role == 0, gt[board], -gt[board])
    gl = env.lambda_returns(extra["v_decl"], gt, 1.0).reshape(-1)
    assert torch.equal(torch.where(role == 0, gl, -gl), mc)
    g0 = env.lambda_returns(extra["v_decl"], gt, 0.0)
    assert torch.equal(g0[:-1], extra["v_decl"][1:]) and torch.equal(g0[-1], gt)


def test_lambda_return_by_hand():
    out = env.lambda_returns(torch.tensor([[9.0], [2.0], [4.0]]), torch.tensor([10.0]), 0.5)
    assert out[:, 0].tolist() == [4.5, 7.0, 10.0]        # G2 = 10; G1 = .5*4 + .5*10; G0 = .5*2 + .5*7


def test_v_decl_is_signed_greedy_value():
    c, (static, Xd, card, role, board, tricks, extra) = _actor_batch(8)
    nets = make_nets(32, 1, seed=5)
    x = torch.cat([static[board], Xd], 1).float()
    with torch.no_grad():
        q = torch.stack([net(x) for net in nets]).gather(0, role[None, :, None].expand(1, len(x), 52)).squeeze(0)
    v = q.gather(1, card[:, None]).squeeze(1)              # eps 0: the played card is the argmax
    assert torch.allclose(torch.where(role == 0, v, -v), extra["v_decl"].reshape(-1), atol=1e-5)


def test_trick_labels():
    c, (static, Xd, card, role, board, tricks, extra) = _actor_batch(16)
    n = len(c)
    lab = train.trick_labels(extra["trick_winner"], c.declarer, role, n).reshape(52, n)
    decl_side = (role == 0).reshape(52, n)
    won_by_decl = torch.where(decl_side, lab, 1 - lab).sum(0) / 4
    assert torch.equal(won_by_decl.long(), tricks)


def test_league_rows_marked_and_played_by_snapshot():
    c = random_contracts(32, seed=9)
    nets, snaps = make_nets(32, 1, seed=1), make_nets(32, 1, seed=2)
    rows = torch.tensor([0, 3, 7, 20]); sdecl = torch.tensor([True, False, True, False])
    static, Xd, card, role, board, tricks, extra = train.actor_round(nets, c, "cpu", 0.0, torch.Generator(),
                                                                     want_v=True, league=(snaps, rows, sdecl))
    n = len(c)
    mine, r = extra["mine"].reshape(52, n), role.reshape(52, n)
    assert mine[:, [i for i in range(n) if i not in rows.tolist()]].all()
    for k, i in enumerate(rows.tolist()):
        assert torch.equal(~mine[:, i], (r[:, i] == 0) == bool(sdecl[k]))
    b = PlayBatch(c.owner, c.trump, c.declarer)
    x = torch.cat([static[board], Xd], 1).float().reshape(52, n, -1)
    card = card.reshape(52, n)
    for t in range(52):
        with torch.no_grad():
            qc = torch.stack([net(x[t]) for net in nets]).gather(0, r[t][None, :, None].expand(1, n, 52)).squeeze(0)
            qs = torch.stack([net(x[t]) for net in snaps]).gather(0, r[t][None, :, None].expand(1, n, 52)).squeeze(0)
        q = torch.where(mine[t][:, None], qc, qs).masked_fill(~b.legal(), -1e9)
        assert torch.equal(q.argmax(1), card[t])
        b.play(card[t])
    assert (~mine).any()


def test_critic_sees_holders_players_do_not():
    c = random_contracts(16, seed=10)
    nets, critic = make_nets(32, 1, seed=1, shared=True, oracle=True), make_nets(32, 1, seed=2, oracle=True)
    static, Xd, card, role, board, tricks, extra = train.actor_round(nets, c, "cpu", 0.0, torch.Generator(),
                                                                     want_v=True, critic=critic)
    o = env.STATIC_DIM
    assert not static[:, o:].any()
    assert torch.equal(extra["static_full"][:, o:], env.oracle_columns(c))
    assert extra["qc"].shape == (52, len(c))


def _rel_inputs(b, decl):
    seat = b.to_play()
    other_seat = torch.where((seat - decl) % 4 == 2, decl, (decl + 2) % 4)
    own, other = b.hand(seat), b.hand(other_seat) & (b.t > 0)
    cur = torch.zeros(b.n, 52, dtype=torch.bool)
    if b.pos:
        cur.scatter_(1, b.trick_cards(), True)
    return own, other, cur


def test_rel_features_counts_and_beats():
    """Higher-card thermometers match a brute-force count; at the 4th card, 'beats' of the card played
    equals 'that seat wins the trick'."""
    c = random_contracts(48, seed=11)
    b = PlayBatch(c.owner, c.trump, c.declarer)
    g = torch.Generator().manual_seed(2)
    checked = 0
    while not b.done:
        own, other, cur = _rel_inputs(b, c.declarer)
        r = env.rel_features(b, own, other, cur)
        assert r.shape == (b.n, env.REL_DIM)
        unseen = b.unplayed & ~own & ~other
        for p_i, m in enumerate((own, other, unseen, cur)):
            for i in range(0, b.n, 7):
                for k in range(52):
                    hi = int(m[i, (k // 13) * 13:k].sum())
                    for th in range(3):
                        assert bool(r[i, (p_i * 3 + th) * 52 + k]) == (hi >= th + 1)
        card = torch.multinomial(b.legal().float(), 1, generator=g).squeeze(1)
        beats = r[:, 12 * 52:].gather(1, card[:, None]).squeeze(1)
        if b.pos == 0:
            assert not r[:, 12 * 52:].any()
        seat = b.to_play()
        last = b.pos == 3
        b.play(card)
        if last:
            assert torch.equal(beats, b.trick_winner[:, b.t // 4 - 1] == seat)
            checked += 1
    assert checked == 13


@pytest.mark.skipif(not Path(env.DATA).exists(), reason="DD dataset missing (data/dds_results_100M.npy or $BRIDGE_DATA)")
def test_dd_tricks_from_real_file():
    owners, tricks = env.load_dataset(env.DATA)            # memory-mapped
    o = np.asarray(owners[env.EVAL_START:env.EVAL_START + 4]); tr = np.asarray(tricks[env.EVAL_START:env.EVAL_START + 4])
    assert (np.stack([(o == s).sum(1) for s in range(4)], 1) == 13).all()
    assert tr.max() <= 13
