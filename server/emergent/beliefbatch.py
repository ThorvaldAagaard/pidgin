"""Belief-net inputs for card play V4: one board -> four per-seat sequences (port of the lab's batch.py).

Kept identical to the trainer's version so serving inputs match training. Step k is the moment
before event k (the calls, then the 52 cards); inputs at step k use only what the viewer sees then:
its own unplayed cards, dummy's once the lead is out, the public auction and play. Hidden cards
enter only once played, so the served position may fill unseen cards with any count-consistent
layout.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_ACT, START, W = 90, 90, 48
E = W + 52
HCP = torch.tensor([max(4 - c % 13, 0) for c in range(52)], dtype=torch.float32)
IMP_BOUNDS = torch.tensor([20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500,
                           1750, 2000, 2250, 2500, 3000, 3500, 4000], dtype=torch.float32)
IN_DIM = 52 + 52 + 31 + 208 + 208 + 14 + 14 + 13 + 4 + 624 + 52
PRIV_DIM = 4 * 52


def imps(points: torch.Tensor) -> torch.Tensor:
    return torch.sign(points) * torch.searchsorted(IMP_BOUNDS.to(points.device), points.abs(), right=True).float()


def to_tensors(sh: dict, idx) -> dict:
    return {k: torch.as_tensor(sh[k][idx]).long() if k not in ("vul_ns", "vul_ew") else torch.as_tensor(sh[k][idx])
            for k in ("owner", "dealer", "vul_ns", "vul_ew", "calls", "n_calls", "cards", "declarer", "level",
                      "strain", "doubled", "score_ns", "tricks", "sys_ns", "sys_ew") if k in sh}


def call_legal(calls, nc, dealer):
    """(B,W,38): legal calls before each call slot, general bridge rules."""
    B = calls.shape[0]
    k = torch.arange(W, device=calls.device)[None].expand(B, -1)
    live = k < nc[:, None]
    c = torch.where(live, calls, -1)

    def last_before(m):                                                         # (B,W): last slot < k where m, else -1
        idx = torch.where(m, k, -1).cummax(1).values
        return F.pad(idx, (1, 0), value=-1)[:, :-1]

    lb = last_before((c >= 0) & (c < 35))
    last = torch.where(lb >= 0, c.gather(1, lb.clamp(min=0)), -1)
    dbl = torch.where(last_before(c == 37) > lb, 2, torch.where(last_before(c == 36) > lb, 1, 0)) * (lb >= 0)
    own = (k - lb) % 2 == 0                                                     # the last bid was made by this side
    out = torch.zeros(B, W, 38, dtype=torch.bool, device=calls.device)
    out[..., :35] = torch.arange(35, device=calls.device)[None, None] > last[..., None]
    out[..., 35] = True
    out[..., 36] = (lb >= 0) & (dbl == 0) & ~own
    out[..., 37] = (dbl == 1) & own & (lb >= 0)
    return out


def build(t: dict) -> dict:
    """Tensors for 4B sequences (board-major: row = 4*b + v), length E."""
    owner, dealer, nc, cards = t["owner"], t["dealer"], t["n_calls"], t["cards"]
    B = owner.shape[0]
    played = t["declarer"] >= 0
    decl = t["declarer"].clamp(min=0)
    dummy = (decl + 2) % 4
    k = torch.arange(E)[None]
    is_call = k < nc[:, None]
    j = (k - nc[:, None]).clamp(0, 51)
    is_card = (~is_call) & (k < nc[:, None] + 52) & (played & (cards[:, 0] >= 0))[:, None]   # bid-only DD boards: contract, no play
    valid = is_call | is_card
    calls_pad = F.pad(t["calls"], (0, E - W), value=-1)
    card_at = cards.gather(1, j).clamp(min=0)                                     # (B,E)
    ev = torch.where(is_call, calls_pad.gather(1, k.expand(B, -1).clamp(max=E - 1)), 38 + card_at)
    ev = torch.where(valid, ev, -1)
    seat = torch.where(is_call, (dealer[:, None] + k) % 4, owner.gather(1, card_at))
    ctrl = torch.where(is_card & (seat == dummy[:, None]), decl[:, None], seat)
    card_oh = F.one_hot(card_at, 52).bool() & is_card[..., None]                   # (B,E,52)
    before = (card_oh.long().cumsum(1) - card_oh.long()) > 0
    hands = F.one_hot(owner, 4).bool().transpose(1, 2)                            # (B,4,52)
    lead_out = is_card & (j > 0)
    dhand = hands[torch.arange(B), dummy]
    dummy_vis = dhand[:, None] & ~before & lead_out[..., None] & played[:, None, None]
    # legal actions
    legal = torch.zeros(B, E, N_ACT, dtype=torch.bool)
    legal[:, :W, :38] = call_legal(t["calls"], nc, dealer)
    actor_rem = hands[torch.arange(B)[:, None], seat.clamp(0, 3)] & ~before         # (B,E,52)
    pos = j % 4
    led = cards.gather(1, (j - pos).clamp(min=0)).clamp(min=0)
    led_suit = led // 13
    suit = torch.arange(52) // 13
    follow = actor_rem & (suit[None, None] == led_suit[..., None])
    must = (pos > 0) & follow.any(-1)
    card_legal = torch.where(must[..., None], follow, actor_rem) & is_card[..., None]
    legal[..., 38:] = card_legal
    legal[..., :38] &= is_call[..., None]
    # per view
    v = torch.arange(4)
    rel = lambda s: (s[:, None] - v[None, :, None]) % 4                          # (B,E) -> (B,4,E)
    own_rem = hands[:, :, None] & ~before[:, None]                                # (B,4,E,52)
    prev_ev = torch.cat([torch.full((B, 1), START), ev[:, :-1].clamp(min=0)], 1)
    prev_seat = torch.cat([torch.full((B, 1), -1), seat[:, :-1]], 1)
    prev_rel = torch.where(prev_seat[:, None] >= 0, (prev_seat[:, None] - v[None, :, None]) % 4, 4)
    vul_us = torch.where(v[None] % 2 == 0, t["vul_ns"][:, None], t["vul_ew"][:, None]).float()
    vul_them = torch.where(v[None] % 2 == 0, t["vul_ew"][:, None], t["vul_ns"][:, None]).float()
    phase = (~is_call & valid).float()
    ctx = [vul_us[..., None, None].expand(B, 4, E, 1), vul_them[..., None, None].expand(B, 4, E, 1),
           F.one_hot((dealer[:, None] - v[None]) % 4, 4).float()[:, :, None].expand(B, 4, E, 4),
           phase[:, None, :, None].expand(B, 4, E, 1),
           F.one_hot(rel(seat.clamp(min=0)), 4).float(),
           (ctrl[:, None] == v[None, :, None]).float()[..., None]]
    pc = phase[:, None, :, None]
    ctx += [F.one_hot((t["level"] - 1).clamp(min=0), 7).float()[:, None, None].expand(B, 4, E, 7) * pc,
            F.one_hot(t["strain"].clamp(min=0), 5).float()[:, None, None].expand(B, 4, E, 5) * pc,
            F.one_hot((decl[:, None] - v[None]) % 4, 4).float()[:, :, None].expand(B, 4, E, 4) * pc,
            F.one_hot(t["doubled"].clamp(min=0), 3).float()[:, None, None].expand(B, 4, E, 3) * pc]
    # play state, the same facts run18 reads (env.dynamic_features + rel_features); zero during the auction
    relseat_e = ((v[:, None] + torch.arange(4)[None]) % 4)                          # [view, r] = seat v + r
    rot = lambda a: a[:, None].expand(B, 4, *a.shape[1:]).gather(
        3, relseat_e[None, :, None, :, None].expand(B, 4, a.shape[1], 4, 52))       # (B,E,4abs,52) -> (B,4,E,4rel,52)
    pos_of = torch.full((B, 53), 99)                                               # slot 52 takes the unplayed (-1)
    pos_of.scatter_(1, torch.where(cards >= 0, cards, 52), torch.arange(52).expand(B, -1).clone())
    pos_of = torch.where(played[:, None], pos_of[:, :52], 99)
    start = (j - pos)[..., None]
    in_cur = (pos_of[:, None] >= start) & (pos_of[:, None] < j[..., None]) & is_card[..., None]   # (B,E,52)
    played_by = rot(before[:, :, None] & hands[:, None]).reshape(B, 4, E, 208)
    cur_by = rot(in_cur[:, :, None] & hands[:, None]).reshape(B, 4, E, 208)
    tc = cards.clamp(min=0).view(B, 13, 4)
    t_suit, t_rank = tc // 13, tc % 13
    trump = torch.tensor([3, 2, 1, 0, 4])[t["strain"].clamp(min=0)]
    sc = torch.where(t_suit == trump[:, None, None], 200 - t_rank,
                     torch.where(t_suit == t_suit[:, :, :1], 100 - t_rank, 0))
    win_pre = torch.stack([tc.gather(2, sc[:, :, :p].argmax(-1, keepdim=True)).squeeze(-1) for p in (1, 2, 3, 4)], -1)
    win_seat = owner.gather(1, win_pre[:, :, 3])                                    # (B,13)
    done = (4 * torch.arange(13) + 4)[None, None] <= j[..., None]                   # trick finished before j
    ns_won = (done & (win_seat % 2 == 0)[:, None]).sum(-1) * is_card
    ew_won = (done & (win_seat % 2 == 1)[:, None]).sum(-1) * is_card
    us_won = torch.where(v[None, :, None] % 2 == 0, ns_won[:, None], ew_won[:, None])
    them_won = torch.where(v[None, :, None] % 2 == 0, ew_won[:, None], ns_won[:, None])
    trick_no = F.one_hot(j // 4, 13) * is_card[..., None]
    pos_oh = F.one_hot(pos, 4) * is_card[..., None]
    wcard = win_pre[torch.arange(B)[:, None], j // 4, (pos - 1).clamp(min=0)]       # card winning the trick so far
    ws, wr = wcard // 13, wcard % 13
    cs, cr = suit[None, None], (torch.arange(52) % 13)[None, None]
    beats = (((cs == ws[..., None]) & (cr < wr[..., None])) | ((cs == trump[:, None, None]) & (ws != trump[:, None])[..., None]))
    beats &= ((pos > 0) & is_card)[..., None]
    unseen = ~before[:, None] & ~own_rem & ~dummy_vis[:, None]
    therm = []
    for m in (own_rem, dummy_vis[:, None].expand(B, 4, E, 52), unseen, in_cur[:, None].expand(B, 4, E, 52)):
        vv = m.reshape(B, 4, E, 4, 13).long()
        hi = (vv.cumsum(-1) - vv).reshape(B, 4, E, 52)                              # higher cards of the same suit in m
        therm += [hi >= 1, hi >= 2, hi >= 3]
    playing = is_card[:, None, :, None]
    play_state = [played_by, cur_by, F.one_hot(us_won, 14) * playing, F.one_hot(them_won, 14) * playing,
                  trick_no[:, None].expand(B, 4, E, 13), pos_oh[:, None].expand(B, 4, E, 4),
                  torch.cat(therm, -1) & playing, beats[:, None].expand(B, 4, E, 52)]
    x = torch.cat([p.to(torch.uint8) for p in
                   [own_rem, dummy_vis[:, None].expand(B, 4, E, 52)] + ctx + play_state], -1)
    # belief target and mask: hidden = not own, not visible dummy, not yet played
    tgt_owner = (owner[:, None] - v[None, :, None]) % 4                            # (B,4,52)
    bmask = ~hands[:, :, None] & ~dummy_vis[:, None] & ~before[:, None] & valid[:, None, :, None]
    decl_side = (v[None] - decl[:, None]) % 2 == 0                                 # (B,4) viewer is on declarer's side
    side_tricks = (torch.where(decl_side, t["tricks"][:, None], 13 - t["tricks"][:, None]) if "tricks" in t
                   else torch.zeros(B, 4, dtype=torch.long))
    side_tricks = torch.where(played[:, None], side_tricks, -1)
    side_score = t["score_ns"].float()[:, None] * torch.where(v % 2 == 0, 1.0, -1.0)[None]
    priv = F.one_hot((owner[:, None] - v[None, :, None]) % 4, 4).float().reshape(B, 4, 208)
    r = lambda a: a.reshape(B * 4, *a.shape[2:])
    sys_ns, sys_ew = (t["sys_ns"], t["sys_ew"]) if "sys_ns" in t else (torch.zeros_like(dealer),) * 2
    sys_us = torch.where(v[None] % 2 == 0, sys_ns[:, None], sys_ew[:, None])
    sys_them = torch.where(v[None] % 2 == 0, sys_ew[:, None], sys_ns[:, None])
    relseat = ((v[:, None] + torch.arange(4)[None]) % 4)[None].expand(B, 4, 4)       # [b, view, r] = seat v + r
    hcp_rel = (hands.float() @ HCP.to(hands.device))[:, None].expand(B, 4, 4).gather(2, relseat)
    lens = hands.reshape(B, 4, 4, 13).sum(-1).float()                               # (B, seat, suit)
    len_rel = lens[:, None].expand(B, 4, 4, 4).gather(2, relseat[..., None].expand(B, 4, 4, 4))
    return dict(x=r(x), prev_ev=r(prev_ev[:, None].expand(B, 4, E)), prev_rel=r(prev_rel),
                act=r(ev[:, None].expand(B, 4, E)), legal=r(legal[:, None].expand(B, 4, E, N_ACT)),
                decide=r((ctrl[:, None] == v[None, :, None]) & valid[:, None]), valid=r(valid[:, None].expand(B, 4, E)),
                is_call=r(is_call[:, None].expand(B, 4, E)), tgt_owner=r(tgt_owner), bmask=r(bmask),
                value=r(imps(side_score) / 10.0), priv=r(priv), nc=nc.repeat_interleave(4),
                dummy_rel=((dummy[:, None] - v[None]) % 4).reshape(-1), lead_out=r(lead_out[:, None].expand(B, 4, E)),
                is_decl=r((v[None] == decl[:, None]) & played[:, None]),
                is_def=r(((v[None] - decl[:, None]) % 2 == 1) & played[:, None]),
                is_lead=r((k == nc[:, None]).expand(B, -1)[:, None].expand(B, 4, E)),
                true_hcp=r(hcp_rel), true_len=r(len_rel), side_tricks=side_tricks.reshape(-1),
                sys=torch.stack([sys_us, sys_them], -1).reshape(B * 4, 2))
