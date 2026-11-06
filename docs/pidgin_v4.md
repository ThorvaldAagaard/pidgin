# Pidgin V4

Pidgin V4 is the default team of the table at <https://bridge.localgeek.jp/> and of the
Brill and BBO APIs. Its team ID is `PidginV4`. It bids with the
[Pidgin V3.1](pidgin_v3_1.md) bidding model (`pidginv3_1_bid_s20000.pt`), without bidding
search, and plays the cards with **Pidgin card play V4**: a belief net and PIMC
(Perfect Information Monte Carlo) search, with [card play V3](card_play.md#card-play-v3)
breaking ties. The table shows it as **Pidgin V4**.

## Card play V4

For each card the seat on turn owes:

1. The belief net (`belief_v4_s34442.pt`, 52M parameters, a GLU MLP 1024 wide and 8 blocks
   deep) reads what the seat sees: its hand, dummy once faced, the auction, the cards played,
   and both pairs' bidding systems. It gives each hidden card's probability of being in each
   hidden hand.
2. Layouts of the hidden cards are drawn card by card from those probabilities, keeping the
   suits a seat has shown out of. Each is weighted by how likely the belief net finds it,
   until the effective sample size reaches 16 (at most 4,096 draws). Then 64 are resampled.
3. The distinct layouts are solved double dummy (DDS through `endplay`), most-drawn first,
   until 0.6 s have passed (`PIMC_BUDGET_S`) and at least 16 are solved. The card with the
   best mean tricks wins. The cap only bites in the first tricks on the 1-vCPU server; the
   measured results below used all 64.
4. Cards within 0.05 tricks of the best count as tied (about 80% of decisions). Card play V3
   (`play_run18_s4000.pt`) picks among them when its own card is one of them; otherwise a
   tied card is drawn at random, seeded by the position.

The code is `server/emergent/playpimc.py` (search and belief net) and
`server/emergent/beliefbatch.py` (the belief net's inputs, the same as in training). Answers
do not depend on which seat names the position. They are deterministic unless the time cap
cuts the solving short, since then how many layouts are solved depends on the machine.

The table at / returns one net card per request and the page asks again, so each card shows
as soon as it is chosen.

### Bidding systems

The belief net reads each pair's system as one of four slots: VieNNa (Pidgin's own bidding),
SAYC, 2/1 game force, and WBridge5. On the Brill API, `ourSystem` / `theirSystem` pick them
(see [serving](serving.md#bidding-systems-on-the-brill-api)). Without them, the opponents
are taken to play 2/1, the system most robots play. The site's own tables use VieNNa at
both sides. The bidding model reads no system and always bids VieNNa.

## How it was trained

The belief net was trained with supervised learning to predict the owner of every hidden
card at every step of a deal, from the seat's view. The data was self-play deals with full
auctions and play: the Pidgin V3.1 bidder (VieNNa); deals where each side drew one of
VieNNa, BEN SAYC and BEN 2/1, with card play V3 playing the cards; and WBridge5's own
deals. Training ran on one GPU for 34,442 steps at batch 1,024. The trainer is in the lab
repository, not here.

## Results

Full boards, Pidgin V3.1 bidding at both tables, so only the card play differs. Same
auctions at both tables on every board. IMPs per board for Pidgin's pair, ± 95% interval:

| Opponent | Boards | Card play V3.1 (V2) | Card play V4 | V4 − V2, paired |
|---|---|---|---|---|
| Lia (2/1) | 200 | −0.12 ± 0.98 | +1.12 ± 0.89 | +1.25 ± 0.82 |
| Brill (2/1, GIB style) | 100 | +0.08 ± 1.34 | +1.23 ± 1.31 | +1.15 ± 1.46 |

The belief net plays about 0.1 s per decision on one CPU core.
