"""The Solo rating: a Bayesian estimate of IMPs per board against the rated team, in shadow mode.

Computed next to the Elo in `players.py` for the same rated boards, stored, not shown. The model,
its parameters and where they come from: docs/solo_rating.md.

    μ = 1000 + 100 · sum(imps) / (n + LAM)        σ = 100 · S / √(n + LAM)

    python -m emergent.solo replay       rebuild solo and solo_history from the Elo's boards
    python -m emergent.solo report       every player: n, μ, σ
    python -m emergent.solo history ID   one player's rated boards, oldest first
"""

import math
import sys
import time

S = 5.0              # IMPs: spread of one board's result around the player's skill
LAM = 8.0            # the prior is worth LAM boards level with the rated team
FORFEIT_IMPS = -7    # a board started and left: the mean lost board against the rated team
BASE = 1000.0        # the rated team
SCALE = 100.0        # points per IMP per board
ORDER_Z = 2.0        # leaderboard order: μ − ORDER_Z·σ

SCHEMA = """
    CREATE TABLE IF NOT EXISTS solo(
        player TEXT PRIMARY KEY, n INTEGER NOT NULL, sum_imps REAL NOT NULL,
        mu REAL NOT NULL, sigma REAL NOT NULL, updated REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS solo_history(
        player TEXT NOT NULL, deal TEXT NOT NULL, imps INTEGER NOT NULL, forfeit INTEGER NOT NULL,
        mu_before REAL NOT NULL, sigma_before REAL NOT NULL, mu_after REAL NOT NULL,
        sigma_after REAL NOT NULL, n_after INTEGER NOT NULL, at REAL NOT NULL,
        PRIMARY KEY(player, deal));
"""


def posterior(n, sum_imps):
    """(μ, σ) in rating points after n boards with this IMP sum."""
    return BASE + SCALE * sum_imps / (n + LAM), SCALE * S / math.sqrt(n + LAM)


START_MU, START_SIGMA = posterior(0, 0)


def schema(conn):
    """Make the tables; a new solo_history next to rated boards is filled from them."""
    conn.executescript(SCHEMA)
    if (conn.execute("SELECT 1 FROM solo_history LIMIT 1").fetchone() is None
            and conn.execute("SELECT 1 FROM boards WHERE state IN ('done','forfeit') LIMIT 1").fetchone()):
        replay(conn)


def rate(conn, pid, deal, imps, forfeit=False, at=None):
    """Rate one result inside the caller's transaction. Returns (μ_before, μ_after), or None
    when this (player, deal) was rated before: a result never counts twice."""
    y = FORFEIT_IMPS if forfeit else int(imps)
    row = conn.execute("SELECT n, sum_imps FROM solo WHERE player=?", (pid,)).fetchone()
    n, total = (row["n"], row["sum_imps"]) if row else (0, 0.0)
    mu0, sd0 = posterior(n, total)
    mu1, sd1 = posterior(n + 1, total + y)
    at = time.time() if at is None else at
    cur = conn.execute("INSERT OR IGNORE INTO solo_history VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (pid, deal, y, int(forfeit), mu0, sd0, mu1, sd1, n + 1, at))
    if cur.rowcount != 1:
        return None
    conn.execute("INSERT INTO solo(player, n, sum_imps, mu, sigma, updated) VALUES(?,?,?,?,?,?) "
                 "ON CONFLICT(player) DO UPDATE SET n=excluded.n, sum_imps=excluded.sum_imps, "
                 "mu=excluded.mu, sigma=excluded.sigma, updated=excluded.updated",
                 (pid, n + 1, total + y, mu1, sd1, at))
    return mu0, mu1


def replay(conn):
    """Rebuild solo and solo_history from the Elo's rated boards, in the order they were rated."""
    rows = conn.execute("SELECT player, deal, state, imps, at FROM boards "
                        "WHERE state IN ('done','forfeit') ORDER BY at, rowid").fetchall()
    own = not conn.in_transaction
    if own:
        conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM solo")
        conn.execute("DELETE FROM solo_history")
        for r in rows:
            rate(conn, r["player"], r["deal"], r["imps"], r["state"] == "forfeit", r["at"])
        if own:
            conn.execute("COMMIT")
    except Exception:
        if own:
            conn.execute("ROLLBACK")
        raise
    return len(rows)


def rating(conn, pid):
    """{mu, sigma, boards} for a player; a new one is at the start."""
    row = conn.execute("SELECT n, mu, sigma FROM solo WHERE player=?", (pid,)).fetchone()
    if row is None:
        return {"mu": START_MU, "sigma": START_SIGMA, "boards": 0}
    return {"mu": row["mu"], "sigma": row["sigma"], "boards": row["n"]}


def history(conn, pid):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM solo_history WHERE player=? ORDER BY at, rowid", (pid,))]


def leaderboard(conn, limit=50):
    """Named players by μ − ORDER_Z·σ. Not served yet."""
    rows = conn.execute("SELECT p.id, p.name, s.n, s.mu, s.sigma FROM solo s JOIN players p ON p.id = s.player "
                        "WHERE p.name IS NOT NULL ORDER BY s.mu - ? * s.sigma DESC, s.n DESC LIMIT ?",
                        (ORDER_Z, limit)).fetchall()
    return [{"id": r["id"], "name": r["name"], "boards": r["n"], "mu": r["mu"], "sigma": r["sigma"]}
            for r in rows]


def main(argv):
    from emergent import players
    conn = players.db()
    cmd = argv[0] if argv else "report"
    if cmd == "replay":
        print(f"replayed {replay(conn)} rated boards")
    elif cmd == "report":
        for r in conn.execute("SELECT s.*, p.name FROM solo s LEFT JOIN players p ON p.id = s.player "
                              "ORDER BY s.mu - ? * s.sigma DESC", (ORDER_Z,)):
            print(f"{r['player']}  {r['name'] or '-':20s}  n {r['n']:4d}  μ {r['mu']:7.1f}  σ {r['sigma']:5.1f}")
    elif cmd == "history" and len(argv) > 1:
        for r in history(conn, argv[1]):
            print(f"{r['deal']}  {r['imps']:+3d}{' forfeit' if r['forfeit'] else ''}  "
                  f"μ {r['mu_before']:7.1f} -> {r['mu_after']:7.1f}  σ {r['sigma_after']:5.1f}  n {r['n_after']}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
