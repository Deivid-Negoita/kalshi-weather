"""Price every settled Kalshi temperature bracket with model.py's forecast, then trade against the archived quotes.

    python backtest.py         # needs data/model/preds.parquet; writes results/ and prints the tables
    python backtest.py boost   # the same on boost.py's forecast (preds_boost.parquet, *_boost outputs)

At each decision time the model's P(YES) is compared with the last hourly yes bid / yes ask at or
before it. A trade buys YES at the ask or NO at (1 - bid) when the expected value after Kalshi's fee
clears a threshold theta. theta is picked on 2022-25 and the result reported on 2026, which is held out.
"""
import functools
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fetch import DATA, _event_date, stations_df
from model import DECISIONS, _read, prob, std_offset

HOLDOUT = pd.Timestamp("2026-01-01")  # 61k of the 94k settled temperature markets are in 2026
STALE_H = 6  # a quote older than this at decision time is no quote
LOT = 10  # fees are charged per order and rounded up to the cent: assume 10-contract orders
THETAS = [0.0, 0.02, 0.05, 0.10, 0.15]
SETS = [("eve", "high"), ("eve", "low"), ("morn", "high"), ("morn", "low")]
TAU = 0.55  # implied(): an open-ended bracket's mean distance past its edge is TAU / (1 - TAU) (Crosier 2026)
QCACHE = DATA / "model" / "quotes_all.parquet"
ALIASES = {"KXHIGHHOU": "HOU"}  # 2024-25 Hobby series, before KXHIGHTHOU
OUT = Path(__file__).parent / "results"
TAG = "_" + sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else ""


def fee(price):
    """Kalshi's quadratic taker fee per contract, for a LOT-contract order at `price` dollars."""
    return math.ceil(round(0.07 * LOT * price * (1 - price) * 100, 6)) / 100 / LOT


def markets():
    st = stations_df()
    code = {s: r.code for r in st.itertuples() for s in str(r.series).split(";")} | ALIASES
    tz = dict(zip(st.code, st.tz))
    out = []
    for f in sorted((DATA / "kalshi" / "markets").glob("*.parquet")):
        if not re.match(r"KX(HIGH|LOWT)", f.stem) or f.stem not in code:
            continue
        m = _read([f], ["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike", "result"])
        m = m[m.strike_type.notna() & m.result.isin(["yes", "no"])]
        out.append(m.reindex(columns=["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike",
                                      "result"]).assign(series=f.stem, code=code[f.stem],
                                                        kind="low" if "LOWT" in f.stem else "high"))
    m = pd.concat(out, ignore_index=True)
    m["day"] = m.event_ticker.map(_event_date)
    m["tz"] = m.code.map(tz)
    return m.dropna(subset=["day"])


def p_yes(r):
    """Model P(YES) for one market row joined to its forecast grid (strikes are whole degrees)."""
    if r.strike_type == "greater":
        return 1 - prob(r.cdf, r.k0, r.floor_strike)
    if r.strike_type == "less":
        return prob(r.cdf, r.k0, r.cap_strike - 1)
    return prob(r.cdf, r.k0, r.cap_strike) - prob(r.cdf, r.k0, r.floor_strike - 1)


def quotes(m):
    """Attach the last hourly quote at or before each row's decision time (unix seconds `t`)."""
    out = []
    for s, g in m.groupby("series"):
        c = _read(sorted((DATA / "kalshi" / "candles" / s).glob("part-*.parquet")),
                  ["ticker", "end_period_ts", "yes_bid.close", "yes_ask.close", "volume"])
        if c.empty:
            continue
        c = c.rename(columns={"yes_bid.close": "bid", "yes_ask.close": "ask", "end_period_ts": "qt"})
        c[["bid", "ask"]] = c[["bid", "ask"]].apply(pd.to_numeric, errors="coerce")  # some parts hold "0.0100"
        c = c[c.ticker.isin(g.ticker)].dropna(subset=["bid", "ask"]).astype({"qt": "int64"})
        g = pd.merge_asof(g.sort_values("t"), c.sort_values("qt"), left_on="t", right_on="qt", by="ticker",
                          direction="backward", tolerance=STALE_H * 3600)
        out.append(g)
    q = pd.concat(out, ignore_index=True).dropna(subset=["qt"])
    q["age_h"] = (q.t - q.qt) / 3600
    return q[(q.bid > 0) | (q.ask < 1)]  # 0 bid and 1 ask: nobody quoting


def market_quotes():
    """Every settled bracket's last quote at or before each decision, model-free (build() minus preds).
    Cached; rebuilt when a Kalshi markets file is newer than the cache."""
    newest = max(f.stat().st_mtime for f in (DATA / "kalshi" / "markets").glob("*.parquet"))
    if QCACHE.exists() and QCACHE.stat().st_mtime > newest:
        return pd.read_parquet(QCACHE)
    m = markets()
    rows = [m.assign(decision=d, t=(m.day + pd.to_timedelta(h - m.tz.map(std_offset), unit="h") - pd.Timestamp(0))
                     // pd.Timedelta(seconds=1)) for d, h in DECISIONS.items()]
    q = quotes(pd.concat(rows, ignore_index=True))
    q["yes"] = (q.result == "yes").astype(int)
    q.to_parquet(QCACHE)
    return q


def implied(q):
    """Market mean and sd of the settlement value per (code, kind, day, decision): brackets at their midpoint,
    weighted by their mids. NaN unless >= 3 brackets are quoted and the mids add to ~1."""
    off = TAU / (1 - TAU)
    x = np.select([q.strike_type == "greater", q.strike_type == "less"],
                  [q.floor_strike + 1 + off, q.cap_strike - 1 - off], (q.floor_strike + q.cap_strike) / 2)
    mid = (q.bid + q.ask) / 2
    q = q.assign(mid=mid, m1=mid * x, m2=mid * x ** 2)
    e = q.groupby(["code", "kind", "day", "decision", "series"]).agg(
        s=("mid", "sum"), n=("mid", "size"), m1=("m1", "sum"), m2=("m2", "sum"))
    e = e[(e.n >= 3) & e.s.between(0.8, 1.25)]
    e["mkt"] = e.m1 / e.s
    e["mkt_sd"] = np.sqrt((e.m2 / e.s - e.mkt ** 2).clip(lower=0))
    return e.groupby(level=[0, 1, 2, 3])[["mkt", "mkt_sd"]].mean().reset_index()


@functools.cache
def market_history():
    return implied(market_quotes())


def build():
    preds = pd.read_parquet(DATA / "model" / f"preds{TAG}.parquet", columns=["code", "decision", "kind", "day", "k0", "cdf"])
    m = markets()
    rows = []
    for decision, kind in SETS:
        g = m[m.kind == kind].merge(preds[(preds.decision == decision) & (preds.kind == kind)],
                                    on=["code", "kind", "day"])
        local = pd.to_timedelta(DECISIONS[decision] - g.tz.map(std_offset), unit="h")
        g["t"] = (g.day + local - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
        rows.append(g.assign(decision=decision))
    m = pd.concat(rows, ignore_index=True)
    m["p"] = [p_yes(r) for r in m.itertuples()]
    m["yes"] = (m.result == "yes").astype(int)
    return quotes(m.drop(columns="cdf"))


def trades(q, theta):
    """Pick the better side when its fee-adjusted edge clears theta; PnL per contract in dollars."""
    ev_yes = np.where(q.ask < 1, q.p - q.ask - q.ask.map(fee), -np.inf)
    ev_no = np.where(q.bid > 0, q.bid - q.p - q.bid.map(fee), -np.inf)
    side = np.where(ev_yes >= ev_no, "yes", "no")
    ev = np.maximum(ev_yes, ev_no)
    t = q.assign(side=side, ev=ev)[ev > theta].copy()
    cost = np.where(t.side == "yes", t.ask, 1 - t.bid)
    t["price"] = cost
    t["pnl"] = np.where(t.side == "yes", t.yes, 1 - t.yes) - cost - pd.Series(cost, index=t.index).map(fee)
    return t


def summary(t):
    """Per-contract PnL and a t-stat over calendar days (brackets on one day are not independent)."""
    if t.empty:
        return pd.Series({"trades": 0})
    d = t.groupby("day").pnl.sum()
    return pd.Series({"trades": len(t), "days": len(d), "win%": 100 * (t.pnl > 0).mean(),
                      "avg price": t.price.mean(), "pnl/contract": t.pnl.mean(), "total $/contract-lot": t.pnl.sum(),
                      "t (by day)": d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 1 else np.nan})


def scores(q):
    """Model vs market mid as forecasts of the result, on the same two-sided quotes."""
    two = q[(q.bid > 0) & (q.ask < 1)]
    mid = ((two.bid + two.ask) / 2).clip(0.01, 0.99)
    p = two.p.clip(0.01, 0.99)
    ll = lambda x: -(two.yes * np.log(x) + (1 - two.yes) * np.log(1 - x)).mean()
    return pd.Series({"markets": len(two), "Brier model": ((p - two.yes) ** 2).mean(),
                      "Brier market": ((mid - two.yes) ** 2).mean(), "logloss model": ll(p), "logloss market": ll(mid),
                      "spread": (two.ask - two.bid).mean()})


def encompassing(q):
    """Logit of the result on logit(mid) and logit(model): does the model add news the price lacks?"""
    import statsmodels.api as sm
    two = q[(q.bid > 0) & (q.ask < 1)]
    lg = lambda x: np.log(x / (1 - x))
    X = sm.add_constant(pd.DataFrame({"market": lg(((two.bid + two.ask) / 2).clip(0.01, 0.99)),
                                      "model": lg(two.p.clip(0.01, 0.99))}))
    r = sm.Logit(two.yes, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": two.day.factorize()[0]})
    return pd.DataFrame({"coef": r.params, "t": r.tvalues}).round(3)


def blend(q, start="2025-01-01"):
    """Market + model: refit logit(result) ~ logit(mid) + logit(model) each month on all earlier months
    of the same set, then use that probability in place of the model's. Needs the quote, so two-sided only."""
    import statsmodels.api as sm
    lg = lambda x: np.log(x / (1 - x))
    two = q[(q.bid > 0) & (q.ask < 1)].copy()
    two["lm"], two["lp"] = lg(((two.bid + two.ask) / 2).clip(.01, .99)), lg(two.p.clip(.01, .99))
    out = []
    for _, g in two.groupby(["decision", "kind"]):
        for m in pd.date_range(start, g.day.max(), freq="MS"):
            tr, te = g[g.day < m], g[(g.day >= m) & (g.day < m + pd.offsets.MonthBegin())]
            if len(tr) < 2000 or te.empty:
                continue
            r = sm.Logit(tr.yes, sm.add_constant(tr[["lm", "lp"]])).fit(disp=0)
            out.append(te.assign(p=r.predict(sm.add_constant(te[["lm", "lp"]], has_constant="add"))))
    return pd.concat(out)


def choose(train):
    """theta per set: the best mean PnL on >= 100 training trades, and only if it made money at all."""
    best = {}
    for k, g in train.groupby(["decision", "kind"]):
        res = {th: t.pnl.mean() for th in THETAS if len(t := trades(g, th)) >= 100}
        if res and max(res.values()) > 0:
            best[k] = max(res, key=res.get)
    return best


def apply(q, best):
    return pd.concat([trades(g, best[k]).assign(theta=best[k]) for k, g in q.groupby(["decision", "kind"])
                      if k in best])


def charts(q, picked, blended):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    two = q[(q.bid > 0) & (q.ask < 1) & (q.day >= HOLDOUT)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    bins = np.linspace(0, 1, 11)
    for name, x in [("model", two.p), ("market mid", (two.bid + two.ask) / 2)]:
        b = pd.cut(x, bins, include_lowest=True)
        r = two.yes.groupby(b, observed=True).mean()
        ax[0].plot(x.groupby(b, observed=True).mean(), r, "o-", label=name)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(title="Reliability, 2026 holdout", xlabel="forecast P(YES)", ylabel="observed YES rate")
    ax[0].legend()
    for a, t, title in [(ax[1], picked, "Model alone"), (ax[2], blended, "Market + model blend")]:
        for (decision, kind), g in t.groupby(["decision", "kind"]):
            a.plot(g.groupby("day").pnl.sum().cumsum(), label=f"{kind} @ {decision}")
        a.axvline(HOLDOUT, color="k", ls=":", lw=1)
        a.set(title=f"{title}: cumulative PnL after fees", ylabel="$ per 1-contract trade")
        a.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / f"backtest{TAG}.png", dpi=120)


def main():
    q = build()
    q.to_parquet(DATA / "model" / f"quotes{TAG}.parquet")
    q["period"] = np.where(q.day >= HOLDOUT, "2026 holdout", "2022-25")
    by = ["period", "decision", "kind"]
    pd.set_option("display.width", 200)
    print(f"{len(q):,} market-decisions with a quote <= {STALE_H}h old (median age {q.age_h.median():.1f}h)\n")
    print("calibration, model vs market mid\n", q.groupby(by).apply(scores, include_groups=False).round(4).to_string())
    print("\ndoes the model add information to the price? (logit, SEs clustered by day)")
    for period, g in q.groupby("period"):
        print(period, "\n", encompassing(g).to_string())
    print("\ntrading by threshold (theta is chosen on 2022-25 only)")
    tab = pd.concat({th: trades(q, th).groupby(by).apply(summary, include_groups=False) for th in THETAS},
                    names=["theta"])
    print(tab.round(4).to_string())
    best = choose(q[q.period == "2022-25"])
    picked = apply(q, best)
    print("\nchosen theta per set, and its holdout result:", best)
    print(picked.groupby(by).apply(summary, include_groups=False).round(4).to_string())
    t = trades(q[(q.decision == "eve") & (q.kind == "high")], 0.05)
    print("\nthe edge decays: model alone, highs at the evening decision, theta 0.05, by year")
    print(t.groupby(t.day.dt.year).apply(summary, include_groups=False).round(4).to_string())

    b = blend(q).assign(period=lambda d: np.where(d.day >= HOLDOUT, "2026 holdout", "2025"))
    print("\nmarket + model blend, refit monthly from 2025 on earlier months only")
    print(b.groupby(by).apply(lambda g: pd.Series({"markets": len(g), "Brier blend": ((g.p - g.yes) ** 2).mean(),
          "Brier market": ((((g.bid + g.ask) / 2) - g.yes) ** 2).mean()}), include_groups=False).round(4).to_string())
    tab = pd.concat({th: trades(b, th).groupby(by).apply(summary, include_groups=False) for th in THETAS[:3]},
                    names=["theta"])
    print(tab.round(4).to_string())
    bbest = choose(b[b.period == "2025"])
    blended = apply(b, bbest)
    print("\nblend: theta chosen on 2025, and its holdout result:", bbest)
    print(blended.groupby(by).apply(summary, include_groups=False).round(4).to_string())
    charts(q, picked, blended)
    print(f"\nwrote {OUT / f'backtest{TAG}.png'}")


if __name__ == "__main__":
    assert fee(0.5) == 0.018 and fee(0.01) == 0.001 and fee(0.99) == 0.001  # 0.175 -> 0.18 per 10
    main()
