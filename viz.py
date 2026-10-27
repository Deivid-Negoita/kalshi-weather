"""Implied-vol surface and market regimes of the temperature brackets, on one page: results/viz.html.

    python viz.py          # ~2 min; reads the candles and data/model/{quotes_all,quotes_boost,preds_boost}.parquet
    python viz.py check    # the sigma inversion on a market that prices a normal exactly

Surface. Every whole event (a tail at each end, no gaps) at each of the HOURS hours before its climate day ends, all
brackets quoted (the last quote within backtest.STALE_H), mids normalised into a distribution. At each bracket edge K
the market's CDF F gives sigma(K) = (K - median) / ppf(F): the normal sd that prices that edge. Flat across F when the
market prices a normal, higher in the wings when it prices fat tails. Strikes sit at F, as FX desks quote vol by delta,
so every hour shares one axis. Cells are medians over snapshots; beside them, realized minus implied P(T <= K), and the
market's sd against the rms error of its mean by hours left (implied vs realized vol).

Regimes. One row per climate day and kind: trailing-week means of what the market shows at the decisions (its sd, the
bracket spread, |model - market| in market sds) and of the settled errors of the market and of the model in market sds,
two days back so the eve decision has them too. Clustered by a Gaussian mixture, lettered by when each regime occurs,
and scored on what the features never see: corr(mu - mkt, y - mkt), the information the model adds, and the
walk-forward blend's one-contract return per $ (backtest.blend at THETA, NO on the favourite skipped).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

import backtest as B
from fetch import DATA
from model import _read, std_offset

HOURS = 40  # a market opens 38-41 h before its climate day ends
F_EDGES = np.array([.02, .06, .12, .2, .3, .4, .5, .6, .7, .8, .88, .94, .98])  # strike axis: the market's P(T <= K)
PERIODS = {"2023": "2024-01-01", "2024": "2025-01-01", "2025": "2026-01-01", "2026 H1": "2026-07-01",
           "Jul 2026 on": "2027-01-01"}  # label: the first day after it; Dec 2022 counts as 2023
MIN_N = 30  # a cell or hour with fewer snapshots is left blank
THETA = 0.08  # the blend's edge threshold, the one the live paper run trades at
K = 3  # regimes; past three, the regime colours stop being told apart under colour blindness
FEATS = ["sd", "spread", "gap", "err_mkt", "err_model"]
OUT = B.OUT / "viz.html"


def js(a, nd):
    """A float array as a JSON-ready list, NaN as null."""
    a = np.round(np.asarray(a, dtype=float), nd)
    return np.where(np.isnan(a), None, a).tolist()


def period(day):
    return pd.cut(day, pd.to_datetime(["2000-01-01", *PERIODS.values()]), labels=list(PERIODS), right=False)


def distribution(q):
    """q: one row per bracket per snapshot (ev, h) with lo/hi edges, value x, bracket count n, settled y and mid.
    Returns (edges, snaps): each interior edge's F, sigma and realized-minus-implied; each snapshot's median, mean, sd."""
    k = ["ev", "h"]
    q = q.sort_values(k + ["hi"])
    g = q.groupby(k).mid
    q = q[(g.transform("size") == q.n) & g.transform("sum").between(.8, 1.25)]  # every bracket quoted, as implied()
    q = q.assign(p=q.mid / q.groupby(k).mid.transform("sum"))
    q["F"] = q.groupby(k).p.cumsum()
    below = q.F - q.p
    # the median, spreading each bracket's mass evenly across its two degrees; a tail holding it has no median
    q["med"] = np.where((below < .5) & (q.F >= .5) & np.isfinite(q.lo + q.hi), q.lo + (.5 - below) / q.p * (q.hi - q.lo),
                        np.nan)
    q["med"] = q.groupby(k).med.transform("max")
    snaps = q.assign(px=q.p * q.x, px2=q.p * q.x ** 2).groupby(k).agg(
        code=("code", "first"), kind=("kind", "first"), day=("day", "first"), y=("y", "first"), med=("med", "first"),
        mean=("px", "sum"), m2=("px2", "sum")).reset_index()
    snaps["sd"] = np.sqrt((snaps.m2 - snaps["mean"] ** 2).clip(lower=0))
    e = q[np.isfinite(q.hi) & q.F.between(F_EDGES[0], F_EDGES[-1]) & ((q.F - .5).abs() > .005) & q.med.notna()]
    e = e.assign(sig=(e.hi - e.med) / norm.ppf(e.F), cal=(e.y <= e.hi) - e.F)
    return e[["code", "kind", "day", "h", "F", "sig", "cal"]], snaps.drop(columns="m2")


def snapshots():
    m = B.markets()
    great, less = m.strike_type == "greater", m.strike_type == "less"
    # strikes are whole degrees, so B72.5 (72-73) spans 71.5..73.5 and T68 ("<68") ends at 67.5
    m["lo"] = np.select([less, great], [-np.inf, m.floor_strike + .5], m.floor_strike - .5)
    m["hi"] = np.select([less, great], [m.cap_strike - .5, np.inf], m.cap_strike + .5)
    off = B.TAU / (1 - B.TAU)
    m["x"] = np.select([great, less], [m.floor_strike + 1 + off, m.cap_strike - 1 - off], (m.floor_strike + m.cap_strike) / 2)
    m = m.sort_values(["event_ticker", "hi"]).reset_index(drop=True)
    g = m.groupby("event_ticker")
    prev = g.hi.shift()
    whole = ((prev.isna() | (m.lo == prev)).groupby(m.event_ticker).transform("all")
             & (g.strike_type.transform("first") == "less") & (g.strike_type.transform("last") == "greater"))
    y = pd.concat([_read([DATA / "kalshi" / "markets" / f"{s}.parquet"], ["event_ticker", "expiration_value"])
                   for s in m.series.unique()])
    y = pd.to_numeric(y.expiration_value, errors="coerce").groupby(y.event_ticker).max()
    m = m[whole].assign(y=lambda x: x.event_ticker.map(y)).dropna(subset=["y"])
    m = m.assign(n=m.groupby("event_ticker").ticker.transform("size"), ev=m.event_ticker.factorize()[0],
                 end=(m.day + pd.to_timedelta(24 - m.tz.map(std_offset), unit="h") - pd.Timestamp(0))
                 // pd.Timedelta(seconds=1))
    cols = ["ticker", "series", "ev", "code", "kind", "day", "lo", "hi", "x", "n", "y"]
    grid = m.loc[m.index.repeat(HOURS + 1), cols + ["end"]].assign(h=np.tile(np.arange(HOURS + 1), len(m)))
    grid["t"] = grid.pop("end") - grid.h * 3600
    q = B.quotes(grid)
    return distribution(q.assign(mid=(q.bid + q.ask) / 2))


def grids(e, s):
    """{"kind|code|period": cells by (hour bin, F bin) and the term structure by hour bin}; "all" pools them."""
    H = np.arange(0, HOURS + 1, 2)
    e = e.assign(hb=e.h // 2 * 2, fb=pd.cut(e.F, F_EDGES, labels=False, include_lowest=True), period=period(e.day),
                 allc="all", allp="all")
    s = s.assign(hb=s.h // 2 * 2, period=period(s.day), v=s.sd ** 2, se=(s.y - s["mean"]) ** 2, allc="all", allp="all")
    out = {}
    for by in (["kind", c, p] for c in ("allc", "code") for p in ("allp", "period")):
        ce = e.groupby(by + ["hb", "fb"], observed=True).agg(sig=("sig", "median"), cal=("cal", "mean"), n=("sig", "size"))
        cs = s.groupby(by + ["hb"], observed=True).agg(v=("v", "mean"), se=("se", "mean"), n=("v", "size"))
        span = s.groupby(by, observed=True).day.agg(["min", "max", "nunique", "size"])
        for key, c in ce.groupby(level=[0, 1, 2], observed=True):
            c = c.droplevel([0, 1, 2])
            c = c[c.n >= MIN_N]
            cell = lambda col, nd: js(c[col].unstack().reindex(index=H, columns=range(len(F_EDGES) - 1)), nd)
            t = cs.loc[key]
            t = t[t.n >= MIN_N].reindex(H)
            sp = span.loc[key]
            out["|".join(map(str, key))] = dict(
                sig=cell("sig", 2), cal=cell("cal", 3), n=cell("n", 0), imp=js(np.sqrt(t.v), 2), real=js(np.sqrt(t.se), 2),
                ns=js(t.n, 0), days=int(sp["nunique"]), snaps=int(sp["size"]), first=f"{sp['min']:%d %b %Y}",
                last=f"{sp['max']:%d %b %Y}")
    return out


def regimes():
    """Per kind: each day's regime, posterior and features; per regime its centroid and score."""
    q = pd.read_parquet(B.QCACHE)
    key = ["code", "kind", "day", "decision"]
    e = pd.read_parquet(DATA / "model" / "preds_boost.parquet", columns=key + ["mu", "y"]).merge(B.implied(q), on=key)
    e = e.dropna(subset=["mu", "y", "mkt", "mkt_sd"])
    two = q[(q.bid > 0) & (q.ask < 1)]
    mid = (two.bid + two.ask) / 2
    spread = (two.ask - two.bid)[mid.between(.05, .95)].groupby([two.kind, two.day]).median().rename("spread")
    daily = e.assign(gap=(e.mu - e.mkt).abs(), err_mkt=(e.y - e.mkt).abs(), err_model=(e.y - e.mu).abs()).groupby(
        ["kind", "day"])[["mkt_sd", "gap", "err_mkt", "err_model"]].median().rename(columns={"mkt_sd": "sd"}).join(spread)
    t = B.trades(B.blend(pd.read_parquet(DATA / "model" / "quotes_boost.parquet")), THETA)
    t = t[~((t.side == "no") & ((t.bid + t.ask) / 2 > .5))]
    out = {}
    for kind, g in daily.groupby(level="kind"):
        r = g.droplevel("kind").asfreq("D").rolling(7, min_periods=3).mean()
        r[["gap", "err_mkt", "err_model"]] = r[["gap", "err_mkt", "err_model"]].div(r.sd, axis=0)
        r = r.shift(1)  # day d's eve decision (18:00 on d-1) has seen the quotes of d-1 and earlier,
        r[["err_mkt", "err_model"]] = r[["err_mkt", "err_model"]].shift(1)  # but only the errors of d-2
        r = r[FEATS].dropna()
        z = (r - r.mean()) / r.std()
        gm = GaussianMixture(K, covariance_type="full", n_init=10, random_state=0).fit(z)
        lab = gm.predict(z)
        order = np.argsort(pd.Series(r.index).groupby(lab).median().reindex(range(K)).values)
        reg, post = pd.Series(np.argsort(order)[lab], r.index), gm.predict_proba(z)[:, order]
        ek, tk = e[e.kind == kind].assign(R=lambda x: x.day.map(reg)), t[t.kind == kind].assign(R=lambda x: x.day.map(reg))
        rows = []
        for j in range(K):
            a, b, days = ek[ek.R == j], tk[tk.R == j], reg.index[reg == j]
            rows.append(dict(name=chr(65 + j), days=len(days), first=f"{days.min():%d %b %Y}", last=f"{days.max():%d %b %Y}",
                             recent=float((reg.iloc[-30:] == j).mean()), **{f: float(r.loc[days, f].mean()) for f in FEATS},
                             info=float(np.corrcoef(a.mu - a.mkt, a.y - a.mkt)[0, 1]) if len(a) > 30 else None,
                             mae=float((a.mu - a.y).abs().mean() - (a.mkt - a.y).abs().mean()) if len(a) else None,
                             trades=len(b), ret=float(b.pnl.sum() / b.price.sum()) if len(b) else None))
        mon = reg.groupby(reg.index.to_period("M")).value_counts(normalize=True).unstack(fill_value=0)
        bic = {k: round(GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit(z).bic(z)) for k in range(2, 7)}
        print(f"\n{kind}: {len(r)} days, {r.index[0]:%d %b %Y} to {r.index[-1]:%d %b %Y}; BIC by number of regimes {bic}")
        print(pd.DataFrame(rows).set_index("name").round(3).to_string())
        out[kind] = dict(names=[x["name"] for x in rows], dates=[f"{d:%Y-%m-%d}" for d in r.index], label=reg.tolist(),
                         p=js(post.max(1), 2), f={f: js(r[f], 3) for f in FEATS}, table=rows,
                         months=[[str(m), *js(v, 3)] for m, v in zip(mon.index, mon.reindex(columns=range(K), fill_value=0).values)],
                         latest=dict(date=f"{r.index[-1]:%d %b %Y}", j=int(reg.iloc[-1]), p=round(float(post[-1].max()), 2)))
    return out


def check():
    """A market pricing a normal(70.3, 2.2) exactly comes back flat at 2.2, with its median at 70.3."""
    hi = np.r_[np.arange(63.5, 78, 2), np.inf]
    lo = np.r_[-np.inf, hi[:-1]]
    q = pd.DataFrame(dict(ev=0, h=0, lo=lo, hi=hi, x=np.clip((lo + hi) / 2, 62, 80), n=len(hi), y=71.0, code="X",
                          kind="high", day=pd.Timestamp("2026-01-01"), mid=np.diff(norm.cdf(np.r_[-np.inf, hi], 70.3, 2.2))))
    e, s = distribution(q)
    assert len(e) == 4 and np.allclose(e.sig, 2.2, rtol=.03), e
    assert abs(s.med.iloc[0] - 70.3) < .05, s
    assert np.allclose(e.cal + e.F, [0, 0, 1, 1]), e  # 71 settles above 67.5 and 69.5, at or below 71.5 and 73.5
    print("check ok: sigma", e.sig.round(3).tolist(), "median", round(s.med.iloc[0], 3))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalshi Weather: Implied Vol and Regimes</title>
<meta name="description" content="Kalshi's daily temperature brackets read as an options book: an implied-volatility surface checked against the settled readings, and three market regimes.">
<meta property="og:title" content="Kalshi's temperature markets, read as an options book">
<meta property="og:description" content="An implied-volatility surface from Kalshi's temperature brackets, checked against the settled readings, and three market regimes.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%232a78d6'/%3E%3Cpath d='M9 23v-5M14 23V12M19 23V9M24 23v-7' stroke='%23fff' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<script>try { const t = localStorage.getItem('kw-theme'); if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t; } catch (e) {}</script>
<style>
:root {
  color-scheme: light;
  --page: #ffffff; --surface: #ffffff; --raised: #ffffff; --ink: #17171a; --ink-2: #4a4a52; --muted: #6c6c75;
  --fill: rgba(118, 118, 128, .12); --fill-2: rgba(118, 118, 128, .2); --rule: rgba(23, 23, 26, .12);
  --grid: #ececf0; --axis: #c8c8cf; --accent: #1460c8; --bar: rgba(255, 255, 255, .72);
  --lift: 0 3px 8px rgba(0, 0, 0, .12), 0 3px 1px rgba(0, 0, 0, .04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --neg: #2a78d6; --mid: #ededf0; --pos: #e34948;
  --ease: cubic-bezier(.23, 1, .32, 1);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #000000; --surface: #000000; --raised: #636366; --ink: #f5f5f7; --ink-2: #c2c2c8; --muted: #8e8e95;
    --fill: rgba(118, 118, 128, .24); --fill-2: rgba(118, 118, 128, .34); --rule: rgba(255, 255, 255, .14);
    --grid: #1d1d20; --axis: #3a3a3f; --accent: #5aa2ff; --bar: rgba(0, 0, 0, .68);
    --lift: 0 3px 8px rgba(0, 0, 0, .32);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #3987e5; --mid: #2c2c2e; --pos: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #000000; --surface: #000000; --raised: #636366; --ink: #f5f5f7; --ink-2: #c2c2c8; --muted: #8e8e95;
  --fill: rgba(118, 118, 128, .24); --fill-2: rgba(118, 118, 128, .34); --rule: rgba(255, 255, 255, .14);
  --grid: #1d1d20; --axis: #3a3a3f; --accent: #5aa2ff; --bar: rgba(0, 0, 0, .68);
  --lift: 0 3px 8px rgba(0, 0, 0, .32);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #3987e5; --mid: #2c2c2e; --pos: #e66767;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body { margin: 0; background: var(--page); color: var(--ink); caret-color: var(--accent);
  font: 16px/1.5 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
::selection { background: color-mix(in srgb, var(--accent) 26%, transparent); }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
h1, h2, h3 { text-wrap: balance; }
p { text-wrap: pretty; }
.wrap { width: 100%; max-width: 1200px; margin: 0 auto; padding-left: 16px; padding-right: 16px; }
@media (min-width: 720px) { .wrap { padding-left: 32px; padding-right: 32px; } }
.vh { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

button { position: relative; height: 32px; padding: 0 12px; border: 0; border-radius: 8px; background: var(--fill); color: var(--ink);
  font: inherit; font-size: 13px; font-weight: 500; line-height: 1; cursor: pointer; }
button::before { content: ""; position: absolute; inset: -4px 0; }  /* a 40px hit area round a 32px button */
button, .seg span, .sel select, .jump a { transition: background-color .15s var(--ease), color .15s var(--ease), transform .12s var(--ease); }
button:active, .seg label:active span { transform: scale(.97); }
button[aria-pressed="true"] { background: var(--ink); color: var(--page); }
.icon { display: inline-grid; place-items: center; width: 40px; height: 40px; padding: 0; border-radius: 50%; background: transparent; color: var(--ink-2); }
.icon::before { content: none; }
.icon svg { width: 20px; height: 20px; }
.icon circle { fill: none; stroke: currentColor; stroke-width: 1.6; }
.icon path { fill: currentColor; }
.ext { flex: none; width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }

.top { display: flex; align-items: center; justify-content: space-between; min-height: 60px; }
.mark { margin: 0; font-size: 15px; font-weight: 600; }
.intro { padding-top: 36px; padding-bottom: 44px; }
@media (min-width: 1000px) {
  .intro { display: grid; grid-template-columns: minmax(0, 6fr) minmax(0, 5fr); column-gap: 64px; align-items: end;
    padding-top: 52px; padding-bottom: 60px; }
}
h1 { max-width: 14em; margin: 0 0 20px; font-size: clamp(36px, 5.2vw, 60px); font-weight: 600; line-height: 1.04; letter-spacing: -.028em; }
@media (min-width: 1000px) { h1 { margin: 0; } }
.dek { max-width: 62ch; margin: 0 0 16px; font-size: 17px; line-height: 1.55; color: var(--ink-2); }
.byline { margin: 0 0 8px; font-size: 14px; color: var(--muted); }
.byline b { color: var(--ink); font-weight: 600; }
.go { display: flex; flex-wrap: wrap; gap: 0 24px; margin: 0; }
.go a { display: inline-flex; align-items: center; gap: 5px; min-height: 40px; font-size: 16px; font-weight: 500; text-decoration: none; }

#pin { height: 1px; margin-bottom: -1px; }
/* clear until content scrolls under it; then a material whose last 16px fade out, a scroll edge in place of a 1px rule */
.bar { position: sticky; top: 0; z-index: 20; margin-bottom: -16px; padding-bottom: 16px; pointer-events: none;  /* the fade passes clicks through */
  -webkit-mask-image: linear-gradient(#000 calc(100% - 16px), transparent); mask-image: linear-gradient(#000 calc(100% - 16px), transparent);
  transition: background-color .2s var(--ease), -webkit-backdrop-filter .2s var(--ease), backdrop-filter .2s var(--ease); }
.bar.stuck { background: var(--bar); -webkit-backdrop-filter: blur(20px) saturate(180%); backdrop-filter: blur(20px) saturate(180%); }
.bar-in { display: flex; align-items: center; gap: 8px; padding-top: 8px; padding-bottom: 8px; pointer-events: auto; }
.seg { position: relative; display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; flex: none; height: 40px; padding: 3px;
  border-radius: 10px; background: var(--fill); }
.seg label { position: relative; z-index: 1; display: flex; cursor: pointer; }
.seg input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: pointer; }
.seg span { display: flex; align-items: center; justify-content: center; width: 100%; padding: 0 12px; border-radius: 7px;
  font-size: 14px; font-weight: 500; color: var(--ink-2); }
.seg input:checked + span { color: var(--ink); }
.seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 0; }
.thumb { position: absolute; top: 3px; bottom: 3px; left: 3px; width: calc(50% - 3px); border-radius: 7px; background: var(--raised);
  box-shadow: var(--lift); transition: transform .26s var(--ease); }
.seg:has(label:nth-of-type(2) input:checked) .thumb { transform: translateX(100%); }
@supports not selector(:has(*)) {
  .thumb { display: none; }
  .seg input:checked + span { background: var(--raised); box-shadow: var(--lift); }
}
.seg.sm { height: 32px; border-radius: 8px; }
.seg.sm span { padding: 0 10px; border-radius: 5px; font-size: 13px; }
.seg.sm .thumb { border-radius: 5px; }
.seg.sm label::before { content: ""; position: absolute; inset: -4px 0; }
.sel { position: relative; display: block; flex: none; }  /* never squeezed past its longest option; a phone scrolls the row */
.sel::before, .sel::after { content: ""; position: absolute; top: 50%; pointer-events: none; background: var(--muted);
  -webkit-mask: center / contain no-repeat; mask: center / contain no-repeat; }
.sel::before { left: 10px; width: 16px; height: 16px; margin-top: -8px; }
.sel::after { right: 9px; width: 10px; height: 14px; margin-top: -7px;
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 14' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 5.2 5 2.2l3 3M2 8.8l3 3 3-3'/%3E%3C/svg%3E");
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 14' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 5.2 5 2.2l3 3M2 8.8l3 3 3-3'/%3E%3C/svg%3E"); }
.sel-s::before {
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.6' stroke-linejoin='round'%3E%3Cpath d='M8 14.4s4.6-4.1 4.6-7.7a4.6 4.6 0 0 0-9.2 0c0 3.6 4.6 7.7 4.6 7.7z'/%3E%3Ccircle cx='8' cy='6.7' r='1.7'/%3E%3C/svg%3E");
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.6' stroke-linejoin='round'%3E%3Cpath d='M8 14.4s4.6-4.1 4.6-7.7a4.6 4.6 0 0 0-9.2 0c0 3.6 4.6 7.7 4.6 7.7z'/%3E%3Ccircle cx='8' cy='6.7' r='1.7'/%3E%3C/svg%3E"); }
.sel-p::before {
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round'%3E%3Crect x='2.3' y='3.3' width='11.4' height='10.4' rx='2.2'/%3E%3Cpath d='M2.3 6.8h11.4M5.4 1.8v2.6M10.6 1.8v2.6'/%3E%3C/svg%3E");
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round'%3E%3Crect x='2.3' y='3.3' width='11.4' height='10.4' rx='2.2'/%3E%3Cpath d='M2.3 6.8h11.4M5.4 1.8v2.6M10.6 1.8v2.6'/%3E%3C/svg%3E"); }
.sel select { -webkit-appearance: none; appearance: none; display: block; width: 100%; height: 40px; margin: 0; padding: 0 26px 0 32px;
  font: inherit; font-size: 15px; font-weight: 500; color: var(--ink); background: var(--fill); border: 0; border-radius: 10px;
  cursor: pointer; text-overflow: ellipsis; }
@media (max-width: 719px) {
  .sel select { padding: 0 23px 0 30px; font-size: 16px; }  /* under 16px, iOS zooms the page on focus */
  .sel::before { left: 9px; }
  .sel::after { right: 8px; }
  .bar .seg span { padding: 0 8px; }
  /* tight enough for a 390px phone; a narrower one scrolls the row, faded at the gutters, rather than clip a code */
  .bar-in { gap: 6px; overflow-x: auto; scrollbar-width: none;
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 16px calc(100% - 16px), transparent);
    mask-image: linear-gradient(90deg, transparent, #000 16px calc(100% - 16px), transparent); }
  .bar-in::-webkit-scrollbar { display: none; }
}
.jump { display: none; margin-left: auto; gap: 2px; }
.jump a { display: flex; align-items: center; min-height: 40px; padding: 0 12px; border-radius: 8px; color: var(--ink-2);
  font-size: 14px; font-weight: 500; text-decoration: none; }
.jump a[aria-current] { color: var(--ink); background: var(--fill); }
@media (min-width: 860px) { .jump { display: flex; } }

.glance { padding-top: 24px; }
.scope { margin: 0 0 14px; font-size: 14px; color: var(--ink-2); }
.scope b { margin-right: 8px; color: var(--ink); font-weight: 600; }
.ledger { display: grid; grid-template-columns: minmax(0, 1fr); column-gap: 48px; margin: 0; border-top: 1px solid var(--rule); }
@media (min-width: 900px) { .ledger { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.lr { display: grid; grid-template-columns: 6.5rem minmax(0, 1fr); column-gap: 16px; align-items: baseline; padding: 16px 0 18px;
  border-bottom: 1px solid var(--rule); }
.lr dt { grid-area: 1 / 2; font-size: 14px; font-weight: 600; line-height: 1.4; }
.lr dd { margin: 0; }
.lr .v { grid-area: 1 / 1 / 3; display: flex; align-items: baseline; gap: 6px; font-size: 32px; font-weight: 500; line-height: 1;
  letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.lr .v small { font-size: 15px; letter-spacing: 0; color: var(--muted); }
.lr .n { grid-area: 2 / 2; font-size: 14px; line-height: 1.45; color: var(--ink-2); }
.tick { display: inline-block; flex: none; width: 4px; height: 16px; border-radius: 2px; background: var(--c); }
.lr .tick { align-self: center; height: 24px; }

.sec { padding-top: 72px; }
@media (min-width: 720px) { .sec { padding-top: 96px; } }
.sec-head { margin-bottom: 28px; }
h2 { margin: 0 0 12px; font-size: clamp(28px, 3.4vw, 40px); font-weight: 600; line-height: 1.1; letter-spacing: -.022em; }
.lede { max-width: 68ch; margin: 0 0 10px; font-size: 17px; line-height: 1.55; color: var(--ink-2); }
.note { margin: 0; font-size: 14px; color: var(--muted); }
.empty { padding: 56px 16px; text-align: center; border-top: 1px solid var(--rule); }
.empty p { margin: 0; color: var(--ink-2); }
.empty b { display: block; margin-bottom: 4px; font-size: 17px; font-weight: 600; color: var(--ink); }

.charts { display: grid; gap: 48px 40px; grid-template-columns: minmax(0, 1fr); }
@media (min-width: 980px) {
  .charts { grid-template-columns: repeat(12, minmax(0, 1fr)); }
  .c-surf, .c-cal { grid-column: span 7; }
  .c-smile, .c-term { grid-column: span 5; }
}
.fig { min-width: 0; margin: 0; padding-top: 14px; border-top: 1px solid var(--rule); }
figcaption { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: 10px 16px; margin-bottom: 12px; }
.ttl { flex: 1 1 240px; min-width: 0; }
.ttl h3 { margin: 0; font-size: 16px; font-weight: 600; line-height: 1.35; }
.cap { margin: 4px 0 0; font-size: 14px; line-height: 1.45; color: var(--muted); }
.tools { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.plot { height: 340px; }
.plot:empty { border-radius: 12px; background: var(--fill); }  /* the frame before plotly draws */
.c-surf .plot { height: min(340px, 75vw); }  /* a phone's box fits its width with no zoom-out */
#reg { height: 190px; }
@media (min-width: 980px) {
  .c-surf .plot, .c-smile .plot { height: 440px; }
  .c-term .plot, .c-cal .plot { height: 390px; }
}
.view { display: none; max-height: 440px; overflow: auto; }
.view, .scroll { scrollbar-width: thin; scrollbar-color: var(--axis) transparent; }
.fig.table .plot { display: none; }
.fig.table .view { display: block; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th { position: sticky; top: 0; z-index: 1; background: var(--page); color: var(--ink-2); font-weight: 600; }
th:first-child, td:first-child { position: sticky; left: 0; text-align: left; background: var(--page); }
th:first-child { z-index: 2; }
td .tick { height: 13px; margin-right: 8px; vertical-align: -2px; }

.cmp { margin-bottom: 48px; }
.cr { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); column-gap: 16px; padding: 12px 0 14px; border-top: 1px solid var(--rule); }
.cr.ch { padding-top: 0; border-top: 0; }
.cr > span { grid-column: 1 / -1; }  /* an empty full-width row: the header keeps its four cells for screen readers */
.ch h3 { display: flex; align-items: center; gap: 8px; margin: 0; font-size: 16px; font-weight: 600; }
.ch p { margin: 2px 0 0; font-size: 13px; color: var(--muted); }
.ch .now { color: var(--ink); font-weight: 600; }
.rl { grid-column: 1 / -1; margin: 0 0 4px; font-size: 13px; color: var(--muted); }
.rv { margin: 0; font-size: 17px; font-weight: 500; font-variant-numeric: tabular-nums; }
.rv small { display: block; margin-top: 2px; font-size: 13px; font-weight: 400; line-height: 1.35; color: var(--muted); }
@media (min-width: 720px) {
  .cr { grid-template-columns: minmax(140px, 1fr) repeat(3, minmax(0, 1.3fr)); align-items: baseline; }
  .cr > span { grid-column: auto; }
  .rl { grid-column: auto; margin: 0; font-size: 14px; color: var(--ink-2); }
}
.more { margin-top: 32px; padding-top: 4px; border-top: 1px solid var(--rule); }
.more summary { display: flex; align-items: center; gap: 8px; min-height: 44px; cursor: pointer; font-size: 15px; font-weight: 600; list-style: none; }
.more summary::-webkit-details-marker { display: none; }
.more summary::before { content: ""; width: 12px; height: 12px; background: var(--muted); transition: transform .2s var(--ease);
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4.5 2.5 8 6l-3.5 3.5'/%3E%3C/svg%3E") center / contain no-repeat;
  mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4.5 2.5 8 6l-3.5 3.5'/%3E%3C/svg%3E") center / contain no-repeat; }
.more[open] summary::before { transform: rotate(90deg); }
.scroll { overflow-x: auto; padding-bottom: 8px; }

.method { display: grid; gap: 24px 40px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); }
.method > div { padding-top: 14px; border-top: 1px solid var(--rule); }
.method h3 { margin: 0 0 6px; font-size: 16px; font-weight: 600; }
.method p { margin: 0; font-size: 15px; line-height: 1.55; color: var(--ink-2); }
.eq { font-style: italic; white-space: nowrap; color: var(--ink); }
code { padding: 2px 6px; border-radius: 6px; background: var(--fill); color: var(--ink); font: 13.5px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.foot { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 24px; margin-top: 96px; padding-top: 20px; padding-bottom: 48px;
  border-top: 1px solid var(--rule); font-size: 13px; color: var(--muted); }
.foot p { margin: 0; }

@media (hover: hover) and (pointer: fine) {
  button:hover, .sel select:hover, .jump a:hover { background-color: var(--fill-2); }
  button[aria-pressed="true"]:hover { background-color: var(--ink); }
  .icon:hover { background-color: var(--fill); color: var(--ink); }
  .seg label:hover input:not(:checked) + span, .jump a:hover { color: var(--ink); }
  .go a:hover { text-decoration: underline; }
}
@media (pointer: coarse) {
  .fine { display: none; }
  #surf-fig:not(.live) #surf { pointer-events: none; }  /* a touch scrolls the page until Rotate is on */
  .seg, .sel select { height: 44px; }
  button, .seg.sm { height: 40px; }
  button { padding: 0 14px; }
}
@media not all and (pointer: coarse) { .coarse { display: none; } }
@media (prefers-reduced-motion: reduce) {
  .thumb, .more summary::before { transition: none; }
  button:active, .seg label:active span { transform: none; }
}
@media (prefers-reduced-transparency: reduce) {
  .bar.stuck { background: var(--page); -webkit-backdrop-filter: none; backdrop-filter: none; }
}
@media (prefers-contrast: more) {
  body { --rule: color-mix(in srgb, var(--ink) 45%, transparent); --muted: var(--ink-2); --fill: color-mix(in srgb, var(--ink) 14%, transparent); }
  .bar { margin-bottom: 0; padding-bottom: 0; -webkit-mask-image: none; mask-image: none; }
  .bar.stuck { background: var(--page); -webkit-backdrop-filter: none; backdrop-filter: none; box-shadow: 0 1px 0 var(--ink); }
}
</style>
</head>
<body>
<header class="top wrap">
  <p class="mark">Kalshi weather</p>
  <button class="icon" id="theme" type="button" aria-label="Switch theme"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z"/></svg></button>
</header>

<main>
<section class="intro wrap">
  <h1>Kalshi's temperature markets, read as an options book</h1>
  <div>
    <p class="dek">Each day Kalshi lists a ladder of brackets on the high and the low at US weather stations, and each bracket pays $1
    if the official reading lands in it. Read together, the prices are the market's distribution for the day. Inverted, they give an
    implied-volatility surface that can be checked against what the thermometer read.</p>
    <p class="byline">By <b>Deivid-Bogdan Negoiță</b><span id="meta"></span></p>
    <p class="go">
      <a href="https://github.com/Deivid-Negoita/kalshi-weather">View the code<svg class="ext" viewBox="0 0 16 16" aria-hidden="true"><path d="M5 11 11 5M6 5h5v5"/></svg></a>
      <a href="https://github.com/Deivid-Negoita/kalshi-weather#readme">Read the write-up<svg class="ext" viewBox="0 0 16 16" aria-hidden="true"><path d="M5 11 11 5M6 5h5v5"/></svg></a>
    </p>
  </div>
</section>

<div id="pin" aria-hidden="true"></div>
<div class="bar" id="controls">
  <div class="wrap bar-in">
    <div class="seg" role="radiogroup" aria-label="Reading">
      <label><input type="radio" name="kind" value="high" checked><span>Highs</span></label>
      <label><input type="radio" name="kind" value="low"><span>Lows</span></label>
      <i class="thumb" aria-hidden="true"></i>
    </div>
    <label class="sel sel-s"><span class="vh">Station</span><select id="code"></select></label>
    <label class="sel sel-p"><span class="vh">Period</span><select id="period"></select></label>
    <nav class="jump" aria-label="Sections"><a href="#surface">Surface</a><a href="#regimes">Regimes</a><a href="#method">Method</a></nav>
  </div>
</div>

<section class="wrap glance" id="glance" aria-label="At a glance">
  <p class="scope" id="sub"></p>
  <dl class="ledger" id="kpis"></dl>
</section>

<section class="wrap sec" id="surface" aria-labelledby="surface-h">
  <header class="sec-head">
    <h2 id="surface-h">The implied-vol surface</h2>
    <p class="lede">σ(K) is the normal sd that prices bracket edge K around the market's own median. A flat surface means the market
    prices a normal; raised wings mean it prices fat tails. Strikes sit at the market's P(T ≤ K), the way FX desks quote vol by
    delta, so every hour shares one axis.</p>
  </header>
  <div class="empty" id="empty" hidden><p><b>No settled events here</b>Pick another period or station.</p></div>
  <div class="charts" id="surfgrid">
    <figure class="fig c-surf" id="surf-fig">
      <figcaption>
        <div class="ttl"><h3>σ by strike and hours left</h3>
          <p class="cap">Median over snapshots, in °F. <span class="fine">Drag to turn it.</span><span class="coarse">Tap Rotate to turn it.</span></p></div>
        <div class="tools">
          <button type="button" class="coarse" id="rotate" aria-pressed="false">Rotate</button>
          <button type="button" id="reset">Reset view</button>
          <div class="seg sm" role="radiogroup" aria-label="View"><label><input type="radio" name="v-surf" value="chart" checked><span>Chart</span></label><label><input type="radio" name="v-surf" value="table"><span>Table</span></label><i class="thumb" aria-hidden="true"></i></div>
        </div>
      </figcaption>
      <div class="plot" id="surf"></div><div class="view" id="surf-t"></div>
    </figure>
    <figure class="fig c-smile">
      <figcaption>
        <div class="ttl"><h3>The smile at three horizons</h3>
          <p class="cap">Slices of the surface in 2-hour bins: the evening decision falls 30–31&nbsp;h out, the morning one 16–17&nbsp;h.</p></div>
        <div class="tools"><div class="seg sm" role="radiogroup" aria-label="View"><label><input type="radio" name="v-smile" value="chart" checked><span>Chart</span></label><label><input type="radio" name="v-smile" value="table"><span>Table</span></label><i class="thumb" aria-hidden="true"></i></div></div>
      </figcaption>
      <div class="plot" id="smile"></div><div class="view" id="smile-t"></div>
    </figure>
    <figure class="fig c-term">
      <figcaption>
        <div class="ttl"><h3>Implied vs realized, by hours left</h3>
          <p class="cap">The market's sd against the rms error of its mean. Implied above realized: the market is too wide.</p></div>
        <div class="tools"><div class="seg sm" role="radiogroup" aria-label="View"><label><input type="radio" name="v-term" value="chart" checked><span>Chart</span></label><label><input type="radio" name="v-term" value="table"><span>Table</span></label><i class="thumb" aria-hidden="true"></i></div></div>
      </figcaption>
      <div class="plot" id="term"></div><div class="view" id="term-t"></div>
    </figure>
    <figure class="fig c-cal">
      <figcaption>
        <div class="ttl"><h3>Realized − implied P(T ≤ K)</h3>
          <p class="cap">How often the reading settled at or below K, less what the market priced. Red: more often than priced; blue: less often.</p></div>
        <div class="tools"><div class="seg sm" role="radiogroup" aria-label="View"><label><input type="radio" name="v-cal" value="chart" checked><span>Chart</span></label><label><input type="radio" name="v-cal" value="table"><span>Table</span></label><i class="thumb" aria-hidden="true"></i></div></div>
      </figcaption>
      <div class="plot" id="cal"></div><div class="view" id="cal-t"></div>
    </figure>
  </div>
</section>

<section class="wrap sec" id="regimes" aria-labelledby="regimes-h">
  <header class="sec-head">
    <h2 id="regimes-h">Three market regimes</h2>
    <p class="lede">Each climate day is a point in five trailing-week features the evening decision can already see: the market's sd,
    the bracket spread, |model − market| in market sds, and the settled errors of the market and of the model two days back. A Gaussian
    mixture splits the days into three regimes, lettered by when they occur. Each is scored on what the features never see: the
    information the model adds, and the walk-forward blend's return per dollar.</p>
    <p class="note">Follows the Highs / Lows switch; station and period apply to the surface only.</p>
  </header>
  <div class="cmp" id="cmp" role="table" aria-label="The three regimes compared"></div>
  <figure class="fig">
    <figcaption>
      <div class="ttl"><h3>Regime by climate day</h3><p class="cap" id="regcap"></p></div>
      <div class="tools"><div class="seg sm" role="radiogroup" aria-label="View"><label><input type="radio" name="v-reg" value="chart" checked><span>Chart</span></label><label><input type="radio" name="v-reg" value="table"><span>Table</span></label><i class="thumb" aria-hidden="true"></i></div></div>
    </figcaption>
    <div class="plot" id="reg"></div><div class="view" id="reg-t"></div>
  </figure>
  <details class="more">
    <summary>Every regime statistic</summary>
    <div class="scroll" id="regtab"></div>
  </details>
</section>

<section class="wrap sec" id="method" aria-labelledby="method-h">
  <header class="sec-head">
    <h2 id="method-h">How it is built</h2>
  </header>
  <div class="method">
    <div><h3>The inversion</h3><p>Bracket mids are normalised into a distribution. At each edge K the market's CDF F gives
      <span class="eq">σ(K) = (K − median) / Φ<sup>−1</sup>(F)</span>, the normal sd that prices that edge.</p></div>
    <div><h3>The strike axis</h3><p>Strikes sit at F, the way FX desks quote vol by delta, so all 40 hours share one axis. Each
      cell is the median over its snapshots.</p></div>
    <div><h3>The data</h3><p>Every whole event (a tail at each end, no gaps, every bracket quoted) at each of the 40 hours before its
      climate day ends. A cell or hour with fewer than <span id="minn">30</span> snapshots is left blank.</p></div>
    <div><h3>The check</h3><p><code>python viz.py check</code> feeds the inversion a market that prices a normal exactly. The surface
      comes back flat at that sd, with the median in place.</p></div>
    <div><h3>The regimes</h3><p>A three-component Gaussian mixture over the standardised features. BIC keeps falling past three
      without an elbow; the page stops at three because past that the regime colours stop being told apart under colour blindness.</p></div>
  </div>
</section>
</main>

<footer class="wrap"><div class="foot">
  <p>Code and the full write-up: <a href="https://github.com/Deivid-Negoita/kalshi-weather">github.com/Deivid-Negoita/kalshi-weather</a>. MIT licence.</p>
  <p id="built"></p>
</div></footer>
<script>/*PLOTLY*/</script>
<script>
const D = /*DATA*/;
const $ = id => document.getElementById(id);
const root = document.documentElement;
const tok = n => getComputedStyle(root).getPropertyValue(n).trim();
const dark = () => getComputedStyle(root).colorScheme === 'dark';
const FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
const CFG = {displaylogo: false, displayModeBar: false, responsive: true, scrollZoom: false};
const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
             '#1c5cab', '#184f95', '#104281', '#0d366b'];
const FEATS = ['sd', 'spread', 'gap', 'err_mkt', 'err_model'];
const SLICES = [[30, '30–31 h left'], [16, '16–17 h left'], [8, '8–9 h left']];
const KIND = {high: 'Highs', low: 'Lows'};
const LO = [0, 1, 2], HI = [9, 10, 11];  // the F bins of the bottom and top tails
const H30 = D.H.indexOf(30);
const st = {kind: 'high', code: 'all', period: 'all'};
const f = (v, d = 2) => v == null ? '–' : v.toFixed(d).replace('-', '−');
const pct = v => v == null ? '–' : (v > 0 ? '+' : '') + (100 * v).toFixed(1).replace('-', '−') + '%';
const cents = v => v == null ? '–' : (100 * v).toFixed(1) + '¢';
const FLAB = D.Fedges.slice(0, -1).map((a, i) => `${a}–${D.Fedges[i + 1]}`);
const HLAB = D.H.map(h => `${h}–${h + 1} h`);
const hl = () => ({bgcolor: tok('--surface'), bordercolor: tok('--axis'), font: {family: FONT, color: tok('--ink')}});

function axis(title, extra) {
  return Object.assign({title: {text: title, font: {color: tok('--ink-2')}}, gridcolor: tok('--grid'), linecolor: tok('--axis'),
    zeroline: false, tickfont: {color: tok('--muted')}, automargin: true, fixedrange: true}, extra);
}
function layout(xa, ya, extra) {  // dragmode false: a touch on a chart scrolls the page instead of zooming it
  return Object.assign({paper_bgcolor: tok('--surface'), plot_bgcolor: tok('--surface'), uirevision: 1, dragmode: false,
    font: {family: FONT, size: 12, color: tok('--ink-2')}, margin: {l: 8, r: 8, t: 8, b: 8}, xaxis: xa, yaxis: ya,
    legend: {orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom'}, hoverlabel: hl()}, extra);
}
const line = (x, y, name, color) => ({x, y, name, type: 'scatter', mode: 'lines+markers', line: {color, width: 2},
  marker: {size: 8, color, line: {color: tok('--surface'), width: 2}}, hovertemplate: `${name}: %{y:.2f} °F<extra></extra>`});
function ends(traces, labels, first) {  // a direct label at each line's right-hand end
  return traces.map((t, i) => {
    const idx = t.y.map((v, k) => v == null ? -1 : k).filter(k => k >= 0);
    if (!idx.length) return null;
    const k = first ? idx[0] : idx[idx.length - 1];
    return {x: t.x[k], y: t.y[k], text: labels[i], showarrow: false, xanchor: 'right', yanchor: 'bottom', yshift: 6,
            font: {size: 11, color: tok('--ink-2')}};
  }).filter(Boolean);
}
function unstack(gd) {  // lines that end close together stack their labels: lift each clear of the one below
  // ponytail: runs per render, so a resize that changes the plot's height keeps the old lift until the next render
  const ya = gd._fullLayout.yaxis, k = ya._length / (ya.range[1] - ya.range[0]), an = gd.layout.annotations, up = {};
  let floor = -Infinity;
  an.map((a, i) => [i, a.y * k]).sort((p, q) => p[1] - q[1]).forEach(([i, px]) => {
    const s = Math.max(0, Math.ceil(floor - px));
    if (s !== an[i].yshift) up[`annotations[${i}].yshift`] = s;
    floor = px + s + 17;  // centre to centre: an 11 px label's box is 14 px, plus a 3 px gap
  });
  return Object.keys(up).length ? Plotly.relayout(gd, up) : gd;
}
function table(el, head, rows) {
  el.innerHTML = '<table><thead><tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
}
const grid = (z, d) => z.map((row, i) => [HLAB[i], ...row.map(v => f(v, d))]);

function tail(S, bins, sign) {  // realized − implied P(T ≤ K) over the bins 16 to 40 h out, weighted by snapshots
  let a = 0, w = 0;
  D.H.forEach((h, i) => h >= 16 && bins.forEach(j => {
    const c = S.cal[i][j], n = S.n[i][j];
    if (c != null && n) { a += c * n; w += n; }
  }));
  return w ? sign * a / w : null;
}
function kpis(S) {
  const R = D.regimes[st.kind], L = R.table[R.latest.j];
  const r = S && S.real[H30] != null && S.imp[H30] != null ? S.real[H30] / S.imp[H30] : null;
  const fr = b => `F ${D.Fedges[b[0]]}–${D.Fedges[b[b.length - 1] + 1]}`;
  const tl = v => v == null ? ['–', 'Too few snapshots.'] :
    [`${Math.abs(100 * v).toFixed(1)}<small>pts</small>`, `${v >= 0 ? 'Less' : 'More'} often than priced, 16–40&nbsp;h out.`];
  const rows = [  // two by two on a wide screen: the headline pair, then the two tails side by side
    ['Realized ÷ implied sd, 30&nbsp;h out', f(r), r == null ? 'Too few snapshots 30&nbsp;h out.' : Math.abs(1 - r) < .005 ?
      'The market prices the day about right.' : `The market prices the day ${Math.round(100 * Math.abs(1 - r))}% too ${r < 1 ? 'wide' : 'narrow'}.`],
    [`Regime on ${R.latest.date}`, `<i class="tick" style="--c:${tok(`--s${R.latest.j + 1}`)}"></i>${L.name}`,
      `Posterior ${f(R.latest.p)}; spread ${cents(L.spread)}, model information ${f(L.info)}.`],
    [`Bottom tail, ${fr(LO)}`, ...tl(S && tail(S, LO, -1))],
    [`Top tail, ${fr(HI)}`, ...tl(S && tail(S, HI, 1))]];
  $('kpis').innerHTML = rows.map(([l, v, n]) =>
    `<div class="lr"><dt>${l}</dt><dd class="v">${v}</dd><dd class="n">${n}</dd></div>`).join('');
}

const bar = () => $('surf').clientWidth >= 520;  // the colorbar repeats the z axis, so a narrow chart drops it
function cam() {  // zoom out when narrow; the colorbar takes ~70 px
  const el = $('surf'), w = el.clientWidth - (bar() ? 70 : 0), k = w > 0 ? Math.max(1, el.clientHeight / w / .92) : 1;
  // the look-at point sits low: the near edges and their titles draw larger than the far ones
  return {eye: {x: 1.55 * k, y: 1.45 * k, z: .75 * k}, center: {x: 0, y: 0, z: -.15}};
}
function surf(S) {
  const cs = (dark() ? [...SEQ].reverse() : SEQ).map((c, i, a) => [i / (a.length - 1), c]);
  const sa = t => ({title: {text: t}, gridcolor: tok('--grid'), linecolor: tok('--axis'), backgroundcolor: tok('--surface'),
    showbackground: true, color: tok('--ink-2'), tickfont: {color: tok('--muted')}});
  Plotly.react('surf', [{type: 'surface', x: D.F, y: D.H, z: S.sig, colorscale: cs, showscale: bar(),
    colorbar: {title: {text: 'σ, °F'}, thickness: 10, len: .7, outlinewidth: 0, tickfont: {color: tok('--muted')}},
    hovertemplate: 'P(T ≤ K) %{x:.2f}, %{y} h left<br>σ %{z:.2f} °F<extra></extra>'}],
    {paper_bgcolor: tok('--surface'), uirevision: 1, font: {family: FONT, size: 11, color: tok('--ink-2')},
     margin: {l: 0, r: 0, t: 0, b: 0}, hoverlabel: hl(),
     scene: {xaxis: sa('P(T ≤ K)'), zaxis: sa(bar() ? 'σ, °F' : ''),  // narrow, the z title clips at the edge; the caption has °F
             // no 40 tick: it sat on the z axis's lowest one
             yaxis: Object.assign(sa('hours left'), {autorange: 'reversed', tickvals: [0, 10, 20, 30]}),
             camera: cam(), aspectratio: {x: 1, y: 1.25, z: .7}}}, CFG);
  table($('surf-t'), ['σ °F, hours left', ...FLAB], grid(S.sig, 2));
}
function cal(S) {
  const m = Math.max(.02, ...S.cal.flat().filter(v => v != null).map(Math.abs));
  Plotly.react('cal', [{type: 'heatmap', x: D.F, y: D.H, z: S.cal, zmin: -m, zmax: m, xgap: 2, ygap: 2, customdata: S.n,
    colorscale: [[0, tok('--neg')], [.5, tok('--mid')], [1, tok('--pos')]], hoverongaps: false,
    colorbar: {title: {text: 'realized − implied'}, thickness: 10, len: .8, outlinewidth: 0, tickformat: '.2f',
               tickfont: {color: tok('--muted')}},  // plotly turns '+.2f' into the invalid '~+.2f'
    hovertemplate: 'P(T ≤ K) %{x:.2f}, %{y} h left<br>realized − implied %{z:.3f}<br>%{customdata} edge-snapshots<extra></extra>'}],
    layout(axis("strike, as the market's P(T ≤ K)"), axis('hours left in the climate day')), CFG);
  table($('cal-t'), ['realized − implied, hours left', ...FLAB], grid(S.cal, 3));
}
function term(S) {
  const tr = [line(D.H, S.imp, 'implied: market sd', tok('--s1')), line(D.H, S.real, 'realized: error of the market mean', tok('--s2'))];
  // past the 0 h end, where nothing is drawn; the lines meet there, so implied goes above and realized below
  const an = ends(tr, ['implied', 'realized'], true).map((a, i) =>
    Object.assign(a, {xanchor: 'left', xshift: 6}, i ? {yanchor: 'top', yshift: -6} : {}));
  Plotly.react('term', tr, layout(axis('hours left in the climate day', {autorange: 'reversed'}), axis('°F', {rangemode: 'tozero'}),
    {hovermode: 'x unified', annotations: an, margin: {l: 8, r: 64, t: 8, b: 8}}), CFG);
  table($('term-t'), ['hours left', 'implied °F', 'realized °F', 'snapshots'],
        D.H.map((h, i) => [HLAB[i], f(S.imp[i]), f(S.real[i]), S.ns[i] == null ? '–' : S.ns[i].toLocaleString()]));
}
function smile(S) {
  const tr = SLICES.map(([h, n], i) => line(D.F, S.sig[D.H.indexOf(h)], n, tok(`--s${i + 1}`)));
  // in the right margin past each line's end, as in the term chart, so a label never sits on the data
  const an = ends(tr, SLICES.map(s => s[1]), false).map(a => Object.assign(a, {xanchor: 'left', xshift: 6, yanchor: 'middle', yshift: 0}));
  Plotly.react('smile', tr, layout(axis("strike, as the market's P(T ≤ K)"), axis('σ, °F'),
    {hovermode: 'x unified', annotations: an, margin: {l: 8, r: 76, t: 8, b: 8}}), CFG).then(unstack);
  table($('smile-t'), ['σ °F', ...FLAB], SLICES.map(([h, n]) => [n, ...S.sig[D.H.indexOf(h)].map(v => f(v))]));
}
const STATS = [  // the compare rows, one per statistic, a column per regime
  ['Days', r => `${r.days.toLocaleString()}<small>${(100 * r.recent).toFixed(0)}% of the last&nbsp;30</small>`],
  ['Bracket spread', r => cents(r.spread)],
  ['Market sd', r => `${f(r.sd)}&nbsp;°F`],
  ['Model information', r => f(r.info)],
  ['MAE vs market', r => `${f(r.mae)}&nbsp;°F`],
  ['Blend backtest', r => r.trades ? `${pct(r.ret)}<small>per $ on ${r.trades.toLocaleString()} trades</small>` : '–<small>no trades</small>']];
function regimes() {
  const R = D.regimes[st.kind], col = j => tok(`--s${j + 1}`), L = R.table[R.latest.j];
  $('cmp').innerHTML = '<div class="cr ch" role="row"><span role="columnheader"></span>' + R.table.map((r, j) =>
    `<div role="columnheader" style="--c:${col(j)}"><h3><i class="tick"></i>Regime ${r.name}</h3>` +
    `<p>${r.first.slice(3)} – ${r.last.slice(3)}</p>${j === R.latest.j ? '<p class="now">Current</p>' : ''}</div>`).join('') + '</div>' +
    STATS.map(([l, v]) => `<div class="cr" role="row"><p class="rl" role="rowheader">${l}</p>` +
      R.table.map(r => `<p class="rv" role="cell">${v(r)}</p>`).join('') + '</div>').join('');
  $('regcap').textContent = `One tick per climate day, in its regime's row. ${R.latest.date}, the last settled day, is in ` +
    `${L.name} with posterior ${f(R.latest.p)}.`;
  const tr = R.names.map((n, j) => {
    const ix = R.label.flatMap((l, i) => l === j ? [i] : []);
    return {x: ix.map(i => R.dates[i]), y: ix.map(() => n), name: n, type: 'scatter', mode: 'markers',
      marker: {symbol: 'line-ns-open', size: 16, color: col(j), line: {width: 2, color: col(j)}},
      customdata: ix.map(i => [R.p[i], ...FEATS.map(k => R.f[k][i])]),
      hovertemplate: `%{x|%d %b %Y}: regime ${n}, posterior %{customdata[0]:.2f}<br>market sd %{customdata[1]:.2f} °F, ` +
        'spread %{customdata[2]:.3f}<br>|model − market| %{customdata[3]:.2f} sd<br>' +
        'errors two days back: market %{customdata[4]:.2f} sd, model %{customdata[5]:.2f} sd<extra></extra>'};
  });
  Plotly.react('reg', tr, layout(axis('', {type: 'date'}), axis('', {type: 'category', categoryorder: 'array',
    categoryarray: [...R.names].reverse()}), {showlegend: false, hovermode: 'closest'}), CFG);
  table($('reg-t'), ['month', ...R.names.map(n => `% of days in ${n}`)],
        R.months.map(m => [m[0], ...m.slice(1).map(v => (100 * v).toFixed(0))]));
  table($('regtab'), ['regime', 'days', 'first', 'last', 'of the last 30', 'market sd °F', 'spread $', '|model − market| sd',
    'market error sd', 'model error sd', 'model information', 'MAE model − market °F', 'blend trades', 'return per $'],
    R.table.map((r, j) => [`<i class="tick" style="--c:${col(j)}"></i>${r.name}`, r.days, r.first, r.last,
      (100 * r.recent).toFixed(0) + '%', f(r.sd), f(r.spread, 3), f(r.gap), f(r.err_mkt), f(r.err_model), f(r.info),
      f(r.mae), r.trades.toLocaleString(), pct(r.ret)]));
}
function render() {
  const S = D.surface[`${st.kind}|${st.code}|${st.period}`];
  $('surfgrid').hidden = !S;
  $('empty').hidden = !!S;
  $('sub').innerHTML = `<b>${KIND[st.kind]} · ${st.code === 'all' ? 'all stations' : st.code} · ` +
    `${st.period === 'all' ? 'all periods' : st.period}</b>` + (S ? `${S.days.toLocaleString()} climate days and ` +
    `${S.snaps.toLocaleString()} snapshots, ${S.first} to ${S.last}.` : 'No settled events.');
  kpis(S);
  if (S) { surf(S); smile(S); term(S); cal(S); }
  regimes();
}
function fill(sel, opts, val) {
  sel.innerHTML = opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('');
  sel.value = opts.some(o => o[0] === val) ? val : opts[0][0];
  return sel.value;
}
const codes = () => [['all', 'All'], ...D.codes[st.kind].map(c => [c, c])];
$('meta').textContent = ` · markets settled to ${D.through.replaceAll(' ', ' ')}`;
$('minn').textContent = D.min_n;
$('built').textContent = `Built ${D.built} by python viz.py.`;
st.kind = document.querySelector('input[name="kind"]:checked').value;  // a reload can restore the other radio
st.code = fill($('code'), codes(), 'all');
st.period = fill($('period'), D.periods.map(p => [p, p === 'all' ? 'All' : p]), 'all');
document.querySelectorAll('input[name="kind"]').forEach(r => r.onchange = () => {
  st.kind = r.value;
  st.code = fill($('code'), codes(), st.code);
  render();
});
$('code').onchange = e => { st.code = e.target.value; render(); };
$('period').onchange = e => { st.period = e.target.value; render(); };
document.querySelectorAll('.seg.sm input').forEach(i => {
  const fig = i.closest('figure');
  i.onchange = () => {
    fig.classList.toggle('table', i.value === 'table');
    if (i.value === 'chart') Plotly.Plots.resize(fig.querySelector('.plot'));
  };
  if (i.checked && i.value === 'table') fig.classList.add('table');  // a reload can restore the Table radio
});
$('reset').onclick = () => Plotly.relayout('surf', {'scene.camera': cam()});
$('rotate').onclick = e => e.currentTarget.setAttribute('aria-pressed', $('surf-fig').classList.toggle('live'));
const themeLabel = () => $('theme').setAttribute('aria-label', `Switch to ${dark() ? 'light' : 'dark'} theme`);
$('theme').onclick = () => {
  const flip = () => {
    root.dataset.theme = dark() ? 'light' : 'dark';
    try { localStorage.setItem('kw-theme', root.dataset.theme); } catch (e) {}
    themeLabel();
    render();
  };
  // a cross-fade, so the page and its charts change brightness together instead of in a jump
  document.startViewTransition ? document.startViewTransition(flip) : flip();
};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { themeLabel(); render(); });
themeLabel();
new IntersectionObserver(([e]) => $('controls').classList.toggle('stuck', e.boundingClientRect.top < 0)).observe($('pin'));
const jumps = document.querySelectorAll('.jump a');  // the section in the middle of the screen is the current one
const spy = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && jumps.forEach(a =>
  a.hash === '#' + e.target.id ? a.setAttribute('aria-current', 'true') : a.removeAttribute('aria-current'))),
  {rootMargin: '-45% 0px -50% 0px'});
['glance', 'surface', 'regimes', 'method'].forEach(id => spy.observe($(id)));
document.addEventListener('touchstart', () => {}, {passive: true});  // lets iOS Safari show :active presses
// plotly measures its text once, so draw after the web font is in (offline, the load resolves empty and the fallback is used)
(document.fonts ? document.fonts.load('12px "IBM Plex Sans"', 'aσ') : Promise.resolve()).catch(() => {}).then(render);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
        sys.exit()
    e, s = snapshots()
    print(f"{s.ev.nunique()} events, {len(s)} snapshots, {len(e)} edges, days {s.day.min():%Y-%m-%d} to {s.day.max():%Y-%m-%d}")
    data = dict(F=js((F_EDGES[:-1] + F_EDGES[1:]) / 2, 3), Fedges=F_EDGES.tolist(), H=list(range(0, HOURS + 1, 2)),
                periods=["all", *PERIODS], codes={k: sorted(g.unique()) for k, g in s.groupby("kind").code}, min_n=MIN_N,
                surface=grids(e, s), regimes=regimes(), through=f"{s.day.max():%d %b %Y}",
                built=pd.Timestamp.now().strftime("%d %b %Y %H:%M"))
    lib = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    OUT.write_text(PAGE.replace("/*DATA*/", json.dumps(data, separators=(",", ":"))).replace("/*PLOTLY*/", lib),
                   encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
