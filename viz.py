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
<title>Kalshi Vol and Regimes</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11, 11, 11, .10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --neg: #2a78d6; --mid: #f0efec; --pos: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255, 255, 255, .10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #3987e5; --mid: #383835; --pos: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255, 255, 255, .10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #3987e5; --mid: #383835; --pos: #e66767;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 24px 16px 56px; }
h1 { font-size: 22px; line-height: 1.25; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 40px 0 4px; }
h3 { font-size: 14px; margin: 20px 0 6px; }
p { margin: 0 0 8px; max-width: 80ch; color: var(--ink-2); }
.meta { color: var(--muted); font-size: 12px; }
.filters { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 10px; }
.filters label { display: grid; gap: 2px; font-size: 12px; color: var(--ink-2); }
select { font: inherit; font-size: 13px; padding: 5px 8px; border-radius: 6px; border: 1px solid var(--axis);
  background: var(--surface); color: var(--ink); min-width: 140px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 540px), 1fr)); gap: 16px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px; min-width: 0; }
figcaption { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
figcaption b { font-size: 14px; font-weight: 600; color: var(--ink); }
.cap { font-size: 12px; margin: 2px 0 6px; }
.plot { height: 380px; }
.tall .plot { height: 460px; }
.view { display: none; overflow: auto; max-height: 460px; }
figure.table .plot { display: none; }
figure.table .view { display: block; }
button { font: inherit; font-size: 12px; color: var(--ink-2); background: none; border: 1px solid var(--axis);
  border-radius: 6px; padding: 2px 10px; cursor: pointer; flex: none; }
button:focus-visible, select:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
th:first-child, td:first-child { text-align: left; }
.chip { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.callout { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 10px 14px; margin: 12px 0; color: var(--ink); }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 4px 4px 8px; overflow-x: auto; }
</style>
</head>
<body>
<main>
<h1>Kalshi temperature brackets: implied vol and market regimes</h1>
<p class="meta" id="meta"></p>
<div class="filters">
  <label>Kind <select id="kind"><option value="high">Daily high</option><option value="low">Daily low</option></select></label>
</div>

<h2>Implied-vol surface</h2>
<p>Each event's brackets at every hour of its last 40, mids normalised into the market's distribution. σ(K) is the
normal sd that prices bracket edge K around the market's own median: a flat surface means the market prices a normal,
higher wings mean it prices fat tails. Strikes sit at the market's P(T ≤ K), the way FX desks quote vol by delta, so
every hour shares one axis.</p>
<div class="filters">
  <label>Station <select id="code"></select></label>
  <label>Period <select id="period"></select></label>
</div>
<p class="meta" id="sub"></p>
<div class="grid" id="surfgrid">
  <figure class="tall"><figcaption><b>σ by strike and hours left</b><button data-t="surf">Table</button></figcaption>
    <p class="cap">Median over snapshots. Drag to turn it.</p>
    <div class="plot" id="surf"></div><div class="view" id="surf-t"></div></figure>
  <figure class="tall"><figcaption><b>Realized − implied P(T ≤ K)</b><button data-t="cal">Table</button></figcaption>
    <p class="cap">How often the value settled at or below K, less what the market priced. Red: settled there more often than priced.</p>
    <div class="plot" id="cal"></div><div class="view" id="cal-t"></div></figure>
  <figure><figcaption><b>Implied vs realized, by hours left</b><button data-t="term">Table</button></figcaption>
    <p class="cap">The market's sd (rms) against the rms error of its mean. Implied above realized: the market is too wide.</p>
    <div class="plot" id="term"></div><div class="view" id="term-t"></div></figure>
  <figure><figcaption><b>Smile at three times</b><button data-t="smile">Table</button></figcaption>
    <p class="cap">Slices of the surface in 2-hour bins: the eve decision falls in 30–31 h left, the morning one in 16–17 h.</p>
    <div class="plot" id="smile"></div><div class="view" id="smile-t"></div></figure>
</div>

<h2>Market regimes</h2>
<p>Each climate day is a point in five trailing-week features known before that day's eve decision: the market's sd,
the bracket spread, |model − market| in market sds, and the settled errors of the market and of the model in market
sds, two days back. A Gaussian mixture splits the days into three regimes, lettered by when they occur. Each is scored
on what the features never see: the information the model adds, corr(mu − mkt, y − mkt), and the walk-forward
blend's one-contract return per $ (THETA, NO on the favourite skipped).</p>
<div class="callout" id="latest"></div>
<figure><figcaption><b>Regime by climate day</b><button data-t="reg">Table</button></figcaption>
  <p class="cap">One tick per day in its regime's row. The table view gives each month's share of days.</p>
  <div class="plot" id="reg" style="height: 200px"></div><div class="view" id="reg-t"></div></figure>
<h3>What each regime looks like, and what the strategy did in it</h3>
<div class="card" id="regtab"></div>
</main>
<script>/*PLOTLY*/</script>
<script>
const D = /*DATA*/;
const $ = id => document.getElementById(id);
const tok = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const CFG = {displaylogo: false, responsive: true};
const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
             '#1c5cab', '#184f95', '#104281', '#0d366b'];
const FEATS = ['sd', 'spread', 'gap', 'err_mkt', 'err_model'];
const SLICES = [[30, '30–31 h left'], [16, '16–17 h left'], [8, '8–9 h left']];
const st = {kind: 'high', code: 'all', period: 'all'};
const f = (v, d = 2) => v == null ? '–' : v.toFixed(d);
const pct = v => v == null ? '–' : (v > 0 ? '+' : '') + (100 * v).toFixed(1) + '%';
const FLAB = D.Fedges.slice(0, -1).map((a, i) => `${a}–${D.Fedges[i + 1]}`);
const HLAB = D.H.map(h => `${h}–${h + 1} h`);

function axis(title, extra) {
  return Object.assign({title: {text: title, font: {color: tok('--ink-2')}}, gridcolor: tok('--grid'),
    linecolor: tok('--axis'), zeroline: false, tickfont: {color: tok('--muted')}, automargin: true}, extra);
}
function layout(xa, ya, extra) {
  return Object.assign({paper_bgcolor: tok('--surface'), plot_bgcolor: tok('--surface'), uirevision: 1,
    font: {family: FONT, size: 12, color: tok('--ink-2')}, margin: {l: 8, r: 8, t: 8, b: 8}, xaxis: xa, yaxis: ya,
    legend: {orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom'},
    hoverlabel: {bgcolor: tok('--surface'), bordercolor: tok('--axis'), font: {family: FONT, color: tok('--ink')}}}, extra);
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
function table(el, head, rows) {
  el.innerHTML = '<table><thead><tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
}
const grid = (z, d) => z.map((row, i) => [HLAB[i], ...row.map(v => f(v, d))]);

function surf(S) {
  const cs = (tok('--surface') === '#1a1a19' ? [...SEQ].reverse() : SEQ).map((c, i, a) => [i / (a.length - 1), c]);
  const sa = t => ({title: {text: t}, gridcolor: tok('--grid'), linecolor: tok('--axis'), backgroundcolor: tok('--surface'),
    showbackground: true, color: tok('--ink-2'), tickfont: {color: tok('--muted')}});
  const el = $('surf'), k = el.clientWidth ? Math.max(1, el.clientHeight / (el.clientWidth - 70) / .92) : 1;  // zoom out when narrow; the colorbar takes ~70 px
  Plotly.react('surf', [{type: 'surface', x: D.F, y: D.H, z: S.sig, colorscale: cs,
    colorbar: {title: {text: 'σ, °F'}, thickness: 10, len: .7, outlinewidth: 0, tickfont: {color: tok('--muted')}},
    hovertemplate: 'P(T ≤ K) %{x:.2f}, %{y} h left<br>σ %{z:.2f} °F<extra></extra>'}],
    {paper_bgcolor: tok('--surface'), uirevision: 1, font: {family: FONT, size: 11, color: tok('--ink-2')},
     margin: {l: 0, r: 0, t: 0, b: 0},
     hoverlabel: {bgcolor: tok('--surface'), bordercolor: tok('--axis'), font: {family: FONT, color: tok('--ink')}},
     scene: {xaxis: sa('P(T ≤ K)'), zaxis: sa('σ, °F'),  // no 40 tick: it sat on the z axis's lowest one
             yaxis: Object.assign(sa('hours left'), {autorange: 'reversed', tickvals: [0, 10, 20, 30]}),
             camera: {eye: {x: 1.55 * k, y: 1.45 * k, z: .75 * k}}, aspectratio: {x: 1, y: 1.25, z: .7}}}, CFG);
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
  Plotly.react('smile', tr, layout(axis("strike, as the market's P(T ≤ K)"), axis('σ, °F'),
    {hovermode: 'x unified', annotations: ends(tr, SLICES.map(s => s[1]), false)}), CFG);
  table($('smile-t'), ['σ °F', ...FLAB], SLICES.map(([h, n]) => [n, ...S.sig[D.H.indexOf(h)].map(v => f(v))]));
}
function regimes() {
  const R = D.regimes[st.kind], col = j => tok(`--s${j + 1}`), L = R.table[R.latest.j];
  $('latest').innerHTML = `<b>${R.latest.date}</b>, the last day with settled data: ${st.kind}s are in regime ` +
    `<span class="chip" style="background:${col(R.latest.j)}"></span><b>${L.name}</b> (posterior ${f(R.latest.p)}). ` +
    `Across ${L.name}'s ${L.days} days the model's information was ${f(L.info)} and the blend made ${pct(L.ret)} per $ on ${L.trades.toLocaleString()} trades.`;
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
    R.table.map((r, j) => [`<span class="chip" style="background:${col(j)}"></span>${r.name}`, r.days, r.first, r.last,
      (100 * r.recent).toFixed(0) + '%', f(r.sd), f(r.spread, 3), f(r.gap), f(r.err_mkt), f(r.err_model), f(r.info),
      f(r.mae), r.trades.toLocaleString(), pct(r.ret)]));
}
function render() {
  const S = D.surface[`${st.kind}|${st.code}|${st.period}`];
  $('surfgrid').hidden = !S;
  $('sub').textContent = S ? `${S.days} climate days, ${S.first} to ${S.last}; ${S.snaps.toLocaleString()} snapshots. ` +
    `Cells with fewer than ${D.min_n} are blank.` : 'No settled events for this station and period.';
  if (S) { surf(S); cal(S); term(S); smile(S); }
  regimes();
}
function fill(sel, opts, val) {
  sel.innerHTML = opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('');
  sel.value = opts.some(o => o[0] === val) ? val : opts[0][0];
  return sel.value;
}
const codes = () => [['all', 'All stations'], ...D.codes[st.kind].map(c => [c, c])];
$('meta').textContent = `Settled markets to ${D.through}; built ${D.built} by python viz.py.`;
st.code = fill($('code'), codes(), 'all');
st.period = fill($('period'), D.periods.map(p => [p, p === 'all' ? 'All periods' : p]), 'all');
$('kind').onchange = e => { st.kind = e.target.value; st.code = fill($('code'), codes(), st.code); render(); };
$('code').onchange = e => { st.code = e.target.value; render(); };
$('period').onchange = e => { st.period = e.target.value; render(); };
document.querySelectorAll('button[data-t]').forEach(b => b.onclick = () => {
  const on = b.closest('figure').classList.toggle('table');
  b.textContent = on ? 'Chart' : 'Table';
  if (!on) Plotly.Plots.resize($(b.dataset.t));
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
render();
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
