"""Forecast each station's settled daily high and low from NWS MOS alone -- no Kalshi data.

    python model.py      # writes data/model/preds.parquet and prints out-of-sample skill

For every station, kind (high/low), decision time and day D:
  1. take the newest NBM (NBS), GFS MOS and NAM MOS run that was public at the decision time, and widen
     each one's daytime max / overnight min to the calendar day Kalshi settles on (see `edges`);
  2. blend them with a linear regression (plus a seasonal term) fitted on the 3 years before D's month;
  3. turn the forecast into a distribution of the integer the climate report will print: same-season
     training errors, scaled by NBM's own spread (xnd), shrunk by KAPPA, smoothed with a Gaussian kernel.
Each choice was picked on 2021-23 forecasts at six stations and checked on 2024-26 (see README).
The output is P(value <= k) on an integer grid around the forecast, so any Kalshi bracket can be priced.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from fetch import DATA, stations_df

# hours after local-standard midnight of day D: the evening before (18:00) and the morning of (07:00)
DECISIONS = {"eve": -6, "morn": 7}
LATENCY = {"NBS": 2, "GFS": 4, "NAM": 4}  # run time -> public, hours; conservative
MODELS = list(LATENCY)
GRID = 20  # CDF stored for k in round(mu) +- GRID
TEST_FROM = "2020-01-01"  # first walk-forward month; NBM archive starts Nov 2018
WINDOW = 3  # training years
KAPPA = 0.95  # error shrink: unshrunk, the 80% interval covered 84% (too wide)


def std_offset(tz):
    """Hours from UTC to local standard time (January has no DST anywhere in the US)."""
    return ZoneInfo(tz).utcoffset(dt.datetime(2025, 1, 15)).total_seconds() / 3600


def decision_utc(days, tz, decision):
    """Naive-UTC decision timestamps for a DatetimeIndex of climate days."""
    return days + pd.to_timedelta(DECISIONS[decision] - std_offset(tz), unit="h")


def _read(files, columns):
    frames = [pd.read_parquet(f, columns=[c for c in columns if c in pq.read_schema(f).names])
              for f in files if pq.read_metadata(f).num_rows]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def truth(code):
    c = _read(sorted((DATA / "cli" / code).glob("*.parquet")), ["valid", "high", "low"])
    c["day"] = pd.to_datetime(c.valid)
    return c.groupby("day")[["high", "low"]].last().apply(pd.to_numeric, errors="coerce")


def mos(code, model):
    """One row per (run, target day, kind) with the max/min forecast; NBS spread `sd` where it exists."""
    d = _read(sorted((DATA / "mos" / model / code).glob("*.parquet")), ["runtime", "ftime", "txn", "n_x", "xnd"])
    if d.empty:
        return d
    v = pd.Series(np.nan, index=d.index)
    for c in ("txn", "n_x"):  # NBS switched n_x -> txn in 2020
        if c in d:
            v = v.fillna(pd.to_numeric(d[c], errors="coerce").where(lambda s: s.between(-80, 140)))  # 999 = missing
    d = d.assign(v=v, sd=pd.to_numeric(d["xnd"], errors="coerce") if "xnd" in d else np.nan).dropna(subset=["v"])
    d["runtime"], d["ftime"] = pd.to_datetime(d.runtime), pd.to_datetime(d.ftime)
    # max is valid at 00Z after the local day, min at 12Z of it, for every US station
    d["kind"] = np.where(d.ftime.dt.hour == 0, "high", np.where(d.ftime.dt.hour == 12, "low", None))
    d["day"] = d.ftime.dt.normalize() - pd.to_timedelta((d.kind == "high").astype(int), unit="D")
    return d.dropna(subset=["kind"])[["runtime", "day", "kind", "v", "sd"]]


def _lst(utc, tz):
    return pd.to_datetime(utc) + pd.to_timedelta(std_offset(tz), unit="h")


def edges(code, model, tz):
    """The MOS max is 07-19 local and the min 19-08, but Kalshi settles midnight to midnight. Per run and
    day: the warmest 3-hourly forecast outside the daytime window and the coldest after 08:00, so a front
    that sets the day's high just after midnight (or its low late in the evening) is not missed."""
    d = _read(sorted((DATA / "mos" / model / code).glob("*.parquet")), ["runtime", "ftime", "tmp"])
    if d.empty:
        return d
    d = d.assign(tmp=pd.to_numeric(d.tmp, errors="coerce").where(lambda s: s.between(-80, 140)), runtime=pd.to_datetime(d.runtime)).dropna()
    lst = _lst(d.ftime, tz)
    d["day"], h = lst.dt.normalize(), lst.dt.hour
    hi = d[(h < 7) | (h > 19)].groupby(["runtime", "day"]).tmp.max().rename("high")
    lo = d[h > 8].groupby(["runtime", "day"]).tmp.min().rename("low")
    return pd.concat([hi, lo], axis=1).stack().rename("edge").rename_axis(["runtime", "day", "kind"]).reset_index()


def observed(code, tz):
    """ASOS max/min from local-standard midnight to the morning decision: already part of the day."""
    a = _read(sorted((DATA / "asos" / code).glob("*.parquet")), ["valid", "tmpf"])
    lst = _lst(a.valid, tz)
    a = a.assign(day=lst.dt.normalize(), tmpf=pd.to_numeric(a.tmpf, errors="coerce"))
    a = a[(lst - a.day) <= pd.Timedelta(hours=DECISIONS["morn"])].dropna()
    return a.groupby("day").tmpf.agg(high="max", low="min").stack().rename("obs").rename_axis(["day", "kind"])


def features(code, tz):
    """Per (decision, kind, day): the newest public run of each model, plus the truth. Each model's
    max/min is widened to the calendar day with its own hourly forecast and, in the morning, what has
    already been observed. `raw` keeps NBM's unadjusted number for comparison."""
    runs = {m: mos(code, m) for m in MODELS}
    edge = {m: edges(code, m, tz) for m in MODELS}
    obs = observed(code, tz)
    y = truth(code)
    out = []
    for decision in DECISIONS:
        cols = []
        for m, d in runs.items():
            if d.empty:
                continue
            d = d[d.runtime + pd.to_timedelta(LATENCY[m], unit="h") <= decision_utc(d.day, tz, decision)]
            d = d.loc[d.groupby(["day", "kind"]).runtime.idxmax()].merge(edge[m], how="left",
                                                                         on=["runtime", "day", "kind"])
            d = d.set_index(["day", "kind"])
            hi = d.index.get_level_values("kind") == "high"
            if decision == "morn":
                o = obs.reindex(d.index).values
                d["edge"] = np.where(hi, np.fmax(d.edge, o), np.fmin(d.edge, o))
            v = np.where(hi, np.fmax(d.v, d.edge), np.fmin(d.v, d.edge))
            cols.append(pd.Series(v, d.index, name=m))
            if m == "NBS":
                cols += [d.sd.rename("sd"), d.v.rename("raw")]
        f = pd.concat(cols, axis=1).reset_index()
        f["y"] = [y.at[r.day, r.kind] if r.day in y.index else np.nan for r in f.itertuples()]
        out.append(f.assign(decision=decision))
    return pd.concat(out, ignore_index=True)


def _design(f):
    fc = f.reindex(columns=MODELS)
    fc = fc.apply(lambda col: col.fillna(fc.mean(axis=1)))  # a missing model borrows the others' mean
    doy = 2 * np.pi * f.day.dt.dayofyear / 365.25
    return np.column_stack([np.ones(len(f)), fc.values, np.sin(doy), np.cos(doy)])


def _kde_cdf(resid, offsets):
    """Smoothed empirical CDF of the residuals, evaluated at `offsets` (Silverman bandwidth)."""
    h = 1.06 * resid.std() * len(resid) ** -0.2
    return norm.cdf((offsets[:, None] - resid[None, :]) / h).mean(axis=1)


def walk_forward(f):
    """Refit each month on everything before it; predict that month. Returns mu and the CDF grid."""
    if "NBS" not in f:  # DJT: no MOS archive at all
        return pd.DataFrame()
    f = f.dropna(subset=["NBS"]).sort_values("day").reset_index(drop=True)
    X, y = _design(f), f.y.values
    s = np.sqrt(1 + f.sd.fillna(f.sd.median()).fillna(2).values ** 2)  # error scale from NBM's spread
    offs = np.arange(-15, 15.001, 0.02)  # standardized error units
    rows = []
    for month in pd.date_range(TEST_FROM, f.day.max(), freq="MS"):
        tr = ((f.day < month) & (f.day >= month - pd.DateOffset(years=WINDOW))).values & ~np.isnan(y)
        te = ((f.day >= month) & (f.day < month + pd.offsets.MonthBegin())).values
        if tr.sum() < 365 or not te.any():
            continue
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        resid = KAPPA * (y[tr] - X[tr] @ beta) / s[tr]
        # errors from the same season: +-45 days of this month's middle, any year
        gap = np.abs(f.day[tr].dt.dayofyear.values - (month + pd.Timedelta(days=14)).dayofyear)
        pool = resid[np.minimum(gap, 365 - gap) <= 45]
        cdf = _kde_cdf(pool if len(pool) >= 150 else resid, offs)
        mu = X[te] @ beta
        k0 = np.round(mu) - GRID
        ks = k0[:, None] + np.arange(2 * GRID + 1)[None, :]
        # P(V <= k) for an integer report V = P(continuous value < k + 0.5)
        grid = np.interp((ks + 0.5 - mu[:, None]) / s[te][:, None], offs, cdf)
        rows.append(pd.DataFrame({"day": f.day[te].values, "mu": mu, "k0": k0,
                                  "cdf": list(grid.astype("float32")), "y": y[te]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prob(cdf, k0, k):
    """P(V <= k) from a stored grid; outside it the answer is 0 or 1."""
    i = int(k - k0)
    return 0.0 if i < 0 else 1.0 if i >= len(cdf) else float(cdf[i])


def skill(p):
    """Out-of-sample checks: MAE of the blend vs raw NBM, log score and 80% interval coverage."""
    p = p.dropna(subset=["y"])
    pmf = [prob(c, k, y) - prob(c, k, y - 1) for c, k, y in zip(p.cdf, p.k0, p.y)]
    lo = [prob(c, k, y - 1) for c, k, y in zip(p.cdf, p.k0, p.y)]
    pit = np.array(lo) + np.random.default_rng(0).random(len(p)) * np.array(pmf)  # randomized PIT
    return pd.Series({"days": len(p), "MAE blend": (p.mu - p.y).abs().mean(),
                      "MAE raw NBM": (p.NBS - p.y).abs().mean(),
                      "log score": -np.log(np.clip(pmf, 1e-6, None)).mean(),
                      "80% cover": ((pit > 0.1) & (pit < 0.9)).mean()})


def predict(stations):
    out = []
    for st in stations.itertuples():
        f = features(st.code, st.tz)
        for (decision, kind), g in f.groupby(["decision", "kind"]):
            p = walk_forward(g)
            if len(p):
                raw = g.set_index("day")["raw"]
                out.append(p.assign(code=st.code, decision=decision, kind=kind, NBS=p.day.map(raw)))
        print(f"  {st.code}: {len(f):,} feature rows", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    preds = predict(stations_df())
    (DATA / "model").mkdir(exist_ok=True)
    preds.to_parquet(DATA / "model" / "preds.parquet")
    print("\nout-of-sample skill, all stations, %s onward" % TEST_FROM)
    print(preds.groupby(["decision", "kind"]).apply(skill, include_groups=False).round(3).to_string())


if __name__ == "__main__":
    main()
