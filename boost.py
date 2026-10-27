"""Second stage: correct model.py's forecast with every other weather source collected, and the market's own guess.

    python boost.py      # needs data/model/preds.parquet; writes data/model/preds_boost.parquet

model.py only reads the NWS MOS. This adds, per station, decision and day:
  - ECMWF IFS, ICON, GEM and GFS from Open-Meteo: the calendar-day max/min of their hourly forecast;
  - the latest ASOS observation at decision time (temperature, dewpoint, wind, cloud), the 24-hour
    change, and the max/min of the last 24 hours;
  - the station's recent forecast bias (7 and 30 days, ending two days back so the truth was public);
  - the weather regime from the latest NBM published LATENCY hours before the decision: the climate day's
    cloud cover, wind vector and speed, rain chance and amount, dewpoint, NBM's spread and the temperature
    change across the day (a front), plus the observed wind vector and rain in the last 6 hours.
    `python regime_test.py` measured them: MAE -4 to -15%, 2026 blend 6.0 -> 9.6 cents per dollar;
  - GFS LAMP (the latest run an hour before the decision), HRRR (morning only), and the mean and spread of
    the temperature implied by the event's bracket mids at the decision (backtest.implied).
    `python feature_test.py` measured them: MAE -2 to -4%, 2026 blend 8.1 -> 10.8 cents per dollar.
One LightGBM per (decision, kind), pooled over all stations, learns the error of model.py's mu.
It is refit every month on all earlier data. The predictive distribution is model.py's, moved to the
new mu and narrowed by how much smaller the new out-of-sample errors are (by their IQR).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest import market_history
from fetch import DATA, stations_df
from model import DECISIONS, GRID, LATENCY, _read, skill, std_offset

OM = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]
# previous_dayN is the run issued >= 24N hours before each hour: day2 is always public by 18:00 the
# evening before; day1 is public by 07:00 on the day, and by 18:00 the evening before for the hours whose
# run was out by then (the evening takes those: research/2026-10-21/eve_honest.py, eve MAE -1.5 to -2.5%, PnL +18%).
LEAD = {"eve": 2, "morn": 1}
# run init -> public, hours: Open-Meteo's meta.json on 21 Oct 2026, rounded up; GEM's is stale, so the slowest
DELAY = {"gfs_seamless": 9, "ecmwf_ifs025": 8, "icon_seamless": 4, "gem_seamless": 9}
SKY = {"CLR": 0, "SKC": 0, "NSC": 0, "NCD": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4}
START = "2021-01-01"  # first month predicted; earlier preds are training only
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def day1_public(t, dec, model):
    """Was the run previous_day1 takes hour t from (the 6-hourly cycle >= 24 h before it) out by dec? All UTC."""
    return (t - pd.Timedelta(hours=24)).dt.floor("6h") + pd.Timedelta(hours=DELAY[model]) <= dec


def openmeteo(code, tz):
    d = pd.read_parquet(DATA / "openmeteo" / f"{code}.parquet")
    t = pd.to_datetime(d.time)
    day = (t + pd.to_timedelta(std_offset(tz), unit="h")).dt.normalize()
    rows = []
    for decision, n in LEAD.items():
        h = pd.DataFrame({m: d.get(f"temperature_2m_previous_day{n}_{m}") for m in OM}).assign(day=day)
        if n == 2:
            dec = day + pd.Timedelta(hours=DECISIONS[decision] - std_offset(tz))
            for m in OM:
                h[m] = d[f"temperature_2m_previous_day1_{m}"].where(day1_public(t, dec, m)).fillna(h[m])
        g = h.groupby("day")
        full = g.count() >= 20  # most of the calendar day forecast
        for kind, agg in (("high", g.max()), ("low", g.min())):
            rows.append(agg.where(full).add_prefix("om_").reset_index().assign(decision=decision, kind=kind))
    return pd.concat(rows, ignore_index=True)


def asos(code):
    a = _read(sorted((DATA / "asos" / code).glob("*.parquet")), ["valid", "tmpf", "dwpf", "sknt", "skyc1"])
    a = a.assign(t=pd.to_datetime(a.valid).astype("datetime64[ns]"), sky=a.skyc1.str.strip().map(SKY))
    a[["tmpf", "dwpf", "sknt"]] = a[["tmpf", "dwpf", "sknt"]].apply(pd.to_numeric, errors="coerce")
    a = a.dropna(subset=["tmpf"]).sort_values("t")[["t", "tmpf", "dwpf", "sknt", "sky"]]
    # max/min of the 24 hours up to each observation, so "the last day" is one asof lookup
    r = a.set_index("t").tmpf.rolling("24h")
    return a.assign(max24=r.max().values, min24=r.min().values)


REGIME = ["f_sky", "f_wsp", "f_u", "f_v", "f_pop", "f_qpf", "f_dpt_minus_mu", "f_tsd", "f_trend", "o_u", "o_v", "o_wet6"]


def wind_uv(deg, kt):
    rad = np.deg2rad(deg)
    return -kt * np.sin(rad), -kt * np.cos(rad)  # the vector the wind blows towards, knots: v > 0 is from the south


def nbm_regime(code, tz):
    """Per NBM run and climate day: what that run forecast for the day's weather."""
    num = ["tmp", "sky", "wdr", "wsp", "p06", "q06", "dpt", "tsd"]
    d = _read(sorted((DATA / "mos" / "NBS" / code).glob("*.parquet")), ["runtime", "ftime"] + num)
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d["runtime"], d["ftime"] = pd.to_datetime(d.runtime), pd.to_datetime(d.ftime)
    d["day"] = (d.ftime + pd.Timedelta(hours=std_offset(tz))).dt.normalize()
    d["u"], d["v"] = wind_uv(d.wdr, d.wsp)
    a = d.groupby(["runtime", "day"]).agg(
        n=("tmp", "size"), f_sky=("sky", "mean"), f_wsp=("wsp", "mean"), f_u=("u", "mean"), f_v=("v", "mean"),
        f_pop=("p06", "max"), f_qpf=("q06", "sum"), f_dpt=("dpt", "mean"), f_tsd=("tsd", "max"),
        f_trend=("tmp", lambda s: s.iloc[-1] - s.iloc[0]))
    a = a[a.n >= 6].drop(columns="n").reset_index().rename(columns={"runtime": "pub"})
    return a.assign(pub=a.pub.astype("datetime64[ns]")).sort_values("pub")


LAMP_LAT = 1  # LAMP is out ~30 min after its cycle hour
LAMP = ["lamp_hi_minus_mu", "lamp_lo_minus_mu", "lamp_n", "hrrr_hi_minus_mu", "hrrr_lo_minus_mu"]
EXTRA = LAMP + ["mkt_minus_mu", "mkt_sd"]  # beyond REGIME; feature_test.py sets it per variant


def lamp(code, tz):
    """Per LAMP run and climate day: max/min of its hourly temperature over the hours of the day it reaches.
    IEM keeps the 00/06/12/18Z runs; each goes 25 hours out, so an evening run covers only part of tomorrow."""
    d = _read(sorted((DATA / "mos" / "LAV" / code).glob("*.parquet")), ["runtime", "ftime", "tmp"])
    if d.empty:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in ("pub", "day") else float)
                             for c in ["pub", "day", "lamp_hi", "lamp_lo", "lamp_n"]})
    d["tmp"] = pd.to_numeric(d.tmp, errors="coerce")
    d["runtime"], d["ftime"] = pd.to_datetime(d.runtime), pd.to_datetime(d.ftime)
    d["day"] = (d.ftime + pd.Timedelta(hours=std_offset(tz))).dt.normalize()
    a = d.dropna(subset=["tmp"]).groupby(["runtime", "day"]).tmp.agg(lamp_hi="max", lamp_lo="min", lamp_n="size")
    a = a[a.lamp_n >= 3].reset_index().rename(columns={"runtime": "pub"})
    return a.assign(pub=a.pub.astype("datetime64[ns]")).sort_values("pub")


def hrrr(code, tz):
    """HRRR's calendar-day max/min from runs >= 24 h before each hour: public by the morning decision only
    (its 48 h horizon leaves previous_day2 empty, and day1 runs are too late for the evening)."""
    f = DATA / "hrrr" / f"{code}.parquet"
    if not f.exists():
        return pd.DataFrame({"day": pd.Series(dtype="datetime64[ns]"), "decision": pd.Series(dtype=object),
                             "hrrr_hi": pd.Series(dtype=float), "hrrr_lo": pd.Series(dtype=float)})
    d = pd.read_parquet(f)
    d["day"] = (pd.to_datetime(d.time) + pd.to_timedelta(std_offset(tz), unit="h")).dt.normalize()
    g = d.groupby("day").temperature_2m_previous_day1
    full = g.count() >= 20
    return pd.DataFrame({"hrrr_hi": g.max().where(full), "hrrr_lo": g.min().where(full)}).reset_index().assign(decision="morn")


def asos_regime(code):
    a = _read(sorted((DATA / "asos" / code).glob("*.parquet")), ["valid", "drct", "sknt", "p01i"])
    a = a.assign(t=pd.to_datetime(a.valid).astype("datetime64[ns]"))
    a[["drct", "sknt", "p01i"]] = a[["drct", "sknt", "p01i"]].apply(pd.to_numeric, errors="coerce")
    a = a.sort_values("t")
    a["o_u"], a["o_v"] = wind_uv(a.drct, a.sknt)
    a["o_wet6"] = (a.set_index("t").p01i.fillna(0) > 0).astype(float).rolling("6h").max().values
    return a.dropna(subset=["o_u"])[["t", "o_u", "o_v", "o_wet6"]]


def add_lamp(p, code, tz):
    """LAMP from the latest run public LAMP_LAT hours before the decision (within 12 h), HRRR at the morning."""
    p = p.assign(pub=(p.t - pd.Timedelta(hours=LAMP_LAT)).astype("datetime64[ns]")).sort_values("pub")
    p = pd.merge_asof(p, lamp(code, tz), on="pub", by="day", tolerance=pd.Timedelta("12h")).drop(columns="pub")
    p = p.merge(hrrr(code, tz), on=["day", "decision"], how="left")
    for k in ("hi", "lo"):
        p[f"lamp_{k}_minus_mu"], p[f"hrrr_{k}_minus_mu"] = p[f"lamp_{k}"] - p.mu, p[f"hrrr_{k}"] - p.mu
    return p.drop(columns=["lamp_hi", "lamp_lo", "hrrr_hi", "hrrr_lo"])


def features(p, code, tz, mkt=None):
    """p: model.py's predictions for one station. Adds every non-MOS feature known at decision time.
    mkt: backtest.implied rows to use (live passes today's on top of the history); default the history."""
    p = p.copy()
    p["t"] = (p.day + pd.to_timedelta(p.decision.map(DECISIONS) - std_offset(tz), unit="h")).astype("datetime64[ns]")
    a = asos(code)
    p = pd.merge_asof(p.sort_values("t"), a, on="t", tolerance=pd.Timedelta("3h"))
    p = pd.merge_asof(p, asos_regime(code), on="t", tolerance=pd.Timedelta("3h"))
    # the NBM run model.py would use: published LATENCY hours before the decision
    p["pub"] = p.t - pd.Timedelta(hours=LATENCY["NBS"])
    p = pd.merge_asof(p.sort_values("pub"), nbm_regime(code, tz), on="pub", by="day", tolerance=pd.Timedelta("36h"))
    p["f_dpt_minus_mu"] = p.f_dpt - p.mu
    p = p.drop(columns=["pub", "f_dpt"])
    if EXTRA:
        p = add_lamp(p, code, tz).merge(market_history() if mkt is None else mkt,
                                        on=["code", "kind", "day", "decision"], how="left")
        p["mkt_minus_mu"] = p.mkt - p.mu
        p = p.drop(columns="mkt")
    p = p.sort_values("t")
    ago = a[["t", "tmpf"]].rename(columns={"tmpf": "tmpf_24h_ago"})
    p = pd.merge_asof(p.assign(t=p.t - pd.Timedelta("24h")), ago, on="t", tolerance=pd.Timedelta("3h"))
    p["dt24"] = p.tmpf - p.tmpf_24h_ago
    p = p.merge(openmeteo(code, tz), on=["day", "decision", "kind"], how="left")
    # recent bias of mu, using only days whose report was out (two days back)
    p = p.sort_values("day")
    for (dec, kind), g in p.groupby(["decision", "kind"]):
        e = (g.y - g.mu).set_axis(g.day).asfreq("D").shift(2)
        for n in (7, 30):
            p.loc[g.index, f"bias{n}"] = e.rolling(n, min_periods=n // 2).mean().reindex(g.day).values
    return p.drop(columns=["t", "tmpf_24h_ago"])


def design(p, codes):
    x = pd.DataFrame({"mu": p.mu, "nbm_minus_mu": p.NBS - p.mu,
                      "obs_minus_mu": p.tmpf - p.mu, "dwpf_minus_mu": p.dwpf - p.mu, "wind": p.sknt, "sky": p.sky,
                      "dt24": p.dt24, "max24_minus_mu": p.max24 - p.mu, "min24_minus_mu": p.min24 - p.mu,
                      "bias7": p.bias7, "bias30": p.bias30,
                      "doy_sin": np.sin(2 * np.pi * p.day.dt.dayofyear / 365.25),
                      "doy_cos": np.cos(2 * np.pi * p.day.dt.dayofyear / 365.25),
                      "station": pd.Categorical(p.code, categories=codes)}, index=p.index)
    om = p[[f"om_{m}" for m in OM]]
    for m in OM:
        x[f"{m}_minus_mu"] = p[f"om_{m}"] - p.mu
    x["om_spread"] = om.std(axis=1)
    for c in REGIME + EXTRA:
        x[c] = p[c].values
    return x


def regrid(p, mu2, c):
    """Move model.py's distribution to mu2 and scale its width by c, on a fresh integer grid."""
    k0n = np.round(mu2) - GRID
    out = []
    for cdf, k0, mu, m2, kn, c in zip(p.cdf, p.k0, p.mu, mu2, k0n, c):
        ks = kn + np.arange(2 * GRID + 1) + 0.5
        out.append(np.interp(mu + (ks - m2) / c, k0 + np.arange(len(cdf)) + 0.5, cdf, left=0, right=1)
                   .astype("float32"))
    return k0n, out


def walk_forward(p, codes):
    p = p.sort_values("day").reset_index(drop=True)
    X, r = design(p, codes), (p.y - p.mu).values
    mu2 = np.full(len(p), np.nan)
    c = np.ones(len(p))
    for month in pd.date_range(START, p.day.max(), freq="MS"):
        tr = ((p.day <= month - pd.Timedelta(days=2)) & ~np.isnan(r)).values
        te = ((p.day >= month) & (p.day < month + pd.offsets.MonthBegin())).values
        if not te.any():
            continue
        m = lgb.LGBMRegressor(**PARAMS).fit(X[tr], r[tr])
        mu2[te] = p.mu.values[te] + m.predict(X[te])
        # width: interquartile range of the new mean's out-of-sample errors vs the old one's, over the
        # months already predicted. IQR, not std: the outliers boosting can't fix kept std-scaled
        # intervals too wide (80% covered 82-86%); IQR gets 82-83%.
        done = tr & ~np.isnan(mu2)
        if done.sum() >= 2000:
            iqr = lambda e: np.subtract(*np.percentile(e, [75, 25]))
            c[te] = iqr(p.y[done] - mu2[done]) / iqr(p.y[done] - p.mu[done])
    p = p.assign(mu2=mu2, c=c).dropna(subset=["mu2"])
    k0, cdf = regrid(p, p.mu2.values, p.c.values)
    return p.assign(mu_old=p.mu, mu=p.mu2, k0=k0, cdf=cdf), m, X.columns


def main():
    st = stations_df()
    preds = pd.read_parquet(DATA / "model" / "preds.parquet")
    codes = sorted(preds.code.unique())
    f = []
    for s in st[st.code.isin(codes)].itertuples():
        f.append(features(preds[preds.code == s.code], s.code, s.tz))
        print(f"  {s.code}", flush=True)
    f = pd.concat(f, ignore_index=True)
    out = []
    for (dec, kind), g in f.groupby(["decision", "kind"]):
        p, m, cols = walk_forward(g, codes)
        out.append(p)
        imp = pd.Series(m.booster_.feature_importance("gain"), cols).sort_values(ascending=False)
        print(f"\n{dec} {kind}: top features by gain (last fit)\n",
              (100 * imp / imp.sum()).round(1).head(8).to_string())
    b = pd.concat(out, ignore_index=True)
    b[["day", "mu", "k0", "cdf", "y", "code", "decision", "kind", "NBS", "mu_old", "c"]].to_parquet(
        DATA / "model" / "preds_boost.parquet")

    # same rows, old vs new, by year
    old = preds.merge(b[["code", "decision", "kind", "day"]], on=["code", "decision", "kind", "day"])
    by = lambda d: [d.decision, d.kind, d.day.dt.year]
    s_old = old.groupby(by(old)).apply(skill, include_groups=False)
    s_new = b.groupby(by(b)).apply(skill, include_groups=False)
    t = pd.concat({"MAE old": s_old["MAE blend"], "MAE new": s_new["MAE blend"], "log old": s_old["log score"],
                   "log new": s_new["log score"], "80% cover new": s_new["80% cover"]}, axis=1)
    print("\nout of sample, model.py (old) vs boost.py (new), same days\n", t.round(3).to_string())


if __name__ == "__main__":
    from scipy.stats import norm
    one = pd.DataFrame({"cdf": [norm.cdf(30 + np.arange(41) + 0.5, 50.2, 3)], "k0": [30.0], "mu": [50.2]})
    k0, (g,) = regrid(one, np.array([50.2]), np.array([1.0]))
    assert k0[0] == 30 and np.allclose(g, one.cdf[0], atol=1e-6)  # no move, no rescale: the same grid
    k0, (g,) = regrid(one, np.array([53.2]), np.array([1.0]))
    assert np.allclose(g, norm.cdf(k0[0] + np.arange(41) + 0.5, 53.2, 3), atol=1e-6)  # a pure shift
    k0, (g,) = regrid(one, np.array([50.2]), np.array([0.5]))
    assert np.allclose(g, norm.cdf(k0[0] + np.arange(41) + 0.5, 50.2, 1.5), atol=5e-3)  # half as wide (linear interp between whole degrees)
    assert np.allclose(wind_uv(180, 10), (0, 10)) and np.allclose(wind_uv(270, 10), (10, 0))  # southerly -> north, westerly -> east
    t = pd.Series(pd.to_datetime(["2026-10-22 05:00", "2026-10-22 23:00", "2026-10-23 04:00"]))  # 00, 18, 23 EST on the 22nd
    dec = pd.Timestamp("2026-10-21 23:00")  # 18:00 EST the evening before
    assert day1_public(t, dec, "gfs_seamless").tolist() == [True, False, False]  # runs 00Z (out 09Z), 18Z (03Z), 00Z the 22nd
    assert day1_public(t, dec, "icon_seamless").tolist() == [True, True, False]  # ICON's 18Z run is out by 22Z
    main()
