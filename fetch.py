"""Pull every history a Kalshi weather edge needs: settlement truth, forecasts, market prices.

    python fetch.py stations                 # Kalshi series -> settlement stations (run first)
    python fetch.py kalshi                   # every market + hourly bid/ask/last candles
    python fetch.py cli asos mos             # IEM: climate reports, hourly obs, MOS/NBM forecasts
    python fetch.py ghcn openmeteo hrrr      # NOAA GHCN-Daily, Open-Meteo previous model runs, HRRR
    python fetch.py all
    python fetch.py status                   # coverage + settlement cross-check

Every step is resumable: finished files are skipped, so just re-run after an interruption.
IEM steps share one server, so run them in one process; kalshi and ghcn/openmeteo can run alongside.
"""
import io, re, sys, time, threading, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, pyarrow.parquet as pq, requests

DATA = Path(__file__).parent / "data"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
IEM = "https://mesonet.agron.iastate.edu"
UA = {"User-Agent": "kalshi-weather-research/0.1 (student project)"}
YEAR = dt.date.today().year
CLI_SINCE, ASOS_SINCE, MOS_SINCE, OM_SINCE = 2000, 2000, 2018, "2022-01-01"
MOS_MODELS = ["GFS", "NAM", "NBS", "NBE", "LAV"]  # LAV = GFS LAMP (IEM keeps the 00/06/12/18Z runs); no ECMWF MOS
OM_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]
HRRR_SINCE = "2024-01-01"  # Open-Meteo's previous-runs archive of HRRR starts here; 48h horizon, so day1 only
EXTRA_STATIONS = ["ORD", "DTW"]  # snow series name a city, not a CLI code

_tl, _lock, _next = threading.local(), threading.Lock(), [0.0]


def get(url, pause=0.0, backoff=2, rate=None, **params):
    """GET with retry on 429/5xx. `rate` = global max requests/sec across threads."""
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
        _tl.s.headers.update(UA)
    for i in range(7):
        if rate:
            with _lock:
                now = time.monotonic()
                wait, _next[0] = _next[0] - now, max(now, _next[0]) + 1 / rate
            time.sleep(max(wait, 0))
        try:
            r = _tl.s.get(url, params=params, timeout=300)
        except requests.RequestException:
            time.sleep(backoff * 2 ** i)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(backoff * 2 ** i)
            continue
        time.sleep(pause)
        r.raise_for_status()
        return r
    raise RuntimeError(f"gave up on {url} {params}")


def kget(path, **params):
    return get(KALSHI + path, rate=5, **params).json()  # measured: 6/s draws 429s, 10/s fails half


def paged(path, key, **params):
    out, cur = [], None
    while True:
        j = kget(path, limit=1000, **params, **({"cursor": cur} if cur else {}))
        out += j.get(key) or []
        cur = j.get("cursor")
        if not cur or not j.get(key):
            return out


def tidy(df):
    """Parquet needs one type per column; IEM mixes numbers with 'M'/'T' and lists."""
    for c in df.columns[df.dtypes == object]:
        df[c] = [None if v is None or v != v else str(v) for v in df[c]]
    return df


def save(df, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    tidy(df).to_parquet(f, index=False)


def stations_df():
    return pd.read_csv(DATA / "stations.csv", dtype=str)


def yearly(kind, code, years, fetch):
    """One parquet per station-year; past years are fetched once, the current year every run."""
    for y in years:
        f = DATA / kind / code / f"{y}.parquet"
        if f.exists() and y < YEAR:
            continue
        save(fetch(y), f)
        print(f"  {kind} {code} {y}", flush=True)


# ---------------------------------------------------------------- stations
def stations():
    series = paged("/series", "series", category="Climate and Weather")
    (DATA / "kalshi").mkdir(parents=True, exist_ok=True)
    pd.json_normalize(series).pipe(tidy).to_parquet(DATA / "kalshi" / "series.parquet", index=False)

    def codes(s):
        t = s["ticker"]
        ms = kget("/markets", series_ticker=t, limit=100).get("markets") or \
            kget("/historical/markets", series_ticker=t, limit=100).get("markets") or []
        rules = " ".join(m.get("rules_primary") or "" for m in ms)
        return t, set(re.findall(r"\b(?:CLI|K)([A-Z]{3})\b", rules))

    with ThreadPoolExecutor(8) as ex:
        hits = dict(ex.map(codes, series))
    by_code = {}
    for t, cs in hits.items():
        for c in cs:
            by_code.setdefault(c, []).append(t)
    for c in EXTRA_STATIONS:
        by_code.setdefault(c, [])

    cli = {f["properties"]["station"]: f["properties"] for f in
           get(f"{IEM}/geojson/cli.py", dt=str(dt.date.today() - dt.timedelta(days=2))).json()["features"]}
    rows, nets = [], {}
    for c, ts in sorted(by_code.items()):
        p = cli.get("K" + c)
        if not p:
            print(f"  skip {c}: no NWS climate report (series {ts})")
            continue
        st = p["state"]
        if st not in nets:
            nets[st] = {f["properties"]["sid"]: f["properties"] for f in
                        get(f"{IEM}/geojson/network/{st}_ASOS.geojson", pause=1).json()["features"]}
        a = nets[st].get(c, {})
        rows.append(dict(code=c, icao="K" + c, name=p["name"], state=st, lat=p["lat"], lon=p["lon"],
                         tz=a.get("tzname"), ghcn=a.get("ncei91") or a.get("ncdc81"),
                         asos_begin=a.get("archive_begin"), series=";".join(sorted(ts))))
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "stations.csv", index=False)
    print(df[["code", "name", "tz", "ghcn", "series"]].to_string())


# ---------------------------------------------------------------- kalshi
def _ts(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _batches(todo):
    """Pack live markets into /markets/candlesticks calls: <=100 markets, markets x window hours <= 10,000.
    That endpoint returns nothing for archived markets (before ~late Jul 2026), so those go one per call."""
    yield from ([m] for m in todo if m["_src"] == "historical")
    b, lo, hi = [], 0, 0
    for m in sorted((m for m in todo if m["_src"] == "live"), key=lambda m: m["open_time"]):
        a, z = _ts(m["open_time"]), _ts(m["close_time"])
        if b and (len(b) == 100 or (len(b) + 1) * ((max(hi, z) - min(lo, a)) // 3600 + 2) > 10000):
            yield b
            b = []
        lo, hi = (min(lo, a), max(hi, z)) if b else (a, z)
        b.append(m)
    if b:
        yield b


def _candles(b):
    a, z = min(_ts(m["open_time"]) for m in b), max(_ts(m["close_time"]) for m in b)
    period = 60 if (z - a) // 3600 + 2 <= 5000 else 1440  # 400 above 5,000 candles/request: daily instead
    if b[0]["_src"] == "historical":
        t = b[0]["ticker"]
        try:
            rows = [dict(c, ticker=t) for c in kget(f"/historical/markets/{t}/candlesticks", start_ts=a, end_ts=z,
                                                     period_interval=period).get("candlesticks") or []]
        except requests.HTTPError as e:  # e.g. a "/" in the ticker: unroutable, skip rather than stop the run
            print(f"  skip {t}: {e.response.status_code}", flush=True)
            rows = []
    else:
        j = kget("/markets/candlesticks", market_tickers=",".join(m["ticker"] for m in b),
                 start_ts=a, end_ts=z, period_interval=period)
        rows = [dict(c, ticker=x["market_ticker"]) for x in j.get("markets") or [] for c in x["candlesticks"]]
    df = pd.json_normalize(rows)
    df.columns = [re.sub(r"_(dollars|fp)$", "", x) for x in df.columns]  # close_dollars -> close
    return [m["ticker"] for m in b], df


def kalshi():
    series = sorted(pd.read_parquet(DATA / "kalshi" / "series.parquet")["ticker"],
                    key=lambda s: not re.match(r"KX(HIGH|LOWT)", s))  # temperature ladders first
    for s in series:
        ms = [dict(m, _src="live") for m in paged("/markets", "markets", series_ticker=s)]
        ms += [dict(m, _src="historical") for m in paged("/historical/markets", "markets", series_ticker=s)]
        if not ms:
            continue
        mdf = pd.json_normalize(ms)
        save(mdf, DATA / "kalshi" / "markets" / f"{s}.parquet")

        cdir = DATA / "kalshi" / "candles" / s
        done_f = cdir / "done.txt"
        done = set(done_f.read_text().split()) if done_f.exists() else set()
        todo = [dict(m, _series=s) for m in ms
                if m["ticker"] not in done and m.get("status") in ("settled", "finalized", "determined", "closed")
                and float(m.get("volume_fp") or m.get("volume") or 0) > 0]
        print(f"{s}: {len(ms)} markets, {len(todo)} need candles", flush=True)
        part = len(list(cdir.glob("part-*.parquet"))) if cdir.exists() else 0
        batches = list(_batches(todo))
        for i in range(0, len(batches), 120):  # one part file per 120 calls
            with ThreadPoolExecutor(6) as ex:
                got = list(ex.map(_candles, batches[i:i + 120]))
            frames = [df for _, df in got if len(df)]
            if frames:
                save(pd.concat(frames, ignore_index=True), cdir / f"part-{part:05d}.parquet")
                part += 1
            cdir.mkdir(parents=True, exist_ok=True)
            with open(done_f, "a") as f:
                f.write("".join(t + "\n" for ts, _ in got for t in ts))
            print(f"  {s} candles {sum(len(b) for b in batches[:i + 120])}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- IEM
def cli():
    for st in stations_df().itertuples():
        yearly("cli", st.code, range(CLI_SINCE, YEAR + 1), lambda y: pd.DataFrame(
            get(f"{IEM}/json/cli.py", pause=1, backoff=10, station=st.icao, year=y).json()["results"]))


def asos():
    cols = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "vsby", "skyc1", "skyl1", "wxcodes", "metar"]
    for st in stations_df().itertuples():
        def one(y):
            r = get(f"{IEM}/cgi-bin/request/asos.py", pause=1, backoff=10, station=st.code, data=cols,
                    year1=y, month1=1, day1=1, year2=y + 1, month2=1, day2=1, tz="Etc/UTC",
                    format="onlycomma", latlon="no", missing="empty", trace="T", report_type=[3, 4])
            d = pd.read_csv(io.StringIO(r.text), dtype=str)
            return d[~d.metar.str.contains("MADISHF", na=False)]  # MADIS copies despite report_type, as live.refresh
        yearly("asos", st.code, range(ASOS_SINCE, YEAR + 1), one)


def mos_csv(text):
    """IEM's mos.py CSV with one time format: a block of midnight runtimes can come date-only ("2026-10-22", LAMP
    at DEN and PHX on 22 Oct), which the readers' inferred format rejects and live._merge's dedupe misses."""
    d = pd.read_csv(io.StringIO(text), dtype=str)
    for c in ("runtime", "ftime"):
        d[c] = pd.to_datetime(d[c], format="ISO8601").dt.strftime("%Y-%m-%d %H:%M:%S")
    return d


def mos():
    for st in stations_df().itertuples():
        for model in MOS_MODELS:
            yearly(f"mos/{model}", st.code, range(MOS_SINCE, YEAR + 1), lambda y: mos_csv(
                get(f"{IEM}/cgi-bin/request/mos.py", pause=1, backoff=10, station=st.icao, model=model,
                    sts=f"{y}-01-01T00:00Z", ets=f"{y + 1}-01-01T00:00Z", format="csv").text))


# ---------------------------------------------------------------- NOAA + Open-Meteo
def ghcn():
    for st in stations_df().dropna(subset=["ghcn"]).itertuples():
        f = DATA / "ghcn" / f"{st.code}.parquet"
        if f.exists():
            continue
        r = get(f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{st.ghcn}.csv")
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        keep = [c for c in df.columns if c.split("_")[0] in
                ("STATION", "DATE", "TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "SNWD")]
        save(df[keep].copy(), f)
        print(f"  ghcn {st.code} {df.DATE.min()}..{df.DATE.max()}", flush=True)


def openmeteo():
    """Real lead-time forecasts: what each model said 1 and 2 days before the hour verified."""
    end = str(dt.date.today() - dt.timedelta(days=1))
    for st in stations_df().itertuples():
        f = DATA / "openmeteo" / f"{st.code}.parquet"
        if f.exists():
            continue
        j = get("https://previous-runs-api.open-meteo.com/v1/forecast", backoff=60, pause=5,
                latitude=st.lat, longitude=st.lon, start_date=OM_SINCE, end_date=end,
                hourly="temperature_2m_previous_day1,temperature_2m_previous_day2",
                models=",".join(OM_MODELS), temperature_unit="fahrenheit", timezone="GMT").json()
        save(pd.DataFrame(j["hourly"]), f)
        print(f"  openmeteo {st.code}", flush=True)


def hrrr():
    end = str(dt.date.today() - dt.timedelta(days=1))
    for st in stations_df().itertuples():
        f = DATA / "hrrr" / f"{st.code}.parquet"
        if f.exists():
            continue
        j = get("https://previous-runs-api.open-meteo.com/v1/forecast", backoff=60, pause=5,
                latitude=st.lat, longitude=st.lon, start_date=HRRR_SINCE, end_date=end,
                hourly="temperature_2m_previous_day1", models="ncep_hrrr_conus",
                temperature_unit="fahrenheit", timezone="GMT").json()
        save(pd.DataFrame(j["hourly"]), f)
        print(f"  hrrr {st.code}", flush=True)


# ---------------------------------------------------------------- status
def _event_date(ev):
    return pd.to_datetime(ev.rsplit("-", 1)[-1], format="%y%b%d", errors="coerce")


def status():
    def summary(kind, datecol):
        root = DATA / kind
        if not root.exists():
            return print(f"{kind:12} -- not fetched")
        files = list(root.rglob("*.parquet"))
        rows = sum(pq.read_metadata(f).num_rows for f in files)
        dates = pd.concat([pd.read_parquet(f, columns=[datecol])[datecol] for f in files
                           if datecol in pq.read_schema(f).names], ignore_index=True).dropna()
        stns = len({f.relative_to(root).parts[-2] if f.parent != root else f.stem for f in files})
        print(f"{kind:12} {stns:3} stations {rows:>11,} rows  {dates.min()} .. {dates.max()}")

    summary("cli", "valid")
    summary("ghcn", "DATE")
    summary("asos", "valid")
    for m in MOS_MODELS:
        summary(f"mos/{m}", "runtime")
    summary("openmeteo", "time")

    mdir = DATA / "kalshi" / "markets"
    if mdir.exists():
        mk = pd.concat([pd.read_parquet(f).assign(series_ticker=f.stem) for f in mdir.glob("*.parquet")],
                       ignore_index=True)  # market rows carry no series field
        cfiles = list((DATA / "kalshi" / "candles").rglob("part-*.parquet"))
        crow = sum(pd.read_parquet(f, columns=["ticker"]).shape[0] for f in cfiles)
        print(f"kalshi       {mk['series_ticker'].nunique() if 'series_ticker' in mk else '?'} series "
              f"{len(mk):,} markets, {crow:,} hourly candles")

        # does our settlement truth (IEM climate report) reproduce Kalshi's expiration_value?
        st = stations_df()
        print("\nsettlement check: Kalshi expiration_value vs IEM CLI, per series")
        for row in st.itertuples():
            files = list((DATA / "cli" / row.code).glob("*.parquet"))
            if not files:
                continue
            c = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if c.empty:
                continue
            c["day"] = pd.to_datetime(c["valid"])
            c = c.set_index("day")
            for s in str(row.series).split(";"):
                kind = "high" if "HIGH" in s else "low" if "LOWT" in s else None
                if not kind or s not in set(mk.get("series_ticker", [])):
                    continue
                e = mk[(mk.series_ticker == s) & mk.expiration_value.notna()].drop_duplicates("event_ticker")
                e = e.assign(day=e.event_ticker.map(_event_date), kv=pd.to_numeric(e.expiration_value, errors="coerce"))
                j = e.join(pd.to_numeric(c[kind], errors="coerce").rename("cli"), on="day").dropna(subset=["kv", "cli"])
                if len(j):
                    ok = (j.kv.round() == j.cli).mean()
                    print(f"  {s:14} {row.code}  {len(j):5} days  exact match {ok:6.1%}  "
                          f"mean |diff| {abs(j.kv - j.cli).mean():.2f}F")


STEPS = dict(stations=stations, kalshi=kalshi, cli=cli, asos=asos, mos=mos, ghcn=ghcn,
             openmeteo=openmeteo, hrrr=hrrr, status=status)

if __name__ == "__main__":
    args = sys.argv[1:] or ["status"]
    if args == ["all"]:
        args = ["stations", "kalshi", "cli", "ghcn", "asos", "mos", "openmeteo", "hrrr", "status"]
    for a in args:
        print(f"== {a}", flush=True)
        STEPS[a]()
