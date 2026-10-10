"""Forecast daily bed requests, backtest against baselines, and convert to census forecasts."""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from common import *
warnings.filterwarnings("ignore")

units = pd.read_csv(DATA / "unit_daily.csv", parse_dates=["date"])
req = units.pivot(index="date", columns="unit", values="bed_requests")
adm = units.pivot(index="date", columns="unit", values="admitted")
census = units.pivot(index="date", columns="unit", values="census")
cap = units.pivot(index="date", columns="unit", values="beds")

T0 = pd.Timestamp(START)


def design(dates, K=3):
    t = ((dates - T0).days.values / 365.25)[:, None]
    doy = dates.dayofyear.values[:, None] / 365.25
    cols = [np.ones_like(t), t]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * doy), np.cos(2 * np.pi * k * doy)]
    dow = pd.get_dummies(dates.dayofweek).reindex(columns=range(7), fill_value=0).values[:, 1:].astype(float)
    return np.hstack(cols + [dow])


def fit_glm(y, dates):
    return sm.GLM(y, design(dates), family=sm.families.Poisson()).fit()


def phi_hat(y, mu):
    return float(max(np.mean(((y - mu) ** 2 - mu) / mu ** 2), 1e-4))


def forecast_models(y_train, d_train, d_future):
    """Return dict model -> forecast array for d_future."""
    out = {}
    # 1) seasonal naive: mean of the last 4 same-weekday values
    s = pd.Series(y_train.values, index=d_train)
    last = {dw: s[s.index.dayofweek == dw].iloc[-4:].mean() for dw in range(7)}
    out["Seasonal naive (4-wk same weekday)"] = np.array([last[d] for d in d_future.dayofweek])
    # 2) seasonal naive 364 days ago
    ly = pd.Series(y_train.values, index=d_train)
    out["Same day last year"] = np.array([ly.get(d - pd.Timedelta(days=364), ly.iloc[-7:].mean()) for d in d_future], float)
    # 3) Poisson GLM: trend + annual Fourier + day-of-week
    res = fit_glm(y_train.values, d_train)
    mu_tr = res.predict(design(d_train))
    base = res.predict(design(d_future))
    out["Poisson GLM"] = base
    # 4) GLM + decaying level correction from the last 28 days
    ratio = y_train.values[-28:].sum() / mu_tr[-28:].sum()
    ratio = 1 + 0.8 * (ratio - 1)
    decay = 0.5 ** (np.arange(1, len(d_future) + 1) / 21.0)
    out["GLM + level correction"] = base * (1 + (ratio - 1) * decay)
    return out, phi_hat(y_train.values, mu_tr)


def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    mae = np.mean(np.abs(a - p))
    wape = np.abs(a - p).sum() / a.sum()
    a7 = pd.Series(a).rolling(7).sum().dropna().values[::7]
    p7 = pd.Series(p).rolling(7).sum().dropna().values[::7]
    wape7 = np.abs(a7 - p7).sum() / a7.sum()
    return mae, wape, wape7


# ---------------- Rolling-origin backtest (90-day horizon) ----------------------------
ORIGINS = ["2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01", "2026-06-30"]
H = 90
rows = []
for o in ORIGINS:
    o = pd.Timestamp(o)
    tr = req.index < o
    fut = req.index[(req.index >= o)][:H]
    for u in UNIT_NAMES:
        fc, _ = forecast_models(req.loc[tr, u], req.index[tr], fut)
        for m, p in fc.items():
            mae, wape, wape7 = metrics(req.loc[fut, u].values, p)
            rows.append(dict(origin=str(o.date()), unit=u, model=m, MAE=mae, WAPE=wape, WAPE_weekly=wape7))
bt = pd.DataFrame(rows)
summary = bt.groupby("model")[["MAE", "WAPE", "WAPE_weekly"]].mean().sort_values("WAPE")
print("Backtest (mean over 5 origins x 3 units, 90-day horizon):\n", summary.round(3))
best = summary.index[0]
bt.to_csv(OUT / "forecast_backtest.csv", index=False)
summary.to_csv(OUT / "forecast_model_summary.csv")

# ---------------- Census forecast backtest through the flow engine -----------------------
rng = np.random.default_rng(11)
RUNS = 300
cb = []
for o in ORIGINS:
    o = pd.Timestamp(o)
    tr = req.index < o
    fut = req.index[(req.index >= o)][:H]
    for u in UNIT_NAMES:
        fc, phi = forecast_models(req.loc[tr, u], req.index[tr], fut)
        warm = adm.loc[tr, u].values[-30:]
        r = run_engine(fc[best], u, RUNS, rng, warm_adm=warm, dispersion=phi)
        c = r["census"]
        p10, p50, p90 = np.percentile(c, [5, 50, 95], axis=0)
        act = census.loc[fut, u].values
        cb.append(dict(origin=str(o.date()), unit=u,
                       census_MAPE=np.mean(np.abs(act - p50) / act),
                       census_MAE=np.mean(np.abs(act - p50)),
                       interval90_coverage=np.mean((act >= p10) & (act <= p90))))
cb = pd.DataFrame(cb)
cb.to_csv(OUT / "census_backtest.csv", index=False)
print("\nCensus forecast via flow engine:\n", cb.groupby("unit")[["census_MAPE", "census_MAE", "interval90_coverage"]].mean().round(3))
naive_census = []
for o in ORIGINS:
    o = pd.Timestamp(o); tr = census.index < o; fut = census.index[census.index >= o][:H]
    for u in UNIT_NAMES:
        last = census.loc[tr, u].values[-7:].mean()
        act = census.loc[fut, u].values
        naive_census.append(np.mean(np.abs(act - last) / act))
print("Naive census MAPE (flat last-7-day mean): %.3f" % np.mean(naive_census))

# ---------------- Production forecast: 1 Oct 2026 -> 31 Mar 2027 (182 days) ----------------
HF = 182
d_future = pd.date_range(pd.Timestamp(END) + pd.Timedelta(days=1), periods=HF, freq="D")
final = {}
lam_out = {}
fc_rows = []
for u in UNIT_NAMES:
    fc, phi = forecast_models(req[u], req.index, d_future)
    lam = fc[best]
    lam_out[u] = lam
    warm = adm[u].values[-30:]
    r = run_engine(lam, u, 500, rng, warm_adm=warm, dispersion=phi, cap=True)
    ru = run_engine(lam, u, 500, rng, warm_adm=warm, dispersion=phi, cap=False)
    p5, p50, p95 = np.percentile(r["census"], [5, 50, 95], axis=0)
    d50 = np.percentile(ru["census"], 50, axis=0)
    for i, d in enumerate(d_future):
        fc_rows.append(dict(date=d, unit=u, forecast_requests=round(float(lam[i]), 2), census_p05=p5[i], census_p50=p50[i],
                            census_p95=p95[i], bed_demand_p50=d50[i], beds=UNITS[u]["beds"]))
    final[u] = dict(phi=phi)
fcdf = pd.DataFrame(fc_rows)
fcdf.to_csv(DATA / "forecast_oct2026_mar2027.csv", index=False)
np.save(OUT / "phi.npy", np.array([final[u]["phi"] for u in UNIT_NAMES]))
np.save(OUT / "lam_forecast.npy", np.array([lam_out[u] for u in UNIT_NAMES]))

tot50 = fcdf.groupby("date")[["census_p50", "bed_demand_p50"]].sum()
print("\nForecast window hospital median occupancy: mean %.3f, peak %.3f on %s" % (
    tot50.census_p50.mean() / TOTAL_BEDS, tot50.census_p50.max() / TOTAL_BEDS, tot50.census_p50.idxmax().date()))
json.dump(dict(best_model=best, backtest=summary.round(4).to_dict("index"),
               census_backtest=cb.groupby("unit")[["census_MAPE", "census_MAE", "interval90_coverage"]].mean().round(4).to_dict("index"),
               naive_census_mape=float(np.mean(naive_census))), open(OUT / "forecast_summary.json", "w"), indent=2)
