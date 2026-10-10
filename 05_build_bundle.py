"""Collect all model outputs into one JSON bundle for the dashboard page."""
import json
import numpy as np, pandas as pd
from common import *

opt = json.load(open(OUT / "optimization_results.json"))
fsum = json.load(open(OUT / "forecast_summary.json"))
phi = np.load(OUT / "phi.npy"); lam = np.load(OUT / "lam_forecast.npy")
units = pd.read_csv(DATA / "unit_daily.csv", parse_dates=["date"])
adm = units.pivot(index="date", columns="unit", values="admitted")
cen = units.pivot(index="date", columns="unit", values="census")
d_future = pd.date_range(pd.Timestamp(END) + pd.Timedelta(days=1), periods=lam.shape[1], freq="D")

# hospital-level forecast band (sum census across units within each simulated path)
rng = np.random.default_rng(21)
tot = 0
for i, u in enumerate(UNIT_NAMES):
    tot = tot + run_engine(lam[i], u, 500, rng, warm_adm=adm[u].values[-30:], dispersion=phi[i])["census"]
occ = tot / TOTAL_BEDS
band = pd.DataFrame({"p05": np.percentile(occ, 5, 0), "p50": np.percentile(occ, 50, 0), "p95": np.percentile(occ, 95, 0)}, index=d_future)
wk_f = band.resample("W-SUN").mean()
hist = (cen.sum(axis=1) / TOTAL_BEDS)
wk_h = hist.resample("W-SUN").mean()
wk_h = wk_h[wk_h.index >= pd.Timestamp("2025-10-01")]
days95 = float((occ >= 0.95).sum(1).mean()); days_full = float((occ >= 0.995).sum(1).mean())

fr = pd.read_csv(OUT / "roster_frontier.csv")
sc = pd.read_csv(OUT / "scenario_results.csv")
se = pd.read_csv(OUT / "sensitivity.csv")
bt = pd.read_csv(OUT / "forecast_model_summary.csv")

bundle = dict(
    hist=dict(week=[d.strftime("%Y-%m-%d") for d in wk_h.index], occ=wk_h.round(4).tolist()),
    fc=dict(week=[d.strftime("%Y-%m-%d") for d in wk_f.index], p05=wk_f.p05.round(4).tolist(), p50=wk_f.p50.round(4).tolist(), p95=wk_f.p95.round(4).tolist(),
            peak_week=wk_f.p50.idxmax().strftime("%d %b %Y"), peak_occ=float(wk_f.p50.max()), mean_occ=float(band.p50.mean()),
            days_ge95=days95, days_full=days_full, window=f"{d_future[0]:%d %b %Y} – {d_future[-1]:%d %b %Y}"),
    backtest=bt.round(4).to_dict("records"), census_bt=fsum["census_backtest"], naive_census=fsum["naive_census_mape"],
    roster=opt["rn_roster"], frontier=fr.round(1).to_dict("records"),
    beds=opt["bed_allocation"], equipment=opt["equipment"], assumptions=opt["assumptions"],
    scenarios=sc.round(3).to_dict("records"), sens=se.round(0).to_dict("records"),
    n_days=int(len(cen)), total_beds=TOTAL_BEDS,
)
json.dump(bundle, open(DATA / "dashboard_bundle.json", "w"))
print("bundle ok", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in bundle.items()})
print("forecast: mean occ %.3f peak %.3f (%s) days>=95%% %.0f full %.0f" % (bundle["fc"]["mean_occ"], bundle["fc"]["peak_occ"], bundle["fc"]["peak_week"], days95, days_full))
r = opt["rn_roster"]
print("roster coverage cur", r["current_holdout"]["coverage"], "opt", r["optimised_holdout"]["coverage"])
print("required mean", r["mean_required_by_weekday"], "p90", r["p90_required_by_weekday"])
print("x_opt", r["x_opt"], "flat total heads 175 vs", sum(r["x_opt"]))
print(sc[["scenario","diverted","agency_shifts","flex_cost","total_cost"]].round(0).to_string())
