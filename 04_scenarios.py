"""Scenario modelling on the forecast window (1 Oct 2026 - 31 Mar 2027)."""
import json
import numpy as np, pandas as pd
from common import *

opt = json.load(open(OUT / "optimization_results.json"))
phi = np.load(OUT / "phi.npy"); lam = np.load(OUT / "lam_forecast.npy")
units = pd.read_csv(DATA / "unit_daily.csv", parse_dates=["date"])
adm = units.pivot(index="date", columns="unit", values="admitted")
HF = lam.shape[1]
d_future = pd.date_range(pd.Timestamp(END) + pd.Timedelta(days=1), periods=HF, freq="D")
dow = np.array(d_future.dayofweek)
x_opt = np.array(opt["rn_roster"]["x_opt"]); x_flat = np.full(7, 25.0)
work = np.ones((7, 7))
for s in range(7):
    work[s, s] = 0; work[s, (s + 1) % 7] = 0
alloc_cur = opt["bed_allocation"]["current"]; alloc_opt = opt["bed_allocation"]["optimised"]
RUNS = 300; WEEKS = HF / 7


def roster_cost(Rreq, x):
    cov_by_dow = (1 - SICK_RATE) * (x[:, None] * work).sum(0)
    cov = cov_by_dow[dow][None, :]
    gap = np.maximum(Rreq - cov, 0)
    core = 5 * CORE_RN_SHIFT_COST * x.sum() * WEEKS
    agency = gap.sum(1).mean() * AGENCY_RN_SHIFT_COST
    return core + agency, gap.sum(1).mean()


def run(name, alloc, los_delta=0.0, uplift=0.0, x=x_flat, flex=None, seed=3):
    rng = np.random.default_rng(seed)
    alloc = dict(alloc)
    if flex:
        for u, n in flex.items():
            alloc[u] += n
    sim = {}
    for i, u in enumerate(UNIT_NAMES):
        sim[u] = run_engine(lam[i] * (1 + uplift), u, RUNS, rng, warm_adm=adm[u].values[-30:], beds=alloc[u],
                            los_mean=UNITS[u]["los_mean"] - los_delta, dispersion=phi[i])
    beds_tot = sum(alloc.values())
    cen = sum(sim[u]["census"] for u in UNIT_NAMES)
    occ = cen / beds_tot
    div = {u: float(sim[u]["diverted"].sum(1).mean()) for u in UNIT_NAMES}
    div_cost = sum(DIVERSION_COST[u] * div[u] for u in UNIT_NAMES)
    Rreq = required_rn_shifts({u: sim[u]["census"] for u in UNIT_NAMES})
    staff_cost, agency = roster_cost(Rreq, x)
    # equipment demand from the same census paths
    cen3 = np.stack([sim[u]["census"] for u in UNIT_NAMES], -1)
    vd = rng.binomial(cen3, [UNITS[u]["vent_p"] for u in UNIT_NAMES]).sum(-1)
    pdm = rng.poisson(cen3 * np.array([UNITS[u]["pump_per_pt"] for u in UNIT_NAMES])).sum(-1)
    extra_beds_cost = 0.0
    if flex:   # carrying cost of flex beds: staffed ~ 1 RN per `ratio` patients is already in staffing; add fixed overhead per bed-day
        extra_beds_cost = sum(flex.values()) * 180 * HF
    return dict(scenario=name, beds=beds_tot, avg_occ=float(occ.mean()), p95_occ=float(np.percentile(occ, 95)),
                days_ge95=float((occ >= 0.95).sum(1).mean()), days_full=float((occ >= 0.995).sum(1).mean()),
                diverted=sum(div.values()), diverted_icu=div["ICU"], diverted_stepdown=div["Step-Down"], diverted_medsurg=div["Med-Surg"],
                agency_shifts=agency, vent_p99=float(np.percentile(vd, 99)), pump_p99=float(np.percentile(pdm, 99)),
                pump_stockout_pct=float((pdm > 340).mean()),
                diversion_cost=div_cost, staffing_cost=staff_cost, flex_cost=extra_beds_cost,
                total_cost=div_cost + staff_cost + extra_beds_cost)

S = [
    run("Status quo (flat roster, current beds)", alloc_cur),
    run("Optimised roster + bed mix", alloc_opt, x=x_opt),
    run("Surge +10% demand (status quo)", alloc_cur, uplift=0.10),
    run("Surge +10% demand (optimised)", alloc_opt, uplift=0.10, x=x_opt),
    run("Surge +10% + 12 flex Med-Surg beds", alloc_opt, uplift=0.10, x=x_opt, flex={"Med-Surg": 12}),
    run("Discharge programme: LOS -0.5 day", alloc_opt, los_delta=0.5, x=x_opt),
    run("Surge +10% with LOS -0.5 day", alloc_opt, uplift=0.10, los_delta=0.5, x=x_opt),
    run("Demand -10% (service-line loss)", alloc_opt, uplift=-0.10, x=x_opt),
]
df = pd.DataFrame(S)
df.to_csv(OUT / "scenario_results.csv", index=False)
base_cost = df.total_cost.iloc[0]
df["vs_status_quo"] = df.total_cost - base_cost
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df[["scenario", "beds", "avg_occ", "days_ge95", "diverted", "agency_shifts", "vent_p99", "pump_p99", "total_cost", "vs_status_quo"]].round(3).to_string(index=False))

# tornado: single-parameter sensitivity of total cost under the optimised plan
sens = []
for label, kw in [("Demand +/-10%", ("uplift", .10, -.10)), ("Length of stay -/+0.5 day", ("los_delta", .5, -.5))]:
    k, hi, lo = kw
    a = run(label, alloc_opt, x=x_opt, **{k: hi}); b = run(label, alloc_opt, x=x_opt, **{k: lo})
    sens.append(dict(driver=label, low=min(a['total_cost'], b['total_cost']),
                     high=max(a['total_cost'], b['total_cost'])))
# cost-parameter sensitivities (no re-simulation needed)
r0 = run("base", alloc_opt, x=x_opt)
agency_cost_part = r0["agency_shifts"] * AGENCY_RN_SHIFT_COST
for label, lo_, hi_ in [("Agency shift cost +/-25%", r0["total_cost"] - .25 * agency_cost_part, r0["total_cost"] + .25 * agency_cost_part),
                        ("Diversion cost +/-25%", r0["total_cost"] - .25 * r0["diversion_cost"], r0["total_cost"] + .25 * r0["diversion_cost"]),
                        ("Core RN shift cost +/-10%", r0["total_cost"] - .10 * 5 * CORE_RN_SHIFT_COST * x_opt.sum() * WEEKS, r0["total_cost"] + .10 * 5 * CORE_RN_SHIFT_COST * x_opt.sum() * WEEKS)]:
    sens.append(dict(driver=label, low=lo_, high=hi_))
sens = pd.DataFrame(sens); sens["base"] = r0["total_cost"]
sens.to_csv(OUT / "sensitivity.csv", index=False)
print(sens.round(0).to_string(index=False))
