"""Optimisation: (A) RN roster, (B) bed allocation, (C) equipment fleet sizing.

Method: sample-average approximation (SAA) on Monte-Carlo scenarios drawn from the
forecast + patient-flow engine, solved as small LPs/ILPs with SciPy (HiGHS).
"""
import json
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from common import *

rng = np.random.default_rng(7)
phi = np.load(OUT / "phi.npy"); lam = np.load(OUT / "lam_forecast.npy")
units = pd.read_csv(DATA / "unit_daily.csv", parse_dates=["date"])
adm = units.pivot(index="date", columns="unit", values="admitted")
HF = lam.shape[1]
d_future = pd.date_range(pd.Timestamp(END) + pd.Timedelta(days=1), periods=HF, freq="D")
RUNS = 400


def simulate(beds=None, los_delta=0.0, uplift=0.0, cap=True, runs=RUNS, rng=rng):
    out = {}
    for i, u in enumerate(UNIT_NAMES):
        b = None if beds is None else beds[u]
        out[u] = run_engine(lam[i] * (1 + uplift), u, runs, rng, warm_adm=adm[u].values[-30:], beds=b,
                            los_mean=UNITS[u]["los_mean"] - los_delta, dispersion=phi[i], cap=cap)
    return out


base = simulate()
dem = simulate(cap=False)
res = {}

# =========================== A. RN roster (two-stage stochastic ILP) ===========================
def rn_required(sim):
    return required_rn_shifts({u: sim[u]["census"] for u in UNIT_NAMES})   # (R, T)

Rreq = rn_required(base)
blocks = Rreq.reshape(RUNS, HF // 7, 7)                  # each block = 7 consecutive days from d_future[0]
dow_of_block_day = np.array([d.dayofweek for d in d_future[:7]])
scen = np.zeros((RUNS * (HF // 7), 7))                   # columns Mon..Sun
for c, dw in enumerate(dow_of_block_day):
    scen[:, dw] = blocks[:, :, c].reshape(-1)
K = 600
scen = scen[rng.choice(len(scen), K, replace=False)]
avail = 1 - SICK_RATE
# work[s][j] = 1 if pattern s works day j (days off = s and s+1)
work = np.ones((7, 7))
for s in range(7):
    work[s, s] = 0; work[s, (s + 1) % 7] = 0


def eval_roster(x, S):
    cov = avail * (x[:, None] * work).sum(0)             # coverage per weekday
    gap = np.maximum(S - cov, 0); idle = np.maximum(cov - S, 0)
    core_cost = 5 * CORE_RN_SHIFT_COST * x.sum()
    return dict(core_fte=float(x.sum()), core_cost_week=float(core_cost),
                agency_shifts_week=float(gap.sum(1).mean()), agency_cost_week=float(gap.sum(1).mean() * AGENCY_RN_SHIFT_COST),
                idle_shifts_week=float(idle.sum(1).mean()),
                total_cost_week=float(core_cost + gap.sum(1).mean() * AGENCY_RN_SHIFT_COST),
                pct_days_core_covers=float((gap == 0).mean()), coverage=cov.round(1).tolist())

# variables: x[0:7] (integer core RNs per days-off pattern), y[7 + k*7 + j] (agency shifts, scenario k, weekday j)
nv = 7 + K * 7
c = np.zeros(nv)
c[:7] = 5 * CORE_RN_SHIFT_COST
c[7:] = AGENCY_RN_SHIFT_COST / K
rows, cols, vals = [], [], []
rhs = scen.reshape(-1)                                   # row r = k*7 + j
for k in range(K):
    for j in range(7):
        r = k * 7 + j
        rows.append(r); cols.append(7 + r); vals.append(1.0)
        for s_ in range(7):
            if work[s_, j]:
                rows.append(r); cols.append(s_); vals.append(avail)
A = sp.csr_matrix((vals, (rows, cols)), shape=(K * 7, nv))
integrality = np.zeros(nv); integrality[:7] = 1
sol = milp(c, constraints=LinearConstraint(A, lb=rhs, ub=np.inf), integrality=integrality, bounds=Bounds(0, np.inf))
assert sol.success, sol.message
x_opt = np.round(sol.x[:7])

# current practice = flat roster sized to the average requirement
x_flat = np.full(7, 125 * 7 / 5 / 7)
cur = eval_roster(x_flat, scen); opt = eval_roster(x_opt, scen)
# robustness check on fresh scenarios
chk = simulate(rng=np.random.default_rng(99), runs=200)
Rc = rn_required(chk).reshape(200, HF // 7, 7)
sc2 = np.zeros((200 * (HF // 7), 7))
for c, dw in enumerate(dow_of_block_day):
    sc2[:, dw] = Rc[:, :, c].reshape(-1)
cur_h = eval_roster(x_flat, sc2); opt_h = eval_roster(x_opt, sc2)
weeks = HF // 7
res["rn_roster"] = dict(weekday_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                        current=cur, optimised=opt, current_holdout=cur_h, optimised_holdout=opt_h,
                        x_opt=x_opt.round(1).tolist(), weeks=weeks,
                        saving_per_week=cur_h["total_cost_week"] - opt_h["total_cost_week"],
                        saving_window=(cur_h["total_cost_week"] - opt_h["total_cost_week"]) * weeks,
                        mean_required_by_weekday=sc2.mean(0).round(1).tolist(),
                        p90_required_by_weekday=np.percentile(sc2, 90, axis=0).round(1).tolist())
# cost vs core-size frontier (agency substitution trade-off)
fr = []
for scale in np.linspace(0.7, 1.35, 27):
    xs = x_opt * scale
    r = eval_roster(xs, sc2)
    fr.append(dict(core_fte=r["core_fte"], total_cost_week=r["total_cost_week"], agency_shifts_week=r["agency_shifts_week"], idle_shifts_week=r["idle_shifts_week"]))
pd.DataFrame(fr).to_csv(OUT / "roster_frontier.csv", index=False)

# =========================== B. Bed allocation (SAA LP) ===========================
Dm = np.stack([dem[u]["census"] for u in UNIT_NAMES], axis=-1).reshape(-1, 3)     # (R*T, 3)
KB = 2500
Ds = Dm[rng.choice(len(Dm), KB, replace=False)].astype(float)
lo = dict(zip(UNIT_NAMES, [24, 40, 170])); hi = dict(zip(UNIT_NAMES, [32, 60, 220]))
w = [DIVERSION_COST[u] for u in UNIT_NAMES]   # LP minimises expected diversion cost (USD)
# variables: b[0:3] beds per unit, s[3 + k*3 + i] shortfall (bed-days short) in scenario k, unit i
nv = 3 + KB * 3
c = np.zeros(nv); c[3:] = np.tile(w, KB) / KB
rows, cols, vals = [], [], []
for k in range(KB):
    for i in range(3):
        r = k * 3 + i
        rows += [r, r]; cols += [3 + r, i]; vals += [1.0, 1.0]
A = sp.csr_matrix((vals, (rows, cols)), shape=(KB * 3, nv))
Aeq = sp.csr_matrix(([1.0, 1.0, 1.0], ([0, 0, 0], [0, 1, 2])), shape=(1, nv))
lb_b = np.r_[[lo[u] for u in UNIT_NAMES], np.zeros(KB * 3)]; ub_b = np.r_[[hi[u] for u in UNIT_NAMES], np.full(KB * 3, np.inf)]
sol = linprog(c, A_ub=-A, b_ub=-Ds.reshape(-1), A_eq=Aeq, b_eq=[TOTAL_BEDS], bounds=list(zip(lb_b, ub_b)), method="highs")
assert sol.status == 0, sol.message
b_lp = sol.x[:3]
b_int = np.floor(b_lp).astype(int)
for i in np.argsort(-(b_lp - b_int))[: TOTAL_BEDS - b_int.sum()]:
    b_int[i] += 1
alloc_opt = dict(zip(UNIT_NAMES, map(int, b_int)))
alloc_cur = {u: UNITS[u]["beds"] for u in UNIT_NAMES}


def bed_kpis(alloc, runs=300, seed=5):
    s = simulate(beds=alloc, runs=runs, rng=np.random.default_rng(seed))
    div = {u: float(s[u]["diverted"].sum(1).mean()) for u in UNIT_NAMES}
    occ = {u: float((s[u]["census"] / alloc[u]).mean()) for u in UNIT_NAMES}
    full = {u: float((s[u]["census"] >= alloc[u]).mean()) for u in UNIT_NAMES}
    wdiv = sum(DIVERSION_COST[u] * div[u] for u in UNIT_NAMES)
    return dict(diverted_window=div, total_diverted=sum(div.values()), weighted_diverted=wdiv,
                occupancy=occ, pct_days_full=full, cost=wdiv)

kp_cur = bed_kpis(alloc_cur); kp_opt = bed_kpis(alloc_opt)
res["bed_allocation"] = dict(current=alloc_cur, optimised=alloc_opt, lp_solution=b_lp.round(2).tolist(),
                             kpi_current=kp_cur, kpi_optimised=kp_opt,
                             bounds=dict(lo=lo, hi=hi), weights=dict(zip(UNIT_NAMES, w)),
                             diversions_avoided_window=kp_cur["total_diverted"] - kp_opt["total_diverted"],
                             value_window=kp_cur["cost"] - kp_opt["cost"])

# =========================== C. Equipment ===========================
cen = np.stack([base[u]["census"] for u in UNIT_NAMES], axis=-1)                  # (R,T,3)
vp = np.array([UNITS[u]["vent_p"] for u in UNIT_NAMES]); pp = np.array([UNITS[u]["pump_per_pt"] for u in UNIT_NAMES])
vent_demand = rng.binomial(cen, vp).sum(-1).ravel()
pump_by_unit = rng.poisson(cen * pp)                                              # (R,T,3)
pump_demand = pump_by_unit.sum(-1).ravel()

fleet = np.arange(8, 45)
p_out = np.array([(vent_demand > f).mean() for f in fleet])
daily_cost = fleet * VENT_ANNUAL_COST / 365 + p_out * VENT_STOCKOUT_COST
f_opt = int(fleet[np.argmin(daily_cost)])
f_99 = int(fleet[np.argmax(p_out <= 0.01)])
vent = dict(current_fleet=36, cost_optimal_fleet=f_opt, fleet_for_1pct_stockout=f_99,
            mean_demand=float(vent_demand.mean()), p99_demand=float(np.percentile(vent_demand, 99)),
            util_current=float(vent_demand.mean() / 36), util_at_optimal=float(vent_demand.mean() / f_opt),
            stockout_prob_at_optimal=float(p_out[fleet == f_opt][0]),
            annual_saving=float((36 - f_opt) * VENT_ANNUAL_COST),
            curve=dict(fleet=fleet.tolist(), p_stockout=p_out.round(5).tolist(), daily_cost=daily_cost.round(1).tolist()))

pf = np.arange(250, 480)
p_pool = np.array([(pump_demand > f).mean() for f in pf])
pool_1 = int(pf[np.argmax(p_pool <= 0.01)])
ded = 0; ded_detail = {}
for i, u in enumerate(UNIT_NAMES):
    d = pump_by_unit[:, :, i].ravel()
    fs = np.arange(0, 400)
    f = int(fs[np.argmax(np.array([(d > q).mean() for q in fs]) <= 0.01)])
    ded += f; ded_detail[u] = f
pump = dict(current_pool=340, util_current=float(pump_demand.mean() / 340), stockout_prob_current=float((pump_demand > 340).mean()),
            pooled_fleet_for_1pct=pool_1, dedicated_fleets_for_1pct=ded_detail, dedicated_total=ded,
            pooling_saving_units=int(ded - pool_1), pooling_saving_annual=float((ded - pool_1) * PUMP_ANNUAL_COST),
            extra_units_vs_current=int(pool_1 - 340), mean_demand=float(pump_demand.mean()),
            curve=dict(fleet=pf.tolist(), p_stockout=p_pool.round(5).tolist()))
res["equipment"] = dict(ventilators=vent, pumps=pump)
res["assumptions"] = dict(core_rn_shift_cost=CORE_RN_SHIFT_COST, agency_rn_shift_cost=AGENCY_RN_SHIFT_COST, sick_rate=SICK_RATE,
                          diversion_cost=DIVERSION_COST, vent_annual_cost=VENT_ANNUAL_COST, vent_stockout_cost=VENT_STOCKOUT_COST,
                          pump_annual_cost=PUMP_ANNUAL_COST, forecast_window=f"{d_future[0].date()} to {d_future[-1].date()}", runs=RUNS)
json.dump(res, open(OUT / "optimization_results.json", "w"), indent=2)

print("=== RN roster (holdout) ===")
for k in ("current_holdout", "optimised_holdout"):
    r = res["rn_roster"][k]; print(k, {a: round(v, 1) if isinstance(v, float) else v for a, v in r.items() if a != "coverage"})
print("x_opt (staff by days-off pattern):", res["rn_roster"]["x_opt"], "saving/wk $%.0f, window $%.0f" % (res["rn_roster"]["saving_per_week"], res["rn_roster"]["saving_window"]))
print("=== Beds ===", alloc_cur, "->", alloc_opt, "LP", b_lp.round(2))
print("diversions in window: current %.0f -> opt %.0f | cost $%.0f -> $%.0f | value $%.0f" % (
    kp_cur["total_diverted"], kp_opt["total_diverted"], kp_cur["cost"], kp_opt["cost"], res["bed_allocation"]["value_window"]))
print("=== Vent ===", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in vent.items() if k != "curve"})
print("=== Pump ===", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pump.items() if k != "curve"})
