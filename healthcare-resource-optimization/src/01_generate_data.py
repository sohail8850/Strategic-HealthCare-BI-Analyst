"""Generate 3.75 years of synthetic daily hospital operations data."""
import numpy as np, pandas as pd
from common import *

rng = np.random.default_rng(2026)
T = len(DATES)

# Unforecastable demand shocks (respiratory waves, local outbreaks): (centre date, width days, amplitude)
EVENTS = [("2023-01-10", 18, 0.14), ("2023-07-22", 10, 0.06), ("2023-12-20", 20, 0.17),
          ("2024-02-05", 14, 0.10), ("2024-09-15", 9, 0.07), ("2024-12-28", 22, 0.19),
          ("2025-02-08", 12, 0.09), ("2025-08-12", 8, 0.06), ("2025-12-18", 20, 0.16),
          ("2026-03-02", 12, 0.08), ("2026-08-05", 9, 0.07)]
shock = np.zeros(T)
x = np.arange(T)
for d, w, a in EVENTS:
    c = (pd.Timestamp(d) - pd.Timestamp(START)).days
    shock += a * np.exp(-0.5 * ((x - c) / (w / 2.2)) ** 2)

trend = 1 + 0.03 * (x / 365.25)
PRE = 60  # burn-in days so the hospital starts the record already "full"
EXT = pd.date_range(pd.Timestamp(START) - pd.Timedelta(days=PRE), END, freq="D")
pre = lambda a: np.concatenate([np.repeat(a[0], PRE), a])
rows_units, true_lam = [], {}
for u in UNIT_NAMES:
    cfg = UNITS[u]
    lam = cfg["base"] * season_factor(EXT) * dow_factor(EXT, cfg["dow_amp"]) * pre(trend) * (1 + pre(shock))
    true_lam[u] = lam
    r = run_engine(lam, u, 1, rng)
    sl = slice(PRE, None)
    df = pd.DataFrame({
        "date": DATES, "unit": u, "beds": cfg["beds"],
        "bed_requests": r["requests"][0][sl], "admitted": r["admitted"][0][sl],
        "diverted": r["diverted"][0][sl], "discharges": r["discharges"][0][sl],
        "census": r["census"][0][sl],
    })
    df["occupancy"] = (df["census"] / df["beds"]).round(4)
    rows_units.append(df)
units = pd.concat(rows_units, ignore_index=True)
units.to_csv(DATA / "unit_daily.csv", index=False)

# --- Staffing: roster built on the *average* requirement, flat across the week ----
wide = units.pivot(index="date", columns="unit", values="census")
required = required_rn_shifts({u: wide[u].values for u in UNIT_NAMES})
roster_flat = np.round(np.mean(required) * 1.03)      # budget view: average need + 3%
core_available = rng.binomial(int(roster_flat), 1 - SICK_RATE, size=T)
agency = np.maximum(required - core_available, 0)
idle = np.maximum(core_available - required, 0)
staff = pd.DataFrame({"date": DATES, "required_rn_shifts": required.astype(int),
                      "scheduled_core_shifts": int(roster_flat), "core_shifts_worked": core_available,
                      "agency_shifts": agency.astype(int), "idle_core_shifts": idle.astype(int)})
staff["cost_usd"] = staff["core_shifts_worked"] * CORE_RN_SHIFT_COST + staff["agency_shifts"] * AGENCY_RN_SHIFT_COST
staff.to_csv(DATA / "staffing_daily.csv", index=False)

# --- Equipment: ventilators (fleet 36) and infusion pumps (pool 340) -----------------
vent_fleet, pump_fleet = 36, 340
vent_use = np.zeros(T, int); pump_use = np.zeros(T, int)
for u in UNIT_NAMES:
    c = wide[u].values
    vent_use += rng.binomial(c, UNITS[u]["vent_p"])
    pump_use += rng.poisson(c * UNITS[u]["pump_per_pt"])
eq = pd.DataFrame({"date": DATES, "vent_fleet": vent_fleet, "vents_in_use": vent_use,
                   "pump_fleet": pump_fleet, "pumps_in_use": pump_use})
eq["vent_util"] = (eq["vents_in_use"] / vent_fleet).round(4)
eq["pump_util"] = (eq["pumps_in_use"] / pump_fleet).round(4)
eq["vent_stockout"] = (eq["vents_in_use"] > vent_fleet).astype(int)
eq["pump_stockout"] = (eq["pumps_in_use"] > pump_fleet).astype(int)
eq.to_csv(DATA / "equipment_daily.csv", index=False)

print(units.groupby("unit").agg(avg_occ=("occupancy", "mean"), diverted=("diverted", "sum"),
                                max_occ=("occupancy", "max"), mean_los_proxy=("census", "mean")))
tot = wide.sum(axis=1) / TOTAL_BEDS
print("hospital avg occupancy %.3f | days >=90%%: %d of %d" % (tot.mean(), (tot >= 0.90).sum(), T))
print("RN required mean %.0f, roster %d | agency shifts/day %.1f, idle core/day %.1f" %
      (required.mean(), roster_flat, agency.mean(), idle.mean()))
print("vent util mean %.2f p99 %.2f stockout days %d | pump util mean %.2f stockout days %d" %
      (eq.vent_util.mean(), eq.vent_util.quantile(.99), eq.vent_stockout.sum(), eq.pump_util.mean(), eq.pump_stockout.sum()))
