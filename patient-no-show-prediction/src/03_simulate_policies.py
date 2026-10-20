"""Replay double-booking policies on held-out provider-days.

For each provider-day, a policy picks which booked appointments to double-book
against (adding a short-notice patient from a waitlist). Using the REAL outcomes
in the held-out window:
  * if the original patient no-shows, the extra patient fills an empty slot (recovered)
  * if the original patient attends, two patients arrive for one slot (collision)
  * the extra patient attends with probability WAITLIST_SHOW
Value of one extra booking = s * [ p*IDLE - (1-p)*COLLISION ], so it only pays when the
risk p exceeds the break-even risk  C / (I + C).
"""
import json
import numpy as np
import pandas as pd
from common import *

df = pd.read_csv(DATA / "appointments.csv").merge(pd.read_csv(OUT / "predictions.csv"), on="appt_id")
val, test = df[df.split == "val"], df[df.split == "test"]
S = WAITLIST_SHOW


def groups(d):
    return [g.sort_values("slot") for _, g in d.groupby(["date", "provider"], sort=True)]


def run(d, picker, idle=IDLE_SLOT_COST, coll=COLLISION_COST, s=S):
    """Totals per 100 provider-days for a policy. picker(g) -> positional indices to double-book,
    or ('flat', k) for random padding evaluated in expectation."""
    gs = groups(d)
    n_pd = len(gs)
    over = rec = col = 0.0
    booked = noshow = 0
    for g in gs:
        ns = g["no_show"].to_numpy()
        booked += len(g)
        noshow += ns.sum()
        sel = picker(g)
        if isinstance(sel, tuple):                       # random padding, in expectation
            k = min(sel[1], len(g))
            over += k
            rec += k * ns.mean() * s
            col += k * (1 - ns.mean()) * s
        else:
            sel = list(sel)
            over += len(sel)
            rec += ns[sel].sum() * s if sel else 0
            col += (1 - ns[sel]).sum() * s if sel else 0
    f = 100.0 / n_pd
    return dict(extra_bookings=over * f, recovered=rec * f, collisions=col * f,
                idle_slots=(noshow - rec) * f, wait_minutes=col * f * WAIT_MIN_PER_COLLISION,
                net_value=(rec * idle - col * coll) * f,
                hit_rate=(rec / (rec + col)) if rec + col > 0 else float("nan"),
                provider_days=n_pd, booked_per_pd=booked / n_pd, noshows_per_pd=noshow / n_pd)


def model_pick(t, cap):
    def f(g):
        r = g["risk"].to_numpy()
        order = np.argsort(-r, kind="stable")
        return [i for i in order[:cap] if r[i] >= t]
    return f


def gut_pick(cap):
    def f(g):
        k = g["prior_no_shows"].to_numpy()
        pr = g["prior_rate"].to_numpy()
        order = np.argsort(-pr, kind="stable")
        return [i for i in order[:cap] if k[i] >= 1]
    return f


none = lambda g: []
flat = lambda cap: (lambda g: ("flat", cap))
PSTAR = BREAK_EVEN_RISK
CAP = MAX_OVERBOOKS_PER_PROVIDER_DAY

R = {"break_even_risk": PSTAR, "cap": CAP,
     "assumptions": dict(idle_slot_cost=IDLE_SLOT_COST, collision_cost=COLLISION_COST, waitlist_show=WAITLIST_SHOW,
                         wait_min_per_collision=WAIT_MIN_PER_COLLISION, providers=N_PROVIDERS,
                         provider_days_per_year=PROVIDER_DAYS_PER_YEAR)}

# ---- headline policies on the test window ----------------------------------
pol = {
    "No double-booking": none,
    "Flat padding (2 at random)": flat(CAP),
    "Front-desk rule (prior miss, max 2)": gut_pick(CAP),
    "Model, top 2 regardless of risk": model_pick(0.0, CAP),
    "Model at break-even risk (max 2)": model_pick(PSTAR, CAP),
}
R["policies"] = [dict(policy=k, **run(test, v)) for k, v in pol.items()]
pd.DataFrame(R["policies"]).to_csv(OUT / "policy_results_test.csv", index=False)
for r in R["policies"]:
    print(f'{r["policy"]:42s} extra {r["extra_bookings"]:6.1f} rec {r["recovered"]:6.1f} col {r["collisions"]:6.1f} net ${r["net_value"]:8.0f} hit {r["hit_rate"]:.2f}')

# ---- cap sweep (model at break-even) ----------------------------------------
R["cap_sweep"] = [dict(cap=c, **run(test, model_pick(PSTAR, c))) for c in range(0, 5)]
pd.DataFrame(R["cap_sweep"]).to_csv(OUT / "cap_sweep_test.csv", index=False)

# ---- threshold sweep, validation and test ----------------------------------
ts = [round(x, 3) for x in np.arange(0.10, 0.651, 0.025)]
R["threshold_sweep"] = [dict(threshold=t, val_net=run(val, model_pick(t, CAP))["net_value"],
                             test_net=run(test, model_pick(t, CAP))["net_value"],
                             test_extra=run(test, model_pick(t, CAP))["extra_bookings"]) for t in ts]
pd.DataFrame(R["threshold_sweep"]).to_csv(OUT / "threshold_sweep.csv", index=False)

# ---- sensitivity to the economic assumptions --------------------------------
scen = [("Base case", {}),
        ("Collision cost $20", dict(coll=20)), ("Collision cost $90", dict(coll=90)),
        ("Idle slot $80", dict(idle=80)), ("Idle slot $160", dict(idle=160)),
        ("Waitlist shows 70%", dict(s=0.70)), ("Waitlist shows 95%", dict(s=0.95))]
sens = []
for name, kw in scen:
    idle, coll, s = kw.get("idle", IDLE_SLOT_COST), kw.get("coll", COLLISION_COST), kw.get("s", S)
    ps = coll / (idle + coll)
    row = dict(scenario=name, break_even=ps)
    for label, picker in [("flat", flat(CAP)), ("gut", gut_pick(CAP)), ("model", model_pick(ps, CAP))]:
        row[label] = run(test, picker, idle=idle, coll=coll, s=s)["net_value"]
    sens.append(row)
R["sensitivity"] = sens
pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)

# ---- who gets flagged? an equity check --------------------------------------
t = test.copy()
t["flag"] = t["risk"] >= PSTAR
t["Age band"] = pd.cut(t["age"], [17, 30, 45, 64, 100], labels=["18-30", "31-45", "46-64", "65+"]).astype(str)
eq = []
for col in ["insurance", "Age band"]:
    for lvl, g in t.groupby(col):
        eq.append(dict(group=col if col != "insurance" else "Insurance", level=lvl,
                       share_of_appts=len(g) / len(t), share_flagged=g["flag"].sum() / t["flag"].sum(),
                       flag_rate=g["flag"].mean(),
                       no_show_if_flagged=float(g.loc[g.flag, "no_show"].mean()) if g.flag.any() else float("nan"),
                       no_show_if_not=float(g.loc[~g.flag, "no_show"].mean())))
R["equity"] = eq
pd.DataFrame(eq).to_csv(OUT / "equity_check.csv", index=False)
gs = groups(test)
R["flags_per_provider_day"] = float(np.mean([(g["risk"] >= PSTAR).sum() for g in gs]))
R["share_pd_with_flag"] = float(np.mean([(g["risk"] >= PSTAR).any() for g in gs]))
R["test_provider_days"] = len(gs)
R["annual_net_model"] = next(p for p in R["policies"] if p["policy"].startswith("Model at"))["net_value"] / 100 * N_PROVIDERS * PROVIDER_DAYS_PER_YEAR
print("flags per provider-day:", round(R["flags_per_provider_day"], 2), "| share with >=1 flag:", round(R["share_pd_with_flag"], 2))
print("annualised net (model, 6 providers):", round(R["annual_net_model"]))
print(pd.DataFrame(R["cap_sweep"])[["cap", "extra_bookings", "recovered", "collisions", "net_value"]].round(1).to_string(index=False))
print(pd.DataFrame(sens).round(0).to_string(index=False))
print(pd.DataFrame(eq).round(3).to_string(index=False))
json.dump(R, open(OUT / "policy_results.json", "w"), indent=1, default=float)
