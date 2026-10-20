"""Generate two years of synthetic outpatient appointments for a six-provider clinic.

Order of generation matters: each appointment's no-show outcome is drawn from
its true risk first, and the patient-history features (prior visits, prior
misses) are then computed ONLY from appointments that had already happened by
the booking date. That keeps the model honest: it never sees the future.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import *

rng = np.random.default_rng(SEED)

# ---- patients ---------------------------------------------------------------
age = np.clip(rng.normal(46, 19, N_PATIENTS), 18, 92).round().astype(int)
ins_names = np.array(["Commercial", "Medicare", "Medicaid", "Self-pay"])
ins = np.empty(N_PATIENTS, dtype=object)
for i, a in enumerate(age):
    p = [0.04, 0.88, 0.05, 0.03] if a >= 65 else [0.62, 0.02, 0.24, 0.12]
    ins[i] = rng.choice(ins_names, p=p)
distance = np.clip(rng.lognormal(np.log(4.5), 0.8, N_PATIENTS), 0.5, 60).round(1)
frailty = rng.normal(0, 0.95, N_PATIENTS)       # unobserved reliability: the source of "history" signal
visit_w = rng.gamma(2.0, 0.5, N_PATIENTS)
visit_w = visit_w / visit_w.sum()

# ---- schedule grid ----------------------------------------------------------
days = pd.bdate_range(START, END)
rows = []
for d in days:
    for pr in range(N_PROVIDERS):
        booked = int(np.clip(round(rng.normal(17.6, 1.8)), 10, SLOTS_PER_DAY))
        slots = np.sort(rng.choice(SLOTS_PER_DAY, size=booked, replace=False))
        for s in slots:
            rows.append((d, pr + 1, int(s)))
df = pd.DataFrame(rows, columns=["date", "provider", "slot"])
n = len(df)
pid = rng.choice(N_PATIENTS, size=n, p=visit_w)
df["patient_id"] = pid
df["age"] = age[pid]
df["insurance"] = ins[pid]
df["distance_km"] = distance[pid]
df["hour"] = 8 + df["slot"] * SLOT_MINUTES / 60.0
df["weekday"] = df["date"].dt.dayofweek
df["weekday_name"] = df["date"].dt.day_name()

types = np.array(["New patient", "Follow-up", "Procedure", "Telehealth"])
df["appt_type"] = rng.choice(types, size=n, p=[0.20, 0.52, 0.10, 0.18])
median_lead = df["appt_type"].map({"New patient": 21, "Follow-up": 10, "Procedure": 28, "Telehealth": 5}).to_numpy()
df["lead_days"] = np.clip(np.round(median_lead * rng.lognormal(0, 0.9, n)), 1, 120).astype(int)
df["booking_date"] = df["date"] - pd.to_timedelta(df["lead_days"], unit="D")

# ---- true no-show risk ------------------------------------------------------
def true_logit(d, z):
    a = d["age"].to_numpy()
    h = d["hour"].to_numpy()
    lt = d["lead_days"].to_numpy()
    x = 0.55 * np.log1p(lt / 7.0)
    x += np.where(a < 31, 0.45, np.where(a >= 65, -0.55, 0.0))
    x += d["appt_type"].map({"New patient": 0.25, "Follow-up": 0.0, "Procedure": -0.50, "Telehealth": 0.35}).to_numpy()
    x += d["weekday"].map({0: 0.15, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.25}).to_numpy()
    x += np.where(h < 8.75, 0.30, 0.0) + np.where(h >= 16.0, 0.35, 0.0) + np.where((h >= 12) & (h < 13), 0.10, 0.0)
    x += d["insurance"].map({"Commercial": 0.0, "Medicare": -0.10, "Medicaid": 0.45, "Self-pay": 0.50}).to_numpy()
    x += 0.018 * np.minimum(d["distance_km"].to_numpy(), 40)
    # interaction: a long wait to a first visit loses patients faster
    x += np.where((d["appt_type"] == "New patient").to_numpy() & (lt > 30), 0.35, 0.0)
    return x + z

z = frailty[pid]
lo, hi = -8.0, 4.0
for _ in range(60):                       # bisection: hit the target clinic-wide rate
    mid = (lo + hi) / 2
    p = 1 / (1 + np.exp(-(true_logit(df, z) + mid)))
    lo, hi = (mid, hi) if p.mean() < TARGET_NO_SHOW_RATE else (lo, mid)
true_p = 1 / (1 + np.exp(-(true_logit(df, z) + mid)))
df["no_show"] = (rng.random(n) < true_p).astype(int)
df["true_p"] = true_p                      # kept in memory only, never written to the CSV

# ---- point-in-time patient history (no future leakage) ----------------------
df = df.sort_values(["patient_id", "date", "provider", "slot"]).reset_index(drop=True)
prior_n = np.zeros(len(df), dtype=int)
prior_k = np.zeros(len(df), dtype=int)
since = np.full(len(df), 365.0)
last_missed = np.zeros(len(df), dtype=int)
d_days = df["date"].to_numpy("datetime64[D]").astype(np.int64)
b_days = df["booking_date"].to_numpy("datetime64[D]").astype(np.int64)
ns = df["no_show"].to_numpy()
starts = np.flatnonzero(np.r_[True, df["patient_id"].to_numpy()[1:] != df["patient_id"].to_numpy()[:-1]])
ends = np.r_[starts[1:], len(df)]
for s, e in zip(starts, ends):
    dd, bb, nn = d_days[s:e], b_days[s:e], ns[s:e]
    cum = np.r_[0, np.cumsum(nn)]
    idx = np.searchsorted(dd, bb, side="left")      # appointments that happened before booking
    prior_n[s:e] = idx
    prior_k[s:e] = cum[idx]
    has = idx > 0
    last = np.where(has, idx - 1, 0)
    since[s:e] = np.where(has, np.minimum(bb - dd[last], 365), 365)
    last_missed[s:e] = np.where(has, nn[last], 0)
df["prior_appts"] = prior_n
df["prior_no_shows"] = prior_k
df["prior_rate"] = ((prior_k + 2 * PRIOR_RATE) / (prior_n + 2)).round(4)
df["days_since_last"] = since.astype(int)
df["last_missed"] = last_missed

df = df.sort_values(["date", "provider", "slot"]).reset_index(drop=True)
df.insert(0, "appt_id", np.arange(1, len(df) + 1))
df["date"] = df["date"].dt.strftime("%Y-%m-%d")
df["booking_date"] = df["booking_date"].dt.strftime("%Y-%m-%d")

cols = ["appt_id", "date", "provider", "slot", "hour", "weekday_name", "patient_id", "age", "insurance",
        "distance_km", "appt_type", "lead_days", "prior_appts", "prior_no_shows", "prior_rate",
        "days_since_last", "last_missed", "no_show"]
df[cols].to_csv(DATA / "appointments.csv", index=False)

# oracle ceiling: the AUC a model would reach if it knew every patient's true risk,
# including the unobserved reliability term. No real model can beat this.
test_mask = (df["date"] > VAL_END).to_numpy()
oracle_auc = float(roc_auc_score(df.loc[test_mask, "no_show"], df.loc[test_mask, "true_p"]))
print("appointments:", len(df), "| no-show rate:", round(df["no_show"].mean(), 4),
      "| patients:", df["patient_id"].nunique(), "| provider-days:", df.groupby(["date", "provider"]).ngroups,
      "| oracle test AUC:", round(oracle_auc, 3))
json.dump({"n_appointments": int(len(df)), "no_show_rate": float(df["no_show"].mean()),
           "oracle_test_auc": oracle_auc,
           "n_patients": int(df["patient_id"].nunique()),
           "provider_days": int(df.groupby(["date", "provider"]).ngroups),
           "start": START, "end": END}, open(OUT / "data_summary.json", "w"), indent=1)
