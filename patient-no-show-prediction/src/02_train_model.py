"""Train and compare no-show models with a strict time-based split.

Train: Oct 2024 - Dec 2025   Validation: Jan - Mar 2026   Test: Apr - Sep 2026
The model never sees a later date than the one it is scored on.
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import *

df = pd.read_csv(DATA / "appointments.csv")
df["split"] = np.where(df["date"] <= TRAIN_END, "train", np.where(df["date"] <= VAL_END, "val", "test"))
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
print("rows  train/val/test:", len(tr), len(va), len(te),
      "| no-show rate:", [round(x.no_show.mean(), 3) for x in (tr, va, te)])


def pipe(estimator, scale):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("num", num, NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("est", estimator)])


models = {
    "Logistic regression": pipe(LogisticRegression(C=1.0, max_iter=2000), scale=True),
    "Gradient boosting": pipe(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                                              l2_regularization=1.0, random_state=SEED), scale=False),
}
for m in models.values():
    m.fit(tr[FEATURES], tr["no_show"])


def top_share(y, s, frac):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(np.asarray(y)[idx]))


def evaluate(name, y, s, is_prob=True):
    r = dict(model=name, roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
             precision_top10=top_share(y, s, 0.10), precision_top20=top_share(y, s, 0.20))
    r["lift_top10"] = r["precision_top10"] / float(np.mean(y))
    if is_prob:
        r["brier"] = brier_score_loss(y, s)
        r["log_loss"] = log_loss(y, np.clip(s, 1e-6, 1 - 1e-6))
    return r


rows_va, rows_te = [], []
for name, m in models.items():
    rows_va.append(evaluate(name, va.no_show, m.predict_proba(va[FEATURES])[:, 1]))
    rows_te.append(evaluate(name, te.no_show, m.predict_proba(te[FEATURES])[:, 1]))
# the front-desk heuristic: rank by the patient's own smoothed miss history
rows_va.append(evaluate("Front-desk rule (miss history)", va.no_show, va["prior_rate"].to_numpy(), False))
rows_te.append(evaluate("Front-desk rule (miss history)", te.no_show, te["prior_rate"].to_numpy(), False))
rows_te.append(dict(model="Clinic average (no model)", roc_auc=0.5, pr_auc=float(te.no_show.mean()),
                    precision_top10=float(te.no_show.mean()), precision_top20=float(te.no_show.mean()),
                    lift_top10=1.0, brier=brier_score_loss(te.no_show, np.full(len(te), tr.no_show.mean())),
                    log_loss=log_loss(te.no_show, np.full(len(te), tr.no_show.mean()))))
comp_va, comp_te = pd.DataFrame(rows_va), pd.DataFrame(rows_te)
comp_te.to_csv(OUT / "model_comparison_test.csv", index=False)
comp_va.to_csv(OUT / "model_comparison_validation.csv", index=False)
print(comp_te.round(3).to_string(index=False))

# choose on VALIDATION; prefer the simpler model unless the complex one is clearly better
v = comp_va.set_index("model")
chosen = "Logistic regression"
if v.loc["Gradient boosting", "roc_auc"] - v.loc["Logistic regression", "roc_auc"] > 0.005:
    chosen = "Gradient boosting"
best = models[chosen]
print("chosen model:", chosen)

df["risk"] = best.predict_proba(df[FEATURES])[:, 1]
df[["appt_id", "split", "risk"]].to_csv(OUT / "predictions.csv", index=False)
te = df[df.split == "test"].copy()
va = df[df.split == "val"].copy()

# calibration by risk decile on the test set
te["decile"] = pd.qcut(te["risk"].rank(method="first"), 10, labels=False) + 1
cal = te.groupby("decile").agg(n=("no_show", "size"), predicted=("risk", "mean"), observed=("no_show", "mean")).reset_index()
cal["decile"] = 11 - cal["decile"]          # decile 1 = highest predicted risk
cal = cal.sort_values("decile")
cal.to_csv(OUT / "calibration_deciles.csv", index=False)

# permutation importance on the test set (drop in AUC when a feature is shuffled)
pi = permutation_importance(best, te[FEATURES], te["no_show"], scoring="roc_auc", n_repeats=8, random_state=SEED)
labels = {"age": "Age", "distance_km": "Distance to clinic", "lead_days": "Booking lead time",
          "prior_appts": "Prior visit count", "prior_rate": "Prior miss rate", "days_since_last": "Days since last visit",
          "last_missed": "Last visit missed", "hour": "Time of day", "appt_type": "Appointment type",
          "insurance": "Insurance type", "weekday_name": "Day of week"}
imp = pd.DataFrame({"feature": [labels[f] for f in FEATURES], "auc_drop": pi.importances_mean,
                    "sd": pi.importances_std}).sort_values("auc_drop", ascending=False)
imp.to_csv(OUT / "feature_importance.csv", index=False)
print(imp.round(4).to_string(index=False))

# where the risk lives: observed no-show rate by segment, on the full history
d = df.copy()
d["Lead time"] = pd.cut(d["lead_days"], [0, 3, 7, 14, 30, 60, 200],
                        labels=["1-3 days", "4-7", "8-14", "15-30", "31-60", "61+"])
d["Time of day"] = pd.cut(d["hour"], [7.9, 9.9, 11.9, 13.9, 15.9, 18.1],
                          labels=["8-10", "10-12", "12-14", "14-16", "16-18"])
d["Age band"] = pd.cut(d["age"], [17, 30, 45, 64, 100], labels=["18-30", "31-45", "46-64", "65+"])
d["Prior misses"] = pd.cut(d["prior_no_shows"], [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
seg = []
for col in ["Lead time", "Time of day", "Age band", "Prior misses", "appt_type", "insurance", "weekday_name"]:
    g = d.groupby(col, observed=True)["no_show"].agg(["size", "mean"]).reset_index()
    g.columns = ["level", "n", "rate"]
    g.insert(0, "segment", {"appt_type": "Appointment type", "insurance": "Insurance", "weekday_name": "Weekday"}.get(col, col))
    seg.append(g)
seg = pd.concat(seg)
seg["level"] = seg["level"].astype(str)
seg.to_csv(OUT / "segment_rates.csv", index=False)

# logistic odds ratios (interpretable view of the chosen / simple model)
lr = models["Logistic regression"]
names = lr.named_steps["pre"].get_feature_names_out()
coef = lr.named_steps["est"].coef_[0]
pd.DataFrame({"term": names, "coef": coef, "odds_ratio": np.exp(coef)}).to_csv(OUT / "logistic_terms.csv", index=False)

json.dump({"chosen": chosen, "n_train": int((df.split == 'train').sum()), "n_val": int((df.split == 'val').sum()),
           "n_test": int((df.split == 'test').sum()),
           "base_rate_train": float(tr.no_show.mean()), "base_rate_test": float(te.no_show.mean())},
          open(OUT / "model_summary.json", "w"), indent=1)
