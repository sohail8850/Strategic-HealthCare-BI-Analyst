"""Render index.html (self-contained interactive project page) from the data bundle."""
import json
import re
from common import *

D = json.load(open(DATA / "dashboard_bundle.json"))
S = D["summary"]
pol = {p["policy"]: p for p in D["policies"]}
flat, gut, mod = pol["Flat padding (2 at random)"], pol["Front-desk rule (prior miss, max 2)"], pol["Model at break-even risk (max 2)"]
models = {m["model"]: m for m in D["models"]}
best = models[S["chosen"]]
rule = models["Front-desk rule (miss history)"]
cal1 = D["cal"][0]
cap = {c["cap"]: c for c in D["cap_sweep"]}
thr = D["thr"]
val_opt = max(thr, key=lambda r: r["val_net"])["threshold"]
test_opt = max(thr, key=lambda r: r["test_net"])["threshold"]
eq = {(e["group"], e["level"]): e for e in D["equity"]}
seg = {(s["segment"], s["level"]): s["rate"] for s in D["segments"]}
sens = D["sens"]

pct0 = lambda v: f"{v*100:.0f}%"
pct1 = lambda v: f"{v*100:.1f}%"
pctx = lambda v: re.sub(r"\.0%$", "%", pct1(v))
money = lambda v: f"{v:,.0f}"
kusd = lambda v: f"${v/1e3:.1f}K"
prec = [e["no_show_if_flagged"] for e in D["equity"] if e["no_show_if_flagged"] is not None]
diff = mod["net_value"] - gut["net_value"]
ann_diff = round(diff / 100 * N_PROVIDERS * PROVIDER_DAYS_PER_YEAR, -2)

tok = {
    "__DATA__": json.dumps(D, separators=(",", ":")),
    "__N_APPTS__": f'{S["n_appointments"]:,}', "__N_PATIENTS__": f'{S["n_patients"]:,}',
    "__BASE__": pct1(S["no_show_rate"]), "__BASE_TEST__": pct1(S["base_rate_test"]),
    "__PD_TEST__": f'{D["test_pd"]:,}',
    "__NET_YEAR__": kusd(D["annual_net_model"]),
    "__FLAT_NET_NEG__": "−" + kusd(abs(flat["net_value"])) if flat["net_value"] < 0 else kusd(flat["net_value"]),
    "__LIFT__": f'{best["lift_top10"]:.1f}', "__TOP10__": pct1(best["precision_top10"]),
    "__PSTAR__": pct0(D["pstar"]), "__IDLE__": str(IDLE_SLOT_COST), "__COLL__": str(COLLISION_COST),
    "__SHOW__": pct0(WAITLIST_SHOW), "__WAITMIN__": str(WAIT_MIN_PER_COLLISION),
    "__AUC__": f'{best["roc_auc"]:.2f}', "__AUC_RULE__": f'{rule["roc_auc"]:.2f}', "__ORACLE__": f'{S["oracle_test_auc"]:.2f}',
    "__CAL1_P__": pct1(cal1["predicted"]), "__CAL1_O__": pct1(cal1["observed"]),
    "__REC_FLAT__": f'{flat["recovered"]:.0f}', "__COL_FLAT__": f'{flat["collisions"]:.0f}', "__HIT_FLAT__": pct0(flat["hit_rate"]),
    "__REC_MODEL__": f'{mod["recovered"]:.0f}', "__COL_MODEL__": f'{mod["collisions"]:.0f}', "__HIT_MODEL__": pct0(mod["hit_rate"]),
    "__MODEL_NET__": money(mod["net_value"]), "__GUT_NET__": money(gut["net_value"]),
    "__DIFF__": money(diff), "__DIFF_PCT__": pct0(diff / gut["net_value"]), "__ANN_DIFF__": money(ann_diff),
    "__VAL_OPT__": pctx(val_opt), "__TEST_OPT__": pctx(test_opt),
    "__MARG_12__": f'{(cap[2]["net_value"]-cap[1]["net_value"])/(cap[2]["extra_bookings"]-cap[1]["extra_bookings"]):.0f}',
    "__MARG_34__": f'{(cap[4]["net_value"]-cap[3]["net_value"])/(cap[4]["extra_bookings"]-cap[3]["extra_bookings"]):.0f}',
    "__SHARE_PD__": pct0(D["share_pd_flag"]), "__FLAGS_PD__": f'{D["flags_pd"]:.1f}',
    "__SENS_N__": str(len(sens)),
    "__SENS_MODEL__": str(sum(r["model"] > 0 for r in sens)), "__SENS_GUT__": str(sum(r["gut"] > 0 for r in sens)),
    "__SENS_FLAT__": str(sum(r["flat"] > 0 for r in sens)),
    "__MCAID_A__": pct0(eq[("Insurance", "Medicaid")]["share_of_appts"]), "__MCAID_F__": pct0(eq[("Insurance", "Medicaid")]["share_flagged"]),
    "__YOUNG_A__": pct0(eq[("Age band", "18-30")]["share_of_appts"]), "__YOUNG_F__": pct0(eq[("Age band", "18-30")]["share_flagged"]),
    "__MEDICARE_F__": pct0(eq[("Insurance", "Medicare")]["share_flagged"]),
    "__PREC_LO__": pct0(min(prec)), "__PREC_HI__": pct0(max(prec)),
    "__NEWPT__": pct0(seg[("Appointment type", "New patient")]), "__YOUNG__": pct0(seg[("Age band", "18-30")]),
    "__MCAID__": pct0(seg[("Insurance", "Medicaid")]), "__SELF__": pct0(seg[("Insurance", "Self-pay")]),
}
html = open(ROOT / "src" / "page_template.html", encoding="utf-8").read()
for k in sorted(tok, key=len, reverse=True):
    html = html.replace(k, tok[k])
left = re.findall(r"__[A-Z0-9_]+__", html.replace("__proto__", ""))
assert not left, left[:5]
(ROOT / "index.html").write_text(html, encoding="utf-8")
print("page written", len(html) // 1024, "KB")
