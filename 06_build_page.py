"""Render docs/index.html (self-contained interactive project page) from the data bundle."""
import json
from common import *

D = json.load(open(DATA / "dashboard_bundle.json"))
R, B, E, S = D["roster"], D["beds"], D["equipment"], D["scenarios"]
usd = lambda v: f"${v/1e6:.2f}M" if abs(v) >= 1e6 else f"${v/1e3:,.0f}K"
bt = {r["model"]: r for r in D["backtest"]}
glm = bt["Poisson GLM"]; naive = bt["Seasonal naive (4-wk same weekday)"]
v, p = E["ventilators"], E["pumps"]
sq, op, sg, sgo, sgf, los = S[0], S[1], S[2], S[3], S[4], S[5]

tok = {
 "__DATA__": json.dumps(D, separators=(",", ":")),
 "__WINDOW__": D["fc"]["window"],
 "__MEANOCC__": f'{D["fc"]["mean_occ"]*100:.0f}%',
 "__DAYS95__": f'{D["fc"]["days_ge95"]:.0f}',
 "__WAPE_W__": f'{glm["WAPE_weekly"]*100:.0f}%', "__WAPE_W_N__": f'{naive["WAPE_weekly"]*100:.0f}%',
 "__WAPE_D__": f'{glm["WAPE"]*100:.0f}%', "__WAPE_D_N__": f'{naive["WAPE"]*100:.0f}%',
 "__ROSTER_SAVE__": usd(R["saving_window"]),
 "__AG_CUR__": f'{R["current_holdout"]["agency_shifts_week"]:.0f}', "__AG_OPT__": f'{R["optimised_holdout"]["agency_shifts_week"]:.0f}',
 "__BED_SAVE__": usd(B["value_window"]),
 "__ALLOC_CUR__": " / ".join(str(B["current"][u]) for u in UNIT_NAMES),
 "__ALLOC_OPT__": " / ".join(str(B["optimised"][u]) for u in UNIT_NAMES),
 "__PUMP_P__": f'{p["stockout_prob_current"]*100:.0f}%', "__PUMP_NEED__": str(p["pooled_fleet_for_1pct"]),
 "__PUMP_EXTRA__": str(p["extra_units_vs_current"]), "__PUMP_POOLSAVE__": str(p["pooling_saving_units"]),
 "__VENT_OPT__": str(v["cost_optimal_fleet"]), "__VENT_P99__": f'{v["p99_demand"]:.0f}', "__VENT_UTIL__": f'{v["util_current"]*100:.0f}%',
 "__VENT_SAVE__": usd(v["annual_saving"]),
 "__LOS_SAVE__": usd(op["total_cost"] - los["total_cost"]),
 "__LOS_DIV__": f'{los["diverted"]:.0f}', "__SQ_DIV__": f'{op["diverted"]:.0f}',
 "__SURGE_DIV__": f'{sgo["diverted"]:.0f}', "__FLEX_DIV__": f'{sgf["diverted"]:.0f}',
 "__FLEX_DELTA__": usd(abs(sgf["total_cost"] - sgo["total_cost"])),
 "__SURGE_COST__": usd(sgo["total_cost"] - op["total_cost"]),
 "__TOTAL_SAVE__": usd(sq["total_cost"] - op["total_cost"]),
}
html = open(ROOT / "src" / "page_template.html").read()
for k, val in tok.items():
    html = html.replace(k, val)
assert "__" not in html.replace("__proto__", ""), [w for w in html.split() if "__" in w][:5]
(ROOT / "index.html").write_text(html)
print("page written", len(html) // 1024, "KB")
