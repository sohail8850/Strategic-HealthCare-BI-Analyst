"""Collect every result into one JSON bundle for the project page."""
import json
import math
import pandas as pd
from common import *


def clean(o):
    """Make NaN / numpy types JSON-safe."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


rd = lambda f: pd.read_csv(OUT / f).astype(object).where(lambda x: x.notna(), None).to_dict("records")
R = json.load(open(OUT / "policy_results.json"))
S = json.load(open(OUT / "data_summary.json"))
M = json.load(open(OUT / "model_summary.json"))

bundle = {
    "summary": {**S, **M},
    "models": rd("model_comparison_test.csv"),
    "cal": rd("calibration_deciles.csv"),
    "imp": rd("feature_importance.csv"),
    "segments": rd("segment_rates.csv"),
    "policies": R["policies"],
    "cap_sweep": R["cap_sweep"],
    "thr": R["threshold_sweep"],
    "sens": R["sensitivity"],
    "equity": R["equity"],
    "assumptions": R["assumptions"],
    "pstar": R["break_even_risk"],
    "cap": R["cap"],
    "flags_pd": R["flags_per_provider_day"],
    "share_pd_flag": R["share_pd_with_flag"],
    "test_pd": R["test_provider_days"],
    "annual_net_model": R["annual_net_model"],
}
json.dump(clean(bundle), open(DATA / "dashboard_bundle.json", "w"), separators=(",", ":"))
print("bundle written", round((DATA / "dashboard_bundle.json").stat().st_size / 1024), "KB")
