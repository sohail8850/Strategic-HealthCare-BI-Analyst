"""Shared configuration and the patient-flow simulation engine.

Everything in this project is SYNTHETIC. Parameters are plausible for a
~280-bed general acute-care hospital but are assumptions, not real data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

START, END = "2023-01-01", "2026-09-30"
DATES = pd.date_range(START, END, freq="D")

# --- Unit configuration (assumed) -------------------------------------------
# beds: staffed beds | los_mean/sigma: lognormal length of stay (days)
# base: mean daily bed requests | ratio: patients per RN per shift
# dow_amp: weekday/weekend swing (elective-driven units swing more)
UNITS = {
    "ICU":       dict(beds=28,  los_mean=4.4, los_sigma=0.75, base=5.4,  ratio=2, dow_amp=0.05, vent_p=0.42, pump_per_pt=3.0, weight=3.0),
    "Step-Down": dict(beds=48,  los_mean=3.9, los_sigma=0.60, base=10.1, ratio=3, dow_amp=0.08, vent_p=0.04, pump_per_pt=1.8, weight=2.0),
    "Med-Surg":  dict(beds=204, los_mean=4.7, los_sigma=0.70, base=35.5, ratio=5, dow_amp=0.14, vent_p=0.00, pump_per_pt=0.9, weight=1.0),
}
UNIT_NAMES = list(UNITS)
TOTAL_BEDS = sum(u["beds"] for u in UNITS.values())

# --- Cost assumptions (USD, illustrative) ------------------------------------
CORE_RN_SHIFT_COST = 560      # loaded cost of a 12-h core RN shift
AGENCY_RN_SHIFT_COST = 980    # agency / premium-pay 12-h shift
SICK_RATE = 0.06              # share of scheduled core shifts lost to sick calls
DIVERSION_COST = {"ICU": 12000, "Step-Down": 6000, "Med-Surg": 3000}  # contribution lost per diverted request, by unit
VENT_ANNUAL_COST = 14000      # annual ownership + maintenance per ventilator
VENT_STOCKOUT_COST = 25000    # penalty per stock-out day (delayed care, emergency rental)
PUMP_ANNUAL_COST = 900        # annual ownership + maintenance per infusion pump

LOS_MAX = 40
DOW_PROFILE = np.array([1.00, 0.98, 0.95, 0.93, 0.90, 0.72, 0.70])  # Mon..Sun, unit-scaled below


def dow_factor(dates, amp):
    """Day-of-week multiplier with mean 1, scaled by `amp` (0 = flat)."""
    raw = DOW_PROFILE[np.asarray(dates.dayofweek)]
    raw = raw / DOW_PROFILE.mean()
    return 1 + (raw - 1) * (amp / 0.14)


def season_factor(dates, amp=0.12):
    doy = np.asarray(dates.dayofyear)
    return 1 + amp * np.cos(2 * np.pi * (doy - 15) / 365.25)


def lognormal_params(mean, sigma):
    mu = np.log(max(mean, 0.5)) - sigma ** 2 / 2
    return mu, sigma


def run_engine(lam, unit, R, rng, warm_adm=None, beds=None, los_mean=None,
               dispersion=0.0036, cap=True):
    """Vectorised patient-flow simulator for one unit.

    lam      : (T,) or (R,T) mean daily bed requests for the future window
    warm_adm : (W,) already-admitted arrivals before the window (known history)
    cap      : if False, beds are unlimited -> returned census is *bed demand*
    Returns dict of (R,T) arrays over the future window.
    """
    cfg = UNITS[unit]
    beds = cfg["beds"] if beds is None else beds
    los_mean = cfg["los_mean"] if los_mean is None else los_mean
    mu, sg = lognormal_params(los_mean, cfg["los_sigma"])
    lam = np.atleast_2d(np.asarray(lam, float))
    if lam.shape[0] == 1:
        lam = np.repeat(lam, R, axis=0)
    T = lam.shape[1]
    W = 0 if warm_adm is None else len(warm_adm)
    k = 1.0 / max(dispersion, 1e-9)
    mult = rng.gamma(k, 1.0 / k, size=lam.shape)
    req = rng.poisson(lam * mult)
    if W:
        req_all = np.concatenate([np.repeat(np.asarray(warm_adm)[None, :], R, 0), req], axis=1)
    else:
        req_all = req
    tot = W + T
    dis = np.zeros((R, tot + LOS_MAX + 2), dtype=np.int32)
    occ = np.zeros(R, dtype=np.int32)
    census = np.zeros((R, tot), dtype=np.int32)
    admitted = np.zeros((R, tot), dtype=np.int32)
    discharged = np.zeros((R, tot), dtype=np.int32)
    ridx = np.arange(R)
    for t in range(tot):
        d = dis[:, t]
        occ = occ - d
        discharged[:, t] = d
        a = req_all[:, t]
        if cap and t >= W:
            adm = np.minimum(a, np.maximum(beds - occ, 0))
        else:
            adm = a
        admitted[:, t] = adm
        amax = int(adm.max())
        if amax > 0:
            los = np.clip(np.rint(rng.lognormal(mu, sg, size=(R, amax))), 1, LOS_MAX).astype(int)
            mask = np.arange(amax)[None, :] < adm[:, None]
            rr = np.broadcast_to(ridx[:, None], los.shape)[mask]
            np.add.at(dis, (rr, (t + los)[mask]), 1)
        occ = occ + adm
        census[:, t] = occ
    sl = slice(W, tot)
    return dict(requests=req_all[:, sl], admitted=admitted[:, sl], diverted=req_all[:, sl] - admitted[:, sl],
                census=census[:, sl], discharges=discharged[:, sl], beds=beds)


def required_rn_shifts(census_by_unit):
    """RN shifts needed per day (day+night) from unit censuses, by ratio."""
    total = 0
    for u, c in census_by_unit.items():
        total = total + 2 * np.ceil(c / UNITS[u]["ratio"])
    return total
