"""Shared configuration for the Patient No-Show Prediction project.

Everything in this project is SYNTHETIC. The clinic, its patients and its costs
are assumptions chosen to be plausible for a mid-sized outpatient clinic, not
real data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SEED = 2026

# --- Clinic configuration (assumed) -----------------------------------------
START, END = "2024-10-01", "2026-09-30"      # two years of weekday schedules
N_PROVIDERS = 6
SLOTS_PER_DAY = 20                           # 30-minute slots, 08:00 to 18:00
SLOT_MINUTES = 30
N_PATIENTS = 7000
TARGET_NO_SHOW_RATE = 0.18                   # clinic-wide base rate (assumed)
PRIOR_RATE = 0.18                            # smoothing prior for patient history

# --- Time-based split -------------------------------------------------------
TRAIN_END = "2025-12-31"                     # train: Oct 2024 - Dec 2025
VAL_END = "2026-03-31"                       # validation: Jan - Mar 2026
                                             # test: Apr - Sep 2026

# --- Economics (USD, illustrative assumptions) ------------------------------
IDLE_SLOT_COST = 120        # contribution lost when a booked slot goes unused
COLLISION_COST = 45         # cost when two patients arrive for one slot:
                            # patient waiting plus clinician overtime pressure
WAITLIST_SHOW = 0.85        # chance an extra, short-notice patient actually attends
WAIT_MIN_PER_COLLISION = 20  # average wait for the patient who is seen second
BREAK_EVEN_RISK = COLLISION_COST / (IDLE_SLOT_COST + COLLISION_COST)
MAX_OVERBOOKS_PER_PROVIDER_DAY = 2
PROVIDER_DAYS_PER_YEAR = 260  # per provider

CATEGORICAL = ["appt_type", "insurance", "weekday_name"]
NUMERIC = ["age", "distance_km", "lead_days", "prior_appts", "prior_rate",
           "days_since_last", "last_missed", "hour"]
FEATURES = NUMERIC + CATEGORICAL
