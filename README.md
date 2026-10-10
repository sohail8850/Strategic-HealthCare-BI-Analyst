# Healthcare Resource Optimization: Beds, Staffing & Equipment

Forecasting, optimization and scenario modelling for a 280-bed hospital's beds, nurse staffing and equipment. The project asks one question: *where is capacity being wasted, and where is it about to run out this winter?*

![Project page preview](images/preview.png)

**Live page:** open `index.html` (interactive charts, light/dark theme, "View data" tables under every chart).

> **All data is synthetic.** A patient-flow simulator generated 3.75 years of daily operations (Jan 2023 – Sep 2026) for ICU, Step-Down and Med-Surg. Unit sizes, costs and length-of-stay figures are assumptions, so the results show how the method works, not what a real hospital should do.

## What it shows

| Skill | Where |
|---|---|
| **Forecasting** | Poisson regression (trend + annual cycle + weekday) backtested over 5 rolling origins against two naive baselines; forecast pushed through a patient-flow simulator for census with uncertainty bands |
| **Optimization** | Nurse roster (stochastic integer program), bed allocation across units (linear program), equipment fleet sizing (cost vs stock-out trade-off) |
| **Scenario modelling** | +10% demand surge, discharge programme (LOS −0.5 day), 12 flex beds, −10% demand, plus a sensitivity tornado |

## Headline results (winter 2026/27, 182 days)

- Winter occupancy forecast at about **94%** on average, with about **93 of 182 days at ≥95%**.
- Poisson model: **10% weekly error** vs 16% for a same-weekday naive forecast.
- Optimised roster + bed mix: about **$1.4M lower** six-month cost (nursing + diversions) than the status quo. The roster alone saves about $0.65M by swapping premium agency shifts for core shifts (agency shifts per week ~111 → ~22).
- Bed re-split (28/48/204 → 32/53/195): about **$0.9M** lower expected diversion cost; total diversions rise slightly, but costly ICU/Step-Down ones fall.
- **Infusion pumps:** the 340-pump pool is short on ~33% of days; ~385 needed for a 1% stock-out rate. Pooling needs ~35 fewer than unit-owned fleets.
- **Ventilators:** utilisation ~34%; cost-optimal fleet is 21 vs 36 owned. Keep a reserve for surges.
- **Length of stay is the strongest lever:** half a day off average stay cuts cost by about $2.9M against the optimised plan and absorbs a 10% surge. Flex beds roughly break even.

## Repository layout

```
healthcare-resource-optimization/
├── index.html              # interactive project page (Chart.js via cdnjs)
├── run_all.sh              # rebuilds everything (~20 s)
├── requirements.txt
├── src/
│   ├── common.py           # assumptions + patient-flow simulation engine
│   ├── 01_generate_data.py # synthetic daily data (units, staffing, equipment)
│   ├── 02_forecast.py      # models, backtest, winter forecast
│   ├── 03_optimize.py      # roster ILP, bed LP, equipment sizing
│   ├── 04_scenarios.py     # scenarios + sensitivity
│   ├── 05_build_bundle.py  # collects results into one JSON
│   ├── 06_build_page.py    # renders index.html from page_template.html
│   └── page_template.html
├── data/                   # unit_daily.csv, staffing_daily.csv, equipment_daily.csv, forecast CSV, bundle JSON
├── outputs/                # backtests, scenario table, sensitivity, optimization_results.json
└── images/preview.png
```

## Run it

```bash
pip install -r requirements.txt
./run_all.sh          # regenerates data, models, results and index.html (fixed seeds, reproducible)
```

## Key assumptions (edit in `src/common.py`)

Core RN shift $560, agency shift $980, sick-call rate 6%; diversion cost $12K / $6K / $3K (ICU / Step-Down / Med-Surg); ventilator $14K/yr, pump $900/yr, stock-out day $25K (ventilators).

## Limitations

- Forecast error is measured on data from the same simulator; real data will be noisier.
- No skill mix, overtime rules, bed turnover time or downstream spill-over.
- Surge scenarios hold the optimised plan fixed rather than re-optimising.
- Bed reallocation hits ICU's upper bound, so physical conversion limits would need real validation.

## Tools

Python 3 (numpy, pandas, statsmodels, scipy/HiGHS) and Chart.js for the page.
