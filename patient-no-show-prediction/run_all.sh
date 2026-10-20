#!/usr/bin/env bash
# Rebuild everything from scratch (about 30 seconds).
set -euo pipefail
cd "$(dirname "$0")/src"
python3 01_generate_data.py
python3 02_train_model.py
python3 03_simulate_policies.py
python3 04_build_bundle.py
python3 05_build_page.py
