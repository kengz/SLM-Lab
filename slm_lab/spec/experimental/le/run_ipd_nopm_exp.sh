#!/usr/bin/env bash

# Run this file
# conda activate lab
# ./slm_lab/spec/experimental/le/run_ipd_nopm_exp.sh


python3 run_lab.py slm_lab/spec/experimental/le/ipd_nopm_rf.json ipd_rf_nopm_le_self_play train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_nopm_rf.json ipd_rf_nopm_le_with_naive_coop train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_nopm_rf.json ipd_rf_nopm_le_with_naive_opponent train

