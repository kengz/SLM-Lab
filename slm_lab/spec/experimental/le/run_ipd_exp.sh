#!/usr/bin/env bash

conda activate lab

python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf_util train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf_le_self_play train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf_le_with_naive_opponent train

#python3 run_lab.py slm_lab/spec/experimental/le/ipd_ppo.json ipd_ppo train
#python3 run_lab.py slm_lab/spec/experimental/le/ipd_ppo.json ipd_ppo_util train
#python3 run_lab.py slm_lab/spec/experimental/le/ipd_ppo.json ipd_ppo_le_self_play train
#python3 run_lab.py slm_lab/spec/experimental/le/ipd_ppo.json ipd_ppo_le_with_naive_opponent train