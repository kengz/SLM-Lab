#!/usr/bin/env bash

# Run this file
# ./slm_lab/spec/experimental/le/run_coin_exp.sh

#conda activate lab

#python3 run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf train
#python3 run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf_util train
#python3 run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf_le_self_play train
#python3 run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf_le_with_naive_opponent train

python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo train
python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_util train
python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_le_self_play train
python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_le_with_naive_opponent train