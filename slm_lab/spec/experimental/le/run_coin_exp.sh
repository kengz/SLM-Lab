#!/usr/bin/env bash

conda activate lab

#python run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf train
#python run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf_util train
python run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf_le_self_play train
python run_lab.py slm_lab/spec/experimental/le/coin_rf.json coin_rf_le_with_naive_opponent train

#python run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo train
#python run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_util train
python run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_le_self_play train
python run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_le_with_naive_opponent train