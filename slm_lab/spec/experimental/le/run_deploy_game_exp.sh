#!/usr/bin/env bash

# Run this file
# ./slm_lab/spec/experimental/le/run_deploy_game_exp.sh

#conda activate lab

python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_default_or_util train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_default_vs_default_or_util train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_util_vs_default_or_util train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_le_self_play train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_le_with_naive_opponent train
python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_nopm_deploy_game_le_with_naive_opponent train

