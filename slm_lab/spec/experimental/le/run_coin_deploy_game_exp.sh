#!/usr/bin/env bash

# Run this file
# conda activate lab
# ./slm_lab/spec/experimental/le/run_coin_deploy_game_exp.sh

python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_default_or_util train
python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_default_vs_default_or_util train
python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_util_vs_default_or_util train
#python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_ppo_deploy_game_pm_le_self_play train
python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_le_with_naive_opponent train
python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_nopm_spl_le_with_naive_opponent train

