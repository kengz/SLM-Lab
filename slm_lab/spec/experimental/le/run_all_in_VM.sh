#!/usr/bin/env bash

# Run this file
# conda activate lab
# ./slm_lab/spec/experimental/le/run_all_in_VM.sh


##./slm_lab/spec/experimental/le/run_ipd_exp.sh
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf_util train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf_le_self_play train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_rf.json ipd_rf_le_with_naive_opponent train
#
#
##./slm_lab/spec/experimental/le/run_nopm_exp.sh
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_nopm_rf.json ipd_rf_nopm_le_self_play train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_nopm_rf.json ipd_rf_nopm_le_with_naive_coop train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_nopm_rf.json ipd_rf_nopm_le_with_naive_opponent train
#
#
##./slm_lab/spec/experimental/le/run_deploy_game_exp.sh
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_default_or_util train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_default_vs_default_or_util train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_util_vs_default_or_util train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_le_self_play train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_deploy_game_le_with_naive_opponent train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_deploy_game_rf.json ipd_rf_nopm_deploy_game_le_with_naive_opponent train
#
#
## ./slm_lab/spec/experimental/le/run_coin_exp.sh
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_util train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_le_self_play train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ppo.json coin_ppo_le_with_naive_opponent train

#./slm_lab/spec/experimental/le/run_coin_nopm_spl_exp.sh
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_nopm_spl_ppo.json coin_ppo_nopm_spl_le_self_play train
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_nopm_spl_ppo.json coin_ppo_nopm_spl_le_with_naive_coop train
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_nopm_spl_ppo.json coin_ppo_nopm_spl_le_with_naive_opponent train


#./slm_lab/spec/experimental/le/run_coin_deploy_game_exp.sh
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_default_or_util train
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_default_vs_default_or_util train
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_util_vs_default_or_util train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_ppo_deploy_game_pm_le_self_play train
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_pm_le_with_naive_opponent train
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_deploy_game_ppo.json coin_deploy_game_ppo_nopm_spl_le_with_naive_opponent train
