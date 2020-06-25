#!/usr/bin/env bash

# Run this file
# conda activate lab
# ./slm_lab/spec/experimental/le/run_coin_study_hyperparam_stability_exp.sh


python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_self_play_0002 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_self_play_0032 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_self_play_0008 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_self_play_0004 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_self_play_0016 train



python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_vs_naive_opp_lr_0032 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_vs_naive_opp_lr_0002 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_vs_naive_opp_lr_0008 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_vs_naive_opp_lr_0004 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_hyperparam_stability.json coin_ppo_nopm_spl_le_vs_naive_opp_lr_0016 train


