#!/usr/bin/env bash

# Run this file
# conda activate lab
# ./slm_lab/spec/experimental/le/run_coin_study_delay_exp.sh


python3 run_lab.py slm_lab/spec/experimental/le/coin_study_delay.json coin_ppo_nopm_spl_le_with_naive_opponent_minus_one_wtout_avg train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_delay.json coin_ppo_nopm_spl_le_with_naive_opponent_avg_50 train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_delay.json coin_ppo_nopm_spl_le_with_naive_opponent_add_one_avg_50 train

python3 run_lab.py slm_lab/spec/experimental/le/coin_study_delay.json coin_ppo_nopm_spl_le_with_naive_opponent_wtout_avg train

python3 run_lab.py slm_lab/spec/experimental/le/coin_study_delay.json coin_ppo_le_with_naive_opponent train
python3 run_lab.py slm_lab/spec/experimental/le/coin_study_delay.json coin_ppo_nopm_spl_le_with_naive_opponent_avg_20 train

python3 run_lab.py slm_lab/spec/experimental/le/coin_study_delay.json coin_ppo_nopm_spl_le_with_naive_opponent_avg_20_SmoothL1Loss train

