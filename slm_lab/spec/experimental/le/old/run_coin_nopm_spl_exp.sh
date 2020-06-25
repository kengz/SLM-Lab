#!/usr/bin/env bash

# Run this file
# conda activate lab
# ./slm_lab/spec/experimental/le/run_coin_nopm_spl_exp.sh


python3 run_lab.py slm_lab/spec/experimental/le/coin_nopm_spl_ppo.json coin_ppo_nopm_spl_le_self_play train
python3 run_lab.py slm_lab/spec/experimental/le/coin_nopm_spl_ppo.json coin_ppo_nopm_spl_le_with_naive_coop train
python3 run_lab.py slm_lab/spec/experimental/le/coin_nopm_spl_ppo.json coin_ppo_nopm_spl_le_with_naive_opponent train