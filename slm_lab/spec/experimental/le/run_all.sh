#!/usr/bin/env bash

# Run this file
# ./slm_lab/spec/experimental/le/run_all.sh

#conda activate lab

./slm_lab/spec/experimental/le/run_nopm_spl_exp.sh
./slm_lab/spec/experimental/le/run_nopm_exp.sh
./slm_lab/spec/experimental/le/run_ipd_exp.sh

./slm_lab/spec/experimental/le/run_deploy_game_exp.sh

./slm_lab/spec/experimental/le/run_coin_exp.sh
./slm_lab/spec/experimental/le/run_coin_nopm_spl_exp.sh

