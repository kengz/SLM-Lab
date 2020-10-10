#!/usr/bin/env bash


#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ppm_le_self_play train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ppm_le_vs_naive_opp train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_base_rf_util train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_base_rf train

nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_no_strat_lr_008  train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_no_strat_lr_002  train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_no_strat_lr_032  train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_no_strat_lr_008 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_no_strat_lr_002 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_no_strat_lr_032 train

nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_strat_2_lr_032 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_strat_2_lr_002 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_strat_2_lr_002 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_strat_2_lr_032 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_strat_2_lr_008 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_strat_2_lr_008 train

nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_exploiter_no_strat_lr_008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_exploiter_strat_5_lr_008 train
nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_exploiter_strat_2_lr_008 train

#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_mb_mpc_cem_le_self_play train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_mb_mpc_cem_le_vs_naive_opp train

#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_strat_5_lr_008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_strat_5_lr_002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_self_play_strat_5_lr_032 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_strat_5_lr_008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_strat_5_lr_002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_short.json ipd_ipm_le_vs_naive_opp_strat_5_lr_032 train





#
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppm_le_self_play train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppm_le_vs_naive_opp train
#
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_base train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_base_util train
#


#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_no_strat_lr_0008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_no_strat_lr_0008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_no_strat_lr_0002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_no_strat_lr_0002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_no_strat_lr_0032 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_no_strat_lr_0032 train
#
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_strat_5_lr_0032 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_strat_5_lr_0002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_strat_5_lr_0002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_strat_5_lr_0032 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_strat_5_lr_0008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_strat_5_lr_0008 train
#
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_exploiter_no_strat_lr_0008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_exploiter_strat_5_lr_0008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_exploiter_strat_2_lr_0008 train

#
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_strat_2_lr_0008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_strat_2_lr_0008 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_strat_2_lr_0032 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_self_play_strat_2_lr_0002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_strat_2_lr_0002 train
#nohup xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_short.json coin_ppo_ipm_le_vs_naive_opp_strat_2_lr_0032 train



