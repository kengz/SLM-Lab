#!/usr/bin/env bash


#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json _ppm_le_self_play train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json _ppm_le_vs_naive_opp train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json _base_rf_util train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json _base_rf train
#
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_exploiter_no_strat_lr_008 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_exploiter_strat_2_lr_008 train
#
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_self_play_no_strat_lr_008  train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_self_play_no_strat_lr_002  train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_self_play_no_strat_lr_032  train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_naive_opp_no_strat_lr_008 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_naive_opp_no_strat_lr_002 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_naive_opp_no_strat_lr_032 train
#
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_self_play_strat_2_lr_002 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_self_play_strat_2_lr_032 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_self_play_strat_2_lr_008 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_naive_opp_strat_2_lr_002 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_naive_opp_strat_2_lr_032 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/ipd_ddqn_boltzmann_search.json ipd_ipm_le_vs_naive_opp_strat_2_lr_008 train
#
#
#
#
#
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppm_le_self_play train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppm_le_vs_naive_opp train
#
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_base train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_base_util train



xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_exploiter_no_strat_lr_0008 train
xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_exploiter_strat_2_lr_0008 train

#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_self_play_no_strat_lr_0008 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_self_play_strat_2_lr_0008 train
#
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_self_play_no_strat_lr_0002 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_self_play_no_strat_lr_0032 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_self_play_strat_2_lr_0032 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_self_play_strat_2_lr_0002 train
#
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_naive_opp_no_strat_lr_0008 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_naive_opp_no_strat_lr_0002 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_naive_opp_no_strat_lr_0032 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_naive_opp_strat_2_lr_0008 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_naive_opp_strat_2_lr_0002 train
#xvfb-run -a python3 run_lab.py slm_lab/spec/experimental/le/coin_ddqn_boltzmann_search.json coin_ppo_ipm_le_vs_naive_opp_strat_2_lr_0032 train
