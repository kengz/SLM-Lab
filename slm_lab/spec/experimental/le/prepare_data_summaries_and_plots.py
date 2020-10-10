import copy
import os
import re
import sys
import traceback
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_MA_WINDOW = 10

MatriceDataPoint = namedtuple("MatriceData", ["y_label", "y_player_1", "y_player_2", "dir_name"])


def find_all_files_with_extension(dir, ext):
    return [el for el in Path(dir).rglob(f'*.{ext}')]


def extract_trial_and_session_from_file_name(file_names_list):
    path_trial_dict = {}

    for path in file_names_list:
        # Extract n° of session (e.g. 0 to 9 for 10 runs with 10 random seeds)
        session_number = re.search(r"_s(\d+)_", path.name)
        if session_number is not None:
            session_number = int(session_number.groups()[0])
        else:
            print("no session_number found for path", path)
            continue

        # Extract n° of trial (always 0 if not using the build-in grid search in SLM-Lab)
        trial_number = re.search(r"_t(\d+)_", path.name)
        if trial_number is not None:
            trial_number = int(trial_number.groups()[0])
        else:
            print("no trial_idx", path)
            continue
        if trial_number not in path_trial_dict.keys():
            path_trial_dict[trial_number] = {"ag_idx_sess_idx": {},
                                             "world": {},
                                             "end_of_session": {}}

        # Agent csv file
        if re.match("^agent_.*", path.name):
            agent_idx = int(re.search(r"^agent_n(\d+)_", path.name).groups()[0])
            if agent_idx not in path_trial_dict[trial_number]["ag_idx_sess_idx"].keys():
                path_trial_dict[trial_number]["ag_idx_sess_idx"][agent_idx] = {}

            assert (session_number not in
                    path_trial_dict[trial_number]["ag_idx_sess_idx"][agent_idx]), (
                f"session_number {session_number} trial_number {trial_number} agent_idx {agent_idx} "
                f"path_trial_dict[trial_idx]['ag_idx_sess_idx'][agent_idx] {path_trial_dict[trial_number]['ag_idx_sess_idx'][agent_idx]}")
            if session_number not in path_trial_dict[trial_number]["ag_idx_sess_idx"][agent_idx]:
                path_trial_dict[trial_number]["ag_idx_sess_idx"][agent_idx][session_number] = path

        # World csv file (not used)
        elif re.match("^world_.*", path.name):
            path_trial_dict[trial_number]["world"][session_number] = path

        # End of session csv file (not used)
        else:
            path_trial_dict[trial_number]["end_of_session"][session_number] = path

    return path_trial_dict


def init_df(csv_nested_dict):
    all_df = {}
    for k, v in csv_nested_dict.items():
        if isinstance(v, dict):
            all_df[k] = init_df(v)
        else:
            all_df[k] = pd.read_csv(v)
    return all_df


def calc_sr_ma(sr):
    '''Calculate the moving-average of a series to be plotted'''
    return sr.rolling(PLOT_MA_WINDOW, min_periods=1).mean()


def get_data_for_all_sessions(data_dict, x_col, y_cols, agent_idx, ma=False):
    x_outputs = {}
    y_outputs = {}

    sessions = data_dict[agent_idx]
    for session_idx, session_df in sessions.items():
        for col in session_df.columns:
            for key in y_cols.keys():
                if key in col:
                    if key in y_outputs.keys():
                        if ma:
                            y_outputs[key].append(calc_sr_ma(session_df[col]))
                        else:
                            y_outputs[key].append(session_df[col])
                    else:
                        if ma:
                            y_outputs[key] = [calc_sr_ma(session_df[col])]
                        else:
                            y_outputs[key] = [session_df[col]]

            for key in x_col.keys():
                if key in col:
                    if key in x_outputs.keys():
                        x_outputs[key].append(session_df[col])
                    else:
                        x_outputs[key] = [session_df[col]]
    return x_outputs, y_outputs


def plot_min_avg_max(x, y_min, y_avg, y_max, color, label):
    plt.plot(x, y_avg, '-', color=color, label=label)
    plt.fill_between(x, y_min, y_max, color=color, alpha=0.2)


def plot_several_series(x, y_cols_aggr, y_cols_colors):
    for k in y_cols_aggr.keys():
        color = y_cols_colors[k]

        not_nan = np.logical_not(np.isnan(np.array(y_cols_aggr[k]["avg"])))
        plot_min_avg_max(x[not_nan], y_min=y_cols_aggr[k]["min"][not_nan],
                         y_avg=y_cols_aggr[k]["avg"][not_nan], y_max=y_cols_aggr[k]["max"][not_nan],
                         color=color, label=k)


def check_and_correct_consistency_of_series(y_cols, k, x):
    sizes = []
    for serie in y_cols[k]:
        sizes.append(len(serie))
    sizes = set(sizes)
    if len(sizes) > 1:
        print("\nWARNING DATA with several length !!! Missing DATA. Remove shorter series by default")
        print("sizes", sizes)
        max_sizes = max(sizes)
        print("n serie before cleaning", len(y_cols[k]))
        x = [x_list for x_list, serie in zip(x, y_cols[k]) if len(serie) == max_sizes]
        y_cols[k] = [serie for serie in y_cols[k] if len(serie) == max_sizes]
        print("n serie after cleaning", len(y_cols[k]))
        sizes = []
        for serie in y_cols[k]:
            sizes.append(len(serie))
        sizes = set(sizes)
        print("corrected sizes", sizes)
        print("\n")
    return y_cols

def compute_plot_min_avg_max(x, y_cols, y_cols_colors):
    y_cols_aggr = {k: {"min": [], "avg": [], "max": []} for k in y_cols.keys()}
    for k in y_cols.keys():
        y_cols = check_and_correct_consistency_of_series(y_cols, k, x)

        col_array = np.array([serie.tolist() for serie in y_cols[k]])

        y_cols_aggr[k]["min"] = col_array.min(axis=0)
        y_cols_aggr[k]["avg"] = col_array.mean(axis=0)
        y_cols_aggr[k]["max"] = col_array.max(axis=0)

    plot_several_series(x[0], y_cols_aggr, y_cols_colors)

def get_tau_val(dir):
    tau = re.search("_temp_[0-9][0-9]?[0-9]?_", dir)[0].split("_")[2]
    if tau == "033":
        tau = 0.33
    else:
        tau = int(tau)
    return tau

def generate_graph_title(current_dir):
    # Extract initial tau of opponent
    if "_temp_" in current_dir:
        tau = get_tau_val(current_dir)

    # if "ipd_" in current_dir:
    #     base_algo = "DDQN"  # ""REINFORCE"
    # elif "coin_" in current_dir:
    #     base_algo = "DDQN"  # "PPO"
    #
    # if "_deploy_game_default_or_util_" in current_dir:
    #     title = "UCB1{selfish, util} in self-play"
    #     player_1, player_2 = f"UCB1({base_algo}){{selfish, util}}", f"UCB1({base_algo}){{selfish, util}}"
    #
    # elif "deploy_game_default_vs_default_or_util" in current_dir:
    #     title = "selfish vs UCB1{selfish, util}"
    #     player_1, player_2 = f"UCB1({base_algo}){{selfish}}", f"UCB1({base_algo}){{selfish, util}}"
    #
    # elif "deploy_game_util_vs_default_or_util" in current_dir:
    #     title = "util vs UCB1{selfish, util}"
    #     player_1, player_2 = f"UCB1({base_algo}){{util}}", f"UCB1({base_algo}){{selfish, util}}"
    #
    # elif "deploy_game_le_self_play" in current_dir:
    #     title = "UCB1{TFT-PPM(wUtil)} in self-play"
    #     player_1, player_2 = f"UCB1(TFT-PPM({base_algo})){{util}}", f"UCB1(TFT-PPM({base_algo})){{util}}"
    #
    # elif "deploy_game_le_with_naive_opponent" in current_dir:
    #     title = "TFT-PPM(wUtil) vs UCB1{{selfish, util}}"
    #     player_1, player_2 = f"UCB1(TFT-PPM({base_algo})){{util}}", f"UCB1({base_algo}){{selfish, util}}"

    if '_ipm_' in current_dir:
        algo_kind = "L-TFT"
    else:
        algo_kind = "TFT-PPM"


    if "_self_play_" in current_dir:
        opponent = "in self-play"
        opponent_name = algo_kind
    elif "_naive_opponent_" in current_dir or "_naive_opp_" in current_dir:
        opponent = "vs. naive"
        opponent_name = "Naive"
    elif "_exploiter_" in current_dir:
        opponent = "vs. Exploiter"
        opponent_name = "Exploiter"
    elif "_naive_coop_" in current_dir:
        opponent = "vs. naive cooperative"
        opponent_name = "Naive(wUtil)"
    elif "_util_" in current_dir:
        title = "baseline(wUtil)"
        player_1, player_2 = f"Naive(wUtil)", f"Naive(wUtil)"
        return title, player_1, player_2
    else:
        title = "baseline"
        player_1, player_2 = f"Naive", f"Naive"
        return title, player_1, player_2




    if "_lr_" in current_dir:
        raise NotImplementedError()
    elif "_temp_" in current_dir:
        if "_no_strat_" in current_dir:
            title = f"{algo_kind} {opponent}, temperature assumed"
            player_1, player_2 = f"L-TFT(1)", f"{opponent_name}({tau})"
        elif "_strat_2_" in current_dir:
            title = f"{algo_kind} {opponent}, temperature estimated"
            player_1, player_2 = f"L-TFT", f"{opponent_name}"
        else:
            raise ValueError()
    else:
        raise ValueError()

    return title, player_1, player_2


def try_to_plot_ipd(input_df, current_dir, trial_idx):
    global ALL_DF_FOR_MATRICES
    title_suffix, player_1, player_2 = generate_graph_title(current_dir)
    title = "IPD: " + title_suffix
    xlabel = "frame"
    ylabel = "prob"
    x_col = {xlabel: []}

    # else:
    n_graphs = 2
    figsize = [6.4, 4.8]
    n_steps_per_episode = 20

    # Detect that it is a IPD directory by the presence of data for "dd" and "cc"
    y_cols = {"dd": [], "cc": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        fig = plt.figure(figsize=figsize)
        plt.subplot(n_graphs, 1, 1)

        y_cols_colors = {"DD": "orange", "CC": "blue"}
        all_y_cols = [el for el in y_cols.keys()]
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0, ma=True)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols = {"CC": y_cols['cc'], "DD": y_cols['dd']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        plt.title(title)
        plt.ylim(-0.1, 1.1)
        plt.ylabel(ylabel)
        plt.legend(loc="upper left")
        plt.subplot(n_graphs, 1, 2)

        ylabel = "payoffs"
        y_cols = {"tot_r_ma": []}
        all_y_cols.extend([el for el in y_cols.keys()])

        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_1: "orange"}
        y_cols['tot_r_ma'] = [serie / n_steps_per_episode for serie in y_cols['tot_r_ma']]
        y_cols_p_1 = copy.deepcopy(y_cols['tot_r_ma'])
        y_cols = {player_1: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        y_cols = {"tot_r_ma": []}
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_2: "blue"}
        y_cols['tot_r_ma'] = [serie / n_steps_per_episode for serie in y_cols['tot_r_ma']]
        y_cols_p_2 = copy.deepcopy(y_cols['tot_r_ma'])
        y_cols = {player_2: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        plt.ylim(-4, 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="lower left")


        graph_name = '_'.join(['summary', "t" + str(trial_idx)] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        plt.close()

        # Gather data for the matrices
        # if bool(re.match("^.*_0\.[0-9][0-9]_0\.[0-9][0-9].*$", current_dir)):
        ALL_DF_FOR_MATRICES.append(MatriceDataPoint(ylabel, y_cols_p_1, y_cols_p_2, current_dir))
            # print(f"added {current_dir} in ALL_DF_FOR_MATRICES")


def try_to_plot_coin_game(input_df, current_dir, trial_idx):
    global ALL_DF_FOR_MATRICES
    title_suffix, player_1, player_2 = generate_graph_title(current_dir)
    title = "Coin Game: " + title_suffix
    xlabel = "frame"
    x_col = {xlabel: []}

    n_graphs = 2
    figsize = [6.4, 4.8]
    n_steps_per_episode = 20

    # Detect that it is a Coin Game directory by the presence of data for "pck_speed"
    y_cols = {"pck_own": [], "pck_speed": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        ylabel = "fraction"
        fig = plt.figure(figsize=figsize)
        plt.subplot(n_graphs, 1, 1)

        # y_cols_colors = {"pick own": "orange", "pick speed": "blue"}
        y_cols_colors = {"pick own": "orange"}
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0, ma=True)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        # y_cols = {"pick own": y_cols['pck_own'], "pick speed": y_cols['pck_speed']}
        y_cols = {"pick own": y_cols['pck_own']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        plt.ylim(-0.1, 1.1)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")
        plt.subplot(n_graphs, 1, 2)

        ylabel = "payoffs"
        y_cols = {"tot_r_ma": []}
        all_y_cols = [el for el in y_cols.keys()]

        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_1: "orange"}
        y_cols['tot_r_ma'] = [serie / n_steps_per_episode for serie in y_cols['tot_r_ma']]
        y_cols_p_1 = copy.deepcopy(y_cols['tot_r_ma'])
        y_cols = {player_1: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        y_cols = {"tot_r_ma": []}
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_2: "blue"}
        y_cols['tot_r_ma'] = [serie / n_steps_per_episode for serie in y_cols['tot_r_ma']]
        y_cols_p_2 = copy.deepcopy(y_cols['tot_r_ma'])
        y_cols = {player_2: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        plt.ylim(-0.5, +0.5)
        plt.yticks(np.arange(-0.5, +0.75, step=0.25))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.legend(loc="lower right")
        plt.legend(loc="lower left")

        graph_name = '_'.join(['summary', "t" + str(trial_idx)] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        plt.close()

        # Gather data for the matrices
        # if bool(re.match("^.*_0\.[0-9][0-9]_0\.[0-9][0-9].*$", current_dir)):
        ALL_DF_FOR_MATRICES.append(MatriceDataPoint(ylabel, y_cols_p_1, y_cols_p_2, current_dir))
            # print(f"added {current_dir} in ALL_DF_FOR_MATRICES")


def try_to_plot_deployment_game(input_df, current_dir, trial_idx):
    title_suffix, player_1, player_2 = generate_graph_title(current_dir)
    title = "Learning Game: " + title_suffix
    xlabel = "frame"
    ylabel = "prob"
    x_col = {xlabel: []}

    if "TFT" in title:
        n_graphs = 3
        figsize = [6.4, 7.2]
    else:
        n_graphs = 2
        figsize = [6.4, 4.8]

    y_cols = {"deployed_algo_idx": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        fig = plt.figure(figsize=figsize)
        plt.subplot(n_graphs, 1, 1)

        all_y_cols = [el for el in y_cols.keys()]
        # Exist for agent_n0 session_n0
        x_col_ag0, y_cols_ag0 = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        x_col_ag1, y_cols_ag1 = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"

        x_col = x_col_ag0
        y_cols = {}

        y_cols_colors = {"dd": "orange", "cc": "blue"}
        assert len(y_cols_ag0.keys()) == 1

        inverse_player_1 = False
        if ("deploy_game_util_vs_default_or_util" in current_dir or
                "deploy_game_le_self_play" in current_dir or
                "deploy_game_le_with_naive_opponent" in current_dir):
            inverse_player_1 = True

        for key in y_cols_ag0.keys():
            if inverse_player_1:
                cc = (1 - np.array(y_cols_ag0[key])) * np.array(y_cols_ag1[key])
            else:
                cc = np.array(y_cols_ag0[key]) * np.array(y_cols_ag1[key])
            for i in range(cc.shape[0]):
                cc_not_nan = np.logical_not(np.isnan(cc))
                cc[i, :][cc_not_nan[i, :]] = calc_sr_ma(pd.Series(cc[i, :][cc_not_nan[i, :]]))
            y_cols["cc"] = cc

            if inverse_player_1:
                dd = np.array(y_cols_ag0[key]) * (1 - np.array(y_cols_ag1[key]))
            else:
                dd = (1 - np.array(y_cols_ag0[key])) * (1 - np.array(y_cols_ag1[key]))
            for i in range(dd.shape[0]):
                dd_not_nan = np.logical_not(np.isnan(dd))
                dd[i, :][dd_not_nan[i, :]] = calc_sr_ma(pd.Series(dd[i, :][dd_not_nan[i, :]]))
            y_cols["dd"] = dd
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend(loc="upper right")
        plt.subplot(n_graphs, 1, 2)

        ylabel = "payoffs"
        y_cols = {"undisc_tot_r": []}
        all_y_cols.extend([el for el in y_cols.keys()])

        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_1: "orange"}
        y_cols = {player_1: y_cols['undisc_tot_r']}
        for key in y_cols.keys():
            col = np.array(y_cols[key])
            for i in range(col.shape[0]):
                col_not_nan = np.logical_not(np.isnan(col))
                col[i, :][col_not_nan[i, :]] = calc_sr_ma(pd.Series(col[i, :][col_not_nan[i, :]]))
            y_cols[key] = col
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        y_cols = {"undisc_tot_r": []}
        x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_2: "blue"}
        y_cols = {player_2: y_cols['undisc_tot_r']}
        for key in y_cols.keys():
            col = np.array(y_cols[key])
            for i in range(col.shape[0]):
                col_not_nan = np.logical_not(np.isnan(col))
                col[i, :][col_not_nan[i, :]] = calc_sr_ma(pd.Series(col[i, :][col_not_nan[i, :]]))
            y_cols[key] = col
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        if "TFT" not in title:
            plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")

        if "TFT" in title:
            plt.subplot(n_graphs, 1, 3)

            ylabel = "symlog(defection metric)"
            y_cols = {"d_carac": []}
            all_y_cols.extend([el for el in y_cols.keys()])

            # Exist for agent_n0 session_n0
            x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
            assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                        for x in x_col[xlabel]]), "x must be identical for all sessions"
            y_cols_colors = {player_1: "orange"}
            if len(y_cols.keys()) > 0:
                y_cols = {player_1: y_cols['d_carac']}
                compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

            y_cols = {"d_carac": []}
            x_col, y_cols = get_data_for_all_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
            assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                        for x in x_col[xlabel]]), "x must be identical for all sessions"
            y_cols_colors = {player_2: "blue"}
            if len(y_cols.keys()) > 0:
                y_cols = {player_2: y_cols['d_carac']}
                compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

            plt.axhline(y=0, linestyle='--', color='grey')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.yscale('symlog')
            plt.legend(loc="lower right")

        graph_name = '_'.join(['summary', "t" + str(trial_idx)] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        plt.close()


def get_y_x_pos_in_matrice(x_values, y_values, current_dir):
    current_dir = current_dir[:-18]
    x_val = float(current_dir[-9:-5])
    y_val = float(current_dir[-4:])
    for i, x_v in enumerate(x_values):
        for j, y_v in enumerate(y_values):
            if x_val == x_v and y_val == y_v:
                return i, j
    raise ValueError()


def plot_table(column_labels, row_labels, values, title, description):
    plt.figure(linewidth=2,
               tight_layout={'pad': 1},
               figsize=(7, 7)
               )

    for player_n in range(N_PLAYERS):
        plt.subplot(2, 1, player_n + 1)

        data = values[player_n]
        column_headers = column_labels
        row_headers = row_labels

        # Format the data
        cell_text = []
        for row_i in range(data.shape[0]):
            cell_text.append([f'{data[row_i,col_i,0]:1.2f}±{data[row_i,col_i,1]:1.3f}' for col_i in range(
                data.shape[1])])

        row_colors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
        col_colors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
        plt.table(cellText=cell_text,
                  rowLabels=row_headers,
                  rowColours=row_colors,
                  rowLoc='right',
                  colColours=col_colors,
                  colLabels=column_headers,
                  loc='center')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.box(on=None)
        plt.suptitle(title + '\n' + description)

    plt.figtext(0.5, 0.90, "Player 1 payoff", ha="center", va="top",
                fontsize=14)
    plt.figtext(0.5, 0.55, "(col Player 1, row Player2)\n\nPlayer 2 payoff", ha="center", va="top",
                fontsize=14)
    plt.draw()
    fig = plt.gcf()
    save_path = f'{title}.png'
    print(f"saving table to {save_path}")
    plt.savefig(save_path,
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=150
                )


def plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_env,
                             x_values, y_values, title, description):
    post_title = None
    # Keep only the relevant data
    selected_data = [data for data in all_matrices_for_matrices
                     if bool(re.match(f"^\./{select_env}_0\.[0-9][0-9]_0\.[0-9][0-9].*$", data.dir_name))]
    print("Data for", len(selected_data), "points in this matrice")
    if len(selected_data) > 0:
        table_values_all_players = []
        for player_n in range(N_PLAYERS):

            table_values = np.zeros(shape=(len(y_values), len(x_values), 2))
            for data in selected_data:
                if post_title is None:
                    post_title, player_1, player_2 = generate_graph_title(data.dir_name)
                    post_title = ', ' + post_title

                if player_n == 0:
                    values = select_function(data.y_player_1)
                elif player_n == 1:
                    values = select_function(data.y_player_2)
                values_mean = values.mean()
                values_std = values.std()
                values_st_error = values_std / np.sqrt(len(values))
                x_pos, y_pos = get_y_x_pos_in_matrice(x_values, y_values, data.dir_name)
                # print(data.dir_name, x_pos, y_pos, "len(values)", len(values), "values_mean", values_mean,
                #       values_st_error)
                table_values[y_pos, x_pos, :] = [values_mean, values_st_error]

            print(f"Player {player_n}")
            print("\tValues")
            print(table_values[:, :, 0])
            print("\tStd")
            print(table_values[:, :, 1])
            table_values_all_players.append(table_values)

        plot_table(column_labels=[str(v) for v in x_values],
                   row_labels=[str(v) for v in y_values],
                   values=table_values_all_players,
                   title=title + post_title,
                   description=description)

def get_y_x_pos_in_table_tau(x_values, y_values, current_dir):
    tau = get_tau_val(current_dir)
    y_val = tau

    for i, x_v in enumerate(x_values):
        for j, y_v in enumerate(y_values):
            if x_v in current_dir and y_val == y_v:
                return i, j
    raise ValueError()

def plot_table_tau(all_matrices_for_matrices, select_function,
                             x_values, y_values, title, description):

    colomns_selectors = ["ipd_ipm_le_self_play_no_strat_temp_", "ipd_ipm_le_self_play_strat_2_temp_",
                         "coin_ppo_ipm_le_self_play_strat_2_temp_", "coin_ppo_ipm_le_self_play_no_strat_temp_"]
    table_values_all_players = []
    for player_n in range(N_PLAYERS):
        table_values = np.zeros(shape=(len(y_values), len(x_values), 2))
        for col_selector in colomns_selectors:
            # post_title = None
            # Keep only the relevant data
            selected_data = [data for data in all_matrices_for_matrices
                             if bool(re.match(f"^\./{col_selector}[0-9]+_202[01].*$", data.dir_name))]
            print("Data for", len(selected_data), "points in this matrice")
            if len(selected_data) > 0:

                # last dim = 2 for mean and std error
                for data in selected_data:
                    # if post_title is None:
                    #     post_title, player_1, player_2 = generate_graph_title(data.dir_name)
                    #     post_title = ', ' + post_title

                    if player_n == 0:
                        values = select_function(data.y_player_1)
                    elif player_n == 1:
                        values = select_function(data.y_player_2)
                    values_mean = values.mean()
                    values_std = values.std()
                    values_st_error = values_std / np.sqrt(len(values))
                    x_pos, y_pos = get_y_x_pos_in_table_tau(colomns_selectors, y_values, data.dir_name)
                    # print(data.dir_name, x_pos, y_pos, "len(values)", len(values), "values_mean", values_mean,
                    #       values_st_error)
                    table_values[y_pos, x_pos, :] = [values_mean, values_st_error]

        print(f"Player {player_n}")
        print("\tValues")
        print(table_values[:, :, 0])
        print("\tStd")
        print(table_values[:, :, 1])
        table_values_all_players.append(table_values)

    plot_table(column_labels=[str(v) for v in x_values],
               row_labels=[str(v) for v in y_values],
               values=table_values_all_players,
               # title=title + post_title,
               title=title,
               description=description)

def get_mean(y):
    v = np.array(y)
    means = []
    for i, v_serie in enumerate(np.split(v, len(v), axis=0)):
        v_serie_not_nan = v_serie[~np.isnan(v_serie)]
        if len(v_serie_not_nan) > 0:
            means.append(v_serie_not_nan.mean())
        else:
            print(f"Warning, v_serie_not_nan {i} has length 0")
    return np.array(means)


def get_last(y):
    v = np.array(y)
    lasts = []
    for i, v_serie in enumerate(np.split(v, len(v), axis=0)):
        v_serie_not_nan = v_serie[~np.isnan(v_serie)]
        if len(v_serie_not_nan) > 0:
            lasts.append(v_serie_not_nan[-1])
        else:
            print(f"Warning, v_serie_not_nan {i} has length 0")
    return np.array(lasts)


def plot_matrices(all_matrices_for_matrices):
    x_values = [0.55, 0.65, 0.75, 0.85, 0.95]
    y_values = [0.55, 0.65, 0.75, 0.85, 0.95]

    print("\nMatrice: Self-play IPD mean payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_mean
    select_prefix = "ipd_ipm_le_self_play_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="IPD, mean payoff",
                             description="")

    print("\nMatrice: Self-play IPD last payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_last
    select_prefix = "ipd_ipm_le_self_play_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="IPD, last moving-average payoff",
                             description="(moving-average over 10 epi = 200 steps)")

    print("\nMatrice: LE vs Exploiter IPD mean payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_mean
    select_prefix = "ipd_ipm_le_vs_exploiter_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="IPD, mean payoff",
                             description="")

    print("\nMatrice: LE vs Exploiter IPD last payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_last
    select_prefix = "ipd_ipm_le_vs_exploiter_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="IPD, last moving-average payoff",
                             description="(moving-average over 10 epi = 200 steps)")

    print("\nMatrice: Coin game mean payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_mean
    select_prefix = "coin_ppo_ipm_le_self_play_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="Coin game, mean payoff",
                             description="")

    print("\nMatrice: Coin game last payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_last
    select_prefix = "coin_ppo_ipm_le_self_play_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="Coin game, last moving-average payoff",
                             description="(moving-average over 10 points in the last 50 epi = 1000 steps)")

    print("\nMatrice: Coin game mean payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_mean
    select_prefix = "coin_ppo_ipm_le_vs_exploiter_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="Coin game, mean payoff",
                             description="")

    print("\nMatrice: Coin game last payoff")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_last
    select_prefix = "coin_ppo_ipm_le_vs_exploiter_strat_2_temp_1"
    plot_matrice_percentiles(all_matrices_for_matrices, select_function, select_prefix,
                             x_values=x_values, y_values=y_values,
                             title="Coin game, last moving-average payoff",
                             description="(moving-average over 10 points in the last 50 epi = 1000 steps)")

    print("\nTable: Varying temperature")
    x_values = ["IPD t assumed", "IPD t estimated",
                "CG t  assumed", "CG t estimated"]
    y_values = [0.33, 1.0, 3.0]
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_mean
    plot_table_tau(all_matrices_for_matrices, select_function,
                             x_values=x_values, y_values=y_values,
                             title="Mean payoff with various temperatures for the opponent",
                             description="")

    print("\nTable: Varying temperature")
    print("col_axis", x_values)
    print("row_axis", np.array(y_values)[:, np.newaxis])
    select_function = get_last
    plot_table_tau(all_matrices_for_matrices, select_function,
                             x_values=x_values, y_values=y_values,
                             title="Last moving-average payoff with various temperatures for the opponent",
                             description="(moving-average over 10 points in the last 50 epi = 1000 steps)")


def plot_graphs(input_df, current_dir):
    for trial_number, session_df in input_df.items():

        try_to_plot_ipd(session_df, current_dir, trial_number)
        try_to_plot_coin_game(session_df, current_dir, trial_number)
        try_to_plot_deployment_game(session_df, current_dir, trial_number)


def process_one_dir(dir_path):
    csv_files = find_all_files_with_extension(dir_path, ext='csv')
    csv_files_sessions = extract_trial_and_session_from_file_name(csv_files)
    input_df = init_df(csv_files_sessions)
    if len(input_df) > 0 and "2020" in dir_path.split('/')[-1]:
        print("Process directory", dir_path)
        assert not all(input_df[0]["ag_idx_sess_idx"][0][0]['tot_r'] == input_df[0]["ag_idx_sess_idx"][1][0]['tot_r'])
        plot_graphs(input_df, dir_path)
    else:
        print(f"Not precessing dir {dir_path}")


def apply_fn_to_all_dir(location, fn_to_apply):
    path_in_location = os.listdir(location)
    for path in path_in_location:
        full_path = os.path.join(location, path)
        if os.path.isdir(full_path):
            if path[0] != ".":
                try:
                    fn_to_apply(full_path)
                except Exception as e:
                    print("Exception", e)
                    traceback.print_exception(*sys.exc_info())


if __name__ == "__main__":
    ALL_DF_FOR_MATRICES = []
    N_PLAYERS = 2

    apply_fn_to_all_dir('.', process_one_dir)
    apply_fn_to_all_dir('.', lambda subdir: apply_fn_to_all_dir(subdir, process_one_dir))

    plot_matrices(ALL_DF_FOR_MATRICES)
