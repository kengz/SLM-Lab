import os
import re
from pathlib import Path

# import slm_lab.lib.viz as viz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_to_all_dir(location, fn_to_apply):
    path_in_location = os.listdir(location)
    for path in path_in_location:
        full_path = os.path.join(location, path)
        if os.path.isdir(full_path):
            if path[0] != ".":
                try:
                    fn_to_apply(full_path)
                except Exception as e:
                    print("Exception", type(e))


def find_all_files_with_ext(dir, ext):
    return [el for el in Path(dir).rglob(f'*.{ext}')]


def extract_trial_and_session_from_file_name(file_names_list):
    path_trial_dict = {}

    for path in file_names_list:
        session_idx = re.search(r"_s(\d+)_", path.name)
        if session_idx is not None:
            session_idx = int(session_idx.groups()[0])
        else:
            print("no session_idx",path)
            continue

        trial_idx = re.search(r"_t(\d+)_", path.name)
        if trial_idx is not None:
            trial_idx = int(trial_idx.groups()[0])
        else:
            print("no trial_idx", path)
            continue

        if trial_idx not in path_trial_dict.keys():
            path_trial_dict[trial_idx] = {"ag_idx_sess_idx": {},
                                         "world": {},
                                         "end_of_session": {}}

        if re.match("^agent_.*", path.name):
            # Agent csv file
            agent_idx = int(re.search(r"^agent_n(\d+)_", path.name).groups()[0])
            if agent_idx not in path_trial_dict[trial_idx]["ag_idx_sess_idx"].keys():
                path_trial_dict[trial_idx]["ag_idx_sess_idx"][agent_idx] = {}
            if session_idx not in path_trial_dict[trial_idx]["ag_idx_sess_idx"][agent_idx]:
                path_trial_dict[trial_idx]["ag_idx_sess_idx"][agent_idx][session_idx] = path
            else:
                raise Exception()
            # print("detected", trial_idx, session_idx,agent_idx)

        elif re.match("^world_.*", path.name):
            # World csv file
            path_trial_dict[trial_idx]["world"][session_idx] = path
        else:
            # End of session csv file
            path_trial_dict[trial_idx]["end_of_session"][session_idx] = path

    return path_trial_dict


def init_df(csv_nested_dict):
    all_df = {}
    for k, v in csv_nested_dict.items():
        if isinstance(v, dict):
            all_df[k] = init_df(v)
        else:
            # print("read_csv",v)
            all_df[k] = pd.read_csv(v)
    return all_df


PLOT_MA_WINDOW = 10


def calc_sr_ma(sr):
    '''Calculate the moving-average of a series to be plotted'''
    return sr.rolling(PLOT_MA_WINDOW, min_periods=1).mean()


def get_data_sessions(data_dict, x_col, y_cols, agent_idx, ma=False):
    x_outputs = {}
    y_outputs = {}
    # for agent_idx, sessions in data_dict.items():
    sessions = data_dict[agent_idx]
    for session_idx, session_df in sessions.items():
        for col in session_df.columns:
            for key in y_cols.keys():
                if key in col:
                    # print("session_df[col]", len(session_df[col]))
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
    # plt.xlim(0, 10)
    # plt.show()


def plot_several_series(x, y_cols_aggr, y_cols_colors):
    for k in y_cols_aggr.keys():
        color = y_cols_colors[k]

        # plot_min_avg_max(x, y_min=y_cols_aggr[k]["min"],
        #                  y_avg=y_cols_aggr[k]["avg"], y_max=y_cols_aggr[k]["max"],
        #                  color=color, label=k)

        not_nan =  np.logical_not(np.isnan(np.array(y_cols_aggr[k]["avg"])))
        plot_min_avg_max(x[not_nan], y_min=y_cols_aggr[k]["min"][not_nan],
                         y_avg=y_cols_aggr[k]["avg"][not_nan], y_max=y_cols_aggr[k]["max"][not_nan],
                         color=color, label=k)


def compute_plot_min_avg_max(x, y_cols, y_cols_colors):
    y_cols_aggr = {k: {"min": [], "avg": [], "max": []} for k in y_cols.keys()}
    for k in y_cols.keys():

        sizes = []
        for serie in y_cols[k]:
            sizes.append(len(serie))
        sizes = set(sizes)

        if len(sizes) > 1:
            print("WARNING DATA with several length !!! Missing DATA. Remove shorter by default")
            print("sizes", sizes)
            max_sizes = max(sizes)
            print("n serie before cleaning", len(y_cols[k]))
            x = [ x_list for x_list, serie in zip(x, y_cols[k]) if len(serie) == max_sizes]
            y_cols[k] = [serie for serie in y_cols[k] if len(serie) == max_sizes]
            print("n serie after cleaning", len(y_cols[k]))

            sizes = []
            for serie in y_cols[k]:
                sizes.append(len(serie))
            sizes = set(sizes)
            print("corrected sizes", sizes)

        col_array = np.array([serie.tolist() for serie in y_cols[k]])
        # print("compute_plot_min_avg_max", np.isnan(col_array).sum(axis=1))

        # print(k, type(col_array), col_array.shape)
        y_cols_aggr[k]["min"] = col_array.min(axis=0)
        y_cols_aggr[k]["avg"] = col_array.mean(axis=0)
        y_cols_aggr[k]["max"] = col_array.max(axis=0)

    # print("len(x), len(x[0]), len(y_cols_aggr)",len(x), len(x[0]), len(y_cols_aggr))
    plot_several_series(x[0], y_cols_aggr, y_cols_colors)


def get_title(current_dir):
    if "ipd_" in current_dir:
        base_algo = "REINFORCE"
    elif "coin_" in current_dir:
        base_algo = "PPO"


    if "_deploy_game_default_or_util_" in current_dir:
        title = "UCB1{selfish, util} in self-play"
        player_1, player_2 = f"UCB1({base_algo}){{selfish, util}}", f"UCB1({base_algo}){{selfish, util}}"

    elif "deploy_game_default_vs_default_or_util" in current_dir:
        title = "selfish vs UCB1{selfish, util}"
        player_1, player_2 = f"UCB1({base_algo}){{selfish}}", f"UCB1({base_algo}){{selfish, util}}"

    elif "deploy_game_util_vs_default_or_util" in current_dir:
        title = "util vs UCB1{selfish, util}"
        player_1, player_2 = f"UCB1({base_algo}){{util}}", f"UCB1({base_algo}){{selfish, util}}"

    elif "deploy_game_le_self_play" in current_dir:
        title = "UCB1{TFT-PPM(wUtil)} in self-play"
        player_1, player_2 = f"UCB1(TFT-PPM({base_algo})){{util}}", f"UCB1(TFT-PPM({base_algo})){{util}}"

    elif "deploy_game_le_with_naive_opponent" in current_dir:
        title = "TFT-PPM(wUtil) vs UCB1{{selfish, util}}"
        player_1, player_2 = f"UCB1(TFT-PPM({base_algo})){{util}}", f"UCB1({base_algo}){{selfish, util}}"

    elif "_self_play_" in current_dir:
        if '_ipm_' in current_dir:
            if "008_" in current_dir:
                title = "L-TFT in self-play, known LR"
            else:
                title = "L-TFT in self-play, unknown LR"
            player_1, player_2 = f"L-TFT({base_algo})", f"L-TFT({base_algo})"
        else:
            if "008_" in current_dir:
                title = "TFT-PPM in self-play, known LR"
            else:
                title = "TFT-PPM in self-play, unknown LR"
            player_1, player_2 = f"TFT-PPM({base_algo})", f"TFT-PPM({base_algo})"
    elif "_naive_opponent_" in current_dir or "_naive_opp_" in current_dir:
        if '_ipm_' in current_dir:
            if "008_" in current_dir:
                title = "L-TFT vs. naive opponent, known LR"
            else:
                title = "L-TFT vs. naive opponent, unknown LR"
            player_1, player_2 = f"L-TFT({base_algo})", f"{base_algo}"
        else:
            if "008_" in current_dir:
                title = "TFT-PPM vs. naive opponent, known LR"
            else:
                title = "TFT-PPM vs. naive opponent, unknown LR"
            player_1, player_2 = f"TFT-PPM({base_algo})", f"{base_algo}"
    elif "_exploiter_" in current_dir :
        if '_ipm_' in current_dir:
            if "008_" in current_dir:
                title = "L-TFT vs. Exploiter, known LR"
            else:
                title = "L-TFT vs. Exploiter, unknown LR"
            player_1, player_2 = f"L-TFT({base_algo})", f"Exploiter({base_algo})"
        else:
            raise ValueError()
            # if "008_" in current_dir:
            #     title = "TFT-PPM vs. Exploiter, known LR"
            # else:
            #     title = "TFT-PPM vs. Exploiter, unknown LR"
            # player_1, player_2 = f"TFT-PPM({base_algo})", f"Exploiter({base_algo})"
    elif "_naive_coop_" in current_dir:
        if '_ipm_' in current_dir:
            if "008_" in current_dir:
                title = "L-TFT vs. naive cooperative, known LR"
            else:
                title = "L-TFT vs. naive cooperative, unknown LR"
            player_1, player_2 = f"L-TFT({base_algo})", f"{base_algo}(wUtil)"
        else:
            if "008_" in current_dir:
                title = "TFT-PPM vs. naive cooperative, known LR"
            else:
                title = "TFT-PPM vs. naive cooperative, unknown LR"
            player_1, player_2 = f"TFT-PPM({base_algo})", f"{base_algo}(wUtil)"
    elif "_util_" in current_dir:
        title = "baseline(wUtil)"
        player_1, player_2 = f"{base_algo}(wUtil)", f"{base_algo}(wUtil)"
    else:
        title = "baseline"
        player_1, player_2 = f"{base_algo}", f"{base_algo}"

    return title, player_1, player_2


def plot_for_ipd(input_df, current_dir, trial_idx):
    title_suffix, player_1, player_2 = get_title(current_dir)
    title = "IPD: " + title_suffix
    xlabel = "frame"
    ylabel = "prob"
    x_col = {xlabel: []}

    # if "TFT" in title:
    #     n_graphs = 3
    #     figsize =[6.4, 7.2]
    # else:
    n_graphs = 2
    figsize = [6.4, 4.8]
    n_steps_per_episode = 20

    y_cols = {"dd": [], "cc": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        fig = plt.figure(figsize=figsize)
        plt.subplot(n_graphs, 1, 1)

        y_cols_colors = {"dd": "orange", "cc": "blue"}
        all_y_cols = [el for el in y_cols.keys()]
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0, ma=True)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        plt.title(title)
        plt.ylim(-0.1, 1.1)
        plt.ylabel(ylabel)
        plt.legend(loc="upper left")
        plt.subplot(n_graphs, 1, 2)

        ylabel = "payoffs"
        y_cols = {"tot_r_ma": []}
        all_y_cols.extend([el for el in y_cols.keys()])

        # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_1: "orange"}
        y_cols = {player_1: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        y_cols = {"tot_r_ma": []}
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_2: "blue"}
        y_cols = {player_2: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        # if "TFT" not in title:
        plt.ylim(-4*n_steps_per_episode, +1*n_steps_per_episode)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="lower left")

        # if "TFT" in title:
        #     plt.subplot(n_graphs, 1, 3)
        #
        #     ylabel = "symlog(defection metric)"
        #     y_cols = {"d_carac": []}
        #     all_y_cols.extend([el for el in y_cols.keys()])
        #
        #     # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        #     # Exist for agent_n0 session_n0
        #     x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        #     assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
        #                 for x in x_col[xlabel]]), "x must be identical for all sessions"
        #     y_cols_colors = {player_1: "orange"}
        #     if len(y_cols.keys()) > 0:
        #         y_cols = {player_1: y_cols['d_carac']}
        #         compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)
        #
        #     y_cols = {"d_carac": []}
        #     x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        #     assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
        #                 for x in x_col[xlabel]]), "x must be identical for all sessions"
        #     y_cols_colors = {player_2: "blue"}
        #     if len(y_cols.keys()) > 0:
        #         y_cols = {player_2: y_cols['d_carac']}
        #         compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)
        #
        #     plt.axhline(y=0, linestyle= '--', color='grey')
        #     plt.xlabel(xlabel)
        #     plt.ylabel(ylabel)
        #     plt.yscale('symlog')
        #     plt.legend(loc="lower right")

        graph_name = '_'.join(['summary', "t" + str(trial_idx)] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        # plt.show()
        plt.close()


def plot_for_coin_game(input_df, current_dir, trial_idx):
    title_suffix, player_1, player_2 = get_title(current_dir)
    title = "Coin Game: " + title_suffix
    xlabel = "frame"
    ylabel = "fraction"
    x_col = {xlabel: []}

    # if "TFT" in title:
    #     n_graphs = 3
    #     figsize =[6.4, 7.2]
    # else:
    n_graphs = 2
    figsize = [6.4, 4.8]

    y_cols = {"pck_own": [], "pck_speed": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        fig = plt.figure(figsize=figsize)
        plt.subplot(n_graphs, 1, 1)

        y_cols_colors = {"pick own": "orange", "pick speed": "blue"}
        all_y_cols = [el for el in y_cols.keys()]
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0, ma=True)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols = {"pick own": y_cols['pck_own'], "pick speed": y_cols['pck_speed']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        plt.ylim(-0.1, 1.1)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")
        plt.subplot(n_graphs, 1, 2)

        ylabel = "payoffs"
        y_cols = {"tot_r_ma": []}
        all_y_cols.extend([el for el in y_cols.keys()])

        # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_1: "orange"}
        y_cols = {player_1: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        y_cols = {"tot_r_ma": []}
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_2: "blue"}
        y_cols = {player_2: y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel], y_cols, y_cols_colors)

        # if "TFT" not in title:
        plt.ylim(-10.5, +10.5)
        plt.yticks(np.arange(-10, +15, step=5.0))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.legend(loc="lower right")
        plt.legend(loc="lower left")

        # if "TFT" in title:
        #     plt.subplot(n_graphs, 1, 3)
        #
        #     ylabel = "symlog(defection metric)"
        #     y_cols = {"d_carac": []}
        #     all_y_cols.extend([el for el in y_cols.keys()])
        #
        #     # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        #     # Exist for agent_n0 session_n0
        #     x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        #     assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
        #                 for x in x_col[xlabel]]), "x must be identical for all sessions"
        #     y_cols_colors = {player_1: "orange"}
        #     if len(y_cols.keys()) > 0:
        #         y_cols = {player_1: y_cols['d_carac']}
        #         compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)
        #
        #     y_cols = {"d_carac": []}
        #     x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        #     assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
        #                 for x in x_col[xlabel]]), "x must be identical for all sessions"
        #     y_cols_colors = {player_2: "blue"}
        #     if len(y_cols.keys()) > 0:
        #         y_cols = {player_2: y_cols['d_carac']}
        #         compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)
        #
        #     plt.axhline(y=0, linestyle= '--', color='grey')
        #     plt.xlabel(xlabel)
        #     plt.ylabel(ylabel)
        #     plt.yscale('symlog')
        #     plt.legend(loc="lower right")

        graph_name = '_'.join(['summary', "t" + str(trial_idx)] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        # plt.show()
        plt.close()




def plot_for_deployment_game(input_df, current_dir, trial_idx):
    title_suffix, player_1, player_2 = get_title(current_dir)
    title = "Learning Game: " + title_suffix
    xlabel = "frame"
    ylabel = "prob"
    x_col = {xlabel: []}


    if "TFT" in title:
        n_graphs = 3
        figsize =[6.4, 7.2]
    else:
        n_graphs = 2
        figsize = [6.4, 4.8]

    y_cols = {"deployed_algo_idx": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        fig = plt.figure(figsize=figsize)
        plt.subplot(n_graphs, 1, 1)

        all_y_cols = [el for el in y_cols.keys()]
        # Exist for agent_n0 session_n0
        x_col_ag0, y_cols_ag0 = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        x_col_ag1, y_cols_ag1 = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
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
                cc = (1-np.array(y_cols_ag0[key])) * np.array(y_cols_ag1[key])
            else:
                cc = np.array(y_cols_ag0[key]) * np.array(y_cols_ag1[key])
            for i in range(cc.shape[0]):
                cc_not_nan = np.logical_not(np.isnan(cc))
                cc[i,:][cc_not_nan[i,:]] = calc_sr_ma( pd.Series(cc[i,:][cc_not_nan[i,:]]))
            y_cols["cc"] = cc

            if inverse_player_1:
                dd = np.array(y_cols_ag0[key]) * (1-np.array(y_cols_ag1[key]))
            else:
                dd = (1-np.array(y_cols_ag0[key])) * (1-np.array(y_cols_ag1[key]))
            for i in range(dd.shape[0]):
                dd_not_nan = np.logical_not(np.isnan(dd))
                dd[i, :][dd_not_nan[i,:]] = calc_sr_ma(pd.Series(dd[i, :][dd_not_nan[i,:]]))
            y_cols["dd"] = dd
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        # for key in y_cols_ag0.keys():
        #     y_cols = {}
        #     y_cols_colors = {player_1: "orange"}
        #     ag0 = np.array(y_cols_ag0[key])
        #     for i in range(ag0.shape[0]):
        #         ag0_not_nan = np.logical_not(np.isnan(ag0))
        #         ag0[i,:][ag0_not_nan[i,:]] = calc_sr_ma( pd.Series(ag0[i,:][ag0_not_nan[i,:]]))
        #     y_cols[player_1] = ag0
        #     compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)
        #
        #     y_cols = {}
        #     y_cols_colors = {player_2: "blue"}
        #     ag1 = (1-np.array(y_cols_ag0[key])) * (1-np.array(y_cols_ag1[key]))
        #     for i in range(ag1.shape[0]):
        #         ag1_not_nan = np.logical_not(np.isnan(ag1))
        #         ag1[i, :][ag1_not_nan[i,:]] = calc_sr_ma(pd.Series(ag1[i, :][ag1_not_nan[i,:]]))
        #     y_cols[player_2] = ag1
        #     compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend(loc="upper right")
        plt.subplot(n_graphs, 1, 2)

        ylabel = "payoffs"
        y_cols = {"undisc_tot_r": []}
        all_y_cols.extend([el for el in y_cols.keys()])

        # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_1: "orange"}
        y_cols = {player_1: y_cols['undisc_tot_r']}
        for key in y_cols.keys():
            col = np.array(y_cols[key])
            for i in range(col.shape[0]):
                col_not_nan = np.logical_not(np.isnan(col))
                col[i,:][col_not_nan[i,:]] = calc_sr_ma( pd.Series(col[i,:][col_not_nan[i,:]]))
            y_cols[key] = col
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        y_cols = {"undisc_tot_r": []}
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {player_2: "blue"}
        y_cols = {player_2: y_cols['undisc_tot_r']}
        for key in y_cols.keys():
            col = np.array(y_cols[key])
            for i in range(col.shape[0]):
                col_not_nan = np.logical_not(np.isnan(col))
                col[i,:][col_not_nan[i,:]] = calc_sr_ma( pd.Series(col[i,:][col_not_nan[i,:]]))
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

            # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
            # Exist for agent_n0 session_n0
            x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
            assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                        for x in x_col[xlabel]]), "x must be identical for all sessions"
            y_cols_colors = {player_1: "orange"}
            if len(y_cols.keys()) > 0:
                y_cols = {player_1: y_cols['d_carac']}
                compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

            y_cols = {"d_carac": []}
            x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
            assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                        for x in x_col[xlabel]]), "x must be identical for all sessions"
            y_cols_colors = {player_2: "blue"}
            if len(y_cols.keys()) > 0:
                y_cols = {player_2: y_cols['d_carac']}
                compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

            plt.axhline(y=0, linestyle= '--', color='grey')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.yscale('symlog')
            plt.legend(loc="lower right")


        graph_name = '_'.join(['summary', "t" + str(trial_idx)] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        # plt.show()
        plt.close()

def plot_all(input_df, current_dir):

    for trial_idx, session_df in input_df.items():

        # Agents
        # name_time_pairs = [
        #         ('payoff', 'frames'),
        #         ('prob', 'frames'),
        #     ]
        # trial_metrics = df_to_trial_metrics(input_df)
        # viz.plot_trial(trial_spec=None, trial_metrics=trial_metrics, ma=False,
        #            prepath="none", graph_prepath="graph_prepath", title=title, name_time_pairs=name_time_pairs)

        # Agents
        plot_for_ipd(session_df, current_dir, trial_idx)
        plot_for_coin_game(session_df, current_dir, trial_idx)
        plot_for_deployment_game(session_df, current_dir, trial_idx)

        # World

        # Session


def process_one_dir(dir_path):
    print("for dir_path", dir_path)
    csv_files = find_all_files_with_ext(dir_path, ext='csv')
    # print("csv_files", csv_files)
    csv_files_sessions = extract_trial_and_session_from_file_name(csv_files)
    # print("csv_files_sessions", csv_files_sessions)
    input_df = init_df(csv_files_sessions)
    assert not all(input_df[0]["ag_idx_sess_idx"][0][0]['tot_r'] == input_df[0]["ag_idx_sess_idx"][1][0]['tot_r'])

    plot_all(input_df, dir_path)


if __name__ == "__main__":
    apply_to_all_dir('.', process_one_dir)
    apply_to_all_dir('.', lambda subdir: apply_to_all_dir(subdir, process_one_dir))
