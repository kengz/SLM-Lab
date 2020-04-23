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
        if os.path.isdir(path):
            if path[0] != ".":
                fn_to_apply(path)


def find_all_files_with_ext(dir, ext):
    return [el for el in Path(dir).rglob(f'*.{ext}')]


def extract_session_from_file_name(file_names_list):
    path_session_dict = {"ag_idx_sess_idx": {},
                         "world": {},
                         "end_of_session": {}}
    for path in file_names_list:
        session_idx = int(re.search(r"_s(\d+)_", path.name).groups()[0])

        if re.match("^agent_.*", path.name):
            # Agent csv file
            agent_idx = int(re.search(r"^agent_n(\d+)_", path.name).groups()[0])
            if agent_idx not in path_session_dict["ag_idx_sess_idx"].keys():
                path_session_dict["ag_idx_sess_idx"][agent_idx] = {}
            path_session_dict["ag_idx_sess_idx"][agent_idx][session_idx] = path

        elif re.match("^world_.*", path.name):
            # World csv file
            path_session_dict["world"][session_idx] = path
        else:
            # End of session csv file
            path_session_dict["end_of_session"][session_idx] = path

    return path_session_dict


def init_df(csv_nested_dict):
    all_df = {}
    for k, v in csv_nested_dict.items():
        if isinstance(v, dict):
            all_df[k] = init_df(v)
        else:
            all_df[k] = pd.read_csv(v)
    return all_df

PLOT_MA_WINDOW = 10

def calc_sr_ma(sr):
    '''Calculate the moving-average of a series to be plotted'''
    return sr.rolling(PLOT_MA_WINDOW, min_periods=1).mean()

def get_data_sessions(data_dict, x_col, y_cols, agent_idx, ma=False):
    # for agent_idx, sessions in data_dict.items():
    sessions = data_dict[agent_idx]
    for session_idx, session_df in sessions.items():
        for col in session_df.columns:
            for key in y_cols.keys():
                if key in col:
                    if ma:
                        y_cols[key].append(calc_sr_ma(session_df[col]))
                    else:
                        y_cols[key].append(session_df[col])

                    # print("append", col)
            for key in x_col.keys():
                if key in col:
                    x_col[key].append(session_df[col])
                    # print("append", col)
    return x_col, y_cols


def plot_min_avg_max(x, y_min, y_avg, y_max, color, label):
    plt.plot(x, y_avg, '-', color=color, label=label)
    plt.fill_between(x, y_min, y_max, color=color, alpha=0.2)
    # plt.xlim(0, 10)
    # plt.show()


def plot_several_series(x, y_cols_aggr, y_cols_colors):
    for k in y_cols_aggr.keys():
        color = y_cols_colors[k]
        plot_min_avg_max(x, y_min=y_cols_aggr[k]["min"],
                         y_avg=y_cols_aggr[k]["avg"], y_max=y_cols_aggr[k]["max"],
                         color=color, label=k)


def compute_plot_min_avg_max(x, y_cols, y_cols_colors):
    y_cols_aggr = {k: {"min": [], "avg": [], "max": []} for k in y_cols.keys()}
    for k in y_cols.keys():
        col_array = np.array([serie.tolist() for serie in y_cols[k]])
        y_cols_aggr[k]["min"] = col_array.min(axis=0)
        y_cols_aggr[k]["avg"] = col_array.mean(axis=0)
        y_cols_aggr[k]["max"] = col_array.max(axis=0)

    plot_several_series(x, y_cols_aggr, y_cols_colors)



def get_title(current_dir):
    if "_self_play_" in current_dir :
        title = "TFT-PM(wUtil) self-play"
    elif "_naive_opponent_" in current_dir :
        title = "TFT-PM(wUtil) naive opponent"
    elif "_util_" in current_dir :
        title = "baseline(wUtil)"
    else :
        title = "baseline"
    return title

def plot_for_ipd(input_df, current_dir):

    title = "IPD " + get_title(current_dir)
    xlabel = "frame"
    ylabel = "prob"
    x_col = {xlabel: []}



    y_cols = {"dd": [], "cc": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):

        fig = plt.figure()
        plt.subplot(2, 1, 1)

        y_cols_colors = {"dd": "orange", "cc": "blue"}
        all_y_cols = [el for el in y_cols.keys()]
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0, ma=True)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend(loc="center right")
        plt.subplot(2, 1, 2)

        ylabel = "payoffs"
        y_cols = {"tot_r_ma": []}
        all_y_cols.extend([el for el in y_cols.keys()])

        # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {"player 1": "orange"}
        y_cols = {"player 1": y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        y_cols = {"tot_r_ma": []}
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {"player 2": "blue"}
        y_cols = {"player 2": y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")

        graph_name = '_'.join(['summary'] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        # plt.show()


def plot_for_coin_game(input_df, current_dir):
    title = "Coin Game " + get_title(current_dir)
    xlabel = "frame"
    ylabel = "fraction"
    x_col = {xlabel: []}

    y_cols = {"pck_own": [], "pck_speed": []}
    if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):

        fig = plt.figure()
        plt.subplot(2, 1, 1)

        y_cols_colors = {"pick own": "orange", "pick speed": "blue"}
        all_y_cols = [el for el in y_cols.keys()]
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0, ma=True)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols = {"pick own": y_cols['pck_own'], "pick speed": y_cols['pck_speed']}
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")
        plt.subplot(2, 1, 2)

        ylabel = "payoffs"
        y_cols = {"tot_r_ma": []}
        all_y_cols.extend([el for el in y_cols.keys()])

        # if any([y in input_df["ag_idx_sess_idx"][0][0].columns for y in y_cols.keys()]):
        # Exist for agent_n0 session_n0
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=0)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {"player 1": "orange"}
        y_cols = {"player 1": y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        y_cols = {"tot_r_ma": []}
        x_col, y_cols = get_data_sessions(input_df["ag_idx_sess_idx"], x_col, y_cols, agent_idx=1)
        assert all([all([x_el == x_col_el for x_el, x_col_el in zip(x, x_col["frame"][0])])
                    for x in x_col[xlabel]]), "x must be identical for all sessions"
        y_cols_colors = {"player 2": "blue"}
        y_cols = {"player 2": y_cols['tot_r_ma']}
        compute_plot_min_avg_max(x_col[xlabel][0], y_cols, y_cols_colors)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")

        graph_name = '_'.join(['summary'] + all_y_cols + ['vs'] + list(x_col.keys()))
        save_to = f"{current_dir }/{graph_name}"
        fig.savefig(save_to, dpi=fig.dpi)
        # plt.show()

def plot_all(input_df, current_dir):
    # Agents
    # name_time_pairs = [
    #         ('payoff', 'frames'),
    #         ('prob', 'frames'),
    #     ]
    # trial_metrics = df_to_trial_metrics(input_df)
    # viz.plot_trial(trial_spec=None, trial_metrics=trial_metrics, ma=False,
    #            prepath="none", graph_prepath="graph_prepath", title=title, name_time_pairs=name_time_pairs)

    # Agents
    plot_for_ipd(input_df, current_dir)
    plot_for_coin_game(input_df, current_dir)

    # World

    # Session


def process_one_dir(dir_path):
    print("for dir_path", dir_path)
    csv_files = find_all_files_with_ext(dir_path, ext='csv')
    # print("csv_files", csv_files)
    csv_files_sessions = extract_session_from_file_name(csv_files)
    # print("csv_files_sessions", csv_files_sessions)
    input_df = init_df(csv_files_sessions)
    # print("input_df", input_df)
    plot_all(input_df, dir_path)


if __name__ == "__main__":
    apply_to_all_dir('.', process_one_dir)
