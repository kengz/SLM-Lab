'''Plots a fan chart of two trials based on mean and std deviation of episode rewards'''

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def get_session_data(session_file):
    df = pd.read_csv(session_file)
    df = df.ix[3:, '0.2']
    df = df.as_matrix().astype(float)
    return df, df.shape[0]


def get_trial_data(path, trial_num):
    session_dfs = [os.path.join(path, f) for f in os.listdir(path) if f.find('session_df') != -1]
    print(f'{len(session_dfs)} sessions in trial {trial_num}')
    data = []
    lengths = []
    for sess in session_dfs:
        d, length = get_session_data(sess)
        data.append(d)
        lengths.append(length)
    max_len = max(lengths)
    print(f'Max length of session: {max_len}')
    data_arr = np.zeros((len(session_dfs), max_len))
    print(f'Data array: {data_arr.shape}')
    early_stop = 0
    for i, d in enumerate(data):
        if d.shape[0] < max_len:
            data_arr[i, :d.shape[0]] = d
            data_arr[i, d.shape[0]:] = np.nan
            early_stop += 1
        else:
            data_arr[i] = d
    mean = np.nanmean(data_arr, axis=0)
    std = np.nanstd(data_arr, axis=0)
    dmin = np.nanmin(data_arr, axis=0)
    dmax = np.nanmax(data_arr, axis=0)
    print(f'Num sessions: {len(session_dfs)}, early stop: {early_stop}')
    return (mean, std, dmin, dmax)


def plot_datagroup(ax, data, color, label):
    num_epis = data[0].shape[0]
    x = list(range(num_epis))
    ax.plot(x, data[0], color=color, label=label, linewidth=1.0)
    ax.fill_between(x, data[0], data[0] + data[1], color=color, alpha=0.2)
    ax.fill_between(x, data[0], data[0] - data[1], color=color, alpha=0.2)


def plot(data_1, data_2, args):
    fig, ax = plt.subplots(dpi=300)
    plot_datagroup(ax, data_1, 'green', args.label1)
    plot_datagroup(ax, data_2, 'blue', args.label2)
    plt.title(args.title)
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Cumulative rewards per episode \nMean and +/- 1 stddev')
    plt.xlabel('Episode')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig(os.path.join(args.savepath, args.figname + '.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fan chart')
    parser.add_argument('--t1', type=str, default="",
                        help='Path to data folder for trial 1')
    parser.add_argument('--t2', type=str, default="",
                        help='Path to data folder for trial 2')
    parser.add_argument('--title', type=str, default="Test",
                        help='Chart title')
    parser.add_argument('--label1', type=str, default="label1",
                        help='Legend label 1')
    parser.add_argument('--label2', type=str, default="label2",
                        help='Legend label 2')
    parser.add_argument('--savepath', type=str, default="./data",
                        help='Where to save the plot')
    parser.add_argument('--figname', type=str, default="test.png",
                        help='Name to save the figure')
    args = parser.parse_args()
    data_1 = get_trial_data(args.t1, 1)
    data_2 = get_trial_data(args.t2, 2)
    plot(data_1, data_2, args)
