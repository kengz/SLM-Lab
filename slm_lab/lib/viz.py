# The data visualization module
# Defines plotting methods for analysis
import os
from glob import glob

import colorlover as cl
import pydash as ps
from plotly import graph_objs as go, io as pio, tools
from plotly.offline import init_notebook_mode, iplot

from slm_lab.lib import logger, util
import numpy as np

logger = logger.get_logger(__name__)

# moving-average window size for plotting
PLOT_MA_WINDOW = 10
USE_Y_LOG_SCALE = ['lr', "param"]

# warn orca failure only once
orca_warn_once = ps.once(lambda e: logger.warning(
    f'Failed to generate graph. Run retro-analysis to generate graphs later. {e}\nIf running on a headless server, prepend your Python command with `xvfb-run -a `, for example `xvfb-run -a python run_lab.py`'))
if util.is_jupyter():
    init_notebook_mode(connected=True)


def calc_sr_ma(sr):
    '''Calculate the moving-average of a series to be plotted'''
    return sr.rolling(PLOT_MA_WINDOW, min_periods=1).mean()


def create_label(y_col, x_col, title=None, y_title=None, x_title=None, legend_name=None):
    '''Create label dict for go.Layout with smart resolution'''
    legend_name = legend_name or y_col
    y_col_list, x_col_list, legend_name_list = ps.map_(
        [y_col, x_col, legend_name], util.cast_list)
    y_title = str(y_title or ','.join(y_col_list))
    x_title = str(x_title or ','.join(x_col_list))
    title = title or f'{y_title} vs {x_title}'

    label = {
        'y_title': y_title,
        'x_title': x_title,
        'title': title,
        'y_col_list': y_col_list,
        'x_col_list': x_col_list,
        'legend_name_list': legend_name_list,
    }
    return label


def create_layout(title, y_title, x_title, x_type=None, width=500, height=500, layout_kwargs=None):
    '''simplified method to generate Layout'''

    layout = go.Layout(
        title=title,
        legend=dict(x=0.0, y=-0.25, orientation='h'),
        yaxis=dict(rangemode='tozero', title=y_title),
        xaxis=dict(type=x_type, title=x_title),
        width=width, height=height,
        margin=go.layout.Margin(l=60, r=30, t=60, b=60),
    )
    layout.update(layout_kwargs)

    if any(selection_mark in y_title for selection_mark in USE_Y_LOG_SCALE):
        layout.update(yaxis_type="log")

    return layout


def get_palette(size):
    '''Get the suitable palette of a certain size'''
    if size <= 8:
        palette = cl.scales[str(max(3, size))]['qual']['Set2']
    else:
        palette = cl.interp(cl.scales['8']['qual']['Set2'], size)
    return palette


def lower_opacity(rgb, opacity):
    return rgb.replace('rgb(', 'rgba(').replace(')', f',{opacity})')


def plot(*args, **kwargs):
    if util.is_jupyter():
        return iplot(*args, **kwargs)


def plot_sr(sr, time_sr, title, y_title, x_title, color=None):
    '''Plot a series'''
    x = time_sr.tolist()
    color = color or get_palette(1)[0]
    main_trace = go.Scatter(
        x=x, y=sr, mode='lines', showlegend=False,
        line={'color': color, 'width': 1},
    )
    data = [main_trace]
    layout = create_layout(title=title, y_title=y_title, x_title=x_title)
    fig = go.Figure(data, layout)
    plot(fig)
    return fig


def plot_mean_sr(sr_list, time_sr, title, y_title, x_title, color=None):
    '''Plot a list of series using its mean, with error bar using std'''
    mean_sr, std_sr = util.calc_srs_mean_std(sr_list)
    max_sr = mean_sr + std_sr
    min_sr = mean_sr - std_sr
    max_y = max_sr.tolist()
    min_y = min_sr.tolist()
    x = time_sr.tolist()
    color = color or get_palette(1)[0]
    main_trace = go.Scatter(
        x=x, y=mean_sr, mode='lines', showlegend=False,
        line={'color': color, 'width': 1},
    )
    envelope_trace = go.Scatter(
        x=x + x[::-1], y=max_y + min_y[::-1], showlegend=False,
        line={'color': 'rgba(0, 0, 0, 0)'},
        fill='tozerox', fillcolor=lower_opacity(color, 0.15),
    )
    data = [main_trace, envelope_trace]
    layout = create_layout(title=title, y_title=y_title, x_title=x_title)
    fig = go.Figure(data, layout)
    return fig

def plot_several_sr(sr_list, time_sr_list, title, y_title, x_title,
                    color=None, name_list=None):
    '''Plot several series'''
    layout = create_layout(title=title, y_title=y_title, x_title=x_title)
    fig = go.Figure(layout=layout)

    for idx, (sr, time_sr, name) in enumerate(
            zip(sr_list, time_sr_list, name_list)):
        x = time_sr.tolist()

        # # Remove Nan
        # not_nan = [not np.isnan(el) for el in sr]
        # x = [el for el, keep in zip(x, not_nan) if keep]
        # sr = [el for el, keep in zip(sr, not_nan) if keep]

        selected_color = color or get_palette(len(sr_list))[idx]
        main_trace = go.Scatter(
            x=x, y=sr, mode='lines', showlegend=True, name=name,
            line={'color': selected_color, 'width': 1},
        )
        fig.add_trace(main_trace)

    plot(fig)
    return fig


def plot_several_mean_sr(sr_list_list, time_sr_list, title, y_title, x_title,
                         color=None, name_list=None):
    '''Plot a list of series using its mean, with error bar using std'''
    layout = create_layout(title=title, y_title=y_title, x_title=x_title)
    fig = go.Figure(layout=layout)

    for idx, (sr_list, time_sr, name) in enumerate(
            zip(sr_list_list, time_sr_list, name_list)):
        mean_sr, std_sr = util.calc_srs_mean_std(sr_list)
        max_sr = mean_sr + std_sr
        min_sr = mean_sr - std_sr
        max_y = max_sr.tolist()
        min_y = min_sr.tolist()
        x = time_sr.tolist()

        # # Remove Nan
        # not_nan = [not np.isnan(el) for el in mean_sr]
        # x = [el for el, keep in zip(x, not_nan) if keep]
        # mean_sr = [el for el, keep in zip(mean_sr, not_nan) if keep]
        # max_y = [el for el, keep in zip(max_y, not_nan) if keep]
        # min_y = [el for el, keep in zip(min_y, not_nan) if keep]

        selected_color = color or get_palette(len(sr_list_list))[idx]
        main_trace = go.Scatter(
            x=x, y=mean_sr, mode='lines', showlegend=True, name=name,
            line={'color': selected_color, 'width': 1},
        )
        envelope_trace = go.Scatter(
            x=x + x[::-1], y=max_y + min_y[::-1], showlegend=False,
            line={'color': 'rgba(0, 0, 0, 0)'},
            fill='tozerox', fillcolor=lower_opacity(selected_color, 0.2),
        )
        fig.add_trace(main_trace)
        fig.add_trace(envelope_trace)

    return fig


def save_image(figure, filepath):
    if os.environ['PY_ENV'] == 'test':
        return
    filepath = util.smart_path(filepath)
    try:
        pio.write_image(figure, filepath, scale=2)
    except Exception as e:
        orca_warn_once(e)


# analysis plot methods
from slm_lab.agent.agent import agent


def add_all_extra_col(name_time_pairs, session_df):
    all_col = []
    for k in session_df.keys():
        all_col.extend(session_df[k].columns)
    all_col = list(set(all_col))

    already_plotted = [el[0] for el in name_time_pairs]
    for df_col in all_col:
        if df_col in already_plotted:
            continue
        if any(df_col == col for col in agent.BASIC_COLS):
            continue
        name_time_pairs.append((df_col, 'frame'))
    return name_time_pairs


def add_summary_plots(name_time_pairs, session_df):
    all_col = []
    for k in session_df.keys():
        all_col.extend(session_df[k].columns)
    all_col = list(set(all_col))

    all_col_for_multiple_alg = [col.split("_alg")[0] for col in all_col if "_alg" in col]
    summary_to_produce = list(set(all_col_for_multiple_alg))

    for summary_name in summary_to_produce:
        summary_col_list = []
        for df_col in all_col:
            if df_col == summary_name:
                summary_col_list.append(df_col)
            elif "_alg" in df_col:
                if df_col.split("_alg")[0] == summary_name:
                    summary_col_list.append(df_col)

        if len(summary_col_list) > 0:
            summary_col_list.sort()
            name_time_pairs.append((summary_col_list, 'frame'))

    return name_time_pairs

def remove_nan(sr,x):
    not_nan = [not np.isnan(el) for el in sr]
    sr = sr[not_nan]
    if len(not_nan) < len(x):
        not_nan_x = [False] * (len(x) - len(not_nan)) + not_nan
    elif len(not_nan) > len(x):
        raise NotImplementedError()
    else:
        not_nan_x = not_nan
    x = x[not_nan_x]

    return sr, x

def plot_session(session_spec, session_metrics, session_df, df_mode='eval', ma=False):
    '''
    Plot the session graphs:
    - mean_returns, strengths, sample_efficiencies, training_efficiencies, stabilities (with error bar)
    - additional plots from session_df: losses, exploration variable, entropy
    '''
    meta_spec = session_spec['meta']
    prepath = meta_spec['prepath']
    graph_prepath = meta_spec['graph_prepath']
    title = f'session graph: {session_spec["name"]} t{meta_spec["trial"]} s{meta_spec["session"]}'

    # local_metrics = session_metrics['local']
    name_time_pairs = [
        ('mean_r', 'frames'),
        ('strengths', 'frames'),
        ('sample_effici.', 'frames'),
        ('train._effici.', 'opt_steps'),
        ('stabilities', 'frames'),
    ]
    for col_names_to_select, time in name_time_pairs:
        str_list = []
        time_sr_list = []
        name_list = []

        for k in session_metrics.keys():
            local_metrics = session_metrics[k]['local']

            if col_names_to_select in local_metrics.keys():
                sr = local_metrics[col_names_to_select]
                x = local_metrics[time]

                # print("len(sr), len(x)", len(sr), len(x), "name",col_names_to_select, "k", k)
                # not_nan = [not np.isnan(el) for el in sr]
                # sr = sr[not_nan]
                # if len(not_nan) < len(x):
                #     not_nan_x = [False] * (len(x) - len(not_nan)) + not_nan
                # elif len(not_nan) > len(x):
                #     raise NotImplementedError()
                # else:
                #     not_nan_x = not_nan
                # x = x[not_nan_x]
                sr, x = remove_nan(sr,x)

                if ma:
                    sr = calc_sr_ma(sr)

                str_list.append(sr)
                time_sr_list.append(x)
                name_list.append(k)

        if ma:
            col_names_to_select = f'{col_names_to_select}_ma'  # for labeling

        fig = plot_several_sr(
            str_list, time_sr_list, title, col_names_to_select, time, name_list=name_list)
        save_image(fig, f'{graph_prepath}_session_graph_{df_mode}_{col_names_to_select}_vs_{time}.png')
        if col_names_to_select in ('mean_r', 'mean_r_ma'):  # save important graphs in prepath directly
            save_image(fig, f'{prepath}_session_graph_{df_mode}_{col_names_to_select}_vs_{time}.png')

    # if ma:
    #     return
    # training plots from session_df
    name_time_pairs = [
        ('loss', 'frame'),
        ('expl_var', 'frame'),
        ('entropy', 'frame'),
    ]
    name_time_pairs = add_all_extra_col(name_time_pairs, session_df)
    name_time_pairs = add_summary_plots(name_time_pairs, session_df)
    if ma:
        ma_exlusion = ['opt_step']
        name_time_pairs = [el for el in name_time_pairs if all([excluded not in el[0] for excluded in ma_exlusion])]
    for col_names_to_select, time in name_time_pairs:
        str_list = []
        time_sr_list = []
        name_list = []

        # Work with a list of col to select per plot
        if not isinstance(col_names_to_select, list):
            col_names_to_select = [col_names_to_select]

        for k in session_df.keys():
            s_df = session_df[k]

            for col in s_df.columns:
                if col in col_names_to_select:
                    # if col_names_to_select in s_df.columns:
                    sr = s_df[col]
                    x = s_df[time]

                    # not_nan = [not np.isnan(el) for el in sr]
                    # sr = sr[not_nan]
                    # if len(not_nan) < len(x):
                    #     not_nan_x = [False] * (len(x) - len(not_nan)) + not_nan
                    # elif len(not_nan) > len(x):
                    #     raise NotImplementedError()
                    # else:
                    #     not_nan_x = not_nan
                    # x = x[not_nan_x]
                    sr, x = remove_nan(sr, x)

                    if ma:
                        sr = calc_sr_ma(sr)

                    str_list.append(sr)
                    time_sr_list.append(x)
                    name_list.append(k + '_' + col)

        # Use the first one as label in plot
        if len(col_names_to_select) == 1:
            name = col_names_to_select[0]
        elif len(col_names_to_select) > 1:
            name = "all_" + col_names_to_select[0]

        if ma:
            name = f'{name}_ma'  # for labeling

        fig = plot_several_sr(
            str_list, time_sr_list, title, name, time, name_list=name_list)
        save_image(fig, f'{graph_prepath}_session_graph_{df_mode}_{name}_vs_{time}.png')


def plot_trial(trial_spec=None, trial_metrics=None, ma=False,
               prepath=None, graph_prepath=None, title=None, name_time_pairs=None):
    '''
    Plot the trial graphs:
    - mean_returns, strengths, sample_efficiencies, training_efficiencies, stabilities (with error bar)
    - consistencies (no error bar)
    '''
    meta_spec = trial_spec['meta'] if trial_spec is not None else None
    prepath = meta_spec['prepath'] if prepath is None else prepath
    graph_prepath = meta_spec['graph_prepath'] if graph_prepath is None else graph_prepath
    title = (f'trial graph: {trial_spec["name"]} t{meta_spec["trial"]} {meta_spec["max_session"]} sessions'
             if title is None else title)

    # local_metrics = trial_metrics['local']
    if name_time_pairs is None:
        name_time_pairs = [
            ('mean_r', 'frames'),
            ('strengths', 'frames'),
            ('sample_effici.', 'frames'),
            ('train._effici.', 'opt_steps'),
            ('stabilities', 'frames'),
            ('consistencies', 'frames'),
        ]
    for name, time in name_time_pairs:
        if name == 'consistencies':
            sr_list = []
            time_list = []
            name_list = []
            for k in trial_metrics.keys():
                if 'local' in trial_metrics[k].keys():
                    local_metrics = trial_metrics[k]['local']
                else:
                    local_metrics = trial_metrics[k]

                if name in local_metrics.keys():
                    sr = local_metrics[name]
                    x = local_metrics[time]

                    # TODO improve this (apply nan with keeping coherent min, max, avg)
                    # Not removing any nan. Otherwise the mean, avg, max means nothing ?
                    # not_nan = [not np.isnan(el) for el in sr]
                    # sr = sr[not_nan]
                    # if len(not_nan) < len(x):
                    #     not_nan_x = [False] * (len(x) - len(not_nan)) + not_nan
                    # elif len(not_nan) > len(x):
                    #     raise NotImplementedError()
                    # else:
                    #     not_nan_x = not_nan
                    # x = x[not_nan_x]

                    if ma:
                        sr = calc_sr_ma(sr)

                    sr_list.append(sr)
                    time_list.append(x)
                    name_list.append(k)
            if ma:
                name = f'{name}_ma'  # for labeling

            fig = plot_several_sr(
                sr_list, time_list, title, name, time, name_list=name_list)

        else:
            sr_list_list = []
            time_list = []
            name_list = []

            for k in trial_metrics.keys():
                if 'local' in trial_metrics[k].keys():
                    local_metrics = trial_metrics[k]['local']
                else:
                    local_metrics = trial_metrics[k]

                if name in local_metrics.keys():
                    sr_list = local_metrics[name]
                    x = local_metrics[time]

                    # TODO improve this (apply nan with keeping coherent min, max, avg)
                    # Not removing any nan. Otherwise the mean, avg, max means nothing ?

                    if ma:
                        # sr_list = [calc_sr_ma(sr.dropna()) for sr in sr_list]
                        sr_list = [calc_sr_ma(sr) for sr in sr_list]

                    sr_list_list.append(sr_list)
                    time_list.append(x)
                    name_list.append(k)

            if ma:
                name = f'{name}_ma'  # for labeling

            fig = plot_several_mean_sr(
                sr_list_list, time_list, title, name, time, name_list=name_list)

        save_image(fig, f'{graph_prepath}_trial_graph_{name}_vs_{time}.png')
        if name in ('mean_r', 'mean_r_ma'):  # save important graphs in prepath directly
            save_image(fig, f'{prepath}_trial_graph_{name}_vs_{time}.png')


def plot_experiment(experiment_spec, experiment_df, metrics_cols):
    '''
    Plot the metrics vs. specs parameters of an experiment, where each point is a trial.
    ref colors: https://plot.ly/python/heatmaps-contours-and-2dhistograms-tutorial/#plotlys-predefined-color-scales
    '''
    y_cols = metrics_cols
    x_cols = ps.difference(experiment_df.columns.tolist(), y_cols + ['trial'])
    fig = tools.make_subplots(rows=len(y_cols), cols=len(x_cols), shared_xaxes=True, shared_yaxes=True,
                              print_grid=False)
    strength_sr = experiment_df['strength']
    min_strength, max_strength = strength_sr.min(), strength_sr.max()
    for row_idx, y in enumerate(y_cols):
        for col_idx, x in enumerate(x_cols):
            x_sr = experiment_df[x]
            guard_cat_x = x_sr.astype(str) if x_sr.dtype == 'object' else x_sr
            trace = go.Scatter(
                y=experiment_df[y], yaxis=f'y{row_idx+1}',
                x=guard_cat_x, xaxis=f'x{col_idx+1}',
                showlegend=False, mode='markers',
                marker={
                    'symbol': 'circle-open-dot', 'color': strength_sr, 'opacity': 0.5,
                    # dump first portion of colorscale that is too bright
                    'cmin': min_strength - 0.5 * (max_strength - min_strength), 'cmax': max_strength,
                    'colorscale': 'YlGnBu', 'reversescale': True
                },
            )
            fig.add_trace(trace, row_idx + 1, col_idx + 1)
            fig.layout[f'xaxis{col_idx+1}'].update(title='<br>'.join(ps.chunk(x, 20)), zerolinewidth=1,
                                                   categoryarray=sorted(guard_cat_x.unique()))
        fig.layout[f'yaxis{row_idx+1}'].update(title=y, rangemode='tozero')
    fig.layout.update(
        title=f'experiment graph: {experiment_spec["name"]}',
        width=100 + 300 * len(x_cols), height=200 + 300 * len(y_cols))
    plot(fig)
    graph_prepath = experiment_spec['meta']['graph_prepath']
    save_image(fig, f'{graph_prepath}_experiment_graph.png')
    # save important graphs in prepath directly
    prepath = experiment_spec['meta']['prepath']
    save_image(fig, f'{prepath}_experiment_graph.png')
    return fig


# TODO modify to support unctions below
def plot_multi_local_metrics(local_metrics_list, legend_list, name, time, title, palette=None, showlegend=True):
    '''Method to plot list local_metrics gathered from multiple trials, with ability to specify custom legend and title. Used by plot_multi_trial'''
    palette = palette or get_palette(len(local_metrics_list))
    all_data = []
    for idx, local_metrics in enumerate(local_metrics_list):
        fig = plot_mean_sr(
            local_metrics[name], local_metrics[time], '', name, time, color=palette[idx])
        if legend_list is not None:
            # update legend for the main trace
            fig.data[0].update({'showlegend': showlegend, 'name': legend_list[idx]})
        all_data += list(fig.data)
    layout = create_layout(title, name, time)
    fig = go.Figure(all_data, layout)
    return fig


def plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=False, name_time_pairs=None,
                     frame_scales=None, palette=None, showlegend=True):
    '''
    Plot multiple trial graphs together
    This method can be used in analysis and also custom plotting by specifying the arguments manually
    @example

    trial_metrics_path_list = [
        'data/dqn_cartpole_2019_06_11_092512/info/dqn_cartpole_t0_trial_metrics.pkl',
        'data/dqn_cartpole_2019_06_11_092512/info/dqn_cartpole_t1_trial_metrics.pkl',
    ]
    legend_list = [
        '0',
        '1',
    ]
    title = f'Multi trial trial graphs'
    graph_prepath = 'data/my_exp'
    viz.plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath)
    '''
    local_metrics_list = [util.read(path)['local'] for path in trial_metrics_path_list]
    # for plotting with async runs to adjust frame scale
    if frame_scales is not None:
        for idx, scale in frame_scales:
            local_metrics_list[idx]['frames'] = local_metrics_list[idx]['frames'] * scale
    name_time_pairs = name_time_pairs or [
        ('mean_r', 'frames'),
        ('strengths', 'frames'),
        ('sample_effici.', 'frames'),
        ('train._effici.', 'opt_steps'),
        ('stabilities', 'frames')
    ]
    for name, time in name_time_pairs:
        if ma:
            for local_metrics in local_metrics_list:
                sr_list = local_metrics[name]
                sr_list = [calc_sr_ma(sr) for sr in sr_list]
                local_metrics[f'{name}_ma'] = sr_list
            name = f'{name}_ma'  # for labeling
        fig = plot_multi_local_metrics(local_metrics_list, legend_list, name, time, title, palette, showlegend)
        save_image(fig, f'{graph_prepath}_multi_trial_graph_{name}_vs_{time}.png')
        if name in ('mean_r', 'mean_r_ma'):  # save important graphs in prepath directly
            prepath = graph_prepath.replace('/graph/', '/')
            save_image(fig, f'{prepath}_multi_trial_graph_{name}_vs_{time}.png')


def get_trial_legends(experiment_df, trial_idxs, metrics_cols):
    '''Format trial variables in experiment_df into legend strings'''
    var_df = experiment_df.drop(metrics_cols, axis=1).set_index('trial')
    trial_legends = []
    for trial_idx in trial_idxs:
        trial_vars = var_df.loc[trial_idx].to_dict()
        var_list = []
        for k, v in trial_vars.items():
            if hasattr(v, '__round__'):
                v = round(v, 8)  # prevent long float digits in formatting
            var_list.append(f'{k.split(".").pop()} {v}')
        var_str = ' '.join(var_list)
        legend = f't{trial_idx}: {var_str}'
        trial_legends.append(legend)
    trial_legends
    return trial_legends


def plot_experiment_trials(experiment_spec, experiment_df, metrics_cols):
    meta_spec = experiment_spec['meta']
    info_prepath = meta_spec['info_prepath']
    trial_metrics_path_list = glob(f'{info_prepath}*_trial_metrics.pkl')
    # sort by trial id
    trial_metrics_path_list = list(sorted(trial_metrics_path_list, key=lambda k: util.prepath_to_idxs(k)[0]))

    # get trial indices to build legends
    trial_idxs = [util.prepath_to_idxs(prepath)[0] for prepath in trial_metrics_path_list]
    legend_list = get_trial_legends(experiment_df, trial_idxs, metrics_cols)

    title = f'multi trial graph: {experiment_spec["name"]}'
    graph_prepath = meta_spec['graph_prepath']
    plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath)
    plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=True)
