# The data visualization module
# Defines plotting methods for analysis
from glob import glob
from plotly import graph_objs as go, io as pio, subplots
from plotly.offline import init_notebook_mode, iplot
from slm_lab.lib import logger, util
from slm_lab.lib.env_var import log_extra
import colorlover as cl
import os
import pydash as ps

logger = logger.get_logger(__name__)
pio.templates.default = 'none'  # set default white background for plots
# moving-average window size for plotting
PLOT_MA_WINDOW = 100
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
        legend=dict(
            x=0.0, y=-0.25,  # further below plot to avoid clipping
            orientation='h',  # horizontal
            font=dict(size=9),
            bgcolor='rgba(0,0,0,0)',  # transparent background (no box)
            bordercolor='rgba(0,0,0,0)',  # no border
            borderwidth=0,
            tracegroupgap=0,  # enable multi-column wrapping
            itemsizing='constant',  # consistent spacing
        ),
        yaxis=dict(rangemode='tozero', title=y_title),
        xaxis=dict(type=x_type, title=x_title),
        width=width, height=height,
        margin=go.layout.Margin(l=60, r=30, t=60, b=140),  # more bottom margin for lower legend
    )
    layout.update(layout_kwargs)
    return layout


def get_palette(size):
    '''Get the suitable palette of a certain size'''
    if size <= 8:
        palette = cl.scales[str(max(3, size))]['qual']['Set2']
    else:
        palette = cl.interp(cl.scales['8']['qual']['Set2'], size)
    return palette


def lower_opacity(rgb, opacity):
    return rgb.replace('rgb(', 'rgba(').replace('hsl(', 'hsla(').replace(')', f', {opacity})')


def plot(*args, **kwargs):
    if util.is_jupyter():
        return iplot(*args, **kwargs)


def plot_sr(sr, time_sr, title, y_title, x_title, color=None):
    '''Plot a series'''
    color = color or get_palette(1)[0]
    main_trace = go.Scatter(
        x=time_sr, y=sr, mode='lines', showlegend=False,
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


def save_image(figure, filepath):
    if os.environ['PY_ENV'] == 'test':
        return
    filepath = util.smart_path(filepath)
    try:
        pio.write_image(figure, filepath, scale=2)
    except Exception as e:
        logger.warning(f'Failed to generate graph. Run retro-analysis to generate graphs later. {e}\nIf running on a headless server, prepend your command with `xvfb-run -a `, for example `xvfb-run -a uv run slm-lab`')


# analysis plot methods

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

    local_metrics = session_metrics['local']
    # Always plot mean_returns
    name_time_pairs = [('mean_returns', 'frames')]
    
    # Only add extra metrics if log_extra is enabled
    if log_extra():
        name_time_pairs.extend([
            ('strengths', 'frames'),
            ('sample_efficiencies', 'frames'),
            ('training_efficiencies', 'opt_steps'),
            ('stabilities', 'frames'),
        ])
    
    for name, time in name_time_pairs:
        sr = local_metrics[name]
        if ma:
            sr = calc_sr_ma(sr)
            name = f'{name}_ma'  # for labeling
        fig = plot_sr(
            sr, local_metrics[time], title, name, time)
        save_image(fig, f'{graph_prepath}_session_graph_{df_mode}_{name}_vs_{time}.png')

    if ma:
        return
    # training plots from session_df
    name_time_pairs = [
        ('loss', 'frame'),
        ('explore_var', 'frame'),
        ('entropy', 'frame'),
    ]
    for name, time in name_time_pairs:
        if name in session_df.columns:  # Only plot if column exists
            fig = plot_sr(
                session_df[name], session_df[time], title, name, time)
            save_image(fig, f'{graph_prepath}_session_graph_{df_mode}_{name}_vs_{time}.png')


def plot_trial(trial_spec, trial_metrics, ma=False):
    '''
    Plot the trial graphs:
    - mean_returns, strengths, sample_efficiencies, training_efficiencies, stabilities (with error bar)
    - consistencies (no error bar)
    '''
    meta_spec = trial_spec['meta']
    prepath = meta_spec['prepath']
    graph_prepath = meta_spec['graph_prepath']
    title = f'trial graph: {trial_spec["name"]} t{meta_spec["trial"]} {meta_spec["max_session"]} sessions'

    local_metrics = trial_metrics['local']
    # Always plot mean_returns
    name_time_pairs = [('mean_returns', 'frames')]
    
    # Only add extra metrics if log_extra is enabled
    if log_extra():
        name_time_pairs.extend([
            ('strengths', 'frames'),
            ('sample_efficiencies', 'frames'),
            ('training_efficiencies', 'opt_steps'),
            ('stabilities', 'frames'),
            ('consistencies', 'frames'),
        ])
    
    for name, time in name_time_pairs:
        if name == 'consistencies':
            sr = local_metrics[name]
            if ma:
                sr = calc_sr_ma(sr)
                name = f'{name}_ma'  # for labeling
            fig = plot_sr(
                sr, local_metrics[time], title, name, time)
        else:
            sr_list = local_metrics[name]
            if ma:
                sr_list = [calc_sr_ma(sr) for sr in sr_list]
                name = f'{name}_ma'  # for labeling
            fig = plot_mean_sr(
                sr_list, local_metrics[time], title, name, time)
        save_image(fig, f'{graph_prepath}_trial_graph_{name}_vs_{time}.png')
        if name in ('mean_returns', 'mean_returns_ma'):  # save important graphs in prepath directly
            save_image(fig, f'{prepath}_trial_graph_{name}_vs_{time}.png')


def plot_experiment(experiment_spec, experiment_df, metrics_cols):
    '''
    Plot the metrics vs. specs parameters of an experiment, where each point is a trial.
    ref colors: https://plot.ly/python/heatmaps-contours-and-2dhistograms-tutorial/#plotlys-predefined-color-scales
    '''
    # Exclude frame from Y-axis (not a meaningful metric to plot vs hparams)
    y_cols = [col for col in metrics_cols if col != 'frame']

    # Filter extra metrics if log_extra is disabled
    if not log_extra():
        extra_metrics = ['strength', 'sample_efficiency', 'training_efficiency', 'stability', 'consistency',
                        'max_strength', 'final_strength']
        y_cols = [col for col in y_cols if col not in extra_metrics]

    # Handle both pandas Index and list columns
    columns = experiment_df.columns.tolist() if hasattr(experiment_df.columns, 'tolist') else list(experiment_df.columns)
    # Exclude all metrics_cols from x-axis (not just y_cols), only show hparams on x-axis
    x_cols = ps.difference(columns, metrics_cols + ['trial'])
    fig = subplots.make_subplots(rows=len(y_cols), cols=len(x_cols), shared_xaxes=True, shared_yaxes=True, print_grid=False)
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
                    'colorscale': 'YlGnBu', 'reversescale': False
                },
            )
            fig.add_trace(trace, row_idx + 1, col_idx + 1)
            fig.update_xaxes(title_text='<br>'.join(ps.chunk(x, 20)), zerolinewidth=1, categoryarray=sorted(guard_cat_x.unique()), row=len(y_cols), col=col_idx+1)
        fig.update_yaxes(title_text=y, rangemode='tozero', row=row_idx+1, col=1)
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


def plot_multi_local_metrics(local_metrics_list, legend_list, name, time, title, palette=None, showlegend=True):
    '''Method to plot list local_metrics gathered from multiple trials, with ability to specify custom legend and title. Used by plot_multi_trial'''
    palette = palette or get_palette(len(local_metrics_list))
    all_data = []
    for idx, local_metrics in enumerate(local_metrics_list):
        fig = plot_mean_sr(
            local_metrics[name], local_metrics[time], '', name, time, color=palette[idx])
        if legend_list is not None:
            # update legend for the main trace - only show line in legend (no rectangle)
            fig.data[0].update({
                'showlegend': showlegend,
                'name': legend_list[idx],
            })
            # Hide envelope trace from legend (it makes rectangles)
            if len(fig.data) > 1:
                fig.data[1].update({'showlegend': False})
        all_data += list(fig.data)
    layout = create_layout(title, name, time)
    fig = go.Figure(all_data, layout)
    # Update layout: preserve trace order in legend, add spacing
    fig.update_layout(legend=dict(traceorder='normal', tracegroupgap=5))
    return fig


def restore_metric_types(local_metrics: dict) -> None:
    """Restore pandas types after JSON deserialization (JSON converts Series to lists)."""
    import pandas as pd

    # Convert time axes (single Series)
    for key in ['frames', 'opt_steps']:
        if isinstance(local_metrics.get(key), list):
            local_metrics[key] = pd.Series(local_metrics[key])

    # Convert metric lists (list of Series)
    for key in ['mean_returns', 'strengths', 'sample_efficiencies', 'training_efficiencies', 'stabilities']:
        if key in local_metrics and isinstance(local_metrics[key], list):
            local_metrics[key] = [pd.Series(sr) if isinstance(sr, list) else sr for sr in local_metrics[key]]

    # consistencies is a single Series
    if isinstance(local_metrics.get('consistencies'), list):
        local_metrics['consistencies'] = pd.Series(local_metrics['consistencies'])


def plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=False, name_time_pairs=None, frame_scales=None, palette=None, showlegend=True):
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
    for local_metrics in local_metrics_list:
        restore_metric_types(local_metrics)
    
    # for plotting with async runs to adjust frame scale
    if frame_scales is not None:
        for idx, scale in frame_scales:
            local_metrics_list[idx]['frames'] = local_metrics_list[idx]['frames'] * scale
    # Always plot mean_returns, only add extra metrics if log_extra is enabled
    if name_time_pairs is None:
        name_time_pairs = [('mean_returns', 'frames')]
        if log_extra():
            name_time_pairs.extend([
                ('strengths', 'frames'),
                ('sample_efficiencies', 'frames'),
                ('training_efficiencies', 'opt_steps'),
                ('stabilities', 'frames')
            ])
    for name, time in name_time_pairs:
        if ma:
            for local_metrics in local_metrics_list:
                sr_list = local_metrics[name]
                sr_list = [calc_sr_ma(sr) for sr in sr_list]
                local_metrics[f'{name}_ma'] = sr_list
            name = f'{name}_ma'  # for labeling
        fig = plot_multi_local_metrics(local_metrics_list, legend_list, name, time, title, palette, showlegend)
        save_image(fig, f'{graph_prepath}_multi_trial_graph_{name}_vs_{time}.png')
        if name in ('mean_returns', 'mean_returns_ma'):  # save important graphs in prepath directly
            prepath = graph_prepath.replace('/graph/', '/')
            save_image(fig, f'{prepath}_multi_trial_graph_{name}_vs_{time}.png')


def get_trial_legends(experiment_df, trial_idxs, metrics_cols):
    '''Format trial legends as simple trial numbers (hparams visible in experiment_graph)'''
    # Just show trial number - hparams are already visible in experiment_graph
    return [f't{trial_idx}' for trial_idx in trial_idxs]


def plot_experiment_trials(experiment_spec, experiment_df, metrics_cols):
    meta_spec = experiment_spec['meta']
    info_prepath = meta_spec['info_prepath']
    
    # Always look in info_prepath for trial_metrics.json files
    # Trials save to the main info directory, not Ray Tune subdirectories
    # For single-session trials, prefer _s0 suffix files, fallback to non-suffixed
    trial_metrics_path_list = glob(f'{info_prepath}*_s0_trial_metrics.json')
    if not trial_metrics_path_list:
        # Fallback to non-suffixed pattern for backward compatibility
        trial_metrics_path_list = glob(f'{info_prepath}*_trial_metrics.json')
        # Filter out _s0 files if they somehow got included
        trial_metrics_path_list = [p for p in trial_metrics_path_list if '_s0_trial_metrics.json' not in p]
    
    if not trial_metrics_path_list:
        logger.warning('No trial metrics found for multi-trial plot')
        return
    
    # sort by trial id (handle Ray Tune paths gracefully)
    def safe_sort_key(path):
        trial_idx = util.prepath_to_idxs(path)[0]
        return trial_idx if trial_idx is not None else 0
    trial_metrics_path_list = list(sorted(trial_metrics_path_list, key=safe_sort_key))

    # get trial indices to build legends - sorted for consistent ordering
    # Use trial indices from experiment_df instead of parsing from paths (Ray Tune compatibility)
    trial_idxs = sorted(experiment_df['trial'].tolist())
    legend_list = get_trial_legends(experiment_df, trial_idxs, metrics_cols)

    title = f'multi trial graph: {experiment_spec["name"]}'
    graph_prepath = meta_spec['graph_prepath']
    plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath)
    plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=True)
