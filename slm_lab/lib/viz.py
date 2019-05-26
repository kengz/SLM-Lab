'''
The data visualization module
'''
from plotly import graph_objs as go, io as pio, tools
from plotly.offline import init_notebook_mode, iplot
from slm_lab.lib import logger, util
import colorlover as cl
import os
import pydash as ps


logger = logger.get_logger(__name__)
# warn orca failure only once
orca_warn_once = ps.once(lambda e: logger.warning(f'Failed to generate graph. Run retro-analysis to generate graphs later.'))
if util.is_jupyter():
    init_notebook_mode(connected=True)


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
        margin=go.layout.Margin(l=60, r=60, t=60, b=60),
    )
    layout.update(layout_kwargs)
    return layout


def get_palette(aeb_count):
    '''Get the suitable palette to plot for some number of aeb graphs, where each aeb is a color.'''
    if aeb_count <= 8:
        palette = cl.scales[str(max(3, aeb_count))]['qual']['Set2']
    else:
        palette = cl.interp(cl.scales['8']['qual']['Set2'], aeb_count)
    return palette


def lower_opacity(rgb, opacity):
    return rgb.replace('rgb(', 'rgba(').replace(')', f',{opacity})')


def plot(*args, **kwargs):
    if util.is_jupyter():
        return iplot(*args, **kwargs)


def plot_sr(sr, time_sr, title, y_title, x_title):
    '''Plot a series'''
    x = time_sr.tolist()
    color = get_palette(1)[0]
    main_trace = go.Scatter(
        x=x, y=sr, mode='lines', showlegend=False,
        line={'color': color, 'width': 1},
    )
    data = [main_trace]
    layout = create_layout(title=title, y_title=y_title, x_title=x_title)
    fig = go.Figure(data, layout)
    plot(fig)
    return fig


def plot_mean_sr(sr_list, time_sr, title, y_title, x_title):
    '''Plot a list of series using its mean, with error bar using std'''
    mean_sr, std_sr = util.calc_srs_mean_std(sr_list)
    max_sr = mean_sr + std_sr
    min_sr = mean_sr - std_sr
    max_y = max_sr.tolist()
    min_y = min_sr.tolist()
    x = time_sr.tolist()
    color = get_palette(1)[0]
    main_trace = go.Scatter(
        x=x, y=mean_sr, mode='lines', showlegend=False,
        line={'color': color, 'width': 1},
    )
    envelope_trace = go.Scatter(
        x=x + x[::-1], y=max_y + min_y[::-1], showlegend=False,
        line={'color': 'rgba(0, 0, 0, 0)'},
        fill='tozerox', fillcolor=lower_opacity(color, 0.2),
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
        pio.write_image(figure, filepath)
    except Exception as e:
        orca_warn_once(e)


# analysis plot methods

def plot_session(session_spec, session_metrics, session_df, df_mode='eval'):
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
    name_time_pairs = [
        ('mean_returns', 'frames'),
        ('strengths', 'frames'),
        ('sample_efficiencies', 'frames'),
        ('training_efficiencies', 'opt_steps'),
        ('stabilities', 'frames')
    ]
    for name, time in name_time_pairs:
        fig = plot_sr(
            local_metrics[name], local_metrics[time], title, name, time)
        save_image(fig, f'{graph_prepath}_session_graph_{df_mode}_{name}_vs_{time}.png')
        if name in ('mean_returns', 'strengths'):  # save important graphs in prepath directly
            save_image(fig, f'{prepath}_session_graph_{df_mode}_{name}_vs_{time}.png')

    if df_mode == 'eval':
        return
    # training plots from session_df
    name_time_pairs = [
        ('loss', 'frame'),
        ('explore_var', 'frame'),
        ('entropy', 'frame'),
    ]
    for name, time in name_time_pairs:
        fig = plot_sr(
            session_df[name], session_df[time], title, name, time)
        save_image(fig, f'{graph_prepath}_session_graph_{df_mode}_{name}_vs_{time}.png')


def plot_trial(trial_spec, trial_metrics):
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
    name_time_pairs = [
        ('mean_returns', 'frames'),
        ('strengths', 'frames'),
        ('sample_efficiencies', 'frames'),
        ('training_efficiencies', 'opt_steps'),
        ('stabilities', 'frames'),
        ('consistencies', 'frames'),
    ]
    for name, time in name_time_pairs:
        if name == 'consistencies':
            fig = plot_sr(
                local_metrics[name], local_metrics[time], title, name, time)
        else:
            fig = plot_mean_sr(
                local_metrics[name], local_metrics[time], title, name, time)
        save_image(fig, f'{graph_prepath}_trial_graph_{name}_vs_{time}.png')
        if name in ('mean_returns', 'strengths'):  # save important graphs in prepath directly
            save_image(fig, f'{prepath}_trial_graph_{name}_vs_{time}.png')


def plot_experiment(experiment_spec, experiment_df, metrics_cols):
    '''
    Plot the metrics vs. specs parameters of an experiment, where each point is a trial.
    ref colors: https://plot.ly/python/heatmaps-contours-and-2dhistograms-tutorial/#plotlys-predefined-color-scales
    '''
    y_cols = metrics_cols
    x_cols = ps.difference(experiment_df.columns.tolist(), y_cols)
    fig = tools.make_subplots(rows=len(y_cols), cols=len(x_cols), shared_xaxes=True, shared_yaxes=True, print_grid=False)
    strength_sr = experiment_df['strength']
    min_strength = strength_sr.values.min()
    max_strength = strength_sr.values.max()
    for row_idx, y in enumerate(y_cols):
        for col_idx, x in enumerate(x_cols):
            x_sr = experiment_df[x]
            guard_cat_x = x_sr.astype(str) if x_sr.dtype == 'object' else x_sr
            trace = go.Scatter(
                y=experiment_df[y], yaxis=f'y{row_idx+1}',
                x=guard_cat_x, xaxis=f'x{col_idx+1}',
                showlegend=False, mode='markers',
                marker={
                    'symbol': 'circle-open-dot', 'color': experiment_df['strength'], 'opacity': 0.5,
                    # dump first quarter of colorscale that is too bright
                    'cmin': min_strength - 0.50 * (max_strength - min_strength), 'cmax': max_strength,
                    'colorscale': 'YlGnBu', 'reversescale': True
                },
            )
            fig.add_trace(trace, row_idx + 1, col_idx + 1)
            fig.layout[f'xaxis{col_idx+1}'].update(title='<br>'.join(ps.chunk(x, 20)), zerolinewidth=1, categoryarray=sorted(guard_cat_x.unique()))
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
