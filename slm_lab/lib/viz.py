'''
The data visualization module
TODO pie, swarm, box plots
'''
from plotly import (
    graph_objs as go,
    offline as py,
    tools,
)
from slm_lab import config
from slm_lab.lib import logger, util
from subprocess import Popen, DEVNULL
import os
import plotly
import pydash as ps
import sys
import ujson as json

PLOT_FILEDIR = util.smart_path('data')
os.makedirs(PLOT_FILEDIR, exist_ok=True)
if util.is_jupyter():
    py.init_notebook_mode(connected=True)
logger = logger.get_logger(__name__)


def plot(*args, **kwargs):
    if util.is_jupyter():
        return py.iplot(*args, **kwargs)
    else:
        kwargs.update({'auto_open': ps.get(kwargs, 'auto_open', False)})
        return py.plot(*args, **kwargs)


def save_image(figure, filepath=None):
    if os.environ['PY_ENV'] == 'test':
        return
    if filepath is None:
        filepath = f'{PLOT_FILEDIR}/{ps.get(figure, "layout.title")}.png'
    filepath = util.smart_path(filepath)
    dirname, filename = os.path.split(filepath)
    try:
        cmd = f'orca graph -o {filename} \'{json.dumps(figure)}\''
        if 'linux' in sys.platform:
            cmd = 'xvfb-run -a -s "-screen 0 1400x900x24" -- ' + cmd
        Popen(cmd, cwd=dirname, shell=True, stderr=DEVNULL, stdout=DEVNULL)
        logger.info(f'Graph saved to {dirname}/{filename}')
    except Exception as e:
        logger.exception(
            'Please install orca for plotly and run retro-analysis to generate graphs.')


def stack_cumsum(df, y_col):
    '''Submethod to cumsum over y columns for stacked area plot'''
    y_col_list = util.cast_list(y_col)
    stack_df = df.copy()
    for idx in range(len(y_col_list)):
        col = y_col_list[idx]
        presum_idx = idx - 1
        if presum_idx > -1:
            presum_col = y_col_list[presum_idx]
            stack_df[col] += stack_df[presum_col]
    return stack_df


def create_label(
        y_col, x_col,
        title=None, y_title=None, x_title=None, legend_name=None):
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


def create_layout(
        title, y_title, x_title, x_type=None,
        width=500, height=350, layout_kwargs=None):
    '''simplified method to generate Layout'''
    layout = go.Layout(
        title=title,
        legend=dict(x=0.0, y=-0.25, orientation='h'),
        yaxis=dict(rangemode='tozero', title=y_title),
        xaxis=dict(type=x_type, title=x_title),
        width=width, height=height,
        margin=go.Margin(l=60, r=60, t=60, b=60),
    )
    layout.update(layout_kwargs)
    return layout


def plot_go(
        df, y_col=None, x_col='index', y2_col=None,
        title=None, y_title=None, x_title=None, x_type=None,
        legend_name=None, width=500, height=350, draw=True,
        save=False, filename=None,
        trace_class='Scatter', trace_kwargs=None, layout_kwargs=None):
    '''
    Quickly plot from df using trace_class, e.g. go.Scatter
    1. create_label() to auto-resolve labels
    2. create_layout() with go.Layout() and update(layout_kwargs)
    3. spread and create go.<trace_class>() and update(trace_kwargs)
    4. Create the figure and plot accordingly
    @returns figure
    '''
    df = df.copy()
    if x_col == 'index':
        df['index'] = df.index.tolist()

    label = create_label(y_col, x_col, title, y_title, x_title, legend_name)
    layout = create_layout(
        x_type=x_type, width=width, height=height, layout_kwargs=layout_kwargs,
        **ps.pick(label, ['title', 'y_title', 'x_title']))
    y_col_list, x_col_list = label['y_col_list'], label['x_col_list']

    if y2_col is not None:
        label2 = create_label(y2_col, x_col, title, y_title, x_title, legend_name)
        layout.update(dict(yaxis2=dict(
            rangemode='tozero', title=label2['y_title'],
            side='right', overlaying='y1', anchor='x1',
        )))
        y2_col_list, x_col_list = label2['y_col_list'], label2['x_col_list']
        label2_legend_name_list = label2['legend_name_list']
    else:
        y2_col_list = []
        label2_legend_name_list = []

    combo_y_col_list = y_col_list + y2_col_list
    combo_legend_name_list = label['legend_name_list'] + label2_legend_name_list
    y_col_num, x_col_num = len(combo_y_col_list), len(x_col_list)
    trace_num = max(y_col_num, x_col_num)
    data = []
    for idx in range(trace_num):
        y_c = ps.get(combo_y_col_list, idx % y_col_num)
        x_c = ps.get(x_col_list, idx % x_col_num)
        df_y, df_x = ps.get(df, y_c), ps.get(df, x_c)
        trace = ps.get(go, trace_class)(y=df_y, x=df_x, name=combo_legend_name_list[idx])
        trace.update(trace_kwargs)
        if idx >= len(y_col_list):
            trace.update(dict(yaxis='y2', xaxis='x1'))
        data.append(trace)

    figure = go.Figure(data=data, layout=layout)
    if draw:
        plot(figure)
    if save:
        save_image(figure, filename=filename)
    return figure


def plot_area(
    *args, fill='tonexty', stack=False,
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot area from df'''
    if stack:
        df, y_col = args[:2]
        stack_df = stack_cumsum(df, y_col)
        args = (stack_df,) + args[1:]
    trace_kwargs = ps.merge(dict(fill=fill, mode='lines', line=dict(width=1)), trace_kwargs)
    layout_kwargs = ps.merge(dict(), layout_kwargs)
    return plot_go(
        *args, trace_class='Scatter',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_bar(
    *args, barmode='stack', orientation='v',
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot bar chart from df'''
    trace_kwargs = ps.merge(dict(orientation=orientation), trace_kwargs)
    layout_kwargs = ps.merge(dict(barmode=barmode), layout_kwargs)
    return plot_go(
        *args, trace_class='Bar',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_line(
    *args,
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot line from df'''
    trace_kwargs = ps.merge(dict(mode='lines', line=dict(width=1)), trace_kwargs)
    layout_kwargs = ps.merge(dict(), layout_kwargs)
    return plot_go(
        *args, trace_class='Scatter',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_scatter(
    *args,
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot scatter from df'''
    trace_kwargs = ps.merge(dict(mode='markers'), trace_kwargs)
    layout_kwargs = ps.merge(dict(), layout_kwargs)
    return plot_go(
        *args, trace_class='Scatter',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_histogram(
    *args, barmode='overlay', xbins=None, histnorm='count', orientation='v',
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot histogram from df'''
    trace_kwargs = ps.merge(dict(orientation=orientation, xbins={}, histnorm=histnorm), trace_kwargs)
    layout_kwargs = ps.merge(dict(barmode=barmode), layout_kwargs)
    return plot_go(
        *args, trace_class='Histogram',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)
