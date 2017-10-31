'''
The data visualization module
TODO pie, swarm, box plots
'''

import os
import plotly
import pydash as _
from plotly import (
    graph_objs as go,
    offline as py,
)
from unity_lab.lib import util
from unity_lab import config

PLOT_FILEDIR = util.smart_path('data')
os.makedirs(PLOT_FILEDIR, exist_ok=True)
if util.is_jupyter():
    py.init_notebook_mode(connected=True)


def save_image(figure, filename=None):
    if filename is None:
        filename = _.get(figure, 'layout.title') + '.png'
    filepath = f'{PLOT_FILEDIR}/{filename}'

    plotly.tools.set_credentials_file(
        username=_.get(config, 'plotly.username'),
        api_key=_.get(config, 'plotly.api_key'))
    plotly.tools.set_config_file(
        world_readable=True, sharing='public')
    return plotly.plotly.image.save_as(figure, filepath)


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
    y_col_list, x_col_list, legend_name_list = _.map_(
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
        legend=dict(x=0.0, y=-0.4, orientation='h'),
        yaxis=dict(rangemode='tozero', title=y_title),
        xaxis=dict(type=x_type, title=x_title),
        width=width, height=height,
        margin=go.Margin(l=70, r=70, t=70, b=70),
    )
    layout.update(layout_kwargs)
    return layout


def plot_go(
        df, y_col=None, x_col='index',
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

    label = create_label(
        y_col, x_col, title, y_title, x_title, legend_name)
    layout = create_layout(
        x_type=x_type, width=width, height=height,
        layout_kwargs=layout_kwargs,
        **_.pick(label, ['title', 'y_title', 'x_title']))

    y_col_list, x_col_list = label['y_col_list'], label['x_col_list']
    trace_num = max(len(y_col_list), len(y_col_list))
    data = []
    for idx in range(trace_num):
        y_c, x_c = _.get(y_col_list, idx), _.get(x_col_list, idx)
        df_y, df_x = _.get(df, y_c), _.get(df, x_c)
        trace = _.get(go, trace_class)(
            y=df_y, x=df_x,
            name=label['legend_name_list'][idx])
        trace.update(trace_kwargs)
        data.append(trace)

    figure = go.Figure(data=data, layout=layout)
    if draw:
        py.iplot(figure)
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
    trace_kwargs = _.merge(dict(fill=fill), trace_kwargs)
    layout_kwargs = _.merge(dict(), layout_kwargs)
    return plot_go(
        *args, trace_class='Scatter',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_bar(
    *args, barmode='stack', orientation='v',
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot bar chart from df'''
    trace_kwargs = _.merge(dict(orientation=orientation), trace_kwargs)
    layout_kwargs = _.merge(dict(barmode=barmode), layout_kwargs)
    return plot_go(
        *args, trace_class='Bar',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_line(
    *args,
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot line from df'''
    trace_kwargs = _.merge(dict(), trace_kwargs)
    layout_kwargs = _.merge(dict(), layout_kwargs)
    return plot_go(
        *args, trace_class='Scatter',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_scatter(
    *args,
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot scatter from df'''
    trace_kwargs = _.merge(dict(mode='markers'), trace_kwargs)
    layout_kwargs = _.merge(dict(), layout_kwargs)
    return plot_go(
        *args, trace_class='Scatter',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)


def plot_histogram(
    *args, barmode='overlay', xbins=None, histnorm='count', orientation='v',
    trace_kwargs=None, layout_kwargs=None,
        **kwargs):
    '''Plot histogram from df'''
    trace_kwargs = _.merge(dict(orientation=orientation,
                                xbins={}, histnorm=histnorm), trace_kwargs)
    layout_kwargs = _.merge(dict(barmode=barmode), layout_kwargs)
    return plot_go(
        *args, trace_class='Histogram',
        trace_kwargs=trace_kwargs, layout_kwargs=layout_kwargs,
        **kwargs)
