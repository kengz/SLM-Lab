import cufflinks as cf
import pydash as _
from plotly import (
    graph_objs as go,
    offline as py,
)
from unity_lab.lib import util

cf.set_config_file(offline=True, world_readable=False)
py.init_notebook_mode(connected=True)


def create_label(
        y_col, x_col,
        title=None, y_title=None, x_title=None, legend_name=None):
    '''Create label dict for go.Layout with smart resolution'''
    y_title = y_title or y_col
    x_title = x_title or x_col
    title = title or f'{y_title} vs {x_title}'
    legend_name = legend_name or y_col
    y_col_list, x_col_list, legend_name_list = _.map_(
        [y_col, x_col, legend_name], util.wrap_list)

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
        width=500, height=350):
    '''simplified method to generate Layout'''
    layout = go.Layout(
        title=title,
        legend=dict(x=0.0, y=-0.2, orientation='h'),
        yaxis=dict(rangemode='tozero', title=y_title),
        xaxis=dict(type=x_type, title=x_title),
        width=width, height=height,
        margin=go.Margin(l=70, r=70, t=70, b=70),
    )
    return layout


def plot_scatter(
        df, y_col, x_col=None,
        title=None, y_title=None, x_title=None, x_type=None,
        legend_name=None, width=500, height=350, draw=True):
    '''Draw scatter plot from df'''
    df = df.copy()
    if x_col is None:
        x_col = 'index'
        df['index'] = df.index.tolist()

    label = create_label(
        y_col, x_col, title, y_title, x_title, legend_name)
    layout = create_layout(
        title=label['title'], y_title=label['y_title'],
        x_title=label['x_title'], x_type=x_type,
        width=width, height=height)

    data = []
    for idx, y_c in enumerate(label['y_col_list']):
        x_c = _.get(label['x_col_list'], idx, default=x_col)
        trace = go.Scatter(
            y=df[y_c], x=df[x_c],
            name=label['legend_name_list'][idx],
        )
        data.append(trace)

    figure = go.Figure(data=data, layout=layout)
    if draw:
        py.iplot(figure)
    return figure
