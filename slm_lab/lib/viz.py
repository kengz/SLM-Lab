'''
The data visualization module
TODO pie, swarm, box plots
'''
from plotly import (
    graph_objs as go,
    offline as py,
    tools,
)
from slm_lab.lib import logger, util
from subprocess import Popen, DEVNULL
import colorlover as cl
import math
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


def get_palette(aeb_count):
    '''Get the suitable palette to plot for some number of aeb graphs, where each aeb is a color.'''
    if aeb_count <= 8:
        palette = cl.scales[str(max(3, aeb_count))]['qual']['Set2']
    else:
        palette = interp(cl.scales['8']['qual']['Set2'], aeb_count)
    return palette


def interp(scl, r):
    '''
    Replacement for colorlover.interp
    Interpolate a color scale "scl" to a new one with length "r"
    Fun usage in IPython notebook:
    HTML( to_html( to_hsl( interp( cl.scales['11']['qual']['Paired'], 5000 ) ) ) )
    '''
    c = []
    SCL_FI = len(scl) - 1  # final index of color scale
    # garyfeng:
    # the following line is buggy.
    # r = [x * 0.1 for x in range(r)] if isinstance( r, int ) else r
    r = [x * 1.0 * SCL_FI / r for x in range(r)] if isinstance(r, int) else r
    # end garyfeng

    scl = cl.to_numeric(scl)

    def interp3(fraction, start, end):
        '''Interpolate between values of 2, 3-member tuples'''
        def intp(f, s, e):
            return s + (e - s) * f
        return tuple([intp(fraction, start[i], end[i]) for i in range(3)])

    def rgb_to_hsl(rgb):
        '''
        Adapted from M Bostock's RGB to HSL converter in d3.js
        https://github.com/mbostock/d3/blob/master/src/color/rgb.js
        '''
        r, g, b = float(rgb[0]) / 255.0,\
            float(rgb[1]) / 255.0,\
            float(rgb[2]) / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        h = s = l = (mx + mn) / 2
        if mx == mn:  # achromatic
            h = 0
            s = 0 if l > 0 and l < 1 else h
        else:
            d = mx - mn
            s = d / (mx + mn) if l < 0.5 else d / (2 - mx - mn)
            if mx == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif mx == g:
                h = (b - r) / d + 2
            else:
                h = r - g / d + 4

        return (int(round(h * 60, 4)), int(round(s * 100, 4)), int(round(l * 100, 4)))

    for i in r:
        # garyfeng: c_i could be rounded up so scl[c_i+1] will go off range
        # c_i = int(i*math.floor(SCL_FI)/round(r[-1])) # start color index
        # c_i = int(math.floor(i*math.floor(SCL_FI)/round(r[-1]))) # start color index
        # c_i = if c_i < len(scl)-1 else hsl_o

        c_i = int(math.floor(i))
        section_min = math.floor(i)
        section_max = math.ceil(i)
        fraction = (i - section_min)  # /(section_max-section_min)

        hsl_o = rgb_to_hsl(scl[c_i])  # convert rgb to hls
        hsl_f = rgb_to_hsl(scl[c_i + 1])
        # section_min = c_i*r[-1]/SCL_FI
        # section_max = (c_i+1)*(r[-1]/SCL_FI)
        # fraction = (i-section_min)/(section_max-section_min)
        hsl = interp3(fraction, hsl_o, hsl_f)
        c.append('hsl' + str(hsl))

    return cl.to_hsl(c)


def lower_opacity(rgb, opacity):
    return rgb.replace('rgb(', 'rgba(').replace(')', f',{opacity})')


def plot(*args, **kwargs):
    if util.is_jupyter():
        return py.iplot(*args, **kwargs)
    else:
        kwargs.update({'auto_open': ps.get(kwargs, 'auto_open', False)})
        return py.plot(*args, **kwargs)


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
        proc = Popen(cmd, cwd=dirname, shell=True, stderr=DEVNULL, stdout=DEVNULL)
        try:
            outs, errs = proc.communicate(timeout=20)
        except TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
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
