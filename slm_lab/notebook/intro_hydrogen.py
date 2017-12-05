'''
Example Hydrogen notebook
Use `lab` as your Hydrogen kernel and run below interactively
'''
import pandas as pd
import pydash as _
from slm_lab.lib import util, viz

df = pd.DataFrame({
    'a': [0, 1, 2, 3, 4],
    'b': [0, 1, 4, 9, 16],
})

fig = viz.plot_area(df, ['a', 'b'])
fig = viz.plot_area(df, ['a'], y2_col=['b'])
fig = viz.plot_area(df, ['a', 'b'], stack=True)
fig = viz.plot_bar(df, ['b', 'a'])
fig = viz.plot_line(df, ['b', 'a'], save=False)
fig = viz.plot_line(df, ['a'], y2_col=['b'])
fig = viz.plot_scatter(df, ['b', 'a'])
fig = viz.plot_histogram(df, ['b'])

# pull plots to make multiple subplots
fig1 = viz.plot_area(df, ['a'], y2_col=['b'], draw=False)
fig2 = viz.plot_area(df, ['b'], draw=False)
fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
fig.append_trace(fig1.data[0], 1, 1)
fig.append_trace(fig1.data[1], 2, 1)
fig.append_trace(fig2.data[0], 3, 1)
fig.layout['yaxis1'].update(fig1.layout['yaxis'])
fig.layout['yaxis2'].update(fig1.layout['yaxis2'])
fig.layout['yaxis1'].update(domain=[0.55, 1])
fig.layout['yaxis3'].update(fig2.layout['yaxis'])
fig.layout['yaxis3'].update(domain=[0, 0.45])
fig.layout.update(_.pick(fig1.layout, ['legend']))
fig.layout.update(title='total_rewards vs time', width=500, height=400)
viz.py.iplot(fig)
