import pandas as pd
import pydash as _
from unity_lab.lib import util, viz

df = pd.DataFrame({
    'x': [0, 1, 2, 3, 4],
    'y': [0, 1, 4, 9, 16],
})
fig = viz.plot_area(df, ['x', 'y'])
fig = viz.plot_area(df, ['x', 'y'], stack=True)
fig = viz.plot_bar(df, ['y', 'x'])
fig = viz.plot_line(df, ['y', 'x'])
fig = viz.plot_scatter(df, ['y', 'x'])
fig = viz.plot_histogram(df, ['y'])
