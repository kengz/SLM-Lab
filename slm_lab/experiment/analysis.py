'''
The analysis module
Handles the analyses of the info and data space for experiment evaluation and design.
'''
import colorlover as cl
import numpy as np
import pandas as pd
import pydash as _
from slm_lab.lib import logger, util, viz

DATA_AGG_FNS = {
    'reward': 'sum',
    'loss': 'mean',
    'explore_var': 'mean',
}


def get_session_data(session):
    '''Gather data from session: MDP, Agent, Env data, and form session_data.'''
    aeb_space = session.aeb_space
    done_h_v = aeb_space.get_history_v('done')
    reward_h_v = aeb_space.get_history_v('reward')
    loss_h_v = aeb_space.get_history_v('loss')
    explore_var_h_v = aeb_space.get_history_v('explore_var')

    session_df_dict = {}
    for aeb in aeb_space.aeb_list:
        # remove last entry (env reset after termination)
        done_h = done_h_v[aeb][:-1]
        reset_idx = np.isnan(done_h)
        nonreset_idx = ~reset_idx
        epi_h = reset_idx.astype(int).cumsum()
        reward_h = reward_h_v[aeb][:-1]
        loss_h = loss_h_v[aeb][:-1]
        explore_var_h = explore_var_h_v[aeb][:-1]
        # TODO save a non-agg data to db for mutual info research
        df = pd.DataFrame({
            'epi': epi_h[nonreset_idx],
            'reward': reward_h[nonreset_idx],
            'loss': loss_h[nonreset_idx],
            'explore_var': explore_var_h[nonreset_idx],
        })
        agg_df = df.groupby('epi').agg(DATA_AGG_FNS)
        agg_df.reset_index(drop=False, inplace=True)
        session_df_dict[aeb] = agg_df
    # multi-indexed with (a,e,b), 3 extra levels
    session_df = pd.concat(session_df_dict, axis=1)
    print(session_df)
    util.write(session_df, f"data/{session.spec['name']}_session_df.csv")
    # to read, use: session_df = util.read(filepath, header=[0, 1, 2, 3])
    # session_df_dict = util.aeb_df_to_df_dict(session_df)
    return session_df_dict


def plot_session(session, session_df_dict):
    aeb_list = sorted(session_df_dict.keys())
    aeb_count = len(aeb_list)
    if aeb_count <= 8:
        palette = cl.scales[str(max(3, aeb_count))]['qual']['Set2']
    else:
        palette = cl.interp(cl.scales['8']['qual']['Set2'], aeb_count)
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
    for idx, (a, e, b) in enumerate(aeb_list):
        aeb_str = f'{a}{e}{b}'
        agg_df = session_df_dict[(a, e, b)]
        fig_1 = viz.plot_line(
            agg_df, 'reward', 'epi', legend_name=aeb_str, draw=False, trace_kwargs={'legendgroup': aeb_str, 'line': {'color': palette[idx]}})
        fig.append_trace(fig_1.data[0], 1, 1)

        fig_2 = viz.plot_line(
            agg_df, ['loss'], 'epi', y2_col=['explore_var'], trace_kwargs={'legendgroup': aeb_str, 'showlegend': False, 'line': {'color': palette[idx]}}, draw=False)
        fig.append_trace(fig_2.data[0], 2, 1)
        fig.append_trace(fig_2.data[1], 3, 1)

    fig.layout['xaxis1'].update(title='epi')
    fig.layout['yaxis1'].update(fig_1.layout['yaxis'])
    fig.layout['yaxis1'].update(domain=[0.55, 1])

    fig.layout['yaxis2'].update(fig_2.layout['yaxis'])
    fig.layout['yaxis2'].update(showgrid=False, domain=[0, 0.45])
    fig.layout['yaxis3'].update(fig_2.layout['yaxis2'])
    fig.layout['yaxis3'].update(overlaying='y2')
    fig.layout.update(_.pick(fig_1.layout, ['legend']))
    fig.layout.update(_.pick(fig_2.layout, ['legend']))
    fig.layout.update(title=session.spec['name'], width=500, height=600)
    viz.plot(fig)
    viz.save_image(fig)


def analyze_session(session):
    '''Gather session data, plot, and return session data'''
    session_df_dict = get_session_data(session)
    plot_session(session, session_df_dict)
    return session_df_dict
