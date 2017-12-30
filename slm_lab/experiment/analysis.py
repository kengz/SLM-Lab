'''
The analysis module
Handles the analyses of the info and data space for experiment evaluation and design.
'''
import numpy as np
import pandas as pd
import pydash as _
from slm_lab.lib import logger, util, viz


def get_session_data(session):
    '''Gather data from session: MDP, Agent, Env data, and form session_data.'''
    aeb_space = session.aeb_space
    reward_h_v = np.stack(
        aeb_space.data_spaces['reward'].data_history, axis=3)
    done_h_v = np.stack(
        aeb_space.data_spaces['done'].data_history, axis=3)
    loss_h_v = np.stack(
        aeb_space.data_spaces['loss'].data_history, axis=3)
    explore_var_h_v = np.stack(
        aeb_space.data_spaces['explore_var'].data_history, axis=3)

    session_df_dict = {}
    for aeb in aeb_space.aeb_list:
        # remove last entry (env reset after termination)
        reward_h = reward_h_v[aeb][:-1]
        loss_h = loss_h_v[aeb][:-1]
        explore_var_h = explore_var_h_v[aeb][:-1]
        reset_idx = np.isnan(reward_h)
        nonreset_idx = ~reset_idx
        epi_h = reset_idx.astype(int).cumsum()
        # TODO save a non-agg data to db for mutual info research
        df = pd.DataFrame({
            'epi': epi_h[nonreset_idx],
            'reward': reward_h[nonreset_idx],
            'loss': loss_h[nonreset_idx],
            'explore_var': explore_var_h[nonreset_idx],
        })
        agg_df = df.groupby('epi').agg(
            {'reward': 'sum', 'loss': 'mean', 'explore_var': 'mean'})
        agg_df.reset_index(drop=False, inplace=True)
        session_df_dict[aeb] = agg_df
    # multi-indexed with (a,e,b), 3 extra levels
    session_df = pd.concat(session_df_dict, axis=1)
    print(session_df)
    util.write(session_df, f"data/{session.spec['name']}_session_df.csv")
    # to read, use: util.read(filepath, header=[0, 1, 2, 3])
    return session_df_dict


def plot_session(session, session_df_dict):
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
    for (a, e, b), agg_df in session_df_dict.items():
        aeb_str = f'{a}{e}{b}'
        # TODO swap plot order, group legend and colors
        agent_fig = viz.plot_line(
            agg_df, ['loss'], y2_col=['explore_var'], legend_name=[f'loss {aeb_str}', f'explore_var {aeb_str}'], draw=False)
        fig.append_trace(agent_fig.data[0], 1, 1)
        fig.append_trace(agent_fig.data[1], 2, 1)

        body_fig = viz.plot_line(
            agg_df, 'reward', 'epi', legend_name=f'reward {aeb_str}', draw=False)
        fig.append_trace(body_fig.data[0], 3, 1)

    fig.layout['yaxis1'].update(agent_fig.layout['yaxis'])
    fig.layout['yaxis1'].update(domain=[0.55, 1])
    fig.layout['yaxis2'].update(agent_fig.layout['yaxis2'])
    fig.layout['yaxis2'].update(showgrid=False)

    fig.layout['yaxis3'].update(body_fig.layout['yaxis'])
    fig.layout['yaxis3'].update(domain=[0, 0.45])
    fig.layout.update(_.pick(agent_fig.layout, ['legend']))
    fig.layout.update(_.pick(body_fig.layout, ['legend']))
    fig.layout.update(title=session.spec['name'], width=500, height=600)
    viz.plot(fig)
    viz.save_image(fig)


def analyze_session(session):
    '''Gather session data, plot, and return session data'''
    session_df_dict = get_session_data(session)
    session_data = pd.DataFrame()
    plot_session(session, session_df_dict)
    return session_data
