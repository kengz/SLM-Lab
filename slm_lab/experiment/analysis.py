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

    mdp_data = {}
    for aeb in aeb_space.aeb_list:
        # remove last entry (env reset after termination)
        reward_h = reward_h_v[aeb][:-1]
        reset_idx = np.isnan(reward_h)
        nonreset_idx = ~reset_idx
        epi_h = reset_idx.astype(int).cumsum()
        df = pd.DataFrame({
            'reward': reward_h[nonreset_idx],
            'epi': epi_h[nonreset_idx],
        })
        agg_df = df.groupby('epi').agg('sum')
        agg_df.reset_index(drop=False, inplace=True)
        # TODO write file properly once session_data is in tabular form
        print(agg_df)
        util.write(agg_df, f"data/{session.spec['name']}_{aeb}_mdp_data.csv")
        mdp_data[aeb] = agg_df

    agent_data = {}
    for agent in aeb_space.agent_space.agents:
        body = agent.flat_nonan_body_a[0]
        aeb = body.aeb
        loss_h = np.array(agent.loss_history)
        explore_var_h = np.array(agent.explore_var_history)

        reward_h = reward_h_v[aeb][:-1]
        reset_idx = np.isnan(reward_h)
        nonreset_idx = ~reset_idx
        epi_h = reset_idx.astype(int).cumsum()
        df = pd.DataFrame({
            'loss': loss_h[nonreset_idx],
            'explore_var': explore_var_h[nonreset_idx],
            'epi': epi_h[nonreset_idx],
        })
        agg_df = df.groupby('epi').agg('mean')
        agg_df.reset_index(drop=False, inplace=True)
        agent_data[agent.a] = agg_df
    # TODO form proper session data for plot and return
    return mdp_data, agent_data


def plot_session(session, mdp_data, agent_data):
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)

    for a, df in agent_data.items():
        agent_fig = viz.plot_line(
            df, ['loss'], y2_col=['explore_var'], draw=False)
        fig.append_trace(agent_fig.data[0], 1, 1)
        fig.append_trace(agent_fig.data[1], 2, 1)

    for aeb, df in mdp_data.items():
        body_fig = viz.plot_line(
            df, 'reward', 'epi', legend_name=str(aeb), draw=False)
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
    mdp_data, agent_data = get_session_data(session)
    session_data = pd.DataFrame()
    plot_session(session, mdp_data, agent_data)
    return session_data
