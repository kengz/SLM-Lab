'''
The analysis module
Handles the analyses of the info and data space for experiment evaluation and design.
'''
from slm_lab.agent import AGENT_DATA_NAMES
from slm_lab.env import ENV_DATA_NAMES
from slm_lab.lib import logger, util, viz
import colorlover as cl
import numpy as np
import pandas as pd
import pydash as _

DATA_AGG_FNS = {
    't': 'sum',
    'reward': 'sum',
    'loss': 'mean',
    'explore_var': 'mean',
}


def get_session_data(session):
    '''Gather data from session: MDP, Agent, Env data, and form session_data.'''
    aeb_space = session.aeb_space
    data_names = AGENT_DATA_NAMES + ENV_DATA_NAMES
    agg_data_names = ['epi'] + list(DATA_AGG_FNS.keys())
    data_h_v_dict = {data_name: aeb_space.get_history_v(data_name)
                     for data_name in data_names}
    session_db_data_dict = {}
    session_data_dict = {}
    for aeb in aeb_space.aeb_list:
        data_h_dict = {data_name: data_h_v[aeb]
                       for data_name, data_h_v in data_h_v_dict.items()}
        reset_idx = np.isnan(data_h_dict['done'])
        nonreset_idx = ~reset_idx
        epi_h = reset_idx.astype(int).cumsum()
        t_h = np.ones(reset_idx.shape)
        data_h_dict['epi'] = epi_h
        data_h_dict['t'] = t_h
        df = pd.DataFrame({data_name: data_h_dict[data_name][nonreset_idx]
                           for data_name in ['epi', 't'] + data_names})
        agg_df = df[agg_data_names].groupby('epi').agg(DATA_AGG_FNS)
        agg_df.reset_index(drop=False, inplace=True)
        # TODO save full data to db
        session_db_data_dict[aeb] = df
        session_data_dict[aeb] = agg_df
    session_data = pd.concat(session_data_dict, axis=1)
    logger.debug(f'{session_data}')
    return session_data


def plot_session(session, session_data):
    aeb_space = session.aeb_space
    aeb_count = len(aeb_space.aeb_list)
    if aeb_count <= 8:
        palette = cl.scales[str(max(3, aeb_count))]['qual']['Set2']
    else:
        palette = util.interp(cl.scales['8']['qual']['Set2'], aeb_count)
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
    for idx, (a, e, b) in enumerate(aeb_space.aeb_list):
        aeb_str = f'{a}{e}{b}'
        agg_df = session_data.loc[:, (a, e, b)]
        fig_1 = viz.plot_line(
            agg_df, 'reward', 'epi', legend_name=aeb_str, draw=False, trace_kwargs={'legendgroup': aeb_str, 'line': {'color': palette[idx]}})
        fig.append_trace(fig_1.data[0], 1, 1)

        fig_2 = viz.plot_line(
            agg_df, ['loss'], 'epi', y2_col=['explore_var'], trace_kwargs={'legendgroup': aeb_str, 'showlegend': False, 'line': {'color': palette[idx]}}, draw=False)
        fig.append_trace(fig_2.data[0], 2, 1)
        fig.append_trace(fig_2.data[1], 3, 1)

    fig.layout['xaxis1'].update(title='epi', zerolinewidth=1)
    fig.layout['yaxis1'].update(fig_1.layout['yaxis'])
    fig.layout['yaxis1'].update(domain=[0.55, 1])

    fig.layout['yaxis2'].update(fig_2.layout['yaxis'])
    fig.layout['yaxis2'].update(showgrid=False, domain=[0, 0.45])
    fig.layout['yaxis3'].update(fig_2.layout['yaxis2'])
    fig.layout['yaxis3'].update(overlaying='y2', anchor='x2')
    fig.layout.update(_.pick(fig_1.layout, ['legend']))
    fig.layout.update(_.pick(fig_2.layout, ['legend']))
    fig.layout.update(title=session.spec['name'], width=500, height=600)
    viz.plot(fig)
    return fig


def save_session_data(session_spec, session_data, session_fig):
    '''
    Save the session data: spec, df, plot.
    session_data is multi-indexed with (a,e,b), 3 extra levels
    to read, use: session_data = util.read(filepath, header=[0, 1, 2, 3])
    session_data = util.aeb_df_to_df_dict(session_data)
    @returns session_data for trial/experiment level agg.
    '''
    # TODO generalize to use experiment timestamp, id, sesison coor in info space, to replace timestamp
    spec_name = session_spec['name']
    prepath = f'data/{spec_name}/{spec_name}_{util.get_timestamp()}'
    logger.info(f'Saving session data to {prepath}_*')
    util.write(session_spec, f'{prepath}_spec.json')
    util.write(session_data, f'{prepath}_session_df.csv')
    viz.save_image(session_fig, f'{prepath}_session_graph.png')


def analyze_session(session):
    '''Gather session data, plot, and return session data (df) for high level agg.'''
    session_data = get_session_data(session)
    session_fig = plot_session(session, session_data)
    save_session_data(session.spec, session_data, session_fig)
    return session_data


def analyze_trial(trial):
    '''Gather trial data, plot, and return trial data (df) for high level agg.'''
    return trial_df


def analyze_experiment(experiment):
    '''Gather experiment data, plot, and return experiment data (df) for high level agg.'''
    return experiment_df
