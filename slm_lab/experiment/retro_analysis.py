'''
The retro analysis module
Runs analysis after a lab run using existing data files
e.g. yarn retro_analyze data/reinforce_cartpole_2018_01_22_211751
'''
from slm_lab.experiment import analysis
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import numpy as np
import os
import pydash as ps
import regex as re

logger = logger.get_logger(__name__)


def session_data_from_file(predir, trial_index, session_index, ckpt=None):
    '''Build session.session_data from file'''
    ckpt_str = '' if ckpt is None else f'_ckpt-{ckpt}'
    for filename in os.listdir(predir):
        if filename.endswith(f'_t{trial_index}_s{session_index}{ckpt_str}_session_df.csv'):
            filepath = f'{predir}/{filename}'
            session_df = util.read(filepath, header=[0, 1, 2, 3], index_col=0)
            session_data = util.session_df_to_data(session_df)
            return session_data


def session_datas_from_file(predir, trial_spec, trial_index, ckpt=None):
    '''Return a dict of {session_index: session_data} for a trial'''
    session_datas = {}
    for s in range(trial_spec['meta']['max_session']):
        session_data = session_data_from_file(predir, trial_index, s, ckpt)
        if session_data is not None:
            session_datas[s] = session_data
    return session_datas


def session_data_dict_from_file(predir, trial_index, ckpt=None):
    '''Build trial.session_data_dict from file'''
    ckpt_str = '' if ckpt is None else f'_ckpt-{ckpt}'
    session_data_dict = {}
    for filename in os.listdir(predir):
        if f'_t{trial_index}_' in filename and filename.endswith(f'{ckpt_str}_session_fitness_df.csv'):
            filepath = f'{predir}/{filename}'
            fitness_df = util.read(filepath, header=[0, 1, 2, 3], index_col=0, dtype=np.float32)
            util.fix_multi_index_dtype(fitness_df)
            session_index = fitness_df.index[0]
            session_data_dict[session_index] = fitness_df
    return session_data_dict


def session_data_dict_for_dist(spec, info_space):
    '''Method to retrieve session_datas (fitness df, so the same as session_data_dict above) when a trial with distributed sessions is done, to avoid messy multiprocessing data communication'''
    prepath = util.get_prepath(spec, info_space)
    predir, _, _, _, _, _ = util.prepath_split(prepath)
    session_datas = session_data_dict_from_file(predir, info_space.get('trial'), ps.get(info_space, 'ckpt'))
    session_datas = [session_datas[k] for k in sorted(session_datas.keys())]
    return session_datas


def trial_data_dict_from_file(predir):
    '''Build experiment.trial_data_dict from file'''
    trial_data_dict = {}
    for filename in os.listdir(predir):
        if filename.endswith('_trial_data.json'):
            filepath = f'{predir}/{filename}'
            exp_trial_data = util.read(filepath)
            trial_index = exp_trial_data.pop('trial_index')
            trial_data_dict[trial_index] = exp_trial_data
    return trial_data_dict


'''
Interface retro methods
'''


def analyze_eval_trial(spec, info_space, predir):
    '''Create a trial and run analysis to get the trial graph and other trial data'''
    from slm_lab.experiment.control import Trial
    trial = Trial(spec, info_space)
    trial.session_data_dict = session_data_dict_from_file(predir, trial.index, ps.get(info_space, 'ckpt'))
    analysis.analyze_trial(trial)


def run_parallel_eval(spec, info_space, ckpt):
    '''
    Calls a subprocess to run lab in eval mode with the constructed ckpt prepath, same as how one would manually run the bash cmd
    @example

    python run_lab.py data/dqn_cartpole_2018_12_19_224811/dqn_cartpole_t0_spec.json dqn_cartpole eval@dqn_cartpole_t0_s1_ckpt-epi10-totalt1000
    '''
    prepath_t = util.get_prepath(spec, info_space, unit='trial')
    prepath_s = util.get_prepath(spec, info_space, unit='session')
    predir, _, prename, spec_name, _, _ = util.prepath_split(prepath_s)
    cmd = f'python run_lab.py {prepath_t}_spec.json {spec_name} eval@{prename}_ckpt-{ckpt}'
    logger.info(f'Running parallel eval for ckpt-{ckpt}')
    return util.run_cmd(cmd)


def run_parallel_eval(session, agent, env):
    '''Plugin to session to run parallel eval for train mode'''
    if util.get_lab_mode() == 'train':
        ckpt = f'epi{env.clock.epi}-totalt{env.clock.total_t}'
        agent.save(ckpt=ckpt)
        # set reference to eval process for handling
        session.eval_proc = run_parallel_eval(session.spec, session.info_space, ckpt)


def try_wait_parallel_eval(session):
    '''Plugin to wait for session's final parallel eval if any'''
    if hasattr(session, 'eval_proc') and session.eval_proc is not None:  # wait for final eval before closing
        util.run_cmd_wait(session.eval_proc)
        session_retro_eval(session)  # rerun failed eval


def run_parallel_eval_from_prepath(prepath):
    '''Used by retro_eval'''
    spec, info_space = util.prepath_to_spec_info_space(prepath)
    ckpt = util.find_ckpt(prepath)
    return run_parallel_eval(spec, info_space, ckpt)


def run_wait_eval(prepath):
    '''Used by retro_eval'''
    eval_proc = run_parallel_eval_from_prepath(prepath)
    util.run_cmd_wait(eval_proc)


def retro_analyze_sessions(predir):
    '''Retro-analyze all session level datas.'''
    logger.info('Retro-analyzing sessions from file')
    from slm_lab.experiment.control import Session, SpaceSession
    for filename in os.listdir(predir):
        if filename.endswith('_session_df.csv'):
            prepath = f'{predir}/{filename}'.replace('_session_df.csv', '')
            spec, info_space = util.prepath_to_spec_info_space(prepath)
            trial_index, session_index = util.prepath_to_idxs(prepath)
            SessionClass = Session if spec_util.is_singleton(spec) else SpaceSession
            session = SessionClass(spec, info_space)
            session_data = session_data_from_file(predir, trial_index, session_index, ps.get(info_space, 'ckpt'))
            analysis.analyze_session(session, session_data)


def retro_analyze_trials(predir):
    '''Retro-analyze all trial level datas.'''
    logger.info('Retro-analyzing trials from file')
    from slm_lab.experiment.control import Trial
    for filename in os.listdir(predir):
        if filename.endswith('_trial_df.csv'):
            filepath = f'{predir}/{filename}'
            prepath = filepath.replace('_trial_df.csv', '')
            spec, info_space = util.prepath_to_spec_info_space(prepath)
            trial_index, _ = util.prepath_to_idxs(prepath)
            trial = Trial(spec, info_space)
            trial.session_data_dict = session_data_dict_from_file(predir, trial_index, ps.get(info_space, 'ckpt'))
            trial_fitness_df = analysis.analyze_trial(trial)

            # write trial_data that was written from ray search
            trial_data_filepath = filepath.replace('_trial_df.csv', '_trial_data.json')
            if os.path.exists(trial_data_filepath):
                fitness_vec = trial_fitness_df.iloc[0].to_dict()
                fitness = analysis.calc_fitness(trial_fitness_df)
                trial_data = util.read(trial_data_filepath)
                trial_data.update({
                    **fitness_vec, 'fitness': fitness, 'trial_index': trial_index,
                })
                util.write(trial_data, trial_data_filepath)


def retro_analyze_experiment(predir):
    '''Retro-analyze all experiment level datas.'''
    logger.info('Retro-analyzing experiment from file')
    from slm_lab.experiment.control import Experiment
    _, _, _, spec_name, _, _ = util.prepath_split(predir)
    prepath = f'{predir}/{spec_name}'
    spec, info_space = util.prepath_to_spec_info_space(prepath)
    if 'search' not in spec:
        return
    experiment = Experiment(spec, info_space)
    experiment.trial_data_dict = trial_data_dict_from_file(predir)
    if not ps.is_empty(experiment.trial_data_dict):
        return analysis.analyze_experiment(experiment)


def retro_analyze(predir):
    '''
    Method to analyze experiment from file after it ran.
    Read from files, constructs lab units, run retro analyses on all lab units.
    This method has no side-effects, i.e. doesn't overwrite data it should not.
    @example

    yarn retro_analyze data/reinforce_cartpole_2018_01_22_211751
    '''
    os.environ['PREPATH'] = f'{predir}/retro_analyze'  # to prevent overwriting log file
    logger.info(f'Retro-analyzing {predir}')
    retro_analyze_sessions(predir)
    retro_analyze_trials(predir)
    retro_analyze_experiment(predir)


def retro_eval(predir, session_index=None):
    '''
    Method to run eval sessions by scanning a predir for ckpt files. Used to rerun failed eval sessions.
    @example

    yarn retro_eval data/reinforce_cartpole_2018_01_22_211751
    '''
    logger.info(f'Retro-evaluate sessions from predir {predir}')
    # collect all unique prepaths first
    prepaths = []
    s_filter = '' if session_index is None else f'_s{session_index}_'
    for filename in os.listdir(predir):
        if filename.endswith('model.pth') and s_filter in filename:
            res = re.search('.+epi(\d+)-totalt(\d+)', filename)
            if res is not None:
                prepath = f'{predir}/{res[0]}'
                if prepath not in prepaths:
                    prepaths.append(prepath)
    if ps.is_empty(prepaths):
        return

    logger.info(f'Starting retro eval')
    np.random.shuffle(prepaths)  # so that CUDA_ID by trial/session index is spread out
    rand_spec = util.prepath_to_spec(prepaths[0])  # get any prepath, read its max session
    max_session = rand_spec['meta']['max_session']
    util.parallelize_fn(run_wait_eval, prepaths, num_cpus=max_session)


def session_retro_eval(session):
    '''retro_eval but for session at the end to rerun failed evals'''
    prepath = util.get_prepath(session.spec, session.info_space, unit='session')
    predir, _, _, _, _, _ = util.prepath_split(prepath)
    retro_eval(predir, session.index)
