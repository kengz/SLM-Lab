# The spec module
# Manages specification to run things in lab
import os

import pydash as ps

from slm_lab import ROOT_DIR
from slm_lab.lib import logger, util
from slm_lab.lib.env_var import lab_mode

SPEC_DIR = 'slm_lab/spec'
'''
All spec values are already param, inferred automatically.
To change from a value into param range, e.g.
- single: "explore_anneal_epi": 50
- continuous param: "explore_anneal_epi": {"min": 50, "max": 100, "dist": "uniform"}
- discrete range: "explore_anneal_epi": {"values": [50, 75, 100]}
'''
SPEC_FORMAT = {
    "agent": {
        "name": str,
        "algorithm": dict,
        "memory": dict,
        "net": dict,
    },
    "env": {
        "name": str,
        "max_t": (type(None), int, float),
        "max_frame": (int, float),
    },
    "meta": {
        "max_session": int,
        "max_trial": (type(None), int),
    },
    "name": str,
}
logger = logger.get_logger(__name__)


def check_comp_spec(comp_spec, comp_spec_format):
    '''Base method to check component spec'''
    for spec_k, spec_format_v in comp_spec_format.items():
        comp_spec_v = comp_spec[spec_k]
        if ps.is_list(spec_format_v):
            v_set = spec_format_v
            assert comp_spec_v in v_set, f'Component spec value {ps.pick(comp_spec, spec_k)} needs to be one of {util.to_json(v_set)}'
        else:
            v_type = spec_format_v
            assert isinstance(comp_spec_v, v_type), f'Component spec {ps.pick(comp_spec, spec_k)} needs to be of type: {v_type}'
            if isinstance(v_type, tuple) and int in v_type and isinstance(comp_spec_v, float):
                # cast if it can be int
                comp_spec[spec_k] = int(comp_spec_v)


def check_compatibility(spec):
    '''Check compatibility among spec setups'''
    # TODO expand to be more comprehensive
    if spec['meta'].get('distributed') == 'synced':
        assert not util.use_gpu(ps.get(spec, 'agent.net.gpu')), 'Distributed mode "synced" works with CPU only. Set gpu: false.'


def check(spec):
    '''Check a single spec for validity'''
    try:
        spec_name = spec.get('name')
        assert set(spec.keys()) >= set(SPEC_FORMAT.keys()), f'Spec needs to follow spec.SPEC_FORMAT. Given \n {spec_name}: {util.to_json(spec)}'
        check_comp_spec(spec['agent'], SPEC_FORMAT['agent'])
        check_comp_spec(spec['env'], SPEC_FORMAT['env'])
        check_comp_spec(spec['meta'], SPEC_FORMAT['meta'])
        check_compatibility(spec)
    except Exception as e:
        logger.exception(f'spec {spec_name} fails spec check')
        raise e
    return True


def check_all():
    '''Check all spec files, all specs.'''
    spec_files = ps.filter_(os.listdir(SPEC_DIR), lambda f: f.endswith('.json') and not f.startswith('_'))
    for spec_file in spec_files:
        spec_dict = util.read(f'{SPEC_DIR}/{spec_file}')
        for spec_name, spec in spec_dict.items():
            # fill-in info at runtime
            spec['name'] = spec_name
            spec = extend_meta_spec(spec)
            try:
                check(spec)
            except Exception as e:
                logger.exception(f'spec_file {spec_file} fails spec check')
                raise e
    logger.info(f'Checked all specs from: {ps.join(spec_files, ",")}')
    return True


def extend_meta_spec(spec, experiment_ts=None):
    '''
    Extend meta spec with information for lab functions
    @param dict:spec
    @param str:experiment_ts Use this experiment_ts if given; used for resuming training
    '''
    extended_meta_spec = {
        'rigorous_eval': ps.get(spec, 'meta.rigorous_eval', 0),
        # reset lab indices to -1 so that they tick to 0
        'experiment': -1,
        'trial': -1,
        'session': -1,
        'cuda_offset': int(os.environ.get('CUDA_OFFSET', 0)),
        'resume': experiment_ts is not None,
        'experiment_ts': experiment_ts or util.get_ts(),
        'prepath': None,
        'git_sha': util.get_git_sha(),
        'random_seed': None,
    }
    spec['meta'].update(extended_meta_spec)
    return spec


def get(spec_file, spec_name, experiment_ts=None):
    '''
    Get an experiment spec from spec_file, spec_name.
    Auto-check spec.
    @param str:spec_file
    @param str:spec_name
    @param str:experiment_ts Use this experiment_ts if given; used for resuming training
    @example

    spec = spec_util.get('demo.json', 'dqn_cartpole')
    '''
    spec_file = spec_file.replace(SPEC_DIR, '')  # guard
    spec_file = f'{SPEC_DIR}/{spec_file}'  # allow direct filename
    spec_dict = util.read(spec_file)
    assert spec_name in spec_dict, f'spec_name {spec_name} is not in spec_file {spec_file}. Choose from:\n {ps.join(spec_dict.keys(), ",")}'
    spec = spec_dict[spec_name]
    # fill-in info at runtime
    spec['name'] = spec_name
    spec = extend_meta_spec(spec, experiment_ts)
    check(spec)
    return spec




def _override_dev_spec(spec):
    spec['meta']['max_session'] = 1
    spec['meta']['max_trial'] = 2
    return spec


def _override_enjoy_spec(spec):
    spec['meta']['max_session'] = 1
    return spec


def _override_test_spec(spec):
    agent_spec = spec['agent']
    # onpolicy freq is episodic
    freq = 1 if agent_spec['memory']['name'] == 'OnPolicyReplay' else 8
    agent_spec['algorithm']['training_frequency'] = freq
    agent_spec['algorithm']['time_horizon'] = freq
    agent_spec['algorithm']['training_start_step'] = 1
    agent_spec['algorithm']['training_iter'] = 1
    agent_spec['algorithm']['training_batch_iter'] = 1
    
    env_spec = spec['env']
    env_spec['max_frame'] = 40
    if env_spec.get('num_envs', 1) > 1:
        env_spec['num_envs'] = 2
    env_spec['max_t'] = 12
    
    spec['meta']['log_frequency'] = 10
    spec['meta']['eval_frequency'] = 10
    spec['meta']['max_session'] = 1
    spec['meta']['max_trial'] = 2
    return spec


def override_spec(spec, mode):
    '''Override spec based on the (lab_)mode, do nothing otherwise.'''
    overrider = {
        'dev': _override_dev_spec,
        'enjoy': _override_enjoy_spec,
        'test': _override_test_spec,
    }.get(mode)
    if overrider is not None:
        overrider(spec)
    return spec


def save(spec, unit='experiment'):
    '''Save spec to proper path. Called at Experiment or Trial init.'''
    prepath = util.get_prepath(spec, unit)
    util.write(spec, f'{prepath}_spec.json')


def tick(spec, unit):
    '''
    Method to tick lab unit (experiment, trial, session) in meta spec to advance their indices
    Reset lower lab indices to -1 so that they tick to 0
    spec_util.tick(spec, 'session')
    session = Session(spec)
    '''
    if lab_mode() == 'enjoy':  # don't tick in enjoy mode
        return spec

    meta_spec = spec['meta']
    if unit == 'experiment':
        meta_spec['experiment_ts'] = util.get_ts()
        meta_spec['experiment'] += 1
        meta_spec['trial'] = -1
        meta_spec['session'] = -1
    elif unit == 'trial':
        if meta_spec['experiment'] == -1:
            meta_spec['experiment'] += 1
        meta_spec['trial'] += 1
        meta_spec['session'] = -1
    elif unit == 'session':
        if meta_spec['experiment'] == -1:
            meta_spec['experiment'] += 1
        if meta_spec['trial'] == -1:
            meta_spec['trial'] += 1
        meta_spec['session'] += 1
    else:
        raise ValueError(f'Unrecognized lab unit to tick: {unit}')
    # set prepath since it is determined at this point
    meta_spec['prepath'] = prepath = util.get_prepath(spec, unit)
    for folder in ('graph', 'info', 'log', 'model'):
        folder_prepath = util.insert_folder(prepath, folder)
        folder_predir = os.path.dirname(f'{ROOT_DIR}/{folder_prepath}')
        os.makedirs(folder_predir, exist_ok=True)
        assert os.path.exists(folder_predir)
        meta_spec[f'{folder}_prepath'] = folder_prepath
    
    # Log the data output path only when first created (experiment, trial, and session all start at 0)
    if meta_spec['experiment'] == 0 and meta_spec['trial'] == 0 and meta_spec['session'] == 0:
        logger.info(f"Data output: {prepath}")
    return spec
