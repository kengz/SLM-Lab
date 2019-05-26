'''
The spec util
Handles the Lab experiment spec: reading, writing(evolution), validation and default setting
Expands the spec and params into consumable inputs in info space for lab units.
'''
from slm_lab.lib import logger, util
from string import Template
import itertools
import json
import numpy as np
import os
import pydash as ps


SPEC_DIR = 'slm_lab/spec'
'''
All spec values are already param, inferred automatically.
To change from a value into param range, e.g.
- single: "explore_anneal_epi": 50
- continuous param: "explore_anneal_epi": {"min": 50, "max": 100, "dist": "uniform"}
- discrete range: "explore_anneal_epi": {"values": [50, 75, 100]}
'''
SPEC_FORMAT = {
    "agent": [{
        "name": str,
        "algorithm": dict,
        "memory": dict,
        "net": dict,
    }],
    "env": [{
        "name": str,
        "max_t": (type(None), int, float),
        "max_frame": (int, float),
    }],
    "body": {
        "product": ["outer", "inner", "custom"],
        "num": (int, list),
    },
    "meta": {
        "eval_frequency": (int, float),
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


def check_body_spec(spec):
    '''Base method to check body spec for AEB space resolution'''
    ae_product = ps.get(spec, 'body.product')
    body_num = ps.get(spec, 'body.num')
    if ae_product == 'outer':
        pass
    elif ae_product == 'inner':
        agent_num = len(spec['agent'])
        env_num = len(spec['env'])
        assert agent_num == env_num, 'Agent and Env spec length must be equal for body `inner` product. Given {agent_num}, {env_num}'
    else:  # custom AEB
        assert ps.is_list(body_num)


def check_compatibility(spec):
    '''Check compatibility among spec setups'''
    # TODO expand to be more comprehensive
    if spec['meta'].get('distributed') == 'synced':
        assert ps.get(spec, 'agent.0.net.gpu') == False, f'Distributed mode "synced" works with CPU only. Set gpu: false.'


def check(spec):
    '''Check a single spec for validity'''
    try:
        spec_name = spec.get('name')
        assert set(spec.keys()) >= set(SPEC_FORMAT.keys()), f'Spec needs to follow spec.SPEC_FORMAT. Given \n {spec_name}: {util.to_json(spec)}'
        for agent_spec in spec['agent']:
            check_comp_spec(agent_spec, SPEC_FORMAT['agent'][0])
        for env_spec in spec['env']:
            check_comp_spec(env_spec, SPEC_FORMAT['env'][0])
        check_comp_spec(spec['body'], SPEC_FORMAT['body'])
        check_comp_spec(spec['meta'], SPEC_FORMAT['meta'])
        check_body_spec(spec)
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


def extend_meta_spec(spec):
    '''Extend meta spec with information for lab functions'''
    extended_meta_spec = {
        # reset lab indices to -1 so that they tick to 0
        'experiment': -1,
        'trial': -1,
        'session': -1,
        'cuda_offset': int(os.environ.get('CUDA_OFFSET', 0)),
        'experiment_ts': util.get_ts(),
        'prepath': None,
        # ckpt extends prepath, e.g. ckpt_str = ckpt-epi10-totalt1000
        'ckpt': None,
        'git_sha': util.get_git_sha(),
        'random_seed': None,
        'eval_model_prepath': None,
    }
    spec['meta'].update(extended_meta_spec)
    return spec


def get(spec_file, spec_name):
    '''
    Get an experiment spec from spec_file, spec_name.
    Auto-check spec.
    @example

    spec = spec_util.get('base.json', 'base_case_openai')
    '''
    spec_file = spec_file.replace(SPEC_DIR, '')  # cleanup
    if 'data/' in spec_file:
        assert spec_name in spec_file, 'spec_file in data/ must be lab-generated and contains spec_name'
        spec = util.read(spec_file)
    else:
        spec_file = f'{SPEC_DIR}/{spec_file}'  # allow direct filename
        spec_dict = util.read(spec_file)
        assert spec_name in spec_dict, f'spec_name {spec_name} is not in spec_file {spec_file}. Choose from:\n {ps.join(spec_dict.keys(), ",")}'
        spec = spec_dict[spec_name]
        # fill-in info at runtime
        spec['name'] = spec_name
        spec = extend_meta_spec(spec)
    check(spec)
    return spec


def get_eval_spec(spec_file, prename):
    '''Get spec for eval mode'''
    predir, _, _, _, _, _ = util.prepath_split(spec_file)
    prepath = f'{predir}/{prename}'
    spec = util.prepath_to_spec(prepath)
    spec['meta']['ckpt'] = 'eval'
    spec['meta']['eval_model_prepath'] = prepath
    return spec


def get_param_specs(spec):
    '''Return a list of specs with substituted spec_params'''
    assert 'spec_params' in spec, 'Parametrized spec needs a spec_params key'
    spec_params = spec.pop('spec_params')
    spec_template = Template(json.dumps(spec))
    keys = spec_params.keys()
    specs = []
    for idx, vals in enumerate(itertools.product(*spec_params.values())):
        spec_str = spec_template.substitute(dict(zip(keys, vals)))
        spec = json.loads(spec_str)
        spec['name'] += f'_{"_".join(vals)}'
        # offset to prevent parallel-run GPU competition, to mod in util.set_cuda_id
        cuda_id_gap = int(spec['meta']['max_session'] / spec['meta']['param_spec_process'])
        spec['meta']['cuda_offset'] += idx * cuda_id_gap
        specs.append(spec)
    return specs


def is_aeb_compact(aeb_list):
    '''
    Check if aeb space (aeb_list) is compact; uniq count must equal shape in each of a,e axes. For b, per unique a,e hash, uniq must equal shape.'''
    aeb_shape = util.get_aeb_shape(aeb_list)
    aeb_uniq = [len(np.unique(col)) for col in np.transpose(aeb_list)]
    ae_compact = np.array_equal(aeb_shape, aeb_uniq)
    b_compact = True
    for ae, ae_b_list in ps.group_by(aeb_list, lambda aeb: f'{aeb[0]}{aeb[1]}').items():
        b_shape = util.get_aeb_shape(ae_b_list)[2]
        b_uniq = [len(np.unique(col)) for col in np.transpose(ae_b_list)][2]
        b_compact = b_compact and np.array_equal(b_shape, b_uniq)
    aeb_compact = ae_compact and b_compact
    return aeb_compact


def is_singleton(spec):
    '''Check if spec uses a singleton Session'''
    return len(spec['agent']) == 1 and len(spec['env']) == 1 and spec['body']['num'] == 1


def override_dev_spec(spec):
    spec['meta']['max_session'] = 1
    spec['meta']['max_trial'] = 2
    return spec


def override_enjoy_spec(spec):
    spec['meta']['max_session'] = 1
    return spec


def override_eval_spec(spec):
    spec['meta']['max_session'] = 1
    # evaluate by episode is set in env clock init in env/base.py
    return spec


def override_test_spec(spec):
    for agent_spec in spec['agent']:
        # onpolicy freq is episodic
        freq = 1 if agent_spec['memory']['name'] == 'OnPolicyReplay' else 8
        agent_spec['algorithm']['training_frequency'] = freq
        agent_spec['algorithm']['training_start_step'] = 1
        agent_spec['algorithm']['training_iter'] = 1
        agent_spec['algorithm']['training_batch_iter'] = 1
    for env_spec in spec['env']:
        env_spec['max_frame'] = 40
        env_spec['max_t'] = 12
    spec['meta']['log_frequency'] = 10
    spec['meta']['eval_frequency'] = 10
    spec['meta']['max_session'] = 1
    spec['meta']['max_trial'] = 2
    return spec


def resolve_aeb(spec):
    '''
    Resolve an experiment spec into the full list of points (coordinates) in AEB space.
    @param {dict} spec An experiment spec.
    @returns {list} aeb_list Resolved array of points in AEB space.
    @example

    spec = spec_util.get('base.json', 'general_inner')
    aeb_list = spec_util.resolve_aeb(spec)
    # => [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
    '''
    agent_num = len(spec['agent']) if ps.is_list(spec['agent']) else 1
    env_num = len(spec['env']) if ps.is_list(spec['env']) else 1
    ae_product = ps.get(spec, 'body.product')
    body_num = ps.get(spec, 'body.num')
    body_num_list = body_num if ps.is_list(body_num) else [body_num] * env_num

    aeb_list = []
    if ae_product == 'outer':
        for e in range(env_num):
            sub_aeb_list = list(itertools.product(range(agent_num), [e], range(body_num_list[e])))
            aeb_list.extend(sub_aeb_list)
    elif ae_product == 'inner':
        for a, e in zip(range(agent_num), range(env_num)):
            sub_aeb_list = list(itertools.product([a], [e], range(body_num_list[e])))
            aeb_list.extend(sub_aeb_list)
    else:  # custom AEB, body_num is a aeb_list
        aeb_list = [tuple(aeb) for aeb in body_num]
    aeb_list.sort()
    assert is_aeb_compact(aeb_list), 'Failed check: for a, e, uniq count == len (shape), and for each a,e hash, b uniq count == b len (shape)'
    return aeb_list


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
    meta_spec['prepath'] = util.get_prepath(spec, unit)
    return spec
