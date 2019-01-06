'''
The spec util
Handles the Lab experiment spec: reading, writing(evolution), validation and default setting
Expands the spec and params into consumable inputs in info space for lab units.
'''
from slm_lab.lib import logger, util
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
        "max_t": (type(None), int),
        "max_tick": int,
        "max_tick_unit": str,
    }],
    "body": {
        "product": ["outer", "inner", "custom"],
        "num": (int, list),
    },
    "meta": {
        "distributed": bool,
        "max_session": int,
        "max_trial": (type(None), int),
        "search": str,
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
            try:
                spec['name'] = spec_name
                spec['git_SHA'] = util.get_git_sha()
                check(spec)
            except Exception as e:
                logger.exception(f'spec_file {spec_file} fails spec check')
                raise e
    logger.info(f'Checked all specs from: {ps.join(spec_files, ",")}')
    return True


def get(spec_file, spec_name):
    '''
    Get an experiment spec from spec_file, spec_name.
    Auto-check spec.
    @example

    spec = spec_util.get('base.json', 'base_case_openai')
    '''
    if 'data/' in spec_file:
        assert spec_name in spec_file, 'spec_file in data/ must be lab-generated and contains spec_name'
        spec = util.read(spec_file)
    else:
        spec_file = f'{SPEC_DIR}/{spec_file}'  # allow direct filename
        spec_dict = util.read(spec_file)
        assert spec_name in spec_dict, f'spec_name {spec_name} is not in spec_file {spec_file}. Choose from:\n {ps.join(spec_dict.keys(), ",")}'
        spec = spec_dict[spec_name]
        spec['name'] = spec_name
        spec['git_SHA'] = util.get_git_sha()
    check(spec)
    return spec


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
    for agent_spec in spec['agent']:
        if 'max_size' in agent_spec['memory']:
            agent_spec['memory']['max_size'] = 100
    # evaluate by episode is set in env clock init in env/base.py
    return spec


def override_test_spec(spec):
    for agent_spec in spec['agent']:
        # covers episodic and timestep
        agent_spec['algorithm']['training_frequency'] = 1
        agent_spec['algorithm']['training_start_step'] = 1
        agent_spec['algorithm']['training_epoch'] = 1
        agent_spec['algorithm']['training_batch_epoch'] = 1
    for env_spec in spec['env']:
        env_spec['max_t'] = 20
        env_spec['max_tick'] = 3
        env_spec['max_tick_unit'] = 'epi'
        env_spec['save_frequency'] = 1000
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
