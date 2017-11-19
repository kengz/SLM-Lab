'''
The spec module
Handles the Lab spec: reading, writing(evolution), validation and default setting
'''
import json
import os
import pydash as _
from slm_lab.lib import logger, util

SPEC_DIR = 'slm_lab/spec'
SPEC_FORMAT = {
    "agent": [{
        "name": str,
        "param": dict
    }],
    "env": [{
        "name": str,
        "param": dict
    }],
    "body": {
        "product": ["outer", "inner", "custom"],
        "num": int
    },
    "meta": {
        "max_timestep": (type(None), int),
        "max_episode": (type(None), int),
        "max_session": int,
        "max_trial": (type(None), int),
        "train_mode": bool
    }
}


def check_comp_spec(comp_spec, comp_spec_format):
    '''Base method to check component spec'''
    for spec_k, spec_format_v in comp_spec_format.items():
        comp_spec_v = comp_spec[spec_k]
        if isinstance(spec_format_v, list):
            v_set = spec_format_v
            assert comp_spec_v in v_set, f'Component spec value {_.pick(comp_spec, spec_k)} needs to be one of {util.to_json(v_set)}'
        else:
            v_type = spec_format_v
            assert isinstance(
                comp_spec_v, v_type), f'Component spec {_.pick(comp_spec, spec_k)} needs to be of type: {v_type}'


def check(spec, spec_name=''):
    assert spec.keys() == SPEC_FORMAT.keys(
    ), f'Spec needs to follow spec.SPEC_FORMAT. Given \n {spec_name}: {util.to_json(spec)}'
    for agent_spec in spec['agent']:
        check_comp_spec(agent_spec, SPEC_FORMAT['agent'][0])
    for env_spec in spec['env']:
        check_comp_spec(env_spec, SPEC_FORMAT['env'][0])
    check_comp_spec(spec['body'], SPEC_FORMAT['body'])
    check_comp_spec(spec['meta'], SPEC_FORMAT['meta'])


def check_all():
    '''Check all spec files, all specs.'''
    spec_files = _.filter_(os.listdir(SPEC_DIR), lambda f: f.endswith('.json'))
    for spec_file in spec_files:
        spec_dict = util.read(f'{SPEC_DIR}/{spec_file}')
        for spec_name, spec in spec_dict.items():
            try:
                check(spec, spec_name)
            except Exception as e:
                logger.exception(
                    f'{spec_file} spec {spec_name} fails spec check')
                raise(e)
    logger.info(f'Checked all specs from: {_.join(spec_files, ",")}')


check_all()


def read(filename):
    # resolve
    # check
    # return

    return
