import json
import os

os.environ['PY_ENV'] = os.environ.get('PY_ENV') or 'development'
CONFIG_NAME_MAP = {
    'test': 'example_default',
    'development': 'default',
    'production': 'production',
}

config_name = CONFIG_NAME_MAP.get(os.environ['PY_ENV'])
with open(os.path.join(os.getcwd(), f'config/{config_name}.json')) as f:
    config = json.load(f)
