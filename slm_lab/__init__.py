import json
import os
import subprocess

os.environ['PY_ENV'] = os.environ.get('PY_ENV') or 'development'
CONFIG_NAME_MAP = {
    'test': 'example_default',
    'development': 'default',
}
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

config_name = CONFIG_NAME_MAP.get(os.environ['PY_ENV'])
config_path = os.path.join(ROOT_DIR, 'config', f'{config_name}.json')
if not os.path.exists(config_path):
    subprocess.call(['bin/copy_config'], cwd=ROOT_DIR)
with open(config_path) as f:
    config = json.load(f)
