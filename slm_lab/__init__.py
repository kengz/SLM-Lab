import os

os.environ['PY_ENV'] = os.environ.get('PY_ENV') or 'development'
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
