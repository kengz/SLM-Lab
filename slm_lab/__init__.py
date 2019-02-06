import os

os.environ['PY_ENV'] = os.environ.get('PY_ENV') or 'development'
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

# valid lab_mode in SLM Lab
EVAL_MODES = ('enjoy', 'eval')
TRAIN_MODES = ('search', 'train', 'dev')
