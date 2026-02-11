# The nets module
# Implements differents types of neural network
from slm_lab.agent.net.conv import *
from slm_lab.agent.net.mlp import *
from slm_lab.agent.net.recurrent import *

# Optional: torcharc-based networks (requires torcharc package)
try:
    from slm_lab.agent.net.torcharc_net import *
except ImportError:
    pass
