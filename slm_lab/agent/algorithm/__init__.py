# The algorithm module
# Contains implementations of reinforcement learning algorithms.
# Uses the nets module to build neural networks as the relevant function approximators
from .actor_critic import *
from .dqn import *
from .ppo import *
from .random import *
from .reinforce import *
from .sac import *
from .sarsa import *
from .sil import *

from .meta_algorithm import *
from .supervised_learning import *
from .supervised_learning_baysian import *

from .ucb import *
from .adaptive_mechanism_design import *
