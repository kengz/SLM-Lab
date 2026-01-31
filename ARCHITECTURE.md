# Architecture

This document describes SLM-Lab's architecture for maintainers and developers extending the framework.

## Overview

SLM-Lab follows a hierarchical control structure with modular, composable components:

```
CLI
 └── Experiment (hyperparameter search via Ray Tune)
      └── Trial (one hyperparameter configuration)
           └── Session (one random seed)
                ├── Agent (algorithm + memory + tracker)
                │    ├── Algorithm (policy, value functions, training logic)
                │    ├── Memory (experience storage and sampling)
                │    └── MetricsTracker (episode/training statistics)
                └── Env (vectorized gymnasium environments)
```

## Core Components

### Experiment Control (`slm_lab/experiment/control.py`)

The control hierarchy manages the training lifecycle:

- **Session**: Single training run with one random seed. Owns the Agent and Env, runs the train loop.
- **Trial**: Runs multiple Sessions with different seeds, aggregates results for statistical robustness.
- **Experiment**: Orchestrates hyperparameter search across Trials using Ray Tune.

```python
# Session train loop (simplified)
while frame < max_frame:
    action = agent.act(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.update(state, action, reward, next_state, terminated, truncated)
    if agent.algorithm.to_train:
        agent.algorithm.train()
```

### Agent (`slm_lab/agent/__init__.py`)

The Agent is a container that wires together:

- **Algorithm**: Policy and value networks, action selection, gradient updates
- **Memory**: Experience buffer (on-policy or replay)
- **MetricsTracker**: Computes episode returns, logs training metrics

The Agent delegates `act()`, `update()`, and `train()` to the Algorithm.

### Algorithm (`slm_lab/agent/algorithm/`)

Abstract base class defines the learning interface:

```python
class Algorithm(ABC):
    @abstractmethod
    def init_nets(self, net_spec): ...      # Create neural networks

    @abstractmethod
    def act(self, state): ...               # Select action from policy

    @abstractmethod
    def train(self): ...                    # Update networks from experience

    def update(self, state, action, reward, next_state, terminated, truncated):
        self.memory.update(...)             # Store transition
```

**Algorithm hierarchy:**

```
Algorithm (base)
 ├── SARSA (tabular baseline)
 ├── VanillaDQN → DQN → DoubleDQN
 └── ActorCritic (policy gradient base)
      ├── A2C (advantage actor-critic)
      ├── PPO (proximal policy optimization)
      └── SAC (soft actor-critic, off-policy)
```

Each algorithm implements its own training logic. For example, PPO:
1. Collects rollouts until `training_frequency` steps
2. Computes advantages with GAE
3. Runs multiple epochs of minibatch updates with clipped objective

### Memory (`slm_lab/agent/memory/`)

Experience storage with two main types:

- **OnPolicyMemory**: Stores single rollout, cleared after training (A2C, PPO)
- **Replay**: Ring buffer with random sampling (DQN, SAC)
  - **PrioritizedReplay**: Samples by TD-error priority

Memory handles vectorized environments, storing batches of transitions.

### Networks (`slm_lab/agent/net/`)

Neural network architectures:

- **MLPNet**: Fully-connected for low-dimensional states
- **ConvNet**: CNN for image observations (Atari)
- **RecurrentNet**: LSTM for partial observability

Networks are configured via spec and support:
- Orthogonal initialization
- Layer normalization (optional)
- Shared or separate actor-critic heads

### Environment (`slm_lab/env/__init__.py`)

Environment setup via `make_env()`:

1. Creates gymnasium environment by name
2. Applies standard wrappers (ClockWrapper for frame counting)
3. Applies Atari wrappers if needed (frame stacking, reward clipping)
4. Vectorizes with `SyncVectorEnv` for parallel stepping

The ClockWrapper tracks `clock.frame` for training schedule coordination.

## Spec System

JSON specs configure experiments declaratively:

```json
{
  "agent": {
    "algorithm": { "name": "PPO", "gamma": 0.99, ... },
    "memory": { "name": "OnPolicyMemory", ... },
    "net": { "type": "MLPNet", "hidden_layers": [64, 64], ... }
  },
  "env": { "name": "CartPole-v1", "num_envs": 4, ... },
  "meta": { "max_frame": 100000, "max_session": 4, ... }
}
```

**Key spec features:**

- **Variable substitution**: `${var}` placeholders replaced via `-s var=value` CLI flag
- **Search blocks**: Define hyperparameter ranges for Ray Tune
- **Reproducibility**: Spec + git SHA fully defines an experiment

Spec loading (`slm_lab/spec/spec_util.py`):
1. Load JSON and resolve variables
2. Flatten nested config for component initialization
3. Track indices via `tick()` for multi-trial/session runs

## Hyperparameter Search (`slm_lab/experiment/search.py`)

Ray Tune integration with ASHA early stopping:

```python
# Search config in spec
"search": {
    "agent.algorithm.gamma__uniform": [0.99, 0.999],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
}
```

- **ASHA scheduler**: Terminates underperforming trials early
- **Trial parallelism**: Configurable GPU/CPU resources per trial
- **Result reporting**: Metrics sent to Ray Tune for scheduling decisions

## Data Flow

### Training Loop

```
1. Session.run() starts training loop
2. Agent.act(state) → Algorithm.act() → network forward pass → action
3. Env.step(action) → next_state, reward, terminated, truncated
4. Agent.update() → Memory.update() stores transition
5. Check training_frequency → if ready, Agent.train()
6. Algorithm.train() → sample from Memory → compute loss → gradient update
7. MetricsTracker logs episode returns
8. Repeat until max_frame
```

### Evaluation

Periodic evaluation runs with deterministic policy (no exploration noise) to measure true performance. Results saved as checkpoints.

## Directory Structure

```
slm_lab/
├── agent/
│   ├── algorithm/      # RL algorithms (DQN, PPO, SAC, etc.)
│   ├── memory/         # Experience buffers
│   └── net/            # Neural network architectures
├── env/                # Environment wrappers and utilities
├── experiment/         # Training control and search
├── spec/               # JSON spec files and utilities
│   └── benchmark/      # Validated benchmark specs
├── lib/                # Shared utilities (math, logging, etc.)
└── cli/                # Command-line interface
```

## Extending SLM-Lab

### Adding a New Algorithm

1. Create `slm_lab/agent/algorithm/your_algo.py`
2. Inherit from appropriate base (`Algorithm`, `ActorCritic`, etc.)
3. Implement `init_nets()`, `act()`, `train()`
4. Register in `slm_lab/agent/algorithm/__init__.py`
5. Create spec file in `slm_lab/spec/`

### Adding Environment Support

SLM-Lab works with any gymnasium-compatible environment. For custom environments:

1. Ensure gymnasium API compliance (`reset()` returns `(obs, info)`, `step()` returns 5-tuple)
2. Register with gymnasium or pass callable to spec
3. Add wrappers in `make_env()` if needed

### Adding a New Network

1. Create network class in `slm_lab/agent/net/`
2. Follow existing patterns (init from spec, forward method)
3. Register in network factory
