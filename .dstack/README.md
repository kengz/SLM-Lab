# GPU Training with dstack

Use [dstack](https://dstack.ai) for GPU-intensive RL training on cloud infrastructure.

## Configuration

Set environment variables to configure your training experiment:

- **SPEC_FILE**: Path to JSON spec file containing experiment configurations
- **SPEC_NAME**: Name of specific experiment within the spec file
- **LAB_MODE**: Training mode (`train` for production, `dev` for debugging)

## Compute

TBD

## Usage

```bash
# initialize dstack repository
dstack init

# create volumes to persist cache and data
dstack apply -f .dstack/cache-volume.yml
dstack apply -f .dstack/data-volume.yml

# start development environment with GPU
dstack apply -f .dstack/dev.yml

# run demo
SPEC_FILE=slm_lab/spec/demo.json SPEC_NAME=dqn_cartpole LAB_MODE=train dstack apply -f .dstack/train.yml
# run ppo pong
SPEC_FILE=slm_lab/spec/benchmark/ppo/ppo_pong.json SPEC_NAME=ppo_pong LAB_MODE=train dstack apply -f .dstack/train.yml
```
