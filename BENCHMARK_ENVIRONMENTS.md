# SLM-Lab Benchmark Environments

This document lists all environment series that were previously configured with Ray Tune search choices but are now run using `--set env=` substitution after implementing ASHA hyperparameter search.

## Atari Environments

### Short List (4 environments)
Used for quick benchmarks and initial testing:
- ALE/Breakout-v5
- ALE/Pong-v5  
- ALE/Qbert-v5
- ALE/Seaquest-v5

### Full Atari Suite (57 environments)
Complete Atari 2600 benchmark suite:
- ALE/Adventure-v5
- ALE/AirRaid-v5
- ALE/Alien-v5
- ALE/Amidar-v5
- ALE/Assault-v5
- ALE/Asterix-v5
- ALE/Asteroids-v5
- ALE/Atlantis-v5
- ALE/BankHeist-v5
- ALE/BattleZone-v5
- ALE/BeamRider-v5
- ALE/Berzerk-v5
- ALE/Bowling-v5
- ALE/Boxing-v5
- ALE/Breakout-v5
- ALE/Carnival-v5
- ALE/Centipede-v5
- ALE/ChopperCommand-v5
- ALE/CrazyClimber-v5
- ALE/Defender-v5
- ALE/DemonAttack-v5
- ALE/DoubleDunk-v5
- ALE/ElevatorAction-v5
- ALE/FishingDerby-v5
- ALE/Freeway-v5
- ALE/Frostbite-v5
- ALE/Gopher-v5
- ALE/Gravitar-v5
- ALE/Hero-v5
- ALE/IceHockey-v5
- ALE/Jamesbond-v5
- ALE/JourneyEscape-v5
- ALE/Kangaroo-v5
- ALE/Krull-v5
- ALE/KungFuMaster-v5
- ALE/MontezumaRevenge-v5
- ALE/MsPacman-v5
- ALE/NameThisGame-v5
- ALE/Phoenix-v5
- ALE/Pitfall-v5
- ALE/Pong-v5
- ALE/Pooyan-v5
- ALE/PrivateEye-v5
- ALE/Qbert-v5
- ALE/Riverraid-v5
- ALE/RoadRunner-v5
- ALE/Robotank-v5
- ALE/Seaquest-v5
- ALE/Skiing-v5
- ALE/Solaris-v5
- ALE/SpaceInvaders-v5
- ALE/StarGunner-v5
- ALE/Tennis-v5
- ALE/TimePilot-v5
- ALE/Tutankham-v5
- ALE/UpNDown-v5
- ALE/Venture-v5
- ALE/VideoPinball-v5
- ALE/WizardOfWor-v5
- ALE/YarsRevenge-v5
- ALE/Zaxxon-v5

## MuJoCo Environments

Standard MuJoCo benchmark suite (7 environments):
- Ant-v5
- HalfCheetah-v5
- Hopper-v5
- InvertedDoublePendulum-v5
- InvertedPendulum-v5
- Reacher-v5
- Walker2d-v5

## Continuous Control Environments

Box2D and classic control environments:
- BipedalWalker-v3
- Pendulum-v1

## Benchmark Commands

### PPO Benchmarks

#### Atari (short list)
```bash
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab --set env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab --set env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab --set env=ALE/Seaquest-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

#### MuJoCo
```bash
slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab --set env=Ant-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab --set env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab --set env=Walker2d-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab --set env=InvertedDoublePendulum-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab --set env=InvertedPendulum-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab --set env=Reacher-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

#### Continuous Control
```bash
slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker train
slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_pendulum train
```

### DQN Benchmarks

#### Atari (all DQN variants)
```bash
# Standard DQN
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/dqn/dqn_atari.json dqn_atari train

# Double DQN
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/dqn/ddqn_atari.json ddqn_atari train

# DQN with Prioritized Experience Replay
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/dqn/dqn_per_atari.json dqn_per_atari train

# Double DQN with PER
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/dqn/ddqn_per_atari.json ddqn_per_atari train

# Dueling Double DQN with PER
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/dqn/dueling_ddqn_per_atari.json dueling_ddqn_per_atari train
```

### SAC Benchmarks

#### MuJoCo
```bash
# Standard SAC
slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco train

# SAC with Prioritized Experience Replay
slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/sac/sac_per_mujoco.json sac_per_mujoco train

# Async SAC
slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/async_sac/async_sac_mujoco.json async_sac_mujoco train
```

### A2C Benchmarks

#### Atari
```bash
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/a2c/a2c_nstep_atari.json a2c_nstep_atari train
```

#### MuJoCo
```bash
slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json a2c_gae_mujoco train
slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/a2c/a2c_nstep_mujoco.json a2c_nstep_mujoco train
```

### A3C Benchmarks

#### Atari
```bash
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/a3c/a3c_gae_atari.json a3c_gae_atari train
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/a3c/a3c_nstep_atari.json a3c_nstep_atari train
```

### DPPO Benchmarks

#### Atari
```bash
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/dppo/dppo_atari.json dppo_atari train
```

## ASHA Hyperparameter Search

All PPO specs now include ASHA search configurations with sophisticated Optuna distributions:

### CartPole and Lunar Lander
```bash
slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole search
slm-lab slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar search
```

### Continuous Control with ASHA
```bash
slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker search
slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_pendulum search
```

### MuJoCo with ASHA
```bash
slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search
slm-lab --set env=Ant-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search
slm-lab --set env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search
slm-lab --set env=Walker2d-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search
```

## Notes

- All environment choices have been removed from search specifications
- Use `--set env=ENVIRONMENT_NAME` to specify environment for multi-environment specs
- ASHA search uses sophisticated Optuna distributions (loguniform, uniform, randint) instead of discrete choices
- Environment substitution works with `${env}` placeholders in spec files
- For comprehensive benchmarks, run all environments in each category