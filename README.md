# SLM Lab
[![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

Modular Deep Reinforcement Learning framework in PyTorch.

|||||
|:---:|:---:|:---:|:---:|
| ![ddqn_beamrider](https://user-images.githubusercontent.com/8209263/49688812-b7e04200-facc-11e8-9a1a-d5c8e512f26c.gif) |  ![ddqn_breakout](https://user-images.githubusercontent.com/8209263/49688819-c29ad700-facc-11e8-842b-1dc6f6f38495.gif) | ![ddqn_enduro](https://user-images.githubusercontent.com/8209263/49688852-3ccb5b80-facd-11e8-80e4-8d86c195d112.gif)|![ddqn_pong](https://user-images.githubusercontent.com/8209263/49688793-54eeab00-facc-11e8-80fe-4b76a12180a0.gif) |
| BeamRider | Breakout | Enduro | Pong |
| ![ddqn_qbert](https://user-images.githubusercontent.com/8209263/49688862-6be1cd00-facd-11e8-849d-61aef598611b.gif) | ![ddqn_seaquest](https://user-images.githubusercontent.com/8209263/49688863-70a68100-facd-11e8-9303-73bea9b9987a.gif) | ![ddqn_spaceinvaders](https://user-images.githubusercontent.com/8209263/49688875-87e56e80-facd-11e8-90be-9d6be7bace03.gif) | |
| Qbert | Seaquest | SpaceInvaders | |

| | | |
|:---:|:---:|:---:|
| ![dqn cartpole ball2d](https://media.giphy.com/media/l0DAIymuiMS3HyW9G/giphy.gif) Multitask DQN solving OpenAI Cartpole-v0 and Unity Ball2D. | ![pong](https://user-images.githubusercontent.com/8209263/49346161-07dd8580-f643-11e8-975c-38972465a587.gif) DQN Atari Pong solution in SLM Lab. | ![lunar](https://user-images.githubusercontent.com/5512945/49346897-8d663300-f64d-11e8-8e9c-97cf079337a3.gif) DDQN Lunar solution in SLM Lab. |

| References | |
|------------|--|
| [Github](https://github.com/kengz/SLM-Lab) | Github repository |
| [Installation](#installation) | How to install SLM Lab |
| [Documentation](https://kengz.gitbooks.io/slm-lab/content/) | Usage documentation |
| [Benchmark](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md)| Benchmark results |
| [Tutorials](https://github.com/kengz/SLM-Lab/blob/master/TUTORIALS.md)| Tutorial resources |
| [Contributing](https://github.com/kengz/SLM-Lab/blob/master/CONTRIBUTING.md)| How to contribute |
| [Roadmap](https://github.com/kengz/SLM-Lab/projects) | Research and engineering roadmap |
| [Gitter](https://gitter.im/SLM-Lab/SLM-Lab) | SLM Lab user chatroom |

SLM Lab is created for deep reinforcement learning research and applications. The design was guided by four principles
- **modularity**
- **simplicity**
- **analytical clarity**
- **reproducibility**

#### Modularity

- makes research easier and more accessible: reuse well-tested components and only focus on the relevant work
- makes learning deep RL easier: the algorithms are complex; SLM Lab breaks them down into more manageable, digestible components
- components get reused maximally, which means less code, more tests, and fewer bugs

#### Simplicity

- the components are designed to closely correspond to the way papers or books discuss RL
- modular libraries are not necessarily simple. Simplicity balances modularity to prevent overly complex abstractions that are difficult to understand and use

#### Analytical clarity

- hyperparameter search results are automatically analyzed and presented hierarchically in increasingly granular detail
- it should take less than 1 minute to understand if an experiment yielded a successful result using the [experiment graph](https://kengz.gitbooks.io/slm-lab/content/analytics/experiment-graph.html)
- it should take less than 5 minutes to find and review the top 3 parameter settings using the [trial](https://kengz.gitbooks.io/slm-lab/content/analytics/trial-graph.html) and [session](https://kengz.gitbooks.io/slm-lab/content/analytics/session-graph.html) graphs

#### Reproducibility

- only the spec file and a git SHA are needed to fully reproduce an experiment
- all the results are recorded in [BENCHMARK.md](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md)
- experiment reproduction instructions are submitted to the Lab via [`result` Pull Requests](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+)
- the full experiment datas contributed are [public on Dropbox ](https://www.dropbox.com/sh/y738zvzj3nxthn1/AAAg1e6TxXVf3krD81TD5V0Ra?dl=0)

## Features

#### [Algorithms](#link-algos)
- numerous canonical algorithms ([listed below](#algorithm))
- reusable and well-tested modular components: algorithm, network, memory, policy
- simple and easy to use for building new algorithms

#### Environments
- supports multiple environments:
    - [OpenAI gym](https://github.com/openai/gym)
    - [VizDoom](https://github.com/mwydmuch/ViZDoom#documentation) (credit: joelouismarino)
    - [Unity environments](https://github.com/Unity-Technologies/ml-agents) with prebuilt binaries
    - *contributions welcome!*
- supports multi-agents, multi-environments
- API for adding custom environments

#### [Experimentation](#experimentation-framework)
- scalable hyperparameter search using [ray](https://github.com/ray-project/ray)
- analytical clarity with auto-generated results and graphs at session, trial, experiment levels
- fitness metric as a richer measurement of an algorithm's performance

## Implementations

SLM Lab implements most of the recent canonical algorithms and various extensions. These are used as the base of research. All the implementations follow this design:

- `Agent`: the base class containing all the components. It has the API methods to interface with the environment.
    - `Algorithm`: the main class containing the implementation details of a specific algorithm. It contains components that are reusable.
        - `Net`: the neural network for the algorithm. An algorithm can have multiple networks, e.g. Actor-Critic, DDQN.
    - `Body`: connects the agent-env, and stores the proper agent-env data, such as entropy/log_prob. Multitask agent will have multiple bodies, each handling a specific environment. Conversely, a multiagent environment will accept multiple bodies from different agents. Essentially, each body keeps track of an agent-env pair.
        - `Memory`: stores the numpy/plain type data produced from the agent-env interactions used for training.

- `BaseEnv`: the environment wrapper class. It has the API methods to interface with the agent. Currently, the Lab contains:
    - `OpenAIEnv` for [OpenAI gym](https://github.com/openai/gym)
    - `UnityEnv` for [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)


### Algorithm<a name="link-algos"></a>

code: [slm_lab/agent/algorithm](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm)

Various algorithms are in fact extensions of some simpler ones, and they are implemented as such. This allows for concise and safer code.

**Policy Gradient:**
- REINFORCE
- AC (Vanilla Actor-Critic)
    - shared or separate actor critic networks
    - plain TD
    - entropy term control
- A2C (Advantage Actor-Critic)
    - extension of AC with with advantage function
    - N-step returns as advantage
    - GAE (Generalized Advantage Estimate) as advantage
- PPO (Proximal Policy Optimization)
    - extension of A2C with PPO loss function
- SIL (Self-Imitation Learning)
    - extension of A2C with off-policy training on custom loss
- PPOSIL
    - SIL with PPO instead of A2C

Using the lab's unified API, **all the algorithms can be distributed hogwild-style**. Session takes the role of workers under a Trial. Some of the distributed algorithms have their own name:

- A3C (Asynchronous A2C / distributed A2C)
- DPPO (Distributed PPO)

**Value-based:**
- SARSA
- DQN (Deep Q Learning)
    - boltzmann or epsilon-greedy policy
- DRQN (DQN + Recurrent Network)
- Dueling DQN
- DDQN (Double DQN)
- DDRQN
- Dueling DDQN
- Hydra DQN (multi-environment DQN)

As mentioned above, **all these algorithms can be turned into distributed algorithms too**, although we do not have special names for them.

Below are the modular building blocks for the algorithms. They are designed to be general, and are reused extensively.

### Memory

code: [slm_lab/agent/memory](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/memory)

`Memory` is a numpy/plain type storage of data which gets reused for more efficient computations (without having to call `tensor.detach()` repeatedly). For storing graph tensor with the gradient, use `agent.body`.

Note that some particular types of algorithm/network need particular types of Memory, e.g. `RecurrentNet` needs any of the `SeqReplay`. See the class definition for more.

For on-policy algorithms (policy gradient):
- OnPolicyReplay
- OnPolicySeqReplay
- OnPolicyBatchReplay
- OnPolicySeqBatchReplay
- OnPolicyConcatReplay
- OnPolicyAtariReplay
- OnPolicyImageReplay (credit: joelouismarino)

For off-policy algorithms (value-based)
- Replay
- SeqReplay
- SILReplay (special Replay for SIL)
- SILSeqReplay (special SeqReplay for SIL)
- ConcatReplay
- AtariReplay
- ImageReplay
- PrioritizedReplay
- AtariPrioritizedReplay

### Neural Network

code: [slm_lab/agent/net](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/net)

These networks are usable for all algorithms, and the lab takes care of the proper initialization with proper input/output sizing. One can swap out the network for any algorithm with just a spec change, e.g. make `DQN` into `DRQN` by substituting the net spec `"type": "MLPNet"` with `"type": "RecurrentNet"`.

- MLPNet (Multi Layer Perceptron, with multi-heads, multi-tails)
- RecurrentNet (with multi-tails support)
- ConvNet (with multi-tails support)

These networks are usable for Q-learning algorithms. For more details see [this paper](http://proceedings.mlr.press/v48/wangf16.pdf).

- DuelingMLPNet
- DuelingConvNet

### Policy

code: [slm_lab/agent/algorithm/policy_util.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/policy_util.py)

The policy module takes the network output `pdparam`, constructs a probability distribution, and samples for it to produce actions. To use a different distribution, just specify it in the algorithm spec `"action_pdtype"`.

- different probability distributions for sampling actions
- default policy
- Boltzmann policy
- Epsilon-greedy policy
- numerous rate decay methods

## Experimentation framework

Deep Reinforcement Learning is highly empirical. The lab enables rapid and massive experimentations, hence it needs a way to quickly analyze data from many trials. The experiment  and analytics framework is the scientific method of the lab.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_experiment_graph.png)
>Experiment graph summarizing the trials in hyperparameter search.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_trial_graph.png)
>Trial graph showing average envelope of repeated sessions.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_session_graph.png)
>Session graph showing total rewards, exploration variable and loss for the episodes.

## Installation

1. Clone the [SLM-Lab repo](https://github.com/kengz/SLM-Lab):
    ```shell
    git clone https://github.com/kengz/SLM-Lab.git
    ```

2. Install dependencies (or inspect `bin/setup_*` first):
    ```shell
    cd SLM-Lab/
    bin/setup
    ```

>For optional extra setup, use `bin/setup extra` instead. E.g. to install Unity environments
>Alternatively, run the content of [`bin/setup_macOS` or `bin/setup_ubuntu`](https://github.com/kengz/SLM-Lab/tree/master/bin) on your terminal manually.
>Docker image and Dockerfile with instructions are also available

>Useful reference: [Debugging](https://kengz.gitbooks.io/slm-lab/content/installation/debugging.html)

### Update

To update SLM Lab, pull the latest git commits and run update:

    ```shell
    git pull
    conda env update -f environment.yml
    ```

>To update Unity environments obtained from the `extra` setup, run `yarn install`

### Demo

Run the demo to quickly see the lab in action (and to test your installation).

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo.png)

It is `DQN` in `CartPole-v0`:

1. See `slm_lab/spec/demo.json` for example spec:
    ```json
    "dqn_cartpole": {
      "agent": [{
        "name": "DQN",
        "algorithm": {
          "name": "DQN",
          "action_pdtype": "Argmax",
          "action_policy": "epsilon_greedy",
        ...
        }
      }]
    }
    ```

2. Launch terminal in the repo directory, run the lab with the demo spec in `dev` lab mode:
    ```shell
    conda activate lab
    python run_lab.py slm_lab/spec/demo.json dqn_cartpole dev
    ```
    >To run any lab commands, conda environment must be activated first. See [Installation](#installation) for more.
    >Spec file is autoresolved from `slm_lab/spec/`, so you may use just `demo.json` too.

    >With extra setup: `yarn start` can be used as a shorthand for `python run_lab.py`

3. This demo will run a single trial using the default parameters, and render the environment. After completion, check the output for data `data/dqn_cartpole_2018_06_16_214527/` (timestamp will differ). You should see some healthy graphs.

    ![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_trial_graph.png)
    >Trial graph showing average envelope of repeated sessions.

    ![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_session_graph.png)
    >Session graph showing total rewards, exploration variable and loss for the episodes.

4. Enjoy mode - when a session ends, a model file will automatically save. You can find the session `prepath` that ends in its trial and session numbers. The example above is trial 1 session 0, and you can see a pytorch model saved at `data/dqn_cartpole_2018_06_16_214527/dqn_cartpole_t1_s0_model_net.pth`. Use the following command to run from the saved folder in `data/`:
    ```bash
    python run_lab.py data/dqn_cartpole_2018_06_16_214527/dqn_cartpole_spec.json dqn_cartpole enjoy@dqn_cartpole_t1_s0
    ```
    >Enjoy mode will automatically disable learning and exploration. Graphs will still save.

    >To run the best model, use the best saved checkpoint `enjoy@dqn_cartpole_t1_s0_ckptbest`

5. The above was in `dev` mode. To run in proper training mode, which is faster without rendering, change the `dev` lab mode to `train`, and the same data is produced.
    ```shell
    python run_lab.py slm_lab/spec/demo.json dqn_cartpole train
    ```

6. Next, perform a hyperparameter search using the lab mode `search`. This runs experiments of multiple trials with hyperparameter search, defined at the bottom section of the demo spec.
    ```bash
    python run_lab.py slm_lab/spec/demo.json dqn_cartpole search
    ```

    When it ends, refer to `{prepath}_experiment_graph.png` and `{prepath}_experiment_df.csv` to find the best trials.

>If the demo fails, consult [Debugging](https://kengz.gitbooks.io/slm-lab/content/installation/debugging.html).

Now the lab is ready for usage.

**Read on: [Github](https://github.com/kengz/SLM-Lab) | [Documentation](https://kengz.gitbooks.io/slm-lab/content/)**

## Citing

If you use `SLM-Lab` in your research, please cite below:

```
@misc{kenggraesser2017slmlab,
    author = {Wah Loon Keng, Laura Graesser},
    title = {SLM-Lab},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/kengz/SLM-Lab}},
}
```

## Contributing

SLM Lab is an MIT-licensed open source project. Contributions are very much welcome, no matter if it's a quick bug-fix or new feature addition. Please see [CONTRIBUTING.md](https://github.com/kengz/SLM-Lab/blob/master/CONTRIBUTING.md) for more info.

If you have an idea for a new algorithm, environment support, analytics, benchmarking, or new experiment design, let us know.

If you're interested in using the lab for **research, teaching or applications**, please contact the [authors](https://twitter.com/kengzwl).
