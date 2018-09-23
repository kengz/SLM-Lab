# Contributing to SLM Lab

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

SLM Lab has the following principles:
- **modularity**: components get reused maximally, which means less code, more tests, fewer bugs
- **simplicity**: the components are designed to closely correspond to the way papers or books discuss RL
- **analytical clarity**: hyper-parameter search results are analyzed automatically and presented hierarchically, in increasingly granular detail
- **reproducibility**: only the spec file and a git SHA are needed to completely reproduce an experiment

We believe that deep RL stands at a wonderful intersection of research and engineering. Hence, all forms of contributions are welcome. If you need a hint on what to contribute, feel free to check out our wish list below.

Contributions are merged into SLM Lab via Pull Requests (PRs). If you are not familiar with it, [don't be shy, here's a guide](https://opensource.guide/how-to-contribute/#opening-a-pull-request). Also, feel free to reach us at the [Gitter channel](https://gitter.im/SLM-Lab/SLM-Lab).

Contributors may follow any of these tracks. They are roughly ordered in increasing difficulty.

## Reproduce Results

Reproducibility is crucial to deep RL, thus it is very valuable to have multiple contributors verifying the results by running experiments. An experiment in the SLM-Lab can be fully reproduced using
1. a spec file
2. a git SHA: check out the SLM Lab to the appropriate branch or commit

An example experiment is [here](todo)

>The only aspect of experiments that are not reproducible are the random seeds. This is by design. Good algorithms and hyper-parameter settings should be reasonably stable across different random seeds.

When the results are available, please upload them by creating a [pull request](https://github.com/kengz/SLM-Lab/pulls) so it is visible to everyone.

Even if your experiments produce negative results, please report it too! *It can be more valuable to know what doesn't work*, as opposed to just know what works. Reproducibility is key even for negative results, so if you can show how to reproduce the it, please submit a PR.

## Run Benchmark Experiments

The next major milestone for SLM Lab is to provide a set of benchmark results on the implemented algorithms in a variety of environments. The aim is to populate this [table](BENCHMARK.md). We encourage contributors to own benchmarking for one or more algorithm-environment combinations. Let us know what you are working on and we will add your name to the benchmark table.

One approach could be to go deep on a single algorithm and tune it for different environments. This can be an excellent way for newcomers to deep RL to learn in depth about a particular algorithm. A high level workflow could go as follows
- Get familiar with the Lab; run the demo, read the [docs](https://kengz.gitbooks.io/slm-lab/content/), look though a couple of [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec)
- Pick an algorithm; change some parameters in a spec file, run a few small scale experiments, see what happens
- Dig deeper
  - read 1 - 2 online tutorials. See [here](TUTORIALS.md) for some suggestions
  - read the [algorithm code](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm)
  - look at the relevant [memory](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/memory) and [net](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/net) code
- Run larger scale experiments

We of course welcome algorithm experts too! Your experience working on deep RL algorithms is extremely helpful and needed.

An alternative approach to to select a single environment and focus on getting good results for a number of different algorithms in that environment.

## Implement Features

Other lab users may request features which could be a generic lab component, or implementation of an RL component. They can be found in [issues](https://github.com/kengz/SLM-Lab/issues) or [roadmaps](https://github.com/kengz/SLM-Lab/projects/3). As with any project, it will take some effort to get familiar with the lab's workflow, but with that you can start implementing features the others can use.

Because deep RL itself is difficult, it is crucial for the components to be robust and easy to reuse across many algorithms. So, it is worth a little extra effort to ensure the implementations have a higher quality.

## Research

Requests for research are found in the [roadmaps](https://github.com/kengz/SLM-Lab/projects/3), and we encourage you to work on any, no matter how big or small. If you have a research idea, feel free to propose in the [Gitter chat](https://gitter.im/SLM-Lab/SLM-Lab) as well.
