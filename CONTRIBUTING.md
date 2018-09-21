# Contributing to SLM Lab

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

SLM Lab has the following goals:
1. make RL reproducible: all you need is the spec file and a git SHA to reproduce a result
2. make research easier and more accessible: reuse well-tested components and only focus on the relevant work
3. make learning RL easier: learning the components piece-wise is easier than building an entire algorithm from scratch

We believe that deep RL stands at a wonderful intersection of research and engineering. Hence, all forms of contributions are welcome. If you need a hint on what to contribute, feel free to check out the:
- [issues](https://github.com/kengz/SLM-Lab/issues)
- [roadmaps](https://github.com/kengz/SLM-Lab/projects), which are not separated into research/engineering because they are tightly coupled.

Contributions are merged into SLM Lab via Pull Requests (PRs). If you are not familiar with it, [don't be shy, here's a guide](https://opensource.guide/how-to-contribute/#opening-a-pull-request). Also, feel free to reach us at the [Gitter channel](https://gitter.im/SLM-Lab/SLM-Lab).

For ease of navigation, contributors may follow any of these tracks:

## Run Benchmark Experiments

Reproducibility is crucial to deep RL, thus it is very valuable to have multiple contributers verifying the results by rerunning it. This can be done by using a spec file and checking out SLM Lab to a specific git SHA commit and running the experiment. When the results are available, please upload them by creating a [pull request](https://github.com/kengz/SLM-Lab/pulls) so it is visible to everyone.

Even if your experiments produce negative results, please report it too! *It can be more valuable to know what doesn't work*, as opposed to just know what works. Reproducibility is key even for negative results, so if you can show how to reproduce the it, please submit a PR.

## Implement Features

Other lab users may request features which could be a generic lab component, or implementation of an RL component. They can be found in issues or roadmaps. As with any project, it will take some effort to get familiar with the lab's workflow, but with that you can start implementing features the others can use.

Because deep RL itself is difficult, it is crucial for the components to be robust and easy to reuse across many algorithms. So, it is worth a little extra effort to ensure the implementations have a higher quality.

## Research

Requests for research are found in the roadmaps, and we encourage you to work on any, no matter how big or small. If you have a research idea, feel free to propose in the [Gitter chat](https://gitter.im/SLM-Lab/SLM-Lab) as well.
