# SLM Lab [![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)
_(Work In Progress)_ An experimental framework for Reinforcement Learning using Unity and PyTorch.

## Installation

1. Clone the repo:
  ```shell
  git clone https://github.com/kengz/SLM-Lab.git
  ```

2. Install dependencies (or inspect `bin/*` before running):
  ```shell
  bin/setup
  source activate lab
  ```

3. Setup config files:
  -  `config/default.json` for local development, used when `grunt` is ran without a production flag.
  -  `config/production.json` for production lab run when `grunt -prod` is ran with the production flag `-prod`.

## Usage

### High Level `yarn` commands

| Function | `command` |
| :------------- | :------------- |
| start the Lab | `yarn start` |
| update dependency file | `yarn update-dep` |
| update environment | `yarn update-env` |
| run tests | `yarn test` |
| clear cache | `yarn clear` |

### Notebook

The Lab uses interactive programming and lit workflow:

1. Install [Atom text editor](https://atom.io/)
2. Install [Hydrogen for Atom](https://atom.io/packages/hydrogen) and [these other Atom packages optionally](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-packages-json-L3)
3. Use this [killer keymap for Hydrogen and other Atom shortcuts](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-keymap-cson-L15-L18)
4. Open and run the example `slm_lab/notebook/hydrogen.py` on Atom using Hydrogen and those keymaps
5. Start working from `slm_lab/notebook/`

### Experiment

_To be set up_
