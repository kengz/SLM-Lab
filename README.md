# SLM Lab [![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)
_(Work In Progress)_ An experimental framework for Reinforcement Learning using Unity and PyTorch.

## Installation

1.  Clone the Lab and the Env repos:
    ```shell
    git clone https://github.com/kengz/SLM-Lab.git
    git clone https://github.com/kengz/SLM-Env.git
    ```
    The Env repo is needed for the environment binaries, ready for Lab usage. Make sure both repo directories are siblings.

2.  Install Lab dependencies (or inspect `bin/*` before running):
    ```shell
    cd SLM-Lab/
    bin/setup
    source activate lab
    ```

3.  Setup config files:
    -   `config/default.json` for local development, used when `grunt` is ran without a production flag.
    -   `config/production.json` for production lab run when `grunt -prod` is ran with the production flag `-prod`.

## Usage

### High Level `yarn` commands

| Function | `command` |
| :------------- | :------------- |
| start the Lab | `yarn start` |
| update dependency file | `yarn update-dep` |
| update environment | `yarn update-env` |
| run tests | `yarn test` |
| clear cache | `yarn clear` |

## Notebook

The Lab uses interactive programming and lit workflow:

1.  Install [Atom text editor](https://atom.io/)
2.  Install [Hydrogen for Atom](https://atom.io/packages/hydrogen) and [these other Atom packages optionally](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-packages-json-L3)
3.  Use this [killer keymap for Hydrogen and other Atom shortcuts](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-keymap-cson-L15-L18).
    -   Install [Sync Settings](https://atom.io/packages/sync-settings)
    -   Fork the keymap gist
    -   Then update the sync settings config
4.  Open and run the example `slm_lab/notebook/intro_hydrogen.py` on Atom using Hydrogen and those keymaps
5.  See `slm_lab/notebook/intro_unity.py` to see example of Unity environment usage
6.  Start working from `slm_lab/notebook/`

## Experiment

_To be set up_

## Unity Environment

If you're just using prebuilt environments for the Lab, get them from the [SLM-Env](https://github.com/kengz/SLM-Env.git) as per the Installation instruction.

To develop and build a new Unity ml-agents environment, clone and use the fork [kengz/ml-agents](https://github.com/kengz/ml-agents):
  ```shell
  git clone https://github.com/kengz/ml-agents.git
  ```

1.  For the most part follow the [original doc](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md#building-unity-environment). Remember the core settings:
    -   `Player > Resolution and Presentation > Run in Background (checked)`
    -   `Player > Resolution and Presentation > Display Resolution Dialog (Disabled)`
    -   `Academy > Brain > External`

2.  Build the following versions of the environment binary (e.g. env name `3DBall`) and save them to the `SLM-Env` repo:
    -   MacOSX version
      -   make `Academy > Training Configuration` as follow (or leave as-is if smaller than `Inference Configuration`):
        -   Width: 128
        -   Height: 72
        -   Quality Level: 0
        -   Time Scale: 100
      -   build directory: `SLM-Env/Build/`
      -   save name: `3DBall`
    -   Linux version
      -   make `Training Configuration` same as MaxOSX
      -   `Headless Mode (checked)`
      -   save name: `3DBall`

3.  Make the sure the built binaries are in `SLM-Env/Build/`. Commit the changes and push to the `SLM-Env` repo. New environment is now ready for Lab usage.
