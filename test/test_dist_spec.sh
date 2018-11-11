#!/bin/bash --login
# Pytest cannot test methods which use multiprocessing when ran from python. But running them individually on the terminal works

# Fail on the first error; killable by SIGINT
set -e
trap "exit" INT

echo "Running distributed algorithm tests separately"
pytest --verbose test/spec/test_spec.py::test_reinforce_dist
pytest --verbose test/spec/test_spec.py::test_reinforce_dist
pytest --verbose test/spec/test_spec.py::test_reinforce_cont_dist
pytest --verbose test/spec/test_spec.py::test_dppo_dist
pytest --verbose test/spec/test_spec.py::test_ppo_cont_dist
pytest --verbose test/spec/test_spec.py::test_ppo_sil_dist
pytest --verbose test/spec/test_spec.py::test_ppo_sil_cont_dist
pytest --verbose test/spec/test_spec.py::test_sil_dist
pytest --verbose test/spec/test_spec.py::test_sil_cont_dist
pytest --verbose test/spec/test_spec.py::test_sarsa_dist
pytest --verbose test/spec/test_spec.py::test_dqn_dist
pytest --verbose test/spec/test_spec.py::test_ddqn_dist
pytest --verbose test/spec/test_spec.py::test_dueling_dqn_dist
pytest --verbose test/spec/test_spec.py::test_multitask_dqn_dist
pytest --verbose test/spec/test_spec.py::test_multitask_dqn_dist