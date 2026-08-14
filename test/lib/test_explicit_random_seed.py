"""`spec.meta.random_seed`, when set, makes a session REGENERABLE. Default behaviour unchanged.

Without this, `set_random_seed` derives the seed from `time.time()`, so no individual run can be
regenerated: re-running the same spec draws a different network init and a different env stream.
That is fine for producing independent seeds and wrong for reproducing a specific result.
"""
from unittest import mock

from slm_lab.lib import ml_util


def _spec(trial=0, session=0, **meta):
    return {"meta": dict(trial=trial, session=session, **meta)}


def test_default_is_time_derived_and_unchanged():
    with mock.patch.object(ml_util.time, "time", return_value=1000.0):
        a = ml_util.set_random_seed(_spec())
    with mock.patch.object(ml_util.time, "time", return_value=2000.0):
        b = ml_util.set_random_seed(_spec())
    assert a == 1000 and b == 2000, "the default must remain exactly time-derived"


def test_explicit_seed_is_regenerable_and_ignores_the_clock():
    with mock.patch.object(ml_util.time, "time", return_value=1000.0):
        a = ml_util.set_random_seed(_spec(random_seed=777))
    with mock.patch.object(ml_util.time, "time", return_value=9e9):
        b = ml_util.set_random_seed(_spec(random_seed=777))
    assert a == b == 777


def test_explicit_seed_still_separates_sessions_and_trials():
    """One supplied number must not collapse every session of an experiment onto one stream."""
    seeds = {ml_util.set_random_seed(_spec(trial=t, session=s, random_seed=777))
             for t, s in ((0, 0), (0, 1), (1, 0))}
    assert len(seeds) == 3


def test_explicit_None_falls_back_to_the_default():
    """`random_seed: null` in a spec must behave as if the key were absent, not as seed 0."""
    with mock.patch.object(ml_util.time, "time", return_value=1234.0):
        assert ml_util.set_random_seed(_spec(random_seed=None)) == 1234
