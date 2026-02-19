import torch

from slm_lab.agent.algorithm.actor_critic import PercentileNormalizer


def test_init_zeros():
    norm = PercentileNormalizer()
    assert norm.perc5 == 0.0
    assert norm.perc95 == 0.0


def test_update_tracks_percentiles():
    norm = PercentileNormalizer(decay=0.0)  # no EMA, instant update
    values = torch.arange(100, dtype=torch.float32)
    norm.update(values)
    # With decay=0, perc5/perc95 should match torch.quantile exactly
    expected_p5 = torch.quantile(values, 0.05).item()
    expected_p95 = torch.quantile(values, 0.95).item()
    assert abs(norm.perc5 - expected_p5) < 1e-4
    assert abs(norm.perc95 - expected_p95) < 1e-4


def test_normalize_divides_by_scale():
    norm = PercentileNormalizer(decay=0.0)
    values = torch.arange(100, dtype=torch.float32)
    norm.update(values)
    scale = max(1.0, norm.perc95 - norm.perc5)
    result = norm.normalize(values)
    expected = values / scale
    assert torch.allclose(result, expected)


def test_ema_decay_converges():
    """After many updates with the same distribution, percentiles should converge"""
    norm = PercentileNormalizer(decay=0.99)
    values = torch.randn(1000)
    for _ in range(500):
        norm.update(values)
    # After convergence, perc5/perc95 should approximate the true quantiles
    true_p5 = torch.quantile(values, 0.05).item()
    true_p95 = torch.quantile(values, 0.95).item()
    assert abs(norm.perc5 - true_p5) < 0.3
    assert abs(norm.perc95 - true_p95) < 0.3


def test_normalize_zero_range():
    """All same values: scale should be max(1.0, 0) = 1.0"""
    norm = PercentileNormalizer(decay=0.0)
    values = torch.ones(100)
    norm.update(values)
    result = norm.normalize(values)
    # scale = max(1.0, perc95 - perc5) = max(1.0, 0.0) = 1.0
    assert torch.allclose(result, values)


def test_normalize_uniform_distribution():
    norm = PercentileNormalizer(decay=0.0)
    values = torch.linspace(0, 100, 1000)
    norm.update(values)
    result = norm.normalize(values)
    scale = max(1.0, norm.perc95 - norm.perc5)
    assert torch.allclose(result, values / scale)


def test_normalize_skewed_distribution():
    """Skewed distribution still produces finite output"""
    norm = PercentileNormalizer(decay=0.0)
    # Exponential-like skew
    values = torch.exp(torch.randn(500))
    norm.update(values)
    result = norm.normalize(values)
    assert torch.all(torch.isfinite(result))
    assert norm.perc95 > norm.perc5
