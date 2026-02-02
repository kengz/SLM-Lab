"""Tests for policy_util module, especially ACTION_PDS configuration."""
import pytest
from torch import distributions

from slm_lab.agent.algorithm import policy_util


class TestActionPds:
    """Tests for ACTION_PDS distribution mapping."""

    def test_multi_continuous_includes_normal(self):
        """Normal should be first option for multi_continuous (standard for SAC/PPO)."""
        pdtypes = policy_util.ACTION_PDS['multi_continuous']
        assert 'Normal' in pdtypes, 'Normal must be available for multi_continuous'
        assert pdtypes[0] == 'Normal', 'Normal should be first (default) for multi_continuous'

    def test_multi_continuous_includes_multivariate_normal(self):
        """MultivariateNormal should also be available for multi_continuous."""
        pdtypes = policy_util.ACTION_PDS['multi_continuous']
        assert 'MultivariateNormal' in pdtypes

    def test_continuous_includes_normal(self):
        """Normal should be first option for continuous."""
        pdtypes = policy_util.ACTION_PDS['continuous']
        assert pdtypes[0] == 'Normal'

    def test_discrete_includes_categorical(self):
        """Categorical should be first option for discrete."""
        pdtypes = policy_util.ACTION_PDS['discrete']
        assert pdtypes[0] == 'Categorical'

    def test_all_action_types_have_defaults(self):
        """All action types should have at least one distribution option."""
        for action_type, pdtypes in policy_util.ACTION_PDS.items():
            assert len(pdtypes) > 0, f'{action_type} has no distributions'


class TestGetActionPdCls:
    """Tests for get_action_pd_cls function."""

    def test_normal_for_multi_continuous(self):
        """Normal should be valid for multi_continuous action types."""
        pd_cls = policy_util.get_action_pd_cls('Normal', 'multi_continuous')
        assert pd_cls == distributions.Normal

    def test_normal_for_continuous(self):
        """Normal should be valid for continuous action types."""
        pd_cls = policy_util.get_action_pd_cls('Normal', 'continuous')
        assert pd_cls == distributions.Normal

    def test_categorical_for_discrete(self):
        """Categorical should be valid for discrete action types."""
        pd_cls = policy_util.get_action_pd_cls('Categorical', 'discrete')
        assert pd_cls == distributions.Categorical

    def test_invalid_pdtype_raises(self):
        """Invalid pdtype for action type should raise assertion."""
        with pytest.raises(AssertionError):
            policy_util.get_action_pd_cls('Categorical', 'continuous')

    def test_invalid_action_type_raises(self):
        """Invalid action type should raise KeyError."""
        with pytest.raises(KeyError):
            policy_util.get_action_pd_cls('Normal', 'invalid_type')


class TestInitActionPd:
    """Tests for init_action_pd function."""

    def test_normal_distribution_init(self):
        """Normal distribution should initialize with loc and scale."""
        import torch
        # pdparam shape: [batch, 2, action_dim] where 2 is [loc, log_scale]
        pdparam = [torch.zeros(2, 2), torch.zeros(2, 2)]  # loc, log_scale
        action_pd = policy_util.init_action_pd(distributions.Normal, pdparam)
        assert isinstance(action_pd, distributions.Normal)
        assert action_pd.loc.shape == (2, 2)

    def test_categorical_distribution_init(self):
        """Categorical distribution should initialize with logits."""
        import torch
        pdparam = torch.randn(2, 4)  # batch_size=2, num_actions=4
        action_pd = policy_util.init_action_pd(distributions.Categorical, pdparam)
        assert isinstance(action_pd, distributions.Categorical)
        assert action_pd.logits.shape == (2, 4)

    def test_normal_1d_continuous_init(self):
        """Normal distribution should handle 1D continuous (Pendulum-like) with tensor input."""
        import torch
        # 1D action: pdparam is [batch, 2] tensor (loc, log_scale concatenated)
        pdparam = torch.randn(256, 2)
        action_pd = policy_util.init_action_pd(distributions.Normal, pdparam)
        assert isinstance(action_pd, distributions.Normal)
        # Shape should be [batch, 1] for consistent sum(-1) behavior
        assert action_pd.loc.shape == (256, 1)
        assert action_pd.scale.shape == (256, 1)

    def test_normal_1d_log_prob_shape(self):
        """1D continuous log_prob should have correct shape for sum(-1)."""
        import torch
        pdparam = torch.randn(256, 2)
        action_pd = policy_util.init_action_pd(distributions.Normal, pdparam)
        actions = action_pd.rsample()
        log_prob = action_pd.log_prob(actions)
        # sum(-1) should produce [batch] shape, not scalar
        result = log_prob.sum(-1)
        assert result.shape == (256,), f'Expected shape (256,), got {result.shape}'

    def test_normal_multidim_continuous_init(self):
        """Normal distribution should handle multi-dim continuous (Lunar-like) with list input."""
        import torch
        # Multi-dim action: pdparam is list of [loc, log_scale] tensors
        pdparam = [torch.randn(256, 2), torch.randn(256, 2)]
        action_pd = policy_util.init_action_pd(distributions.Normal, pdparam)
        assert isinstance(action_pd, distributions.Normal)
        assert action_pd.loc.shape == (256, 2)
        assert action_pd.scale.shape == (256, 2)

    def test_normal_multidim_log_prob_shape(self):
        """Multi-dim continuous log_prob should have correct shape for sum(-1)."""
        import torch
        pdparam = [torch.randn(256, 2), torch.randn(256, 2)]
        action_pd = policy_util.init_action_pd(distributions.Normal, pdparam)
        actions = action_pd.rsample()
        log_prob = action_pd.log_prob(actions)
        result = log_prob.sum(-1)
        assert result.shape == (256,), f'Expected shape (256,), got {result.shape}'

    def test_entropy_1d_continuous_shape(self):
        """1D continuous entropy should have correct shape for sum(-1)."""
        import torch
        pdparam = torch.randn(256, 2)
        action_pd = policy_util.init_action_pd(distributions.Normal, pdparam)
        entropy = action_pd.entropy()
        # Shape should be [batch, 1] for consistent sum(-1) behavior
        assert entropy.shape == (256, 1), f'Expected shape (256, 1), got {entropy.shape}'

    def test_entropy_multidim_continuous_shape(self):
        """Multi-dim continuous entropy should have correct shape for sum(-1)."""
        import torch
        pdparam = [torch.randn(256, 6), torch.randn(256, 6)]  # HalfCheetah-like
        action_pd = policy_util.init_action_pd(distributions.Normal, pdparam)
        entropy = action_pd.entropy()
        assert entropy.shape == (256, 6), f'Expected shape (256, 6), got {entropy.shape}'

    def test_entropy_sum_then_mean_pattern(self):
        """Entropy sum(-1).mean() should scale with action dimensions (CleanRL standard).

        This is the correct pattern for computing entropy in policy gradient methods:
        - Sum entropy across action dimensions first
        - Then take mean over batch

        The OLD incorrect pattern was .mean() on all dims, which made entropy
        contribution N times weaker for N-dimensional action spaces.
        """
        import torch
        batch_size = 256

        # 1D continuous (Pendulum-like)
        pdparam_1d = torch.randn(batch_size, 2)
        action_pd_1d = policy_util.init_action_pd(distributions.Normal, pdparam_1d)
        entropy_1d = action_pd_1d.entropy()
        if entropy_1d.dim() > 1:
            entropy_1d = entropy_1d.sum(dim=-1)
        entropy_1d_result = entropy_1d.mean()

        # 6D continuous (HalfCheetah-like)
        pdparam_6d = [torch.zeros(batch_size, 6), torch.zeros(batch_size, 6)]  # zero mean, unit std
        action_pd_6d = policy_util.init_action_pd(distributions.Normal, pdparam_6d)
        entropy_6d = action_pd_6d.entropy()
        if entropy_6d.dim() > 1:
            entropy_6d = entropy_6d.sum(dim=-1)
        entropy_6d_result = entropy_6d.mean()

        # Entropy should scale approximately with action dims
        # Using zeros for loc and zeros for log_scale (so scale=exp(0)=1)
        # gives consistent entropy per dimension
        ratio = entropy_6d_result / entropy_1d_result
        # Should be close to 6.0 (6 dims vs 1 dim)
        assert 5.5 < ratio < 6.5, f'Entropy should scale ~6x, got ratio={ratio:.2f}'
