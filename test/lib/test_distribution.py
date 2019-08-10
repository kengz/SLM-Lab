from flaky import flaky
from slm_lab.lib import distribution
import pytest
import torch


@pytest.mark.parametrize('pdparam_type', [
    'probs', 'logits'
])
def test_argmax(pdparam_type):
    pdparam = torch.tensor([1.1, 10.0, 2.1])
    # test both probs or logits
    pd = distribution.Argmax(**{pdparam_type: pdparam})
    for _ in range(10):
        assert pd.sample().item() == 1
    assert torch.equal(pd.probs, torch.tensor([0., 1., 0.]))


@flaky
@pytest.mark.parametrize('pdparam_type', [
    'probs', 'logits'
])
def test_gumbel_categorical(pdparam_type):
    pdparam = torch.tensor([1.1, 10.0, 2.1])
    pd = distribution.GumbelSoftmax(**{pdparam_type: pdparam})
    for _ in range(10):
        assert torch.is_tensor(pd.sample())


@pytest.mark.parametrize('pdparam_type', [
    'probs', 'logits'
])
def test_multicategorical(pdparam_type):
    pdparam0 = torch.tensor([10.0, 0.0, 0.0])
    pdparam1 = torch.tensor([0.0, 10.0, 0.0])
    pdparam2 = torch.tensor([0.0, 0.0, 10.0])
    pdparams = [pdparam0, pdparam1, pdparam2]
    # use a probs
    pd = distribution.MultiCategorical(**{pdparam_type: pdparams})
    assert isinstance(pd.probs, list)
    # test probs only since if init from logits, probs will be close but not precise
    if pdparam_type == 'probs':
        assert torch.equal(pd.probs[0], torch.tensor([1., 0., 0.]))
        assert torch.equal(pd.probs[1], torch.tensor([0., 1., 0.]))
        assert torch.equal(pd.probs[2], torch.tensor([0., 0., 1.]))
    for _ in range(10):
        assert torch.equal(pd.sample(), torch.tensor([0, 1, 2]))
