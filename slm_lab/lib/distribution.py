# Custom distribution classes to extend torch.distributions
# Mainly used by policy_util action distribution
from torch import distributions
import torch


class Argmax(distributions.Categorical):
    '''
    Special distribution class for argmax sampling, where probability is always 1 for the argmax.
    NOTE although argmax is not a sampling distribution, this implementation is for API consistency.
    '''

    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            new_probs = torch.zeros_like(probs, dtype=torch.float)
            new_probs[probs == probs.max(dim=-1, keepdim=True)[0]] = 1.0
            probs = new_probs
        elif logits is not None:
            new_logits = torch.full_like(logits, -1e8, dtype=torch.float)
            new_logits[logits == logits.max(dim=-1, keepdim=True)[0]] = 1.0
            logits = new_logits

        super().__init__(probs=probs, logits=logits, validate_args=validate_args)


class GumbelCategorical(distributions.Categorical):
    '''
    Special Categorical using Gumbel distribution to simulate softmax categorical for discrete action.
    Similar to OpenAI's https://github.com/openai/baselines/blob/98257ef8c9bd23a24a330731ae54ed086d9ce4a7/baselines/a2c/utils.py#L8-L10
    Explanation http://amid.fish/assets/gumbel.html
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel softmax sampling'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=0)


class MultiCategorical(distributions.Categorical):
    '''MultiCategorical as collection of Categoricals'''

    def __init__(self, probs=None, logits=None, validate_args=None):
        self.categoricals = []
        if probs is None:
            probs = [None] * len(logits)
        elif logits is None:
            logits = [None] * len(probs)
        else:
            raise ValueError('Either probs or logits must be None')

        for sub_probs, sub_logits in zip(probs, logits):
            categorical = distributions.Categorical(probs=sub_probs, logits=sub_logits, validate_args=validate_args)
            self.categoricals.append(categorical)

    @property
    def logits(self):
        return [cat.logits for cat in self.categoricals]

    @property
    def probs(self):
        return [cat.probs for cat in self.categoricals]

    @property
    def param_shape(self):
        return [cat.param_shape for cat in self.categoricals]

    @property
    def mean(self):
        return torch.stack([cat.mean for cat in self.categoricals])

    @property
    def variance(self):
        return torch.stack([cat.variance for cat in self.categoricals])

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([cat.sample(sample_shape=sample_shape) for cat in self.categoricals])

    def log_prob(self, value):
        value_t = value.transpose(0, 1)
        return torch.stack([cat.log_prob(value_t[idx]) for idx, cat in enumerate(self.categoricals)])

    def entropy(self):
        return torch.stack([cat.entropy() for cat in self.categoricals])

    def enumerate_support(self):
        return [cat.enumerate_support() for cat in self.categoricals]
