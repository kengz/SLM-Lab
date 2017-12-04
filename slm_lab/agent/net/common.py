import torch

# Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2
def flatten_params(net):
    return torch.cat([param.data.view(-1) for param in net.parameters()], 0)

# Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2
def load_params(net, flattened):
    offset = 0
    for param in net.parameters():
        param.data.copy_(flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()
    return net
