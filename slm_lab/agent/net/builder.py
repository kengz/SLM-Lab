# Module to quickly build neural networks with relatively simple architecture
import inspect
import torch
import torch.nn as nn
import pydash as ps
import torch.nn.functional as F

# TODO rebuild dependencies too

# general net spec, can be composed from programs
net_spec = {
    "#heads": {
        "image": {
            "type": "conv2d",
            "layers": [
                [32, 8, 4, 0, 1],
                [64, 4, 2, 0, 1],
                [32, 3, 1, 0, 1]
            ],
            "activation": "relu",
            "batch_norm": None,
            "init_fn": "orthogonal_",
        },
        "gyro": {
            "type": "mlp",
            "layers": [64, 64],
            "activation": "relu",
            "batch_norm": None,
            "init_fn": "orthogonal_",
        },
    },
    "#body": {  # this is special since it contains a models op
        "join": "film",  # or concat
        "type": "mlp",
        "layers": [],
        "activation": "relu",
        "init_fn": "orthogonal_",
    },
    "#tails": {
        "v": {
            "type": "mlp",
            "layers": [],
            "activation": "relu",
            "out_activation": None,
            # TODO out_layer activation
        },
        "pi": {
            "type": "mlp",
            "layers": [],
            "activation": "relu",
            "out_activation": None,
        }
    }
}


def get_nn_name(uncased_name):
    '''Helper to get the proper name in PyTorch nn given a case-insensitive name'''
    for nn_name in nn.__dict__:
        if uncased_name.lower() == nn_name.lower():
            return nn_name
    raise ValueError(f'Name {uncased_name} not found in {nn.__dict__}')


def build_activation_layer(activation):
    '''Helper to build an activation function layer for net'''
    if activation is None:
        return None
    else:
        ActivationClass = getattr(nn, get_nn_name(activation))
        return ActivationClass()


def resolve_activation_layer(net_spec, is_last_layer):
    '''
    Help resolve the activation layer depending if it's the last layer
    if not the last layer:
        use net_spec['activation']
    if the last layer and:
        if net_spec['out_activation'] is specified, use it
        else use net_spec['activation']
    @param dict:net_spec Specifying 'activation' and optionally 'out_activation'
    @return Layer:Activation
    '''
    activation, out_activation = ps.at(net_spec, *['activation', 'out_activation'])
    if not is_last_layer:
        return build_activation_layer(activation)
    else:  # last layer
        if 'out_activation' in net_spec:  # specified explicitly
            return build_activation_layer(out_activation)
        else:
            return build_activation_layer(activation)


def get_init_weights(init_fn, activation=None):
    '''
    Helper to get the init_weights function given init_fn and activation to be applied as net.apply(init_weights)
    @param str:init_fn
    @param str:activation
    @returns function:init_weights
    '''
    def init_weights(module):
        if (not hasattr(module, 'weight')) or 'BatchNorm' in module.__class__.__name__:
            # skip if not a module with weight, or if it's a BatchNorm layer
            return

        # init weight accordingly
        weight_init_fn = getattr(nn.init, init_fn)
        init_fn_args = inspect.getfullargspec(weight_init_fn).args
        if activation is None:
            weight_init_fn(module.weight)
        elif 'gain' in init_fn_args:
            weight_init_fn(module.weight, gain=nn.init.calculate_gain(activation))
        elif 'nonlinearity' in init_fn_args:
            weight_init_fn(module.weight, nonlinearity=activation)
        else:
            weight_init_fn(module.weight)
        # init bias
        if module.bias is not None:
            nn.init.normal_(module.bias)
    return init_weights


def check_net_spec(net_spec):
    # CHECK FOR KEYS
    pass


def build_mlp_model(net_spec):
    '''
    Build an MLP model given net_spec
    @param dict:net_spec With the following format/example:
    net_spec = {
        "type": "mlp",
        "in_shape": 4,  # input shape
        "out_shape": 2,  # optional: output shape if this is a full model
        "layers": [64, 64],  # hidden layers
        "batch_norm": True,  # optional: apply BatchNorm before activation
        "activation": "relu",  # activation function
        "out_activation": None,  # optional: specify to override 'activation' for the last layer, useful for output model
        "init_fn": "orthogonal_",  # weight initialization
    }
    '''
    check_net_spec(net_spec)
    in_shape, out_shape, layers, batch_norm, activation, out_activation, init_fn = ps.at(net_spec, *['in_shape', 'out_shape', 'layers', 'batch_norm', 'activation', 'out_activation', 'init_fn'])

    nn_layers = []
    # if out_shape is specified in net_spec (a full network), copy layers and append out_shape to iterate
    layers = layers + [out_shape] if out_shape else layers
    for idx, out_shape in enumerate(layers):
        is_last_layer = (idx == (len(layers) - 1))
        nn_layers.append(nn.Linear(in_shape, out_shape))
        if batch_norm and not is_last_layer:
            nn_layers.append(nn.BatchNorm1d(out_shape))
        nn_layers.append(resolve_activation_layer(net_spec, is_last_layer=is_last_layer))
        in_shape = out_shape  # update in_shape
        net_spec['out_shape'] = out_shape
    nn_layers = ps.compact(nn_layers)  # remove None
    mlp_model = nn.Sequential(*nn_layers)

    if init_fn:  # initialize weights if specified
        init_weights = get_init_weights(init_fn, activation)
        mlp_model.apply(init_weights)
    return mlp_model


net_spec = {
    "type": "mlp",
    "in_shape": 4,
    "layers": [64, 64],
    "activation": "relu",
    "init_fn": "orthogonal_",
}
mlp_model = build_mlp_model(net_spec)
assert net_spec['out_shape'] == net_spec['layers'][-1]
layer_names = ['Linear', 'ReLU', 'Linear', 'ReLU']
for nn_layer, layer_name in zip(mlp_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "mlp",
    "in_shape": 4,
    "layers": [64, 64],
    "activation": "relu",
    "out_activation": None,
    "init_fn": "orthogonal_",
}
mlp_model = build_mlp_model(net_spec)
assert net_spec['out_shape'] == net_spec['layers'][-1]
layer_names = ['Linear', 'ReLU', 'Linear']
for nn_layer, layer_name in zip(mlp_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "mlp",
    "in_shape": 4,
    "out_shape": 2,
    "layers": [64, 64],
    "batch_norm": True,
    "activation": "relu",
    "out_activation": "sigmoid",
    "init_fn": "orthogonal_",
}
mlp_model = build_mlp_model(net_spec)
assert net_spec['out_shape'] == net_spec['out_shape']
layer_names = ['Linear', 'BatchNorm1d', 'ReLU', 'Linear', 'BatchNorm1d', 'ReLU', 'Linear', 'Sigmoid']
for nn_layer, layer_name in zip(mlp_model, layer_names):
    assert nn_layer._get_name() == layer_name


def get_conv_out_shape(conv_model, in_shape):
    '''Helper to calculate the output shape of a conv model with flattened last layer given an input shape'''
    x = torch.rand(in_shape).unsqueeze(dim=0)
    if isinstance(conv_model, list):  # if passed nn_layers, build tmp model
        conv_model = nn.Sequential(*ps.compact(conv_model))
    y = conv_model(x).squeeze(dim=0)
    out_shape = y.shape
    if len(out_shape) == 1:
        return out_shape[0]
    else:
        return list(out_shape)


def build_conv_model(net_spec):
    '''
    Build a Conv1d, Conv2d (vision), or Conv3d model given net_spec
    Conv params: layer = [in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1]
    Layer contains arguments to Conv except the in_channels, which is inferred. Feel free to provide layer with as many params, since they are applied with *layer
    @param dict:net_spec With the following format/example:
    net_spec = {
        "type": "conv2d",  # options: 'conv1d', 'conv2d', conv3d
        "in_shape": [3, 84, 84],  # channel, height, width
        "out_shape": 2,  # optional: output shape if this is a full model. This must go with "flatten": True.
        "layers": [
            [32, 8, 4, 0, 1],  # out_channels, kernel_size, stride, padding, dilation
            [64, 4, 2, 0, 1],
            [32, 3, 1, 0, 1]
        ],
        "batch_norm": True,  # optional: apply BatchNorm before activation
        "activation": "relu",  # activation function
        "out_activation": None,  # optional: specify to override 'activation' for the last layer, useful for output model
        "flatten": False,  # optional: flatten the output layer
        "init_fn": "orthogonal_",  # weight initialization
    }
    '''
    check_net_spec(net_spec)
    in_shape, out_shape, layers, batch_norm, activation, out_activation, flatten, init_fn = ps.at(net_spec, *['in_shape', 'out_shape', 'layers', 'batch_norm', 'activation', 'out_activation', 'flatten', 'init_fn'])
    if net_spec['type'] == 'conv1d':
        ConvClass, BNClass = nn.Conv1d, nn.BatchNorm1d
    elif net_spec['type'] == 'conv2d':
        ConvClass, BNClass = nn.Conv2d, nn.BatchNorm2d
    elif net_spec['type'] == 'conv3d':
        ConvClass, BNClass = nn.Conv3d, nn.BatchNorm3d
    else:
        raise ValueError(f"type: {net_spec['type']} is not supported")

    in_c = in_shape[0]  # PyTorch image input shape is (c,h,w)
    nn_layers = []
    layers = layers + [out_shape] if out_shape else layers
    for idx, layer in enumerate(layers):
        is_last_layer = (idx == (len(layers) - 1))
        if isinstance(layer, list):
            out_c = layer[0]
            nn_layers.append(ConvClass(in_c, *layer))
            if batch_norm:
                nn_layers.append(BNClass(out_c))
            nn_layers.append(resolve_activation_layer(net_spec, is_last_layer=is_last_layer))
            in_c = out_c  # update
            if is_last_layer and (not out_shape) and flatten:
                nn_layers.append(nn.Flatten())
        else:  # if specified net_spec['out_shape] for output
            assert is_last_layer and out_shape and flatten
            # flatten conv model and get conv_out_shape
            nn_layers.append(nn.Flatten())
            conv_out_shape = get_conv_out_shape(nn_layers, in_shape)
            # add the mlp output layer if specified in net_spec
            nn_layers.append(nn.Linear(conv_out_shape, out_shape))
            nn_layers.append(resolve_activation_layer(net_spec, is_last_layer=is_last_layer))
    nn_layers = ps.compact(nn_layers)  # remove None
    conv_model = nn.Sequential(*nn_layers)
    net_spec['out_shape'] = get_conv_out_shape(conv_model, in_shape)

    if init_fn:  # initialize weights if specified
        init_weights = get_init_weights(init_fn, activation)
        conv_model.apply(init_weights)
    return conv_model


net_spec = {
    "type": "conv1d",
    "in_shape": [3, 20],
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "activation": "relu",
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], list)
layer_names = ['Conv1d', 'ReLU', 'Conv1d', 'ReLU']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "conv1d",
    "in_shape": [3, 20],
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "activation": "relu",
    "flatten": True,
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], int)
layer_names = ['Conv1d', 'ReLU', 'Conv1d', 'ReLU', 'Flatten']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "conv1d",
    "in_shape": [3, 20],
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "batch_norm": True,
    "activation": "relu",
    "out_activation": "sigmoid",
    "flatten": True,
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], int)
layer_names = ['Conv1d', 'BatchNorm1d', 'ReLU', 'Conv1d', 'BatchNorm1d', 'Sigmoid', 'Flatten']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name

# TODO rename to flatten_conv
net_spec = {
    "type": "conv1d",
    "in_shape": [3, 20],
    "out_shape": 2,
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "batch_norm": True,
    "activation": "relu",
    "out_activation": "sigmoid",
    "flatten": True,
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], int)
layer_names = ['Conv1d', 'BatchNorm1d', 'ReLU', 'Conv1d', 'BatchNorm1d', 'ReLU', 'Flatten', 'Linear', 'Sigmoid']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "conv2d",
    "in_shape": [3, 20, 20],
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "activation": "relu",
    "flatten": True,
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], int)
layer_names = ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'Flatten']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "conv2d",
    "in_shape": [3, 20, 20],
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "batch_norm": True,
    "activation": "relu",
    "out_activation": "sigmoid",
    "flatten": True,
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], int)
layer_names = ['Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Sigmoid', 'Flatten']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "conv3d",
    "in_shape": [3, 20, 20, 20],
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "activation": "relu",
    "flatten": True,
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], int)
layer_names = ['Conv3d', 'ReLU', 'Conv3d', 'ReLU', 'Flatten']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name

net_spec = {
    "type": "conv3d",
    "in_shape": [3, 20, 20, 20],
    "layers": [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    "batch_norm": True,
    "activation": "relu",
    "out_activation": "sigmoid",
    "flatten": True,
    "init_fn": "orthogonal_",
}
conv_model = build_conv_model(net_spec)
assert isinstance(net_spec['out_shape'], int)
layer_names = ['Conv3d', 'BatchNorm3d', 'ReLU', 'Conv3d', 'BatchNorm3d', 'Sigmoid', 'Flatten']
for nn_layer, layer_name in zip(conv_model, layer_names):
    assert nn_layer._get_name() == layer_name


def build_recurrent_model(net_spec):
    '''
    Build a Recurrent model given net_spec
    @param dict:net_spec With the following format/example:
    net_spec = {
        "type": "rnn",  # options: 'rnn', 'lstm', 'gru'
        "in_shape": 3,  # the number of features in x
        "layers": [64, 64],  # the hidden layers, must be the same for all layers
        "bidirectional": False,  # whether to make network bidirectional
        "init_fn": "orthogonal_",  # weight initialization
    }
    '''
    check_net_spec(net_spec)
    in_shape, layers, bidirectional, init_fn = ps.at(net_spec, *['in_shape', 'layers', 'bidirectional', 'init_fn'])

    assert len(ps.uniq(layers)) == 1, f'layers must specify the same number of hidden units for each layer, but got {layers}'
    hidden_size = layers[0]
    recurrent_model = getattr(nn, get_nn_name(net_spec['type']))(
        input_size=in_shape, hidden_size=hidden_size, num_layers=len(layers),
        batch_first=True, bidirectional=bidirectional)
    # y.shape = (batch, seq_len, num_directions * hidden_size)
    # h_n.shape = (num_layers * num_directions, batch, hidden_size)
    # NOTE seq_len, which is flexible, is not considered part of in_shape and out_shape. h_n shape is not useful - we'd have direct access to it
    num_dir = 2 if recurrent_model.bidirectional else 1
    net_spec['out_shape'] = num_dir * hidden_size

    if init_fn:  # initialize weights if specified
        init_weights = get_init_weights(init_fn)
        mlp_model.apply(init_weights)
    return recurrent_model


net_spec = {
    "type": "rnn",  # options: 'rnn', 'lstm', 'gru'
    "in_shape": 3,  # the number of features in x
    "layers": [64, 64],  # the hidden layers, must be the same for all layers
    "bidirectional": False,  # whether to make network bidirectional
    "init_fn": "orthogonal_",  # weight initialization
}
recurrent_model = build_recurrent_model(net_spec)
num_dir = 2 if recurrent_model.bidirectional else 1
hidden_size = net_spec['layers'][0]
seq_len = 10
assert net_spec['out_shape'] == num_dir * hidden_size
x = torch.rand([seq_len, net_spec['in_shape']]).unsqueeze(dim=0)
y, h_n = recurrent_model(x)
torch.equal(torch.tensor(y.shape), torch.tensor([1, seq_len, hidden_size]))
assert torch.is_tensor(h_n)

net_spec = {
    "type": "rnn",  # options: 'rnn', 'lstm', 'gru'
    "in_shape": 3,  # the number of features in x
    "layers": [64, 64],  # the hidden layers, must be the same for all layers
    "bidirectional": True,  # whether to make network bidirectional
    "init_fn": "orthogonal_",  # weight initialization
}
recurrent_model = build_recurrent_model(net_spec)
num_dir = 2 if recurrent_model.bidirectional else 1
hidden_size = net_spec['layers'][0]
seq_len = 10
assert net_spec['out_shape'] == num_dir * hidden_size
x = torch.rand([seq_len, net_spec['in_shape']]).unsqueeze(dim=0)
y, h_n = recurrent_model(x)
torch.equal(torch.tensor(y.shape), torch.tensor([1, seq_len, hidden_size]))
assert torch.is_tensor(h_n)

net_spec = {
    "type": "gru",  # options: 'rnn', 'lstm', 'gru'
    "in_shape": 3,  # the number of features in x
    "layers": [64, 64],  # the hidden layers, must be the same for all layers
    "bidirectional": False,  # whether to make network bidirectional
    "init_fn": "orthogonal_",  # weight initialization
}
recurrent_model = build_recurrent_model(net_spec)
num_dir = 2 if recurrent_model.bidirectional else 1
hidden_size = net_spec['layers'][0]
seq_len = 10
assert net_spec['out_shape'] == num_dir * hidden_size
x = torch.rand([seq_len, net_spec['in_shape']]).unsqueeze(dim=0)
y, h_n = recurrent_model(x)
torch.equal(torch.tensor(y.shape), torch.tensor([1, seq_len, hidden_size]))
assert torch.is_tensor(h_n)

net_spec = {
    "type": "rnn",  # options: 'rnn', 'lstm', 'gru'
    "in_shape": 3,  # the number of features in x
    "layers": [64, 64],  # the hidden layers, must be the same for all layers
    "bidirectional": False,  # whether to make network bidirectional
    "init_fn": "orthogonal_",  # weight initialization
}
recurrent_model = build_recurrent_model(net_spec)
num_dir = 2 if recurrent_model.bidirectional else 1
hidden_size = net_spec['layers'][0]
seq_len = 10
assert net_spec['out_shape'] == num_dir * hidden_size
x = torch.rand([seq_len, net_spec['in_shape']]).unsqueeze(dim=0)
y, (h_n, c_n) = recurrent_model(x)
torch.equal(torch.tensor(y.shape), torch.tensor([1, seq_len, hidden_size]))
assert torch.is_tensor(h_n)
assert torch.is_tensor(c_n)


class FiLM(nn.Module):
    '''
    Feature-wise Linear Modulation layer https://distill.pub/2018/feature-wise-transformations/
    Takes a feature tensor and affine-transforms it with a conditioner tensor:
    output = cond_scale * feature + cond_shift
    The conditioner is always a vector with length = number of features or channels (image), and the operation is element-wise on feature or channel-wide (image)
    '''

    def __init__(self, num_feat, num_cond):
        # conditioner params with output shape matching num_feat, and
        # num_feat = feat.shape[1]
        # num_cond = cond.shape[1]
        nn.Module.__init__(self)
        self.cond_scale = nn.Linear(num_cond, num_feat)
        self.cond_shift = nn.Linear(num_cond, num_feat)

    def forward(self, feat, cond):
        cond_scale_x = self.cond_scale(cond)
        cond_shift_x = self.cond_shift(cond)
        # use view to ensure cond transform will broadcast consistently across entire feature/channel
        view_shape = list(cond_scale_x.shape) + [1] * (feat.dim() - cond.dim())
        x = cond_scale_x.view(*view_shape) * feat + cond_shift_x.view(*view_shape)
        assert x.shape == feat.shape
        return x


# an example vector and a 5D conditioning vector
feat = torch.rand(10).unsqueeze(dim=0)
cond = torch.rand(5).unsqueeze(dim=0)
num_feat = feat.shape[1]
num_cond = cond.shape[1]
film = FiLM(num_feat, num_cond)
x = film(feat, cond)
assert x.shape == feat.shape

# an example image and a 5D conditioning vector
feat = torch.rand(3, 10, 10).unsqueeze(dim=0)
cond = torch.rand(5).unsqueeze(dim=0)
num_feat = feat.shape[1]
num_cond = cond.shape[1]
film = FiLM(num_feat, num_cond)
x = film(feat, cond)
assert x.shape == feat.shape


class Concat(nn.Module):
    '''Flatten all input tensors and concatenate them'''

    def forward(self, *tensors):
        return torch.cat([t.flatten(start_dim=1) for t in tensors], dim=-1)


# any tensors: vectors
t1 = torch.rand(10).unsqueeze(dim=0)
t2 = torch.rand(5).unsqueeze(dim=0)
t3 = torch.rand(2).unsqueeze(dim=0)
concat = Concat()
x = concat(t1, t2, t3)
assert x.dim() == 2
assert x.shape[1] == t1.shape[1] + t2.shape[1] + t3.shape[1]

# any tensors: an image and a vector
t1 = torch.rand(3, 10, 10).unsqueeze(dim=0)
t2 = torch.rand(5).unsqueeze(dim=0)
concat = Concat()
x = concat(t1, t2)
assert x.dim() == 2
assert x.shape[1] == 305
