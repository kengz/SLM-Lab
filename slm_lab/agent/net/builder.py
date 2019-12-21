# Module to quickly build neural networks with relatively simple architecture
import torch.nn as nn
import pydash as ps

# general net spec, can be composed from programs
net_spec = {
    "#heads": {
        "image": {
            "type": "conv",
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
            "init_fn": "orthogonal_",
        },
    },
    "#body": {  # this is special since it contains a models op
        "bottleneck": {
            "input_op": "concat",
            "type": "mlp",
            "layers": [],
            "activation": "relu",
            "init_fn": "orthogonal_",
        }
    },
    "#tails": {
        "v": {
            "type": "mlp",
            "layers": [],
            "activation": "relu",
        },
        "pi": {
            "type": "mlp",
            "layers": [],
            "activation": "relu",
        }
    }
}


def get_nn_name(uncased_name):
    '''Helper to get the proper name in PyTorch nn given a case-insensitive name'''
    for nn_name in nn.__dict__:
        if uncased_name.lower() == nn_name.lower():
            return nn_name
    raise ValueError(f'Name {uncased_name} not found in {nn.__dict__}')


def get_conv_out_shape(conv_model, in_shape):
    '''Helper to calculate the output shape of a conv model with flattened last layer given an input shape'''
    x = torch.rand(in_shape).unsqueeze(dim=0)
    y = conv_model(x)
    return y.shape[1]


def build_activation_layer(activation):
    '''Helper to build an activation function layer for net'''
    ActivationClass = getattr(nn, get_nn_name(activation))
    return ActivationClass()


def check_net_spec(net_spec):
    # CHECK FOR KEYS
    pass

# TODO do init here too

def build_mlp_model(net_spec):
    '''
    Build an MLP model given net_spec
    @param dict:net_spec With the following format/example:
    {
        "type": "mlp",
        "layers": [64, 64],
        "activation": "relu",
        "init_fn": "orthogonal_",
    }
    '''
    check_net_spec(net_spec)
    in_shape, activation = ps.at(net_spec, *['in_shape', 'activation'])
    layers = []
    for out_shape in net_spec['layers']:
        layers.append(nn.Linear(in_shape, out_shape))
        if activation is not None:
            layers.append(build_activation_layer(activation))
        in_shape = out_shape  # update
    net_spec['out_shape'] = out_shape
    mlp_model = nn.Sequential(*layers)
    return mlp_model


gyro_net_spec = net_spec['#heads']['gyro']
# programmatic insertion
gyro_net_spec['in_shape'] = 10
build_mlp_model(gyro_net_spec)


def build_conv_model(net_spec):
    '''
    Build a Conv2d (vision) model given net_spec
    Conv2D params: [in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1]
    Layer contains arguments to Conv2D except the in_channels, which is inferred. Feel free to provide layer with as many params, since they are applied with *layer
    '''
    check_net_spec(net_spec)
    in_shape, batch_norm, activation = ps.at(net_spec, *['in_shape', 'batch_norm', 'activation'])
    in_c = in_shape[0]  # PyTorch image input shape is (c,h,w)
    layers = []
    for layer in net_spec['layers']:
        out_c = layer[0]
        layers.append(nn.Conv2d(in_c, *layer))
        if batch_norm is not None:
            layers.append(nn.BatchNorm2d(out_c))
        if activation is not None:
            layers.append(build_activation_layer(activation))
        in_c = out_c  # update
    layers.append(nn.Flatten())  # add flatten layer automatically
    conv_model = nn.Sequential(*layers)
    net_spec['out_shape'] = get_conv_out_shape(conv_model, in_shape)
    return conv_model


image_net_spec = net_spec['#heads']['image']
image_net_spec['in_shape'] = [3, 84, 84]
conv_model = build_conv_model(image_net_spec)
conv_model
in_shape = [3, 84, 84]


def build_rnn_model(net_spec):
    pass
