from slm_lab.agent.net import builder
import pytest
import torch


@pytest.mark.parametrize('net_spec,layer_names,out_shape', [
    (
        {  # basic
            "type": "mlp",
            "in_shape": [4],
            "layers": [64, 64],
            "activation": "relu",
            "init_fn": "orthogonal_",
        },
        ['Linear', 'ReLU', 'Linear', 'ReLU'],
        [64]
    ), (
        {  # out_activation
            "type": "mlp",
            "in_shape": [4],
            "layers": [64, 64],
            "activation": "relu",
            "out_activation": None,
            "init_fn": "orthogonal_",
        },
        ['Linear', 'ReLU', 'Linear'],
        [64]
    ), (
        {  # out_shape, out_activation, batch_norm
            "type": "mlp",
            "in_shape": [4],
            "out_shape": [2],
            "layers": [64, 64],
            "batch_norm": True,
            "activation": "relu",
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Linear', 'BatchNorm1d', 'ReLU', 'Linear', 'BatchNorm1d', 'ReLU', 'Linear', 'Sigmoid'],
        [2]
    ), (
        {  # no layers
            "type": "mlp",
            "in_shape": [4],
            "layers": [],
            "init_fn": "orthogonal_",
        },
        ['Identity'],
        [4]
    ), (
        {  # no layers but has out_shape, out_activation
            "type": "mlp",
            "in_shape": [4],
            "out_shape": [2],
            "layers": [],
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Linear', 'Sigmoid'],
        [2]
    ),
])
def test_build_mlp_model(net_spec, layer_names, out_shape):
    mlp_model = builder.build_mlp_model(net_spec)
    for nn_layer, layer_name in zip(mlp_model, layer_names):
        assert nn_layer._get_name() == layer_name
    # second call for idempotent test
    rebuild_mlp_model = builder.build_mlp_model(net_spec)
    for m1, m2 in zip(mlp_model, rebuild_mlp_model):
        assert m1._get_name() == m2._get_name()
    batch = 8
    x = torch.rand([batch, *net_spec['in_shape']])
    y = mlp_model(x)
    assert list(y.shape) == [batch, *net_spec['_out_shape']]
    assert net_spec['_out_shape'] == out_shape


@pytest.mark.parametrize('net_spec,layer_names,y_shape,out_shape', [
    (
        {  # basic unflattened
            "type": "conv1d",
            "in_shape": [3, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "activation": "relu",
            "init_fn": "orthogonal_",
        },
        ['Conv1d', 'ReLU', 'Conv1d', 'ReLU'],
        [8, 16, 6], [16, 6]
    ), (
        {  # basic flattened
            "type": "conv1d",
            "in_shape": [3, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "activation": "relu",
            "init_fn": "orthogonal_",
        },
        ['Conv1d', 'ReLU', 'Conv1d', 'ReLU', 'Flatten'],
        [8, 96], 96
    ), (
        {  # batch_norm and out_activation
            "type": "conv1d",
            "in_shape": [3, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "batch_norm": True,
            "activation": "relu",
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Conv1d', 'BatchNorm1d', 'ReLU', 'Conv1d', 'BatchNorm1d', 'Sigmoid', 'Flatten'],
        [8, 96], 96
    ), (
        {  # out_shape and flattened
            "type": "conv1d",
            "in_shape": [3, 20],
            "out_shape": 2,
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "batch_norm": True,
            "activation": "relu",
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Conv1d', 'BatchNorm1d', 'ReLU', 'Conv1d', 'BatchNorm1d', 'ReLU', 'Flatten', 'Linear', 'Sigmoid'],
        [8, 2], 2
    ),
])
def test_build_conv_model_1d(net_spec, layer_names, y_shape, out_shape):
    conv_model = builder.build_conv_model(net_spec)
    for nn_layer, layer_name in zip(conv_model, layer_names):
        assert nn_layer._get_name() == layer_name
    # second call for idempotent test
    rebuild_conv_model = builder.build_conv_model(net_spec)
    for m1, m2 in zip(conv_model, rebuild_conv_model):
        assert m1._get_name() == m2._get_name()
    batch = 8
    x = torch.rand([batch, *net_spec['in_shape']])
    y = conv_model(x)
    assert list(y.shape) == y_shape
    assert net_spec['_out_shape'] == out_shape


@pytest.mark.parametrize('net_spec,layer_names,y_shape,out_shape', [
    (
        {  # basic unflattened
            "type": "conv2d",
            "in_shape": [3, 20, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "activation": "relu",
            "init_fn": "orthogonal_",
        },
        ['Conv2d', 'ReLU', 'Conv2d', 'ReLU'],
        [8, 16, 6, 6], [16, 6, 6]
    ), (
        {  # basic flattened
            "type": "conv2d",
            "in_shape": [3, 20, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "activation": "relu",
            "init_fn": "orthogonal_",
        },
        ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'Flatten'],
        [8, 576], 576
    ), (
        {  # batch_norm and out_activation
            "type": "conv2d",
            "in_shape": [3, 20, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "batch_norm": True,
            "activation": "relu",
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Sigmoid', 'Flatten'],
        [8, 576], 576
    ), (
        {  # out_shape and flattened
            "type": "conv2d",
            "in_shape": [3, 20, 20],
            "out_shape": 2,
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "batch_norm": True,
            "activation": "relu",
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Flatten', 'Linear', 'Sigmoid'],
        [8, 2], 2
    ),
])
def test_build_conv_model_2d(net_spec, layer_names, y_shape, out_shape):
    conv_model = builder.build_conv_model(net_spec)
    for nn_layer, layer_name in zip(conv_model, layer_names):
        assert nn_layer._get_name() == layer_name
    # second call for idempotent test
    rebuild_conv_model = builder.build_conv_model(net_spec)
    for m1, m2 in zip(conv_model, rebuild_conv_model):
        assert m1._get_name() == m2._get_name()
    batch = 8
    x = torch.rand([batch, *net_spec['in_shape']])
    y = conv_model(x)
    assert list(y.shape) == y_shape
    assert net_spec['_out_shape'] == out_shape


@pytest.mark.parametrize('net_spec,layer_names,y_shape,out_shape', [
    (
        {  # basic unflattened
            "type": "conv3d",
            "in_shape": [3, 20, 20, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "activation": "relu",
            "init_fn": "orthogonal_",
        },
        ['Conv3d', 'ReLU', 'Conv3d', 'ReLU'],
        [8, 16, 6, 6, 6], [16, 6, 6, 6]
    ), (
        {  # basic flattened
            "type": "conv3d",
            "in_shape": [3, 20, 20, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "activation": "relu",
            "init_fn": "orthogonal_",
        },
        ['Conv3d', 'ReLU', 'Conv3d', 'ReLU', 'Flatten'],
        [8, 3456], 3456
    ), (
        {  # batch_norm and out_activation
            "type": "conv3d",
            "in_shape": [3, 20, 20, 20],
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "batch_norm": True,
            "activation": "relu",
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Conv3d', 'BatchNorm3d', 'ReLU', 'Conv3d', 'BatchNorm3d', 'Sigmoid', 'Flatten'],
        [8, 3456], 3456
    ), (
        {  # out_shape and flattened
            "type": "conv3d",
            "in_shape": [3, 20, 20, 20],
            "out_shape": 2,
            "layers": [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            "flatten_out": True,
            "batch_norm": True,
            "activation": "relu",
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        ['Conv3d', 'BatchNorm3d', 'ReLU', 'Conv3d', 'BatchNorm3d', 'ReLU', 'Flatten', 'Linear', 'Sigmoid'],
        [8, 2], 2
    ),
])
def test_build_conv_model_3d(net_spec, layer_names, y_shape, out_shape):
    conv_model = builder.build_conv_model(net_spec)
    for nn_layer, layer_name in zip(conv_model, layer_names):
        assert nn_layer._get_name() == layer_name
    # second call for idempotent test
    rebuild_conv_model = builder.build_conv_model(net_spec)
    for m1, m2 in zip(conv_model, rebuild_conv_model):
        assert m1._get_name() == m2._get_name()
    batch = 8
    x = torch.rand([batch, *net_spec['in_shape']])
    y = conv_model(x)
    assert list(y.shape) == y_shape
    assert net_spec['_out_shape'] == out_shape


# @pytest.mark.parametrize('cell_type', ['rnn', 'gru', 'lstm'])
@pytest.mark.parametrize('net_spec,y_shape,out_shape', [
    (
        {  # basic
            "type": "cell_type",
            "in_shape": 3,
            "layers": [64, 64],
            "bidirectional": False,
            "init_fn": "orthogonal_",
        },
        [8, 10, 1 * 64],  # [batch, seq_len, num_dir * hidden_size],
        1 * 64,  # of y without batch and seq_len, so = num_dir * hidden_size
    ), (
        {  # bidirectional
            "type": "cell_type",
            "in_shape": 3,
            "layers": [64, 64],
            "bidirectional": True,
            "init_fn": "orthogonal_",
        },
        [8, 10, 2 * 64],  # [batch, seq_len, num_dir * hidden_size],
        2 * 64,  # of y without batch and seq_len, so = num_dir * hidden_size
    ), (
        {  # out_shape and out_activation
            "type": "cell_type",
            "in_shape": 3,
            "out_shape": 2,
            "layers": [64, 64],
            "bidirectional": False,
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        [8, 2],  # out_shape specified
        2,  # of y.shape from mlp with out_shape from net_spec
    ), (
        {  # out_shape and out_activation and bidirectional
            "type": "cell_type",
            "in_shape": 3,
            "out_shape": 2,
            "layers": [64, 64],
            "bidirectional": True,
            "out_activation": "sigmoid",
            "init_fn": "orthogonal_",
        },
        [8, 2],  # out_shape specified
        2,  # of y.shape from mlp with out_shape from net_spec
    ),
])
@pytest.mark.parametrize('cell_type', ['rnn', 'gru', 'lstm'])
def test_build_recurrent_model(cell_type, net_spec, y_shape, out_shape):
    net_spec = net_spec.copy()
    net_spec['type'] = cell_type
    recurrent_model = builder.build_recurrent_model(net_spec)
    # second call for idempotent test
    rebuild_recurrent_model = builder.build_recurrent_model(net_spec)
    for m1, m2 in zip(recurrent_model.modules(), rebuild_recurrent_model.modules()):
        assert m1._get_name() == m2._get_name()
    num_dir = 2 if net_spec['bidirectional'] else 1
    hidden_size = net_spec['layers'][0]
    batch = 8
    seq_len = 10
    x = torch.rand([batch, seq_len, net_spec['in_shape']])
    y, h_out = recurrent_model(x)
    # test passing h_out as input
    next_y, next_h_out = recurrent_model(x, h_out)
    assert list(y.shape) == y_shape
    assert y.shape[-1] == out_shape
    assert net_spec['_out_shape'] == out_shape

    if 'out_activation' in net_spec:  # using specified out_shape and out_activation, has mlp and flattened
        assert recurrent_model.recurrent_model._get_name() == cell_type.upper()
        for nn_layer, layer_name in zip(recurrent_model.mlp_model, ['Linear', 'Sigmoid']):
            assert nn_layer._get_name() == layer_name


@pytest.mark.parametrize('feat,cond', [
    # an example vector and a 5D conditioning vector
    (torch.rand(8, 10), torch.rand(8, 5)),
    # an example image and a 5D conditioning vector
    (torch.rand(8, 3, 10, 10), torch.rand(8, 5)),
])
def test_film(feat, cond):
    num_feat = feat.shape[1]
    num_cond = cond.shape[1]
    film = builder.FiLM(num_feat, num_cond)
    x = film(feat, cond)
    assert x.shape == feat.shape


def test_concat_vectors():
    # any tensors: vectors
    t1 = torch.rand(10).unsqueeze(dim=0)
    t2 = torch.rand(5).unsqueeze(dim=0)
    t3 = torch.rand(2).unsqueeze(dim=0)
    concat = builder.Concat()
    x = concat(t1, t2, t3)
    assert x.dim() == 2
    assert x.shape[1] == t1.shape[1] + t2.shape[1] + t3.shape[1]


def test_concat_image_and_vector():
    # any tensors: an image and a vector
    t1 = torch.rand(3, 10, 10).unsqueeze(dim=0)
    t2 = torch.rand(5).unsqueeze(dim=0)
    concat = builder.Concat()
    x = concat(t1, t2)
    assert x.dim() == 2
    assert x.shape[1] == 305
