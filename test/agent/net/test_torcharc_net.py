import pytest
import torch

try:
    import torcharc

    HAS_TORCHARC = True
except ImportError:
    HAS_TORCHARC = False


@pytest.mark.skipif(not HAS_TORCHARC, reason="torcharc not installed")
class TestTorchArcNet:
    def test_build_mlp(self):
        from slm_lab.agent.net.torcharc_net import TorchArcNet

        net_spec = {
            "type": "TorchArcNet",
            "arc": {
                "modules": {
                    "body": {
                        "Sequential": [
                            {"LazyLinear": {"out_features": 64}},
                            {"ReLU": {}},
                            {"LazyLinear": {"out_features": 64}},
                            {"ReLU": {}},
                        ]
                    }
                },
                "graph": {
                    "input": "x",
                    "modules": {"body": ["x"]},
                    "output": "body",
                },
            },
            "loss_spec": {"name": "MSELoss"},
            "optim_spec": {"name": "Adam", "lr": 0.001},
            "gpu": False,
        }

        # Test with discrete action space (single int out_dim)
        net = TorchArcNet(net_spec, in_dim=4, out_dim=2)
        x = torch.randn(8, 4)
        out = net(x)
        assert out.shape == (8, 2)

    def test_build_multi_output(self):
        from slm_lab.agent.net.torcharc_net import TorchArcNet

        net_spec = {
            "type": "TorchArcNet",
            "arc": {
                "modules": {
                    "body": {
                        "Sequential": [
                            {"LazyLinear": {"out_features": 64}},
                            {"Tanh": {}},
                        ]
                    }
                },
                "graph": {
                    "input": "x",
                    "modules": {"body": ["x"]},
                    "output": "body",
                },
            },
            "loss_spec": {"name": "MSELoss"},
            "optim_spec": {"name": "Adam", "lr": 0.001},
            "gpu": False,
        }

        # Test with actor-critic output (list out_dim)
        net = TorchArcNet(net_spec, in_dim=4, out_dim=[2, 1])
        x = torch.randn(8, 4)
        out = net(x)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].shape == (8, 2)
        assert out[1].shape == (8, 1)

    def test_with_optim(self):
        from slm_lab.agent.net.torcharc_net import TorchArcNet
        from slm_lab.agent.net import net_util

        net_spec = {
            "type": "TorchArcNet",
            "arc": {
                "modules": {
                    "body": {
                        "Sequential": [
                            {"LazyLinear": {"out_features": 32}},
                            {"ReLU": {}},
                        ]
                    }
                },
                "graph": {
                    "input": "x",
                    "modules": {"body": ["x"]},
                    "output": "body",
                },
            },
            "loss_spec": {"name": "MSELoss"},
            "optim_spec": {"name": "Adam", "lr": 0.001},
            "gpu": False,
        }
        net = TorchArcNet(net_spec, in_dim=4, out_dim=2)
        optim = net_util.get_optim(net, net.optim_spec)
        assert optim is not None

        # Test backward pass
        x = torch.randn(8, 4)
        out = net(x)
        loss = out.sum()
        loss.backward()
        optim.step()

    def test_log_std(self):
        from slm_lab.agent.net.torcharc_net import TorchArcNet

        net_spec = {
            "type": "TorchArcNet",
            "arc": {
                "modules": {
                    "body": {
                        "Sequential": [
                            {"LazyLinear": {"out_features": 64}},
                            {"Tanh": {}},
                        ]
                    }
                },
                "graph": {
                    "input": "x",
                    "modules": {"body": ["x"]},
                    "output": "body",
                },
            },
            "loss_spec": {"name": "MSELoss"},
            "optim_spec": {"name": "Adam", "lr": 0.001},
            "gpu": False,
            "log_std_init": 0.0,
        }

        # Test with state-independent log_std (continuous actions)
        net = TorchArcNet(net_spec, in_dim=4, out_dim=[3, 3])
        x = torch.randn(8, 4)
        out = net(x)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].shape == (8, 3)  # mean
        assert out[1].shape == (8, 3)  # log_std (expanded)

    def test_conv_with_init_fn(self):
        from slm_lab.agent.net.torcharc_net import TorchArcNet

        conv_arc = {
            "modules": {
                "body": {
                    "Sequential": [
                        {
                            "LazyConv2d": {
                                "out_channels": 32,
                                "kernel_size": 8,
                                "stride": 4,
                            }
                        },
                        {"ReLU": {}},
                        {
                            "LazyConv2d": {
                                "out_channels": 64,
                                "kernel_size": 4,
                                "stride": 2,
                            }
                        },
                        {"ReLU": {}},
                        {
                            "LazyConv2d": {
                                "out_channels": 64,
                                "kernel_size": 3,
                                "stride": 1,
                            }
                        },
                        {"ReLU": {}},
                        {"Flatten": {}},
                        {"LazyLinear": {"out_features": 512}},
                        {"ReLU": {}},
                    ]
                }
            },
            "graph": {
                "input": "x",
                "modules": {"body": ["x"]},
                "output": "body",
            },
        }
        base_spec = {
            "type": "TorchArcNet",
            "arc": conv_arc,
            "loss_spec": {"name": "MSELoss"},
            "optim_spec": {"name": "Adam", "lr": 0.001},
            "gpu": False,
            "hid_layers_activation": "relu",
        }

        # Build without init_fn (default init)
        net_default = TorchArcNet(base_spec, in_dim=(4, 84, 84), out_dim=[6, 1])
        # Grab a copy of a conv weight under default init
        default_weight = None
        for m in net_default.modules():
            if isinstance(m, torch.nn.Conv2d):
                default_weight = m.weight.data.clone()
                break

        # Build with orthogonal init
        ortho_spec = {**base_spec, "init_fn": "orthogonal_"}
        net_ortho = TorchArcNet(ortho_spec, in_dim=(4, 84, 84), out_dim=[6, 1])
        ortho_weight = None
        for m in net_ortho.modules():
            if isinstance(m, torch.nn.Conv2d):
                ortho_weight = m.weight.data.clone()
                break

        assert default_weight is not None and ortho_weight is not None
        # Orthogonal init should produce different weights than default
        assert not torch.equal(default_weight, ortho_weight)

        # Verify forward pass shape
        x = torch.randn(4, 4, 84, 84)
        out = net_ortho(x)
        assert isinstance(out, list)
        assert out[0].shape == (4, 6)
        assert out[1].shape == (4, 1)

    def test_shared_param(self):
        from slm_lab.agent.net.torcharc_net import TorchArcNet

        net_spec = {
            "type": "TorchArcNet",
            "arc": {
                "modules": {
                    "body": {
                        "Sequential": [
                            {"LazyLinear": {"out_features": 64}},
                            {"ReLU": {}},
                        ]
                    }
                },
                "graph": {
                    "input": "x",
                    "modules": {"body": ["x"]},
                    "output": "body",
                },
            },
            "loss_spec": {"name": "MSELoss"},
            "optim_spec": {"name": "Adam", "lr": 0.001},
            "gpu": False,
        }

        # Default: shared should be False
        net_default = TorchArcNet(net_spec, in_dim=4, out_dim=2)
        assert net_default.shared is False

        # With shared=True
        shared_spec = {**net_spec, "shared": True}
        net_shared = TorchArcNet(shared_spec, in_dim=4, out_dim=2)
        assert net_shared.shared is True

    def test_hid_layers_activation(self):
        from slm_lab.agent.net.torcharc_net import TorchArcNet

        net_spec = {
            "type": "TorchArcNet",
            "arc": {
                "modules": {
                    "body": {
                        "Sequential": [
                            {"LazyLinear": {"out_features": 64}},
                            {"ReLU": {}},
                        ]
                    }
                },
                "graph": {
                    "input": "x",
                    "modules": {"body": ["x"]},
                    "output": "body",
                },
            },
            "loss_spec": {"name": "MSELoss"},
            "optim_spec": {"name": "Adam", "lr": 0.001},
            "gpu": False,
        }

        # Default: hid_layers_activation should be "relu"
        net_default = TorchArcNet(net_spec, in_dim=4, out_dim=2)
        assert net_default.hid_layers_activation == "relu"

        # With tanh
        tanh_spec = {**net_spec, "hid_layers_activation": "tanh"}
        net_tanh = TorchArcNet(tanh_spec, in_dim=4, out_dim=2)
        assert net_tanh.hid_layers_activation == "tanh"
