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
