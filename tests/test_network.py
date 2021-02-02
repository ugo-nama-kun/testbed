import torch

from testbed.ppo.network import ValueNetwork, PolicyNetwork


def test_value_network():
    vf = ValueNetwork(
        dim_observation=3,
        hidden_size1=10,
        hidden_size2=10
    )

    assert (5, 1) == vf.forward(torch.zeros(5, 3)).size()


def test_policy_network():
    pol = PolicyNetwork(
        dim_observation=3,
        hidden_size1=10,
        hidden_size2=10,
        dim_action=2
    )

    assert (20, 2) == pol.forward(torch.zeros(20, 3)).size()
