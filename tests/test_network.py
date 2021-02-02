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
    mean, var = pol.forward(torch.zeros(1, 3))
    assert (1, 2) == mean.size()
    assert (1, 2) == var.size()


def test_sample_from_policy():
    pol = PolicyNetwork(
        dim_observation=3,
        hidden_size1=10,
        hidden_size2=10,
        dim_action=2
    )
    action = pol.sample(torch.zeros(3))
    print(f"sample: {action}")
    assert action.size()[0] == 2