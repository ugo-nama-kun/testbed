import torch
import torch.nn.functional as F


class ValueNetwork(torch.nn.Module):
    def __init__(self,
                 dim_observation,
                 hidden_size1,
                 hidden_size2):
        super(ValueNetwork, self).__init__()

        self._fc1 = torch.nn.Linear(in_features=dim_observation,
                                    out_features=hidden_size1)
        self._fc2 = torch.nn.Linear(in_features=hidden_size1,
                                    out_features=hidden_size2)
        self._fc3 = torch.nn.Linear(in_features=hidden_size2,
                                    out_features=1)

    def forward(self, observation):
        x = F.leaky_relu(self._fc1(observation))
        x = F.leaky_relu(self._fc2(x))
        v = self._fc3(x)
        return v
