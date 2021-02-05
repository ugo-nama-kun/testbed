import torch
import torch.nn.functional as F
from torch import Tensor


class ValueNetwork(torch.nn.Module):
    def __init__(self,
                 dim_observation: int,
                 hidden_size1: int,
                 hidden_size2: int):
        super(ValueNetwork, self).__init__()
        self._dim_observation = dim_observation

        self._fc1 = torch.nn.Linear(in_features=dim_observation,
                                    out_features=hidden_size1)
        self._fc2 = torch.nn.Linear(in_features=hidden_size1,
                                    out_features=hidden_size2)
        self._fc3 = torch.nn.Linear(in_features=hidden_size2,
                                    out_features=1)

    @property
    def dim_observation(self) -> int:
        return self._dim_observation

    def forward(self, observation: Tensor) -> Tensor:
        # TODO: use shared feature extraction with policy
        x = F.leaky_relu(self._fc1(observation))
        x = F.leaky_relu(self._fc2(x))
        v = self._fc3(x)
        return v


class PolicyNetwork(torch.nn.Module):
    def __init__(self,
                 dim_observation: int,
                 hidden_size1: int,
                 hidden_size2: int,
                 dim_action: int,
                 is_constant_var=False):
        super(PolicyNetwork, self).__init__()
        self._dim_observation = dim_observation
        self._dim_action = dim_action

        self._fc1 = torch.nn.Linear(in_features=dim_observation,
                                    out_features=hidden_size1)
        self._fc2 = torch.nn.Linear(in_features=hidden_size1,
                                    out_features=hidden_size2)
        self._fc3_mean = torch.nn.Linear(in_features=hidden_size2,
                                         out_features=self._dim_action)

        self._is_constant_var = is_constant_var
        if not is_constant_var:
            self._log_var = torch.nn.Linear(in_features=1,  # use constant input 1
                                            out_features=self._dim_action,
                                            bias=False)
            self._log_var.weight.data.fill_(1.0)

    @property
    def dim_observation(self) -> int:
        return self._dim_observation

    @property
    def dim_action(self) -> int:
        return self._dim_action

    def forward(self, observation: Tensor) -> (Tensor, Tensor):
        # TODO: use shared feature extraction with VF
        x = F.leaky_relu(self._fc1(observation))
        x = F.leaky_relu(self._fc2(x))
        mean_action = torch.tanh(self._fc3_mean(x))
        if self._is_constant_var:
            var_action = 0.5 * torch.ones(1, self._dim_action)   # constant variance
        else:
            var_action = torch.exp(self._log_var(torch.ones(observation.size()[0], 1)))
        return mean_action, var_action

    def log_prob(self, observation: Tensor, action: Tensor) -> Tensor:
        """
        Gaussian probability. Single observation is supported
        # TODO: support multiple inputs
        :param observation:
        :param action:
        :return:
        """
        mean, var = self.forward(observation)
        p = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.diag(var[0]))
        return p.log_prob(action)

    def sample(self, observation: Tensor) -> Tensor:
        """
        Sample from Gaussian policy. Single observation is supported
        # TODO: support multiple inputs
        :param observation:
        :return:
        """
        mean, var = self.forward(observation)
        p = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.diag(var[0]))
        action = p.sample()
        return action[0].detach()
