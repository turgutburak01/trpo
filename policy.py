import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn

torch.set_default_tensor_type('torch.DoubleTensor')
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Policy, self).__init__()
        self.inputLayer = nn.Linear(num_inputs, hidden_size)
        self.hiddenLayer = nn.Linear(hidden_size, hidden_size)
        self.hiddenLayer2 = nn.Linear(hidden_size, hidden_size)
        self.outputLayer = nn.Linear(hidden_size, num_outputs)
        self.logStd = nn.Parameter(torch.zeros(1, num_outputs))
        
    def forward(self, x):
        """
        Parameters:
        states (torch.Tensor): n_state x n_sample
        
        Returns:
        torch.Tensor: n_action x n_sample | mean of the action
        torch.Tensor: n_action x n_sample | log(std) of action
        torch.Tensor: n_action x n_sample | std of action
        """
        x = torch.tanh(self.inputLayer(x))
        x = torch.tanh(self.hiddenLayer(x))
        x = torch.tanh(self.hiddenLayer2(x))
        action_mean = self.outputLayer(x)
        action_logStd = self.logStd.expand_as(action_mean)
        action_std = torch.exp(self.logStd)
        
        return action_mean, action_logStd, action_std