from yahoofinancials import YahooFinancials

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import pandas as pd
import os
import math
import datetime

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter



class FlowModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(FlowModel, self).__init__()

        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dim)

        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)

        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))

        flow_values = self.softplus(self.output_layer(x))
        return flow_values
    

# Exemple d'utilisation
state_dim = 10
action_dim = 4

model = FlowModel(state_dim, action_dim)

# Exemple d'état
state = torch.randn(1, state_dim)
action = torch.randn(1, action_dim)

# Génération du Flux
fluxes = model(state, action)

print(f"Flux pour chaque action: {fluxes}")