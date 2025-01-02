import torch
import torch.nn as nn
import torch.distributions as dist

#importation du modèle
from models.gflownet import GFlowNet

#initialisation du modèle
state_dim = 3
action_dim = 3
model = GFlowNet(state_dim=state_dim, action_dim=action_dim)

#générer un batch d'état aléatoires
batch_size = 5
states = torch.rand(batch_size, state_dim)
