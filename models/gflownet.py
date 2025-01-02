import torch
import torch.nn as nn
import torch.distributions as dist

""" le modèle présenté est un effort pour former un gflownet continu, le mode de fonctionnement
    consiste à générer séquentiellement des actions et de fournir le chemin des actions au modèle.
    L'action est générée selon une distribution de Dirichlet. 
"""


class GFlowNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__(self)
        """
        Initialisation du GflowNet avec deux têtes et gestion des itérations.
        """

        #embedding pour le vecteur d'état
        self.state_encoder = nn.Linear(state_dim, 128)

        #embedding pour la matrice des actions (alphas et temps)
        self.action_matrix_encoder = nn.Linear(action_dim, 128)

        #fusion des embedding (state + actions)
        self.fusion_layer = nn.Linear(256, 128)

        #générateur des paramètres alpha pour Dirichlet
        self.alpha_generator = nn.Linear(128, action_dim)

    def forward(self, state, action_matrix):

        #encoder l'état
        state_embedding = torch.relu(self.state_encoder(state)) # (batch_size, 128)

        #encoder chaque ligne de la matrice d'actions
        batch_size, n_actions, alpha_dim = action_matrix.size()
        action_matrix_flat = action_matrix.view(-1, alpha_dim) # (batch_size * n_actions, alpha_dim)
        encoded_actions = torch.relu(self.action_encoder(action_matrix_flat)) # (batch_size * n_actions, 128)
        encoded_actions = encoded_actions.view(batch_size, n_actions, 128) # (batch_size, n_actions, 128)

        #combiner toutes les actions encodées par une moyenne
        action_embedding = torch.mean(encoded_actions, dim=1) # (batch_size, 128)

        #fusion des embeddings état + actions
        fusion = torch.cat([state_embedding, action_embedding], dim=-1) # (batch_size, 256)
        latent = torch.relu(self.fusion_layer(fusion)) # (batch_size, 128)

        #génération des paramètres alpha pour la distribution dirichlet avec Softplus
        alpha = torch.nn.functional.softplus(self.alpha_generator(latent)) + 1e-6 #éviter des zéros

        return alpha
    
    def sample_action(self, state, n_steps, action_dim):
        """
        Génère un chemin complet d'actions séquentiellement
        """
        batch_size = state.size(0)

        #initialiser une matrice vide pour stocker les actions générées
        action_matrix = torch.zeros(batch_size, 0, action_dim)

        #liste pour stocker les log prob
        log_probs_list = []

        for _ in range(n_steps):
            if action_matrix.size(1) == 0:
                #si c'est la première itération, on crée une matrice d'action initiale vide
                padded_action_matrix = torch.zeros(batch_size, 1, action_dim)
            else:
                #ajouter une dimension supplémentaire pour l'entrée
                padded_action_matrix = action_matrix

            #passer l'état et la matrice d'actions à travers le modèle pour obtenir les paramètres alpha
            alpha = self.forward(state, padded_action_matrix)

            #échantillonner une nouvelle action dans la distribution Dirichlet
            dirichlet_dist = dist.Dirichlet(alpha)
            new_action = dirichlet_dist.sample() # (batch_size, action_dim)

            #calculer les log-probabilités des actions échantillonnées
            log_probs = dirichlet_dist.log_prob(new_action) # (batch_size,)

            #ajouter la nouvelle action à la matrice des actions
            new_action = new_action.unsqueeze(1) # (batch_size, action_dim)
            action_matrix = torch.cat([action_matrix, new_action], dim=1) # (batch_size, n_steps, action_dim)

            #stocker les log prob
            log_probs_list.append(log_probs)

        return action_matrix, log_probs_list


