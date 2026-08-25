"""Neural network architectures for offline RL action prediction.

Contains the Actor, Critic, and ActorCritic models that use a GAT instead of a shared GIN
backbone for molecular graph encoding, with dense heads for action prediction
(actor) and action scoring (critic).
"""
import torch
import torch.nn as nn
from torchdrug import models 

class NeuralNet(nn.Module):
    """Feed-forward network with batch normalization."""
    def __init__(self, input_size, output_size, num_hidden=1, hidden_size=50):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            self.hidden_layers.append(nn.ReLU())
        self.last_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.last_layer(out)
        return out


class ActorNetwork(nn.Module):
    """Actor network using GAT backbone."""
    def __init__(self, input_dim=66, hidden_dims=[256, 256, 256], num_heads=8, num_hidden=3, dense_hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.GAT = models.GAT(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_head=num_heads,
            concat_hidden=False
        )
        self.output_dim = hidden_dims[-1]
        
        self.DENSE = NeuralNet(self.output_dim * 2, self.output_dim * 2,
                               num_hidden=num_hidden, hidden_size=dense_hidden_size)

    @property
    def actor(self):
        return self.DENSE

    def forward(self, x1, x2, *args):
        out1 = self.GAT(x1, x1.node_feature.float())["graph_feature"]
        out2 = self.GAT(x2, x2.node_feature.float())["graph_feature"]

        out = torch.cat([out1, out2], dim=1)
        out = self.DENSE(out)
        return out


class CriticNetwork(nn.Module):
    """Critic network using GAT backbone."""
    def __init__(self, input_dim=66, hidden_dims=[256, 256, 256], num_heads=8, num_hidden=2, dense_hidden_size=256):
        super(CriticNetwork, self).__init__()
        self.GAT = models.GAT(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_head=num_heads,
            concat_hidden=False
        )
        self.output_dim = hidden_dims[-1]
        
        self.DENSE = NeuralNet(self.output_dim * 4, 1,
                               num_hidden=num_hidden, hidden_size=dense_hidden_size)

    def forward(self, x1, x2, x3, x4, *args):
        out1 = self.GAT(x1, x1.node_feature.float())["graph_feature"]
        out2 = self.GAT(x2, x2.node_feature.float())["graph_feature"]
        out3 = self.GAT(x3, x3.node_feature.float())["graph_feature"]
        out4 = self.GAT(x4, x4.node_feature.float())["graph_feature"]

        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.DENSE(out)
        return out


class ActorCritic(nn.Module):
    """Combined actor-critic with shared GAT backbone."""
    def __init__(self, input_dim=66, hidden_dims=[256, 256, 256], num_heads=8, 
                 actor_num_hidden=3, critic_num_hidden=2, dense_hidden_size=256):
        super(ActorCritic, self).__init__()
        
        # Shared GAT Encoder
        self.GAT = models.GAT(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_head=num_heads,
            concat_hidden=False
        )
        self.output_dim = hidden_dims[-1]

        self.actor = NeuralNet(self.output_dim * 2, self.output_dim * 2,
                               num_hidden=actor_num_hidden, hidden_size=dense_hidden_size)
        self.critic = NeuralNet(self.output_dim * 4, 1,
                                num_hidden=critic_num_hidden, hidden_size=dense_hidden_size)

    def forward(self, reac, prod, rsig, psig, out_type="both"):
        reac_out = self.GAT(reac, reac.node_feature.float())["graph_feature"]
        prod_out = self.GAT(prod, prod.node_feature.float())["graph_feature"]

        output = []
        if out_type in ["both", "actor"]:
            output.append(self.actor(torch.cat([reac_out, prod_out], dim=1)))

        if out_type in ["both", "critic"]:
            psig_out = self.GAT(psig, psig.node_feature.float())["graph_feature"]
            rsig_out = self.GAT(rsig, rsig.node_feature.float())["graph_feature"]
            output.append(self.critic(torch.cat([reac_out, prod_out, rsig_out, psig_out], dim=1)))

        if len(output) == 1:
            return output[0]
        return output