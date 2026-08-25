"""GAT-based Actor-Critic for PPO.

Actor scores each candidate action against the state (action set size
varies per molecule); critic shares the same GAT backbone for the value
baseline. Not compatible with the legacy offline trainer's forward() API.
"""
import torch
import torch.nn as nn

from reactionrl.models.gat_encoder import GATMoleculeEncoder


def _mlp(in_dim, hidden_size, out_dim, num_hidden):
    layers = []
    dim = in_dim
    for _ in range(num_hidden):
        layers += [nn.Linear(dim, hidden_size), nn.ReLU()]
        dim = hidden_size
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class GATActorCritic(nn.Module):
    """Shared-backbone actor-critic used by PPOTrainer."""
    def __init__(self, hidden_size=256, gat_num_layers=3, gat_num_heads=4,
                 actor_num_hidden=3, critic_num_hidden=2, **kwargs):
        super().__init__()
        self.encoder = GATMoleculeEncoder(hidden_size=hidden_size, num_layers=gat_num_layers, num_heads=gat_num_heads)
        embed_dim = self.encoder.embed_dim
        # actor input = concat(state_embedding, action_embedding=[rsig_emb; psig_emb])
        self.actor = _mlp(embed_dim + 2 * embed_dim, hidden_size, 1, actor_num_hidden)
        self.critic = _mlp(embed_dim, hidden_size, 1, critic_num_hidden)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def GIN(self):
        """Alias for the shared backbone; kept for code expecting `.GIN`."""
        return self.encoder.gat

    def encode_states(self, smiles_batch):
        """smiles_batch -> (batch, embed_dim)."""
        return self.encoder(smiles_batch)

    def encode_actions(self, rsig_smiles, psig_smiles):
        """rsig/psig SMILES -> (batch, 2*embed_dim) action embeddings."""
        rsig_emb = self.encoder(rsig_smiles)
        psig_emb = self.encoder(psig_smiles)
        return torch.cat([rsig_emb, psig_emb], dim=-1)

    def action_logits(self, state_emb, action_embs):
        """Score every candidate action against one state.

        Returns (num_actions,) logits for Categorical(logits=...).
        """
        state_rep = state_emb.unsqueeze(0).expand(action_embs.shape[0], -1)
        return self.actor(torch.cat([state_rep, action_embs], dim=-1)).squeeze(-1)

    def value(self, state_emb):
        """(embed_dim,) or (batch, embed_dim) -> value estimate."""
        squeeze_batch = state_emb.dim() == 1
        if squeeze_batch:
            state_emb = state_emb.unsqueeze(0)
        v = self.critic(state_emb).squeeze(-1)
        return v.squeeze(0) if squeeze_batch else v
