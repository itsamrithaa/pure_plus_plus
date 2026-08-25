"""Model registry.

Was previously absent (gitignored, never committed). Now provides the
GAT-based actor-critic used by PPOTrainer.
"""
from reactionrl.models.gat_encoder import GATMoleculeEncoder
from reactionrl.models.actor_critic import GATActorCritic

MODEL_REGISTRY = {
    "actor-critic": GATActorCritic,
    "gat-actor-critic": GATActorCritic,
}

__all__ = ["GATMoleculeEncoder", "GATActorCritic", "MODEL_REGISTRY"]
