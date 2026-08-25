"""
Central configuration for the reactionrl package.

Directory paths are auto-detected from the package location, or can be
overridden via the MAIN_DIR environment variable. TrainingConfig holds
all hyperparameters needed for offline RL training.
"""
import os
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

# Directory paths - auto-detect from package root, or use MAIN_DIR env var
_PACKAGE_DIR = Path(__file__).resolve().parent
MAIN_DIR = Path(os.getenv("MAIN_DIR", str(_PACKAGE_DIR.parent)))
DATASETS_DIR = MAIN_DIR / "datasets"
PRETRAINED_MODELS_DIR = MAIN_DIR / "pretrained_models"
OUTPUT_DIR = Path("/fs/scratch/PAS2030/itsamrithaa/GAT_CHECKPOINTS")


@dataclass
class TrainingConfig:
    """Configuration for offline RL training (Updated for GAT).

    Controls hyperparameters, model architecture (GAT vs GIN), and training behavior.

    Attributes:
        backbone_type: One of "GAT" or "GIN".
        input_dim: Number of atom features (default 66 for TorchDrug).
        num_heads: Number of attention heads for GAT.
        gat_hidden_dims: List of dimensions for GAT layers.
        dropout: Dropout rate for GAT layers.
        
        actor_lr: Learning rate for actor optimizer.
        critic_lr: Learning rate for critic optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        topk: Number of top-k negative samples for PG loss.
        steps: Trajectory length (corresponds to dataset filename).
        model_type: One of "actor", "critic", or "actor-critic".
        actor_loss: Loss type for actor - "mse" or "PG" (policy gradient).
        negative_selection: Strategy for negative sampling.
        seed: Random seed for reproducibility.
        device: Torch device string (e.g. "cpu", "cuda:0").
        num_workers: Number of multiprocessing workers for data preparation.
        train_frac: Fraction of data used for training (rest is validation).
        gin_model_path: Path to pretrained GIN model (if backbone_type="GIN").
        hidden_size: Hidden layer size for actor/critic dense networks.
        actor_num_hidden: Number of hidden layers in actor network.
        critic_num_hidden: Number of hidden layers in critic network.
    """
    # --- PIVOT: Backbone Configuration ---
    backbone_type: str = "GAT"  # Change to "GIN" to use the original architecture
    input_dim: int = 22         # Standard TorchDrug atom feature dimension
    num_heads: int = 8          # Attention heads for GAT
    gat_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    dropout: float = 0.1        # Dropout rate for attention layers

    # Optimization
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    topk: int = 10

    # Data
    steps: int = 5
    train_frac: float = 0.8
    num_workers: int = min(cpu_count(), 35)

    # Model Architecture
    model_type: str = "actor-critic"
    actor_loss: str = "PG"
    negative_selection: str = "combined"
    gin_model_path: str = ""  
    hidden_size: int = 256     # Hidden size for the dense (MLP) heads
    actor_num_hidden: int = 3
    critic_num_hidden: int = 2

    # Runtime
    seed: int = 42
    device: str = "cpu"

    def get_backbone_path(self) -> str:
        """Returns the model path for GIN or GAT."""
        if self.backbone_type == "GIN":
            if self.gin_model_path:
                return self.gin_model_path
            return str(PRETRAINED_MODELS_DIR / "zinc2m_gin.pth")
        
        # For GAT, if you eventually pretrain it, return that path here
        return self.gin_model_path if self.gin_model_path else "none"

    def get_gin_model_path(self) -> str:
        """Alias to satisfy the Trainer class which expects this specific name."""
        return self.get_backbone_path()
