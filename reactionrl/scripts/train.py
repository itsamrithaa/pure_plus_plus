"""CLI entry point for offline RL training. (Updated for GAT pivot)"""
import argparse
import numpy as np
import torch

from reactionrl.config import TrainingConfig, DATASETS_DIR
from reactionrl.models import MODEL_REGISTRY
from reactionrl.data.dataset import OfflineRLDataset
from reactionrl.training.trainer import OfflineRLTrainer

def get_args():
    parser = argparse.ArgumentParser(description="Train an offline RL model for action prediction.")
    parser.add_argument("--steps", type=int, required=True, help="Trajectory length")
    parser.add_argument("--model-type", type=str, choices=["actor", "critic", "actor-critic"], required=True)
    parser.add_argument("--actor-loss", type=str, choices=["mse", "PG"], default="PG")
    
    # --- NEW: Backbone Selection ---
    parser.add_argument("--backbone", type=str, choices=["GIN", "GAT"], default="GAT", help="Encoder backbone")
    parser.add_argument("--heads", type=int, default=8, help="Number of GAT attention heads")
    
    parser.add_argument("--negative-selection", type=str, choices=["random", "closest", "e-greedy", "combined"], default="combined")
    parser.add_argument("--cuda", type=int, default=-1, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()

def main():
    args = get_args()

    if args.cuda >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    print(f"Using device {device} with {args.backbone} backbone")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Config - Now passing backbone-specific params
    config = TrainingConfig(
        steps=args.steps,
        model_type=args.model_type,
        actor_loss=args.actor_loss,
        backbone_type=args.backbone, # Updated
        num_heads=args.heads,        # Updated
        negative_selection=args.negative_selection,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
    )

    csv_path = str(DATASETS_DIR / f"offlineRL/{args.steps}steps_train.csv")
    dataset = OfflineRLDataset(csv_path, device=device)
    num_workers = args.num_workers if args.num_workers is not None else config.num_workers
    dataset.prepare(num_workers=num_workers)

    # --- UPDATED: Init model logic ---
    model_cls = MODEL_REGISTRY[args.model_type]
    
    # Branching logic to build kwargs based on Backbone
    if config.backbone_type == "GAT":
        model_kwargs = {
            "input_dim": config.input_dim,
            "hidden_dims": config.gat_hidden_dims,
            "num_heads": config.num_heads,
            "dense_hidden_size": config.hidden_size
        }
    else:
        model_kwargs = {
            "gin_model_path": config.get_backbone_path(),
            "hidden_size": config.hidden_size
        }

    # Add head-specific depth
    if args.model_type == "actor":
        model_kwargs["num_hidden"] = config.actor_num_hidden
    elif args.model_type == "critic":
        model_kwargs["num_hidden"] = config.critic_num_hidden
    elif args.model_type == "actor-critic":
        model_kwargs["actor_num_hidden"] = config.actor_num_hidden
        model_kwargs["critic_num_hidden"] = config.critic_num_hidden
        
    model = model_cls(**model_kwargs).to(torch.device(device))

    train_split, valid_split = dataset.split(train_frac=config.train_frac)

    print(f"Backbone: {args.backbone} | Heads: {args.heads if args.backbone=='GAT' else 'N/A'}")
    print(f"Training on {train_split.reactants.batch_size} samples...")

    trainer = OfflineRLTrainer(model, dataset, config)
    trainer.train(train_split, valid_split)
    trainer.save()

if __name__ == "__main__":
    main()
