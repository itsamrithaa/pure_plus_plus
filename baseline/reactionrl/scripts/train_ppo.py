"""CLI entry point for online PPO training of the GAT-based actor-critic.

python -m reactionrl.scripts.train_ppo --eval-smiles "..." --updates 200
Reward = weighted QED/DRD2/logP/SA (set a weight to 0 to disable it).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from reactionrl.config import PPOConfig
from reactionrl.models import MODEL_REGISTRY
from reactionrl.training.ppo import PPOTrainer
from reactionrl.evaluation.ppo_report import generate_report


def _load_smiles_list(path):
    if path.endswith(".pickle") or path.endswith(".pkl"):
        series = pd.read_pickle(path)
        return list(series)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        return df[col].tolist()
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_args():
    parser = argparse.ArgumentParser(description="Train the GAT-based actor-critic online with PPO.")
    parser.add_argument("--qed-weight", type=float, default=2.0, help="Weight for QED in the combined reward (default: 2.0)")
    parser.add_argument("--drd2-weight", type=float, default=1.0, help="Weight for DRD2 activity (default: 1.0)")
    parser.add_argument("--logp-weight", type=float, default=1.0, help="Weight for logP (default: 1.0)")
    parser.add_argument("--sa-weight", type=float, default=1.0, help="Weight for synthetic accessibility (default: 1.0)")
    parser.add_argument("--start-smiles-file", type=str, default=None,
                         help="Pool of starting molecules for training rollouts (.pickle/.csv/.txt). "
                              "Defaults to datasets/my_uspto/unique_start_mols.pickle if present, "
                              "else falls back to --eval-smiles.")
    parser.add_argument("--eval-smiles", type=str, default=None,
                         help="Comma-separated SMILES to report per-molecule metrics for.")
    parser.add_argument("--eval-smiles-file", type=str, default=None,
                         help="File with one SMILES per line, alternative to --eval-smiles.")
    parser.add_argument("--num-decode", type=int, default=20, help="Decodes per molecule in the eval report (default: 20)")

    parser.add_argument("--max-steps", type=int, default=5, help="Max transformation steps per episode")
    parser.add_argument("--similarity-threshold", type=float, default=0.4)
    parser.add_argument("--similarity-penalty-weight", type=float, default=1.0)

    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gat-num-layers", type=int, default=3)
    parser.add_argument("--gat-num-heads", type=int, default=4)
    parser.add_argument("--actor-num-hidden", type=int, default=3)
    parser.add_argument("--critic-num-hidden", type=int, default=2)

    parser.add_argument("--cuda", type=int, default=-1, help="GPU device index (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: output/ppo/<active-properties>/...)")
    return parser.parse_args()


def main():
    args = get_args()

    device = f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    property_weights = {
        "qed": args.qed_weight, "drd2": args.drd2_weight, "logp": args.logp_weight, "sa": args.sa_weight,
    }
    config = PPOConfig(
        property_weights=property_weights,
        similarity_threshold=args.similarity_threshold,
        similarity_penalty_weight=args.similarity_penalty_weight,
        max_steps=args.max_steps,
        episodes_per_update=args.episodes_per_update,
        updates=args.updates,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        hidden_size=args.hidden_size,
        gat_num_layers=args.gat_num_layers,
        gat_num_heads=args.gat_num_heads,
        actor_num_hidden=args.actor_num_hidden,
        critic_num_hidden=args.critic_num_hidden,
        seed=args.seed,
        device=device,
    )

    # Eval molecules (the fixed set to report per-molecule metrics for)
    if args.eval_smiles:
        eval_smiles = [s.strip() for s in args.eval_smiles.split(",") if s.strip()]
    elif args.eval_smiles_file:
        eval_smiles = _load_smiles_list(args.eval_smiles_file)
    else:
        raise ValueError("Provide --eval-smiles or --eval-smiles-file to specify the molecules to report on.")

    # Training pool
    if args.start_smiles_file:
        start_smiles = _load_smiles_list(args.start_smiles_file)
    else:
        default_pool = os.path.join("datasets", "my_uspto", "unique_start_mols.pickle")
        if os.path.exists(default_pool):
            start_smiles = _load_smiles_list(default_pool)
        else:
            print("No --start-smiles-file given and datasets/my_uspto/unique_start_mols.pickle not found; "
                  "training on --eval-smiles directly.")
            start_smiles = list(eval_smiles)

    model_cls = MODEL_REGISTRY["actor-critic"]
    model = model_cls(
        hidden_size=config.hidden_size, gat_num_layers=config.gat_num_layers, gat_num_heads=config.gat_num_heads,
        actor_num_hidden=config.actor_num_hidden, critic_num_hidden=config.critic_num_hidden,
    ).to(torch.device(device))

    trainer = PPOTrainer(model, config)
    trainer.train(start_smiles)
    folder = trainer.save(args.output)

    print("\nTraining complete. Generating per-molecule evaluation report...")
    generate_report(
        model, eval_smiles, property_weights=config.property_weights,
        similarity_threshold=config.similarity_threshold, max_steps=config.max_steps,
        num_decode=args.num_decode, device=device, output_dir=os.path.join(folder, "eval_report"),
    )


if __name__ == "__main__":
    main()
