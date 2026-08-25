"""PPO trainer for online molecule-editing RL.

Rolls out MoleculeEditEnv, computes GAE(lambda), and optimizes the
clipped surrogate + value loss + entropy bonus. Action sets vary per
state, so transitions are rescored individually, not batched.
"""
import os
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tabulate import tabulate

from reactionrl.config import PPOConfig
from reactionrl.envs.molecule_env import MoleculeEditEnv


class PPOTrainer:
    """Trains a GATActorCritic online with PPO against MoleculeEditEnv."""
    def __init__(self, model, config: PPOConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.env = MoleculeEditEnv(
            property_weights=config.property_weights,
            similarity_threshold=config.similarity_threshold,
            similarity_penalty_weight=config.similarity_penalty_weight,
            max_steps=config.max_steps,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.history = []
        self.best_improvement = -float("inf")
        self.best_model = None
        self.best_update = 0

    def _collect_rollouts(self, start_smiles_list):
        """Rolls out the current policy from each start molecule.

        Returns (buffer, episode_stats).
        """
        buffer = []
        episode_stats = []

        self.model.eval()
        with torch.no_grad():
            for smiles in start_smiles_list:
                try:
                    self.env.reset(smiles)
                except ValueError:
                    continue

                total_reward = 0.0
                last_info = {"similarity_to_origin": 1.0, "property": self.env.origin_property, "improvement": 0.0}
                steps_taken = 0
                buffer_start = len(buffer)

                for _ in range(self.env.max_steps):
                    state_smiles = self.env.smiles
                    actions_df = self.env.applicable_actions()
                    if len(actions_df) == 0:
                        break

                    state_emb = self.model.encode_states([state_smiles])[0]
                    action_embs = self.model.encode_actions(
                        actions_df["rsig"].tolist(), actions_df["psig"].tolist()
                    )
                    logits = self.model.action_logits(state_emb, action_embs)
                    dist = Categorical(logits=logits)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                    value = self.model.value(state_emb)

                    action_row = actions_df.iloc[int(action_idx.item())][
                        ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
                    ]
                    next_smiles, reward, done, info = self.env.step(action_row)

                    buffer.append({
                        "state_smiles": state_smiles,
                        "actions_df": actions_df,
                        "action_idx": int(action_idx.item()),
                        "old_log_prob": log_prob.item(),
                        "old_value": value.item(),
                        "reward": reward,
                        "done": done,
                    })
                    total_reward += reward
                    steps_taken += 1
                    if "error" not in info:
                        last_info = info
                    if done:
                        break

                # Episode can end early (no actions left) without done=True.
                # Force it so GAE doesn't bootstrap into the next molecule.
                if len(buffer) > buffer_start:
                    buffer[-1]["done"] = True

                episode_stats.append({
                    "smiles": smiles,
                    "steps": steps_taken,
                    "total_reward": total_reward,
                    "final_similarity": last_info["similarity_to_origin"],
                    "final_property": last_info["property"],
                    "improvement": last_info["improvement"],
                })

        return buffer, episode_stats

    @staticmethod
    def _compute_gae(buffer, gamma, lam):
        n = len(buffer)
        rewards = [t["reward"] for t in buffer]
        values = [t["old_value"] for t in buffer]
        dones = [t["done"] for t in buffer]

        advantages = [0.0] * n
        gae = 0.0
        for i in reversed(range(n)):
            next_value = 0.0 if dones[i] else values[i + 1]
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * lam * (0.0 if dones[i] else gae)
            advantages[i] = gae
        returns = [advantages[i] + values[i] for i in range(n)]
        return advantages, returns

    def _evaluate_transition(self, transition):
        """Recompute log-prob/entropy/value under the current policy."""
        state_emb = self.model.encode_states([transition["state_smiles"]])[0]
        actions_df = transition["actions_df"]
        action_embs = self.model.encode_actions(actions_df["rsig"].tolist(), actions_df["psig"].tolist())
        logits = self.model.action_logits(state_emb, action_embs)
        dist = Categorical(logits=logits)
        action_idx = torch.tensor(transition["action_idx"], device=self.device)
        return dist.log_prob(action_idx), dist.entropy(), self.model.value(state_emb)

    def _update(self, buffer, advantages, returns):
        config = self.config
        n = len(buffer)
        indices = np.arange(n)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

        self.model.train()
        for _ in range(config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, config.minibatch_size):
                mb_idx = indices[start:start + config.minibatch_size]

                new_log_probs, entropies, new_values = [], [], []
                for i in mb_idx:
                    nlp, ent, val = self._evaluate_transition(buffer[i])
                    new_log_probs.append(nlp)
                    entropies.append(ent)
                    new_values.append(val)
                new_log_probs = torch.stack(new_log_probs)
                entropies = torch.stack(entropies)
                new_values = torch.stack(new_values)

                old_log_probs = torch.tensor(
                    [buffer[i]["old_log_prob"] for i in mb_idx], device=self.device, dtype=torch.float32
                )
                mb_advantages = torch.tensor(
                    [advantages[i] for i in mb_idx], device=self.device, dtype=torch.float32
                )
                mb_returns = torch.tensor(
                    [returns[i] for i in mb_idx], device=self.device, dtype=torch.float32
                )
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(new_values, mb_returns)
                entropy_bonus = entropies.mean()

                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > config.clip_eps).float().mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_bonus.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_fraction)

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def train(self, start_smiles_list):
        """Runs config.updates PPO iterations over start_smiles_list.

        Returns self.history (list of per-update metric dicts).
        """
        config = self.config
        start_smiles_list = list(start_smiles_list)
        if len(start_smiles_list) == 0:
            raise ValueError("start_smiles_list is empty")

        for update in range(1, config.updates + 1):
            start_time = time.time()

            sampled = np.random.choice(start_smiles_list, size=config.episodes_per_update, replace=True)
            buffer, episode_stats = self._collect_rollouts(sampled)

            if len(buffer) == 0:
                print(f"Update {update}/{config.updates}: no applicable actions found for any sampled molecule, skipping.")
                continue

            advantages, returns = self._compute_gae(buffer, config.gamma, config.gae_lambda)
            opt_metrics = self._update(buffer, advantages, returns)

            mean_reward = float(np.mean([e["total_reward"] for e in episode_stats]))
            mean_improvement = float(np.mean([e["improvement"] for e in episode_stats]))
            mean_similarity = float(np.mean([e["final_similarity"] for e in episode_stats]))
            mean_steps = float(np.mean([e["steps"] for e in episode_stats]))

            row = {
                "update": update,
                "mean_reward": mean_reward,
                "mean_improvement": mean_improvement,
                "mean_similarity": mean_similarity,
                "mean_steps": mean_steps,
                **opt_metrics,
                "time(min)": (time.time() - start_time) / 60,
            }
            self.history.append(row)

            if update % max(1, config.updates // 20) == 0 or update == 1:
                print(tabulate(pd.DataFrame([row]), headers="keys", tablefmt="grid", showindex=False))

            if mean_improvement > self.best_improvement:
                self.best_improvement = mean_improvement
                self.best_model = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_update = update

        return self.history

    def save(self, folder=None):
        """Saves the model, training curve, and config to disk."""
        config = self.config
        if folder is None:
            active_properties = "-".join(k for k, w in config.property_weights.items() if w) or "none"
            folder = os.path.join(
                "output", "ppo", active_properties,
                f"steps={config.max_steps}_seed={config.seed}"
            )
        os.makedirs(folder, exist_ok=True)

        torch.save(self.model, os.path.join(folder, "model.pth"))
        if self.best_model is not None:
            torch.save(self.best_model, os.path.join(folder, "best_model_state_dict.pth"))
        pd.DataFrame(self.history).to_csv(os.path.join(folder, "ppo_metrics.csv"), index=False)
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=2)
        print("Saved PPO run at", folder)
        return folder
