"""Per-molecule evaluation report for a trained PPO policy.

Decodes num_decode candidates per molecule and reports QED/DRD2/logP/SA
before vs. after, Tanimoto similarity, and aggregate metrics.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from rdkit import Chem
from tabulate import tabulate
from matplotlib import pyplot as plt

from reactionrl.envs.molecule_env import MoleculeEditEnv
from reactionrl.evaluation.evaluate import evaluate_metric

RAW_PROPERTIES = ["qed", "drd2", "logp", "sa"]


def _decode_one_episode(model, env, smiles, greedy=False):
    """Runs one rollout from `smiles` under the current policy.

    Returns (final_smiles, info) - see MoleculeEditEnv.step.
    """
    env.reset(smiles)
    info = {
        "similarity_to_origin": 1.0, "property": env.origin_property,
        "raw_scores": dict(env.origin_raw_scores), "improvement": 0.0,
    }

    with torch.no_grad():
        for _ in range(env.max_steps):
            actions_df = env.applicable_actions()
            if len(actions_df) == 0:
                break
            state_emb = model.encode_states([env.smiles])[0]
            action_embs = model.encode_actions(actions_df["rsig"].tolist(), actions_df["psig"].tolist())
            logits = model.action_logits(state_emb, action_embs)
            action_idx = int(torch.argmax(logits).item()) if greedy else int(Categorical(logits=logits).sample().item())

            action_row = actions_df.iloc[action_idx][
                ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
            ]
            _, _, done, step_info = env.step(action_row)
            if "error" not in step_info:
                info = step_info
            if done:
                break

    return env.smiles, info


def decode_molecules(model, smiles_list, property_weights=None, similarity_threshold=0.4,
                      max_steps=5, num_decode=20, device="cpu"):
    """Decodes num_decode candidate edits per source molecule.

    Returns one row per decode, in the columns evaluate_metric expects
    plus raw per-property before/after columns. Invalid SMILES are skipped.
    """
    model.eval()
    model.to(device)
    env = MoleculeEditEnv(
        property_weights=property_weights, similarity_threshold=similarity_threshold, max_steps=max_steps
    )

    checked_smiles = []
    for smiles in smiles_list:
        if Chem.MolFromSmiles(smiles) is None:
            print(f"WARNING: skipping invalid SMILES: {smiles!r}")
            continue
        checked_smiles.append(smiles)

    rows = []
    for smiles in checked_smiles:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        for k in range(num_decode):
            target_smiles, info = _decode_one_episode(model, env, smiles, greedy=(k == 0))
            row = {
                "source": canonical, "target": target_smiles,
                "similarity": info["similarity_to_origin"],
                "prop_target": info["property"], "prop_source": env.origin_property,
            }
            for prop in RAW_PROPERTIES:
                row[f"{prop}_source"] = env.origin_raw_scores[prop]
                row[f"{prop}_target"] = info["raw_scores"][prop]
            rows.append(row)

    columns = ["source", "target", "similarity", "prop_target", "prop_source"]
    columns += [f"{p}_source" for p in RAW_PROPERTIES] + [f"{p}_target" for p in RAW_PROPERTIES]
    return pd.DataFrame(rows, columns=columns)


def per_molecule_report(df_generated):
    """Best-candidate summary table, per property, before vs. after."""
    records = []
    for source, group in df_generated.groupby("source", sort=False):
        base = {"source": source, "combined_before": group["prop_source"].iloc[0]}
        for prop in RAW_PROPERTIES:
            base[f"{prop}_before"] = group[f"{prop}_source"].iloc[0]

        valid = group[(group["target"] != source) & (group["similarity"] > 0) & (group["similarity"] < 1)]
        if len(valid) == 0:
            records.append({
                **base, "best_target": None, "combined_after": None, "combined_improvement": None,
                "similarity": None, "valid_candidates": 0, "num_decode": len(group),
                **{f"{prop}_after": None for prop in RAW_PROPERTIES},
            })
            continue

        best = valid.loc[valid["prop_target"].idxmax()]
        records.append({
            **base, "best_target": best["target"], "combined_after": best["prop_target"],
            "combined_improvement": best["prop_target"] - base["combined_before"],
            "similarity": best["similarity"], "valid_candidates": len(valid), "num_decode": len(group),
            **{f"{prop}_after": best[f"{prop}_target"] for prop in RAW_PROPERTIES},
        })
    return pd.DataFrame(records)


def _plot_before_after(per_mol, path):
    labels = [f"mol {i + 1}" for i in range(len(per_mol))]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, prop in zip(axes.flat, RAW_PROPERTIES):
        before = per_mol[f"{prop}_before"].astype(float).tolist()
        after = per_mol[f"{prop}_after"].fillna(per_mol[f"{prop}_before"]).astype(float).tolist()
        ax.bar(x - width / 2, before, width, label="original")
        ax.bar(x + width / 2, after, width, label="PPO-optimized")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(prop.upper())
        ax.legend(fontsize=8)
    fig.suptitle("Before vs. after PPO optimization")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_report(model, smiles_list, property_weights=None, similarity_threshold=0.4,
                     max_steps=5, num_decode=20, device="cpu", output_dir=None):
    """Decodes candidates, prints the report, and optionally saves it.

    Returns per_molecule/all_decoded/metrics/success_rate.
    """
    df_generated = decode_molecules(
        model, smiles_list, property_weights=property_weights, similarity_threshold=similarity_threshold,
        max_steps=max_steps, num_decode=num_decode, device=device,
    )
    per_mol = per_molecule_report(df_generated)

    # evaluate_metric expects exactly these 5 columns, in this order.
    eval_input = df_generated[["source", "target", "similarity", "prop_target", "prop_source"]]
    aggregate = evaluate_metric(
        eval_input, smiles_train_high=set(), num_decode=num_decode,
        list_threshold_sim=[similarity_threshold],
    )

    print(f"\n=== Per-molecule results ({num_decode} decodes each) ===")
    print(tabulate(per_mol, headers="keys", tablefmt="grid", showindex=False, floatfmt=".3f"))
    print("\n=== Aggregate metrics (combined weighted score) ===")
    print(tabulate(aggregate["metrics"], headers="keys", tablefmt="grid", floatfmt=".3f"))
    print("\n=== Success rates (similarity >= threshold AND combined score improved/above 0) ===")
    print(tabulate(aggregate["success_rate"], headers="keys", tablefmt="grid", floatfmt=".3f"))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        per_mol.to_csv(os.path.join(output_dir, "per_molecule_metrics.csv"), index=False)
        df_generated.to_csv(os.path.join(output_dir, "all_decoded_candidates.csv"), index=False)
        aggregate["metrics"].to_csv(os.path.join(output_dir, "aggregate_metrics.csv"), index=False)
        aggregate["success_rate"].to_csv(os.path.join(output_dir, "success_rate.csv"), index=False)
        _plot_before_after(per_mol, os.path.join(output_dir, "before_after_property.png"))
        print(f"\nSaved report to {output_dir}")

    return {"per_molecule": per_mol, "all_decoded": df_generated, **aggregate}
