"""Beam search generation for molecule optimization.

Corrected for GAT backbone compatibility and H100 GPU device handling.
"""
import numpy as np
import pandas as pd
import torch
import tqdm
import functools
from multiprocessing import Pool
from rdkit import Chem
from torchdrug import data

from reactionrl.data.molecule_utils import molecule_from_smile
from reactionrl.actions.action_space import get_default_action_space
from reactionrl.training.embedding_helpers import (
    get_action_dataset_embeddings,
    get_emb_indices_and_correct_idx,
)
from reactionrl.evaluation.properties import similarity as tanimoto_similarity


def prepare_action_data(action_space=None):
    """Pack action signatures into torchdrug molecules for embedding computation."""
    if action_space is None:
        action_space = get_default_action_space()
    action_dataset = action_space.dataset[
        ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
    ]
    action_rsigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["rsig"])))
    action_psigs = data.Molecule.pack(list(map(molecule_from_smile, action_dataset["psig"])))
    return action_dataset, action_rsigs, action_psigs


def get_topk_predictions(model, source_list, target_list,
                         action_rsigs, action_psigs,
                         device, topk_actor=50, topk_critic=5,
                         batch_size=1024, num_workers=8):
    """Get top actions via actor filtering + critic re-ranking."""
    n = len(source_list)

    # Pack source and target molecules
    sources = data.Molecule.pack(list(map(molecule_from_smile, source_list)))
    targets = data.Molecule.pack(list(map(molecule_from_smile, target_list)))

    # Step 1: Actor predictions in batches (moved to CPU for distance math)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([
            model(
                sources[i:min(i + batch_size, n)].to(device),
                targets[i:min(i + batch_size, n)].to(device),
                None, None, "actor"
            ).detach().cpu()
            for i in range(0, n, batch_size)
        ], dim=0)

    # Identify backbone dynamically and move embeddings to CPU to match 'pred'
    backbone = model.GAT if hasattr(model, 'GAT') else model.GIN
    action_embeddings = get_action_dataset_embeddings(
        backbone, action_rsigs, action_psigs
    ).cpu()

    # Step 2: Get applicable action indices per source
    applicable_action_indices_list = []
    rows = [{"reactant": source_list[i]} for i in range(n)]
    with Pool(num_workers) as p:
        for idxes in p.imap(
            functools.partial(get_emb_indices_and_correct_idx, no_correct_idx=True),
            rows, chunksize=10
        ):
            applicable_action_indices_list.append(idxes)

    # Step 3: Actor filtering — top B_A by Euclidean distance
    filtered_indices = {}
    for i in range(n):
        adi = applicable_action_indices_list[i]
        if len(adi) == 0:
            filtered_indices[i] = np.array([], dtype=np.int64)
            continue
        # Standardized 'dim' keyword and ensured CPU-CPU math
        dist = torch.linalg.norm(action_embeddings[adi] - pred[i], dim=1)
        top_k = min(topk_actor, len(adi))
        filtered_indices[i] = adi[torch.argsort(dist)[:top_k].numpy().astype(np.int64)]

    # Step 4: Critic re-ranking — top B by Q-value
    all_action_indices = np.concatenate([filtered_indices[i] for i in range(n)])
    all_state_indices = np.concatenate([
        np.full(len(filtered_indices[i]), i, dtype=np.int64) for i in range(n)
    ])

    if len(all_action_indices) == 0:
        return [np.array([], dtype=np.int64) for _ in range(n)]

    with torch.no_grad():
        critic_qs = torch.cat([
            model(
                sources[all_state_indices[j:j + batch_size]].to(device),
                targets[all_state_indices[j:j + batch_size]].to(device),
                action_rsigs[all_action_indices[j:j + batch_size]].to(device),
                action_psigs[all_action_indices[j:j + batch_size]].to(device),
                "critic"
            ).detach().cpu()
            for j in range(0, len(all_action_indices), batch_size)
        ], dim=0).numpy().reshape(-1)

    result = []
    offset = 0
    for i in range(n):
        count = len(filtered_indices[i])
        if count == 0:
            result.append(np.array([], dtype=np.int64))
        else:
            i_qs = critic_qs[offset:offset + count]
            top_k = min(topk_critic, count)
            top_idx = i_qs.argsort()[::-1][:top_k]
            result.append(filtered_indices[i][top_idx])
        offset += count

    return result


def _apply_actions_worker(args):
    """Worker to apply actions to molecules."""
    reactant_smi, action_indices = args
    if len(action_indices) == 0:
        return []

    action_space = get_default_action_space()
    action_dataset = action_space.dataset[
        ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
    ]
    mol = Chem.MolFromSmiles(reactant_smi)
    results = []
    for idx in action_indices:
        try:
            product = action_space.apply_action(mol, *action_dataset.iloc[idx])
            results.append((Chem.MolToSmiles(product), int(idx)))
        except Exception:
            pass
    return results


def generate_molecules(model, source_smiles, target_smiles,
                       action_rsigs, action_psigs,
                       device, steps=5, topk_actor=50, topk_critic=5,
                       num_workers=8):
    """Run Algorithm 2 beam search."""
    n = len(source_smiles)
    assert len(target_smiles) == n

    trajectory_dict = {str(i): source_smiles[i] for i in range(n)}
    source_keys = [str(i) for i in range(n)]
    current_sources = list(source_smiles)
    target_idx_list = list(range(n))

    for step in range(1, steps + 1):
        print(f"Generation step {step}/{steps} — {len(current_sources)} source molecules")

        if len(current_sources) == 0:
            break

        current_targets = [target_smiles[ti] for ti in target_idx_list]
        pred_indices = get_topk_predictions(
            model, current_sources, current_targets,
            action_rsigs, action_psigs, device,
            topk_actor=topk_actor, topk_critic=topk_critic,
            num_workers=num_workers,
        )

        new_keys = []
        new_sources = []
        new_target_idx = []
        with Pool(num_workers) as p:
            work = list(zip(current_sources, pred_indices))
            for i, products in enumerate(p.imap(
                _apply_actions_worker, work, chunksize=10
            )):
                parent_key = source_keys[i]
                ti = target_idx_list[i]
                for product_smi, action_idx in products:
                    key = f"{parent_key}_{action_idx}"
                    trajectory_dict[key] = product_smi
                    new_keys.append(key)
                    new_sources.append(product_smi)
                    new_target_idx.append(ti)

        source_keys = new_keys
        current_sources = new_sources
        target_idx_list = new_target_idx

    print(f"Computing similarities for {len(trajectory_dict)} molecules...")
    similarity_dict = {}
    for key, smi in tqdm.tqdm(trajectory_dict.items()):
        ti = int(key.split("_")[0])
        similarity_dict[key] = tanimoto_similarity(smi, target_smiles[ti])

    return trajectory_dict, similarity_dict
