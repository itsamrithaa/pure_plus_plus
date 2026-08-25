import torch
import numpy as np
import pandas as pd
import tqdm
from rdkit import Chem
from torchdrug import data

from reactionrl.data.molecule_utils import molecule_from_smile
from reactionrl.actions import get_applicable_actions

def get_mol_embedding(model, smiles):
    """Gets graph-level embeddings. Handles GAT/GIN backbones dynamically."""
    if isinstance(smiles, str):
        mol = molecule_from_smile(smiles)
    elif isinstance(smiles, list) or isinstance(smiles, pd.Series):
        mol = list(map(molecule_from_smile, smiles))
        mol = data.Molecule.pack(mol)
    else:
        mol = smiles
    
    # FIX: Safely get device from model parameters
    device = next(model.parameters()).device
    mol = mol.to(device)

    # FIX: Handle if the full ActorCritic model is passed instead of just the backbone
    encoder = model.GAT if hasattr(model, 'GAT') else (model.GIN if hasattr(model, 'GIN') else model)
    
    emb = encoder(mol, mol.node_feature.float())["graph_feature"]
    return emb.detach()


def get_atom_embedding(model, smiles, idx):
    """Gets atom-level embeddings."""
    device = next(model.parameters()).device
    encoder = model.GAT if hasattr(model, 'GAT') else (model.GIN if hasattr(model, 'GIN') else model)

    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        mol = mol.to(device)
        emb = encoder(mol, mol.node_feature.float())["node_feature"][idx]
    except Exception:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        mol = mol.to(device)
        emb = encoder(mol, mol.node_feature.float())["node_feature"][idx]
    return emb.detach()


def get_action_embedding(model, action_df):
    """Concatenates rsig and psig embeddings. Uses torch.cat for stability."""
    rsub, rcen, rsig, _, psub, pcen, psig, __ = [action_df[c] for c in action_df.columns]
    # Changed concatenate to cat
    embedding = torch.cat([get_mol_embedding(model, rsig), get_mol_embedding(model, psig)], dim=1)
    return embedding


def get_action_embedding_from_packed_molecule(model, rsig, psig):
    """Batch version for packed molecules."""
    # Changed concatenate to cat
    embedding = torch.cat([get_mol_embedding(model, rsig), get_mol_embedding(model, psig)], dim=1)
    return embedding


def get_action_dataset_embeddings(model, action_rsigs, action_psigs, batch_size=2048):
    """Computes embeddings for entire action dataset."""
    action_embeddings = []
    for i in tqdm.tqdm(range(0, action_rsigs.batch_size, batch_size)):
        batch_rsig = action_rsigs[i:min(i + batch_size, action_rsigs.batch_size)]
        batch_psig = action_psigs[i:min(i + batch_size, action_psigs.batch_size)]
        action_embeddings.append(get_action_embedding_from_packed_molecule(model, batch_rsig, batch_psig))
    
    # Changed concatenate to cat
    action_embeddings = torch.cat(action_embeddings, dim=0)
    return action_embeddings

def get_emb_indices_and_correct_idx(row, no_correct_idx=False):
    """Returns applicable action indices and correct action index for training.

    Uses the lazily-initialized default action space, so it works in multiprocessing workers.
    """
    from reactionrl.actions.action_space import get_default_action_space
    action_space = get_default_action_space()
    action_dataset = action_space.dataset[
        ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
    ]

    if isinstance(row, tuple):  # For pandas iterrows
        row = row[1]

    # Applicable indices
    applicable_actions_df = get_applicable_actions(Chem.MolFromSmiles(row["reactant"]))
    if applicable_actions_df.shape[0] == 0:
        # If there are no applicable actions detected (rdkit problems)
        if no_correct_idx is False:
            indices_used_for_data = np.where((action_dataset.index == row.name))[0]
            correct_applicable_idx = 0
            correct_action_idx = indices_used_for_data[0]
        else:
            indices_used_for_data = []
    else:
        indices_used_for_data = np.where(action_dataset.index.isin(applicable_actions_df.index))[0]

        if no_correct_idx is False:
            # Correct index
            applicable_actions_df = applicable_actions_df.loc[action_dataset.iloc[indices_used_for_data].index]
            correct_applicable_idx = (applicable_actions_df.index == row.name).argmax()
            correct_action_idx = indices_used_for_data[correct_applicable_idx]

    if no_correct_idx is True:
        return indices_used_for_data
    return indices_used_for_data, correct_applicable_idx, correct_action_idx
