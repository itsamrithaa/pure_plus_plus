"""GAT-based molecular graph encoder.

Uses torchdrug's default continuous features, not the legacy GIN's
pretrain-only featurization - no external checkpoint needed.
"""
import torch
import torch.nn as nn
from torchdrug import data, models


def featurize(smiles):
    """SMILES -> torchdrug Molecule with continuous atom/bond features."""
    try:
        return data.Molecule.from_smiles(smiles, atom_feature="default", bond_feature="default")
    except Exception:
        return data.Molecule.from_smiles(smiles, atom_feature="default", bond_feature="default", with_hydrogen=True)


class GATMoleculeEncoder(nn.Module):
    """GAT that maps a batch of SMILES to fixed-size embeddings."""
    def __init__(self, hidden_size=256, num_layers=3, num_heads=4, readout="mean"):
        super().__init__()
        input_dim = featurize("CC").node_feature.shape[-1]
        self.gat = models.GAT(
            input_dim=input_dim,
            hidden_dims=[hidden_size] * num_layers,
            num_head=num_heads,
            readout=readout,
        )
        self.embed_dim = hidden_size

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, smiles):
        """smiles: str or list of str -> (batch, embed_dim) tensor."""
        smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
        mols = data.Molecule.pack([featurize(s) for s in smiles_list]).to(self.device)
        return self.gat(mols, mols.node_feature.float())["graph_feature"]
