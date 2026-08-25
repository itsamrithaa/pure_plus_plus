"""Single-molecule editing environment for online (PPO) RL training.

Wraps ActionSpace as a finite-horizon MDP. Reward is a weighted
combination of QED/DRD2/logP/SA, penalized below a similarity threshold.
"""
from rdkit import Chem

from reactionrl.actions.action_space import get_default_action_space
from reactionrl.evaluation.properties import similarity as smiles_tanimoto_similarity
from reactionrl.rewards.properties import qed, drd2, logP, SA

# logP/SA are unbounded, so z-score normalize before combining.
# Same constants used elsewhere in this repo (ZINC250k reference).
LOGP_MEAN, LOGP_STD = 2.4570953396190123, 1.434324401111988
SA_MEAN, SA_STD = -3.0525811293166134, 0.8335207024513095

DEFAULT_PROPERTY_WEIGHTS = {"qed": 2.0, "drd2": 1.0, "logp": 1.0, "sa": 1.0}

_drd2_broken = False


def _safe_drd2(mol):
    """drd2_scorer's classifier can fail under modern sklearn.
    Falls back to 0.0 and warns once instead of crashing."""
    global _drd2_broken
    if _drd2_broken:
        return 0.0
    try:
        return drd2(mol)
    except Exception as e:
        _drd2_broken = True
        print(f"WARNING: drd2 scorer failed ({e!r}); returning 0.0 for all molecules "
              f"this run. Set drd2's weight to 0 to exclude it from the reward.")
        return 0.0


def raw_property_scores(mol):
    """Un-normalized property values, as reported (qed/drd2 in [0, 1], logP
    unbounded, SA in [1 (easy), 10 (hard)])."""
    return {"qed": qed(mol), "drd2": _safe_drd2(mol), "logp": logP(mol), "sa": SA(mol)}


def combined_score(raw_scores, property_weights, clip=1.5):
    """Weighted sum of normalized property scores; higher is better.

    Clips each term to [-clip, clip] first so one unbounded property
    (e.g. logP) can't dominate regardless of weight. clip=None disables it.
    """
    normalized = {
        "qed": raw_scores["qed"],
        "drd2": raw_scores["drd2"],
        "logp": (raw_scores["logp"] - LOGP_MEAN) / LOGP_STD,
        "sa": (-raw_scores["sa"] - SA_MEAN) / SA_STD,  # lower SA = easier = better
    }
    if clip is not None:
        normalized = {k: max(-clip, min(clip, v)) for k, v in normalized.items()}
    return sum(property_weights.get(k, 0.0) * v for k, v in normalized.items())


class MoleculeEditEnv:
    """Gym-style environment: state = current molecule SMILES."""
    def __init__(self, property_weights=None, similarity_threshold=0.4,
                 similarity_penalty_weight=1.0, max_steps=5, action_space=None):
        self.property_weights = dict(property_weights) if property_weights else dict(DEFAULT_PROPERTY_WEIGHTS)
        self.similarity_threshold = similarity_threshold
        self.similarity_penalty_weight = similarity_penalty_weight
        self.max_steps = max_steps
        self.action_space = action_space or get_default_action_space()

        self.origin_smiles = None
        self.origin_raw_scores = None
        self.origin_property = None
        self.mol = None
        self.smiles = None
        self.step_count = 0

    def reset(self, smiles):
        """Start a new episode from `smiles`. Returns the canonical starting SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid starting SMILES: {smiles!r}")
        self.origin_smiles = Chem.MolToSmiles(mol)
        self.origin_raw_scores = raw_property_scores(mol)
        self.origin_property = combined_score(self.origin_raw_scores, self.property_weights)
        self.mol = mol
        self.smiles = self.origin_smiles
        self.step_count = 0
        return self.smiles

    def applicable_actions(self):
        """DataFrame of actions applicable to the current molecule (may be empty)."""
        return self.action_space.get_applicable_actions(self.mol)

    def step(self, action_row):
        """Apply one action to the current molecule.

        Returns (next_smiles, reward, done, info).
        """
        self.step_count += 1
        try:
            next_mol = self.action_space.apply_action(self.mol, *action_row)
        except Exception as e:
            # Template matched during search but failed to apply cleanly.
            return self.smiles, -1.0, True, {"error": str(e)}

        prev_property = combined_score(raw_property_scores(self.mol), self.property_weights)
        next_raw = raw_property_scores(next_mol)
        next_property = combined_score(next_raw, self.property_weights)
        next_smiles = Chem.MolToSmiles(next_mol)
        sim_to_origin = smiles_tanimoto_similarity(self.origin_smiles, next_smiles)

        reward = next_property - prev_property
        if sim_to_origin < self.similarity_threshold:
            reward -= self.similarity_penalty_weight * (self.similarity_threshold - sim_to_origin)

        self.mol = next_mol
        self.smiles = next_smiles
        done = self.step_count >= self.max_steps

        info = {
            "similarity_to_origin": sim_to_origin,
            "property": next_property,
            "raw_scores": next_raw,
            "improvement": next_property - self.origin_property,
        }
        return self.smiles, reward, done, info
