# PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation

PURE uses:
1. A Graph Isomorphism Network (GIN) to encode molecular structures as graphs.
2. Template-based molecular simulations, extracted from the USPTO-MIT reaction database, which constrain exploration to chemically valid, synthesizable reactions.
3. A policy-guided actor-critic RL setup, where molecular similarity naturally emerges from the learned representations rather than being hard-coded through external metrics.

For more details, please refer to
**Gupta, A., Lenin, B., Current, S., Batra, R., Ravindran, B., Raman, K., & Parthasarathy, S. (2025). PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation. bioRxiv, 2025-05.** ([PDF](https://www.biorxiv.org/content/10.1101/2025.05.21.655002v1.full.pdf)).

---

## Repository Structure

```
reactionrl/                  # Main package
├── actions/                 # ActionSpace: find & apply molecular transformations
├── embeddings/              # GIN and MPNN molecular embedders
├── models/                  # Actor, Critic, ActorCritic networks
├── rewards/                 # Property scorers (logP, QED, DRD2, SA, similarity)
├── data/                    # Dataset loading & trajectory generation
├── training/                # Trainer, losses, ranking metrics
├── evaluation/              # PURE vs baseline comparison metrics
├── preprocessing/           # Data pipeline scripts (USPTO extraction)
├── scripts/                 # CLI entry points (train, generate_data, preprocess)
├── utils/                   # Shared molecule utilities
└── config.py                # Paths & TrainingConfig dataclass
pretrained_models/           # GIN and MPNN pretrained weights
datasets/                    # Action dataset & training data
notebooks/                   # Experimental notebooks
```

---

## Requirements

Python 3.8 with the following core dependencies:

```
torch
torchdrug
rdkit
pandas
numpy
networkx
scikit-learn
tabulate
tqdm
filehash
matplotlib
```

Optional (for MPNN embedder):
```
deepchem
dgl
```

Install via conda:
```bash
conda env create -f environment.yml
conda activate reactionrl
```

Or install manually:
```bash
conda create -n reactionrl python=3.8
conda activate reactionrl
pip install torch==2.1.1
pip install torchdrug==0.2.1 rdkit-pypi pandas numpy networkx scikit-learn tabulate tqdm filehash matplotlib
```

---

## Quick Start

### 1. Preprocess the action dataset (one-time setup)

Extract reaction templates from USPTO-MIT:

```bash
bash reactionrl/scripts/preprocess.sh
```

This runs the full pipeline: download transformations, extract reaction centres and signatures, generate and filter the action dataset, dump starting molecules, and compute action embeddings.

### 2. Generate training data

Roll out a random policy to collect (reactant, action, product) trajectories:

```bash
python -m reactionrl.scripts.generate_data --steps 5 --train-samples 100000
```

Options:
- `--steps`: Number of transformation steps per trajectory
- `--train-samples`: Number of training samples to generate (default: 100,000)
- `--processes`: Number of parallel workers (default: 80% of CPU cores)

### 3. Train an offline RL model

```bash
python -m reactionrl.scripts.train --steps 5 --model-type actor-critic --actor-loss PG --cuda 0
```

Options:
- `--steps`: Trajectory length (must match the generated data)
- `--model-type`: `actor`, `critic`, or `actor-critic`
- `--actor-loss`: `mse` or `PG` (policy gradient with negative sampling)
- `--epochs`: Number of training epochs (default: 50)
- `--cuda`: GPU index (-1 for CPU, default: -1)
- `--seed`: Random seed (default: 42)
- `--negative-selection`: `random`, `closest`, `e-greedy`, or `combined`
- `--num-workers`: Parallel workers for data preparation

---

## Online RL with PPO (GAT-based actor-critic)

In addition to the offline actor-critic trainer above, `reactionrl` includes
an online PPO trainer (`reactionrl.training.ppo.PPOTrainer`). Instead of
learning from a fixed trajectory CSV, it rolls out a GAT-based actor-critic
(`reactionrl.models.GATActorCritic`) against `reactionrl.envs.MoleculeEditEnv`
- a small MDP built on top of the existing `ActionSpace` where each step
applies one applicable reaction-template action to the current molecule -
and optimizes the clipped PPO surrogate objective with a GAE(lambda)
advantage estimate.

The GAT backbone is trained from scratch (no ZINC-pretrained checkpoint is
required, unlike the GIN backbone used by the offline path).

### Train

The reward is a *weighted combination* of QED, DRD2 activity, logP, and SA
(synthetic accessibility) - set any weight to `0` to ignore that property.
For example, to optimize sorafenib with QED weighted twice as heavily as
the others (the default weights):

```bash
python -m reactionrl.scripts.train_ppo \
    --qed-weight 2.0 --drd2-weight 1.0 --logp-weight 1.0 --sa-weight 1.0 \
    --eval-smiles "CNC(=O)c1cc(ccn1)Oc2ccc(cc2)NC(=O)Nc3ccc(c(c3)C(F)(F)F)Cl" \
    --updates 200 --episodes-per-update 32 --cuda 0
```

Key options:
- `--qed-weight` / `--drd2-weight` / `--logp-weight` / `--sa-weight`: weights in the combined reward (defaults: 2.0/1.0/1.0/1.0). logP and SA are rescaled onto a comparable range to QED/DRD2 before being combined (see `combined_score` in `envs/molecule_env.py`).
- `--eval-smiles` / `--eval-smiles-file`: the molecule(s) to report per-molecule before/after metrics for.
- `--start-smiles-file`: pool of starting molecules for training rollouts (`.pickle`/`.csv`/`.txt`). Defaults to `datasets/my_uspto/unique_start_mols.pickle` (see `preprocessing/dump_start_mols.py`) if present, else falls back to `--eval-smiles` (i.e. training rollouts start from the same molecule(s) being evaluated - the right choice when optimizing one specific molecule like above).
- `--similarity-threshold`: minimum Tanimoto similarity to the original molecule before the reward is penalized (keeps edits structurally constrained).
- `--updates`, `--episodes-per-update`, `--ppo-epochs`, `--minibatch-size`, `--gamma`, `--gae-lambda`, `--clip-eps`, `--entropy-coef`, `--value-coef`, `--lr`: standard PPO hyperparameters (see `PPOConfig` in `config.py`).

Each update prints a metrics row (mean reward, mean combined-score
improvement, mean similarity to origin, policy/value loss, entropy, approx
KL, clip fraction). `trainer.save()` writes `ppo_metrics.csv`, `model.pth`,
and `config.json` to `output/ppo/<active-properties>/...`.

### Evaluate / report

After training, `train_ppo.py` automatically calls
`reactionrl.evaluation.ppo_report.generate_report`, which decodes 20
candidates per evaluation molecule (1 greedy + 19 sampled, matching the
existing `num_decode=20` convention in `evaluation/evaluate.py`) and prints:

- **Per-molecule table**: original vs. best-generated QED, DRD2, logP, and SA (each broken out individually), the combined-score improvement, and Tanimoto similarity to the original, for each evaluation molecule.
- **Aggregate metrics**: validity, average combined score, average improvement, average similarity, novelty, diversity (via the existing `evaluate_metric`).
- **Success rates**: fraction of molecules with a similarity-gated valid/improved edit.

It also saves `per_molecule_metrics.csv`, `all_decoded_candidates.csv`,
`aggregate_metrics.csv`, `success_rate.csv`, and a `before_after_property.png`
2x2 bar chart (one panel per property) to `output/ppo/<active-properties>/.../eval_report/`.

To generate a report from an already-trained model without retraining:

```python
import torch
from reactionrl.evaluation.ppo_report import generate_report

model = torch.load("output/ppo/qed-drd2-logp-sa/steps=5_seed=42/model.pth")
generate_report(model, ["CNC(=O)c1cc(ccn1)Oc2ccc(cc2)NC(=O)Nc3ccc(c(c3)C(F)(F)F)Cl"], output_dir="my_report")
```

---

## How to Experiment

The codebase is modular -- here's how to swap or modify components:

### Swap the embedding model

All models use a GIN backbone loaded from `pretrained_models/zinc2m_gin.pth`. To use a different pretrained model:

```python
from reactionrl.config import TrainingConfig

config = TrainingConfig(gin_model_path="/path/to/your/model.pth")
```

To implement a completely new embedder, subclass `BaseEmbeddingClass`:

```python
from reactionrl.embeddings.base import BaseEmbeddingClass

class MyEmbedder(BaseEmbeddingClass):
    def mol_to_embedding(self, mol):
        ...
    def atom_to_embedding(self, mol, idx):
        ...
```

### Change model architecture

Architecture parameters are configurable:

```python
from reactionrl.models import ActorCritic

model = ActorCritic(
    gin_model_path="pretrained_models/zinc2m_gin.pth",
    actor_num_hidden=4,      # default: 3
    critic_num_hidden=3,     # default: 2
    hidden_size=512,         # default: 256
)
```

Or use `TrainingConfig`:

```python
config = TrainingConfig(
    hidden_size=512,
    actor_num_hidden=4,
    critic_num_hidden=3,
)
```

### Add a new model type

1. Create a new `nn.Module` in `reactionrl/models/` with a `.GIN` attribute and `.actor` property
2. Register it in `reactionrl/models/__init__.py`:

```python
MODEL_REGISTRY["my-model"] = MyModel
```

### Try different reward functions

Property scorers are in `reactionrl/rewards/`:

```python
from reactionrl.rewards import logP, qed, drd2, SA, similarity
```

### Use the action space programmatically

```python
from rdkit import Chem
from reactionrl.actions import get_applicable_actions, apply_action

mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")  # biphenyl
actions = get_applicable_actions(mol)
print(f"{actions.shape[0]} applicable actions")

# Apply the first action
product = apply_action(mol, *actions.iloc[0])
print(Chem.MolToSmiles(product))
```

### Customize training

```python
from reactionrl.config import TrainingConfig
from reactionrl.models import ActorCritic
from reactionrl.data.dataset import OfflineRLDataset
from reactionrl.training import OfflineRLTrainer

config = TrainingConfig(
    steps=5,
    model_type="actor-critic",
    actor_loss="PG",
    epochs=100,
    batch_size=256,
    actor_lr=1e-4,
    device="cuda:0",
)

dataset = OfflineRLDataset("datasets/offlineRL/5steps_train.csv", device="cuda:0")
dataset.prepare(num_workers=config.num_workers)
train_split, valid_split = dataset.split(train_frac=config.train_frac)

model = ActorCritic(gin_model_path=config.get_gin_model_path())
model = model.to("cuda:0")

trainer = OfflineRLTrainer(model, dataset, config)
trainer.train(train_split, valid_split)
trainer.save("output/my_experiment")
```

---

## Key Concepts

**Action Space**: Molecular transformations are defined as reaction signature pairs (rsig -> psig) extracted from USPTO. Given a molecule, the `ActionSpace` identifies which transformations are applicable by matching substructures via articulation point decomposition.

**Offline RL Training**: The actor network learns to predict the correct action embedding (concatenated rsig + psig GIN embeddings) given a (reactant, product) pair. The critic network learns to score whether a given action is correct for a transition. Policy gradient loss with negative sampling trains the actor to distinguish correct actions from similar but incorrect ones.

**Evaluation**: Ranking metrics (euclidean and cosine distance) measure how well the predicted action embedding ranks the correct action among all applicable actions. Lower rank = better prediction.

---

## Citation

```bibtex
@article{gupta2025pure,
  title={PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation},
  author={Gupta, Abhor and Lenin, Bhargav and Current, Sean and Batra, Ritwik and Ravindran, Balaraman and Raman, Karthik and Parthasarathy, Srinivasan},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.05.21.655002}
}
```
