import pickle
import glob
import os
import sys
import pandas as pd

# --- Pandas 2.0+ Compatibility Patch ---
import pandas.core.indexes.base as base
sys.modules['pandas.core.indexes.numeric'] = base
if not hasattr(base, 'Int64Index'):
    base.Int64Index = base.Index

# The 9 "Top Molecules" from the original PURE Paper
paper_top_9 = [
    "N#Cc1nccc(Oc1ccc(NC(=O)NC(=O)C1CC1)cc1)n1",
    "NC(=O)Nc1ccc(Oc1ccnc(C(=O)N(C)C)c1)cc1",
    "CNC(=O)Nc1ccc(Oc1ccnc(C(=O)C)c1)cc1",
    "N#Cc1nccc(Oc1ccc(NC(=O)NC(=O)C)cc1)n1",
    "CNC(=O)Nc1ccc(Oc1ccnc(C(=O)OC)c1)cc1",
    "O=C(Nc1ccc(F)cc1)Nc1ccc(Oc1cccnc1)cc1",
    "CCNC(=O)Nc1ccc(Oc1ccnc(C#N)c1)cc1",
    "CNC(=O)Nc1ccc(Oc1ccnc(C#N)c1)cc1",
    "N#Cc1nccc(Oc1ccc(NC(=O)NC(=O)C(F)F)cc1)n1"
]

results_path = "/fs/scratch/PAS2030/itsamrithaa/GAT_FULL_RESULTS/qed_chunk_*"
files = glob.glob(os.path.join(results_path, "results_x1_y1.pickle"))

print(f"Searching {len(files)} result files for the Paper's Top 9 molecules...")

found_smiles = set()
total_checked = 0

for f_path in files:
    try:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
            # 'target' column in the dataframe contains the generated molecules
            generated_list = data['df']['target'].unique().tolist()
            total_checked += len(generated_list)
            
            for smi in paper_top_9:
                if smi in generated_list:
                    found_smiles.add(smi)
    except:
        continue

print("\n" + "="*40)
print("  REVERSION ANALYSIS (GAT vs GIN)")
print("="*40)
print(f"Total GAT candidates searched: {total_checked}")
print(f"Paper molecules recovered:     {len(found_smiles)} / 9")
print("="*40)

if found_smiles:
    print("\nMatches Found:")
    for s in found_smiles:
        print(f" - {s}")
else:
    print("\nNo direct matches found. GAT explored different chemical space.")

