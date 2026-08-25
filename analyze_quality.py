import pickle, glob, os, sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from tqdm import tqdm

# --- Pandas 2.0+ Compatibility Patch ---
import pandas.core.indexes.base as base
sys.modules['pandas.core.indexes.numeric'] = base
if not hasattr(base, 'Int64Index'): base.Int64Index = base.Index

def get_quality_metrics(smi):
    """Calculates molecular weight and Rule of 5 violations."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol: return None
        mw = Descriptors.MolWt(mol)
        qed = Descriptors.qed(mol)
        
        # Lipinski Rule of 5 Violations
        violations = 0
        if mw > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5: violations += 1
        if Lipinski.NumHDonors(mol) > 5: violations += 1
        if Lipinski.NumHAcceptors(mol) > 10: violations += 1
        
        return {'mw': mw, 'qed': qed, 'ro5_violations': violations}
    except:
        return None

path = "/fs/scratch/PAS2030/itsamrithaa/GAT_FULL_RESULTS/qed_chunk_*/results_x1_y1.pickle"
files = glob.glob(path)

all_leads = []
print(f"Auditing quality for {len(files)*25} molecules...")

for f_path in tqdm(files):
    with open(f_path, 'rb') as f:
        data = pickle.load(f)
        # 'df' contains the top molecules chosen by the model
        df = data['df']
        
        for _, row in df.iterrows():
            metrics = get_quality_metrics(row['target'])
            if metrics:
                all_leads.append({
                    'smiles': row['target'],
                    'similarity': row['similarity'],
                    'qed': metrics['qed'],
                    'mw': metrics['mw'],
                    'violations': metrics['ro5_violations']
                })

leads_df = pd.DataFrame(all_leads)

# DEFINE "HIGH QUALITY"
# 1. High QED (> 0.90)
# 2. Good Similarity (0.4 - 0.7) - The "Novelty Zone"
# 3. MW < 500
top_tier = leads_df[(leads_df['qed'] > 0.90) & 
                    (leads_df['similarity'] >= 0.4) & 
                    (leads_df['mw'] < 500)]

print("\n" + "="*50)
print("  GAT QUALITY AUDIT RESULTS")
print("="*50)
print(f"Total Molecules Screened:     {len(leads_df)}")
print(f"Average Molecular Weight:     {leads_df['mw'].mean():.2f} Da")
print(f"Average QED:                  {leads_df['qed'].mean():.4f}")
print(f"Molecules with 0 violations:  {len(leads_df[leads_df['violations']==0])}")
print(f"TOP TIER LEADS DISCOVERED:    {len(top_tier)}")
print("="*50)

print("\nYOUR GAT 'TOP 5' FOR THE PRESENTATION:")
print(top_tier.sort_values(by='qed', ascending=False).head(5).to_string(index=False))
