import sys
import pandas as pd
import glob
import pickle
import os

# --- Pandas 2.0+ Compatibility Patch ---
import pandas.core.indexes.base as base
sys.modules['pandas.core.indexes.numeric'] = base
if not hasattr(base, 'Int64Index'):
    base.Int64Index = base.Index
# --------------------------------------

path = "/fs/scratch/PAS2030/itsamrithaa/GAT_FULL_RESULTS/qed_chunk_*"
combos = ['x0_y1', 'x1_y0', 'x1_y1']

print(f"Merging results for 800 targets across 32 chunks...")

for combo in combos:
    metrics_list = []
    sr_list = []
    folders = sorted(glob.glob(path))
    
    for folder in folders:
        f = os.path.join(folder, f"results_{combo}.pickle")
        if os.path.exists(f):
            with open(f, 'rb') as rb:
                try:
                    data = pickle.load(rb)
                    metrics_list.append(data['results']['metrics'])
                    sr_list.append(data['results']['success_rate'])
                except:
                    continue

    if metrics_list:
        final_metrics = pd.concat(metrics_list).mean().to_frame().T
        final_sr = pd.concat(sr_list).mean().to_frame().T
        
        print(f"\n{'='*30}")
        print(f" SCENARIO: {combo}")
        print(f"{'='*30}")
        print("\nAVERAGE METRICS (N=800):")
        print(final_metrics.to_string(index=False))
        print("\nAVERAGE SUCCESS RATES:")
        print(final_sr.to_string(index=False))
    else:
        print(f"No data found for {combo}")
