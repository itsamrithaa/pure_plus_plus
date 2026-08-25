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

BASE_PATH = "/fs/scratch/PAS2030/itsamrithaa/GAT_FULL_RESULTS"
PROPERTIES = ['qed', 'drd2', 'logp04', 'logp06']

# Map properties to their specific coefficient combos
COMBOS_MAP = {
    "qed": ['x0_y1', 'x1_y0', 'x1_y1'],
    "drd2": ['x0_y1', 'x1_y0', 'x1_y1'],
    "logp04": ['x0_y1', 'x1_y0', 'x25_y1'],
    "logp06": ['x0_y1', 'x1_y0', 'x25_y1']
}

print("Starting Universal Merge for all PURE++ Benchmarks...\n")

for prop in PROPERTIES:
    print(f"\n" + "#"*60)
    print(f" PROPERTY: {prop.upper()}")
    print("#"*60)
    
    prop_path = os.path.join(BASE_PATH, prop)
    if not os.path.exists(prop_path):
        print(f"Directory not found for {prop}, skipping...")
        continue

    combos = COMBOS_MAP.get(prop, [])
    
    for combo in combos:
        metrics_list = []
        sr_list = []
        
        # Search for pickles in all chunk subfolders for this property
        search_pattern = os.path.join(prop_path, "*chunk*", f"results_{combo}.pickle")
        files = glob.glob(search_pattern)
        
        for f in files:
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
            
            # Count total molecules processed
            total_mols = len(metrics_list) * 25 
            
            print(f"\n--- Scenario: {combo} (Total Targets: {total_mols}) ---")
            print("METRICS:")
            print(final_metrics.to_string(index=False))
            print("SUCCESS RATES:")
            print(final_sr.to_string(index=False))
        else:
            print(f"\n[!] No data found for {prop} scenario {combo}")

print("\nAll properties merged successfully.")
