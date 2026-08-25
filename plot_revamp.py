import pickle, glob, os, sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Pandas 2.0+ Compatibility Patch ---
import pandas.core.indexes.base as base
sys.modules['pandas.core.indexes.numeric'] = base
if not hasattr(base, 'Int64Index'): base.Int64Index = base.Index

path = "/fs/scratch/PAS2030/itsamrithaa/GAT_FULL_RESULTS/qed_chunk_*/results_x1_y1.pickle"
files = glob.glob(path)

all_data = []
print(f"Collecting data from {len(files)} chunks...")

for f_path in files:
    try:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
            df = data['df']
            # We want similarity vs property_target (QED)
            all_data.append(df[['similarity', 'property_target']])
    except:
        continue

plot_df = pd.concat(all_data)

# Create the plot
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Scatter plot with transparency to handle 16,000 points (mimics panel d)
sns.scatterplot(
    data=plot_df, 
    x='similarity', 
    y='property_target', 
    alpha=0.4, 
    edgecolor=None, 
    color='#66c2a5',
    s=10
)

plt.title("GAT Generative Distribution: QED vs. Similarity", fontsize=14)
plt.xlabel("Tanimoto Similarity to Target", fontsize=12)
plt.ylabel("QED Score", fontsize=12)
plt.xlim(0.3, 0.7) # Focus on the paper's 'Novelty Zone'
plt.ylim(0.2, 1.0)

plt.tight_layout()
plt.savefig("GAT_distribution_plot.png", dpi=300)
print("Plot saved as GAT_distribution_plot.png")
