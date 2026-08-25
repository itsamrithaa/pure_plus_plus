#!/bin/bash
#SBATCH --job-name=GAT_BENCH
#SBATCH --account=PAS2030
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40    
#SBATCH --gpus-per-node=1
#SBATCH --cluster=pitzer

# Pitzer-specific modules
module load miniconda3/24.1.2-py310
module load cuda/12.1.1

# Activate environment
source activate /users/PAS2030/itsamrithaa/envs/reactionrl_author

cd /users/PAS2030/itsamrithaa/ReactionRL
export PYTHONPATH=$PYTHONPATH:.

# CHUNK_NAME is e.g., drd2_chunk_aa
CHUNK_NAME=$(basename $1)

# Extract the base property (drd2, logp04, etc) for the output path
BASE_PROP=$(echo $CHUNK_NAME | cut -d'_' -f1)

MODEL_PATH="/users/PAS2030/itsamrithaa/ReactionRL/output/supervised/actor-critic/steps=5_actor_loss=PG_neg=combined_seed=42/model.pth"

echo "Processing $BASE_PROP | Chunk: $CHUNK_NAME"

# We use the scratch space but organized by property
OUTPUT_DIR="/fs/scratch/PAS2030/itsamrithaa/GAT_FULL_RESULTS/$BASE_PROP/$CHUNK_NAME"
mkdir -p "$OUTPUT_DIR"

# Run the evaluation
python -m reactionrl.scripts.evaluate \
    --model-path "$MODEL_PATH" \
    --property "$CHUNK_NAME" \
    --cuda 0 \
    --num-workers 40 \
    --output-dir "$OUTPUT_DIR"

echo "Finished $CHUNK_NAME"
