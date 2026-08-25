#!/bin/bash
#SBATCH --job-name=GAT_CHUNK
#SBATCH --account=PAS2030
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40    
#SBATCH --gpus-per-node=1
#SBATCH --cluster=pitzer

module load miniconda3/24.1.2-py310
module load cuda/12.3.2
source activate /users/PAS2030/itsamrithaa/envs/reactionrl_author

cd /users/PAS2030/itsamrithaa/ReactionRL
export PYTHONPATH=$PYTHONPATH:.

# Variable passed from the loop (e.g., qed_chunk_aa)
CHUNK_NAME=$(basename $1)

MODEL_PATH="/users/PAS2030/itsamrithaa/ReactionRL/output/supervised/actor-critic/steps=5_actor_loss=PG_neg=combined_seed=42/model.pth"

echo "Evaluating Chunk: $CHUNK_NAME"

# The script will look in datasets/coma/qed_chunk_aa/rdkit_test.txt
python -m reactionrl.scripts.evaluate \
    --model-path "$MODEL_PATH" \
    --property "$CHUNK_NAME" \
    --cuda 0 \
    --num-workers 40 \
    --output-dir "/fs/scratch/PAS2030/itsamrithaa/GAT_FULL_RESULTS/${CHUNK_NAME}"

echo "Chunk $CHUNK_NAME finished."
