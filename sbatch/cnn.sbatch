#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=deepchess_cnn        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=log/deepchess_cnn_%j.log      # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=30:00:00              # Max execution time
#

cd $SLURM_SUBMIT_DIR
# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/theotaj/anaconda3/envs/deepchess

# Run your Python script
python src/train_CNN.py \
    --lr 0.0001 \
    --dropout 0.05 \
    --alpha 1.0 \
    --model_name "CNN_5M" \
    --conv_filters 32 64 128 \
    --fc_layers 1024 512 256 \
    --epochs 100 \
    --patience 10 \
    --batch_size 128 \
    --dataset_path "data/dataset_5000000.parquet" \
    --scheduler_factor 0.5 \
    --scheduler_patience 4