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
#SBATCH --time=03:00:00              # Max execution time
#

cd $SLURM_SUBMIT_DIR
# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/theotaj/anaconda3/envs/deepchess

# Run your Python script
python src/train_CNN.py \
    --lr 0.0005 \
    --dropout 0.4 \
    --alpha 1.0 \
    --model_name "CNN_TUNE_36" \
    --conv_filters 32 64 128 \
    --fc_layers 512 256