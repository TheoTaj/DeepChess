#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=deepchess_cnn        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=log/deepchess_cnn_%j.log      # Log-file (important!)
#SBATCH --cpus-per-task=8          # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --exclude=compute-04
#SBATCH --time=60:00:00              # Max execution time
#

cd $SLURM_SUBMIT_DIR
# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/theotaj/anaconda3/envs/deepchess

# Run your Python script
python src/train_CNN.py --lr 0.00005 --dropout 0.05 --alpha 1.0 --model_name "CNN_5M_5" --conv_filters 64 128 256 --fc_layers 512 256 128 --epochs 200 --patience 15 --batch_size 512 --dataset_path "data/kaggle_5M_300.parquet" --K 300 --scheduler_factor 0.5 --scheduler_patience 5