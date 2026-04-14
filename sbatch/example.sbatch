#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=deepchess        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=outputs/dc-output.log      # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=1:00:00              # Max execution time
#

# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/<your_username>/anaconda3/envs/deepchess # CHANGE HERE

# Run your Python script
python test.py