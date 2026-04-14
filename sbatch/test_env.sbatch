#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=deepchess_test_env        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=log/deepchess_%j.log      # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=00:10:00              # Max execution time
#

cd $SLURM_SUBMIT_DIR
# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/theotaj/anaconda3/envs/deepchess # CHANGE HERE

# Run your Python script
python test.py