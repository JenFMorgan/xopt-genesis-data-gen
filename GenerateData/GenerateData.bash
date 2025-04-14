#!/bin/bash -x
#SBATCH --account=ad:beamphysics           # Adjust your account name if needed
#SBATCH --partition=roma                 # Partition name
#SBATCH --job-name=generate_data_job            # Job name
#SBATCH --output=generate_data_output.log       # Standard output
#SBATCH --error=generate_data_job_%j.err        # Error log (%j inserts job ID)
#SBATCH --ntasks=1                         # Single task
#SBATCH --cpus-per-task=10 # Number of CPU cores per task
#SBATCH --mem-per-cpu=4g
#SBATCH --time=0-96:00:00                  # Time limit, adjust if necessary

# Load the necessary modules (adjust the Python version as necessary)
module load python/3.x                     # Load Python module

# Activate your Python environment (adjust the path to your conda environment)
source /sdf/home/j/jmorgan/conda.sh        # Activate conda environment

conda activate ~/conda/envs/Xopt_3.10/

# Set environment variables (adjust paths as needed)
export LCLS_LATTICE=/sdf/home/j/jmorgan/lclsrepo_dontsharethese/lcls-lattice
export SCRATCH=/sdf/scratch/users/j/jmorgan
export PMIX_MCA_psec=^munge
# Execute the Python script with the neural network model
python GenerateData.py


# Notify when the job is completed
echo "Neural network training completed."

