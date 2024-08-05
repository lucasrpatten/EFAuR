#!/bin/bash --login

#SBATCH --time=32:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=6
#SBATCH --mem=30G   # memory
#SBATCH -J "EfaurTrain"   # job name
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --array=0-4   # Array range based on the number of configurations

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Load modules and activate environment
mamba activate efaur

# Read the configuration file
config_file="job_array_config.txt"
config=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $config_file)

# Extract parameters from the configuration line
IFS=' ' read -r index learning_rate activation pooling <<< "$config"

# Run the script with the specified parameters and redirect output and errors
torchrun --nproc_per_node=6 parser.py --batch_size=20 --epochs=32 --learning_rate="$learning_rate" --activation="$activation" --pooling="$pooling"
