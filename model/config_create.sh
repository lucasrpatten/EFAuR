#!/bin/bash

# Define learning rates, activation functions, and pooling methods
# Change these values to vary the configuration
#!TODO Find the best learning rate using binary search
learning_rates=("0.00025") #"0.0002" "0.0005")
#!TODO Find the best activation function
activations=("relu" "leakyrelu" "swish")
#!TODO Find the best pooling method (probably attention)
pooling_methods=("attention") # "mean" "max"

# Define the output configuration file
config_file="job_array_config.txt"

# Clear the configuration file if it exists
> $config_file

# Initialize index counter
index=0

# Create every combination of learning rates, activations, and pooling methods
for lr in "${learning_rates[@]}"; do
  for act in "${activations[@]}"; do
    for pool in "${pooling_methods[@]}"; do
      echo "$index $lr $act $pool" >> $config_file
      index=$((index + 1))
    done
  done
done

echo "Configuration file '$config_file' has been created with all combinations."
