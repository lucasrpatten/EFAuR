#!/bin/bash
# License: GNU GENERAL PUBLIC LICENSE, Version 3.0
# Script to download the Gutenberg dataset utilizing https://github.com/pgcorpus/gutenberg


dataset_location="../compute/gutenberg/data"
repo_location="../compute/gutenberg"

# helpFunction() {
#     echo ""
#     echo "Usage: $0 -d <dataset_location> -g <repo_location>"
#     echo "  -d Location to save the dataset [Default: $dataset_location]"
#     echo "  -g Location to save the repo [Default: $repo_location]"
#     exit 0
# }

# while getopts "d:g:h:u" opt
# do
#     case "$opt" in
#         d) dataset_location="$OPTARG" ;;
#         g) repo_location="$OPTARG" ;;
#         h) helpFunction ;; # Print helpFunction in case parameter is non-existent
#     esac
# done

git clone https://github.com/pgcorpus/gutenberg.git "$repo_location/"
mv "$repo_location"/data/ "$dataset_location"/

python3 "$repo_location"/get_data.py
python3 "$repo_location"/process_data.py
mv "$repo_location"/data/ "$dataset_location"/
mv "$repo_location"/metadata/ "$dataset_location"/metadata/
