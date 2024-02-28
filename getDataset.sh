#!/bin/bash
# License: GNU GENERAL PUBLIC LICENSE, Version 3.0
# Script to download the Gutenberg dataset utilizing https://github.com/pgcorpus/gutenberg


dataset_location="../compute/gutenberg/data"
repo_location="../compute/gutenberg"

git clone https://github.com/pgcorpus/gutenberg.git "$repo_location/"
mv "$repo_location"/data/ "$dataset_location"/

python3 "$repo_location"/get_data.py
python3 "$repo_location"/process_data.py
mv "$repo_location"/data/ "$dataset_location"/
mv "$repo_location"/metadata/ "$dataset_location"/metadata/