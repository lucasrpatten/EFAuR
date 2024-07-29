# EFAuR
Embeddings For Authorship using RoBERTa

- [EFAuR](#efaur)
- [Dataset](#dataset)
  - [Dataset Collection Process](#dataset-collection-process)
- [Models](#models)
  - [Activations and Definitions](#activations-and-definitions)

# Dataset
The dataset used in this project consists of public domain books written in English, with a single, known author, sourced from Project Gutenberg.

## Dataset Collection Process
1. Mirror all *.txt* files from a Project Gutenberg Mirror
2. Download the metadata tarball from gutenberg.org
3. Extract the metadata tarball and parse all the RDF metadata files within
    - Select English books labeled for public domain use with a single, known author
    - Write the ID of each book, along with the corresponding author, to a CSV file
4. Combine all books written by an author into one file
    - Combine all books by the same author into a single file
    - Remove Project Gutenberg headers and footers
    - Remove readability formatting that may obscure the texts
5. Create a dataset directory
    - Create subdirectories for train, test, and val
    - Each subdirectory has files containing a single data point for the model
    - Each subdirectory has text1, text2, and a label specifying same or different authorship
      - *label is 0 if same authorship, 1 if different*

# Models

## Activations and Definitions