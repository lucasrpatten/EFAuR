"""
Combines all (English) books by author into one file per author
"""

import os
import pandas as pd


def by_author(data_path: str = "../compute/gutenberg/data") -> None:
    """Gets all English books, and writes them to a file with the author of the book as the title

    Args:
        data_path (str, optional): The path to the data dir. Default: "../compute/gutenberg/data".
    """
    meta_path = os.path.join(data_path, "metadata", "metadata.csv")
    authorship_path = os.path.join(data_path, "en_by_author")
    if not os.path.exists(authorship_path):
        os.mkdir(authorship_path)
    df = pd.read_csv(meta_path)
    en_df = df[(df['type'] == 'Text') & (df['language'].str.contains('en')) & (~df['author'].isin(['Anonymous', 'Unknown', 'Various'])) & (df['author'].notna())]
    print(en_df)
    authors = set()
    for _, row in en_df.iterrows():
        author = row["author"]
        print(author)
        if not os.path.exists(os.path.join(data_path, "text", row["id"] + "_text.txt")):
            continue
        with open(
            os.path.join(data_path, "text", row["id"] + "_text.txt"), "r", encoding="utf-8"
        ) as f:
            text = f.read()
        if author not in authors:
            authors.add(author)
            with open(
                os.path.join(authorship_path, str(author) + ".txt"), "w", encoding="utf-8"
            ) as f:
                f.write(text + "\n")
        else:

            with open(
                os.path.join(authorship_path, str(author) + ".txt"), "a", encoding="utf-8"
            ) as f:
                f.write(text + "\n")


if __name__ == "__main__":
    by_author()
