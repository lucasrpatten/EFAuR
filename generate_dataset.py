"""
Combines all (English) books by author into one file per author
"""

import os
import random
import re
import pandas as pd
from multiprocessing import Pool


def group_by_author(data_path: str = "../compute/gutenberg/data") -> None:
    """Gets all English books, and writes them to a file with the author of the book as the title

    Args:
        data_path (str, optional): The path to the data dir. Default: "../compute/gutenberg/data".
    """
    meta_path = os.path.join(data_path, "metadata", "metadata.csv")
    authorship_path = os.path.join(data_path, "en_by_author")
    if not os.path.exists(authorship_path):
        os.mkdir(authorship_path)
    df = pd.read_csv(meta_path)
    en_df = df[
        (df["type"] == "Text")
        & (df["language"].str.contains("en"))
        & (~df["author"].isin(["Anonymous", "Unknown", "Various"]))
        & (df["author"].notna())
    ]
    print(en_df)
    authors = set()
    for _, row in en_df.iterrows():
        author = row["author"]
        print(author)
        if not os.path.exists(os.path.join(data_path, "text", row["id"] + "_text.txt")):
            continue
        with open(
            os.path.join(data_path, "text", row["id"] + "_text.txt"),
            "r",
            encoding="utf-8",
        ) as f:
            text = f.read()
        if author not in authors:
            authors.add(author)
            with open(
                os.path.join(authorship_path, str(author) + ".txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text + "\n")
        else:

            with open(
                os.path.join(authorship_path, str(author) + ".txt"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(text + "\n")


# pylint: disable=dangerous-default-value
def split_text(
    author: str, data_path: str = "../compute/gutenberg/data", memo={}
) -> list:
    """Splits a text by sentence (splits on . ! ?)

    Args:
        author (str): author to split
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data"
        memo (dict, optional): Memo Dict. Defaults to {}.

    Returns:
        list: list of sentences
    """
    file_path = os.path.join(data_path, "en_by_author", f"{author}.txt")
    if not os.path.exists(file_path):
        file_path = os.path.join(data_path, "en_by_author", f"{author}..txt")
    if file_path in memo:
        return memo[file_path]
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(r"(?<=[.!?]) +", text)
    sentences = [s.strip() + p for s, p in zip(sentences, re.findall(r"[.!?]", text))]
    memo[file_path] = sentences
    return sentences


def get_single_data(texts: list, max_length=512) -> str:
    """Gets a single data point (1/2 of a pair)

    Args:
        texts (list): list of texts to choose from
        max_length (int, optional): Max string length. Defaults to 512.

    Returns:
        str: Single text chunk of length 12 <= length <= max_length
    """
    text_len = random.randint(12, max_length)
    text_idx = random.randint(0, len(texts))
    text = texts[text_idx]
    while (
        len(texts[text_idx].split()) < text_len
        and text_idx + 1 < len(texts)
        and len(texts[text_idx].split() + texts[text_idx + 1].split()) <= text_len
    ):
        text += " " + texts[text_idx + 1]
    return text


def get_random_pair(
    authors: list,
    data_path: str = "../compute/gutenberg/data",
) -> tuple[str, str, int]:
    """Gets a random pair of texts

    Args:
        authors (list): list of authors to choose from
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data"

    Returns:
        (tuple[str, str, int]): text1, text2, boolean label (0 is same, 1 is different)
    """

    # same author
    if random.randint(0, 1) == 0:
        texts = split_text(random.choice(authors), data_path)
        text1 = get_single_data(texts)
        text2 = get_single_data(texts)
        return text1, text2, 0

    # different author
    author1 = random.choice(authors)
    author2 = random.choice(authors)
    while author1 == author2:
        author2 = random.choice(authors)
    texts1 = split_text(author1, data_path)
    texts2 = split_text(author2, data_path)
    text1 = get_single_data(texts1)
    text2 = get_single_data(texts2)
    return text1, text2, 1


def generate_segment(
    segment_index,
    segment_count,
    split_size,
    leftover,
    authors,
    dataset_type,
    dataset_path,
    data_path,
):
    df = pd.DataFrame(columns=["text1", "text2", "label"])
    segment_size = split_size if segment_index != segment_count - 1 else leftover
    for _ in range(segment_size):
        text1, text2, label = get_random_pair(authors, data_path)
        df = pd.concat(
            [
                df,
                pd.DataFrame({"text1": [text1], "text2": [text2], "label": [label]}),
            ]
        )
    file_path = os.path.join(dataset_path, f"{dataset_type}_{segment_index}.csv")
    df.to_csv(file_path, index=False)


def generate_dataset(
    dataset_type: str,
    dataset_size: int,
    authors: list,
    split_size: int = 1000,
    data_path: str = "../compute/gutenberg/data",
    num_processes=os.cpu_count(),
) -> None:
    """Generates a dataset of random pairs
    For each pair:
        text1, text2, label
    Saved into csv files of size split_size

    Args:
        dataset_type (str): Dataset type ("train", "val", or "test")
        dataset_size (int): Number of pairs to generate for the dataset.
        authors (list): List of authors to choose from.
        split_size (int, optional): Number of pairs per segment file. Defaults to 1000
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data"
    """
    segment_count = dataset_size // split_size
    leftover = dataset_size % split_size
    if leftover != 0:
        segment_count += 1
    dataset_path = os.path.join(data_path, dataset_type)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    with Pool(num_processes) as pool:
        pool.starmap(
            generate_segment,
            [
                (
                    i,
                    segment_count,
                    split_size,
                    leftover,
                    authors,
                    dataset_type,
                    dataset_path,
                    data_path,
                )
                for i in range(segment_count)
            ],
        )


def authorship_split(
    dataset_size: int = 100000,
    train_percent: float = 0.8,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
    data_path: str = "../compute/gutenberg/data",
) -> None:
    """Generates three datasets for training, validation, and testing

    Args:
        dataset_size (int, optional): Total number of pairs. Defaults to 100000.
        train_percent (float, optional): Percent of dataset to use for training. Defaults to 0.8.
        val_percent (float, optional): Percent of dataset to use for validation. Defaults to 0.1.
        test_percent (float, optional): Percent of dataset to use for testing. Defaults to 0.1.
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data"
    """
    authorship_path = os.path.join(data_path, "en_by_author/")
    authors = [
        f[:-5] if f.endswith("..txt") else f[:-4] for f in os.listdir(authorship_path)
    ]
    author_count = len(authors)
    train_size = int(train_percent * author_count)
    val_size = int(val_percent * author_count)
    test_size = int(test_percent * author_count)

    remaining_authors = author_count - (train_size + val_size + test_size)
    train_size += remaining_authors

    random.shuffle(authors)
    train_authors = authors[:train_size]
    val_authors = authors[train_size : train_size + val_size]
    test_authors = authors[train_size + val_size :]
    generate_dataset(
        "train", int(dataset_size * train_percent), train_authors, 10000, data_path
    )
    generate_dataset(
        "val", int(dataset_size * val_percent), val_authors, 10000, data_path
    )
    generate_dataset(
        "test", int(dataset_size * test_percent), test_authors, 10000, data_path
    )


if __name__ == "__main__":
    # group_by_author()
    authorship_split()
