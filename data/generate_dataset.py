"""
Contains functions for populating a directory with random pairs of texts.
These pairs are used to train a model.

Author: Lucas Patten
"""

import json
import os
import random
import pickle
import re

from concurrent.futures import ThreadPoolExecutor


def authorship_split(
    authorship_dir: str,
    dataset_dir: str,
    train_percent: float,
    val_percent: float,
    test_percent: float,
) -> tuple[list[str], list[str], list[str]]:
    """Selects which authors to use for training, validation, and testing.
    Saves these lists in pickle files in the dataset directory

    Args:
        authorship_dir (str): The directory containing the authorship data
        dataset_dir (str, optional): The directory to save the dataset.
        train_percent (float, optional): Percent of authors to use for training. Defaults to 0.8.
        val_percent (float, optional): Percent of authors to use for validation. Defaults to 0.1.
        test_percent (float, optional): Percent of authors to use for testing. Defaults to 0.1.

    Returns:
        tuple[list[str], list[str], list[str]]: train_authors, val_authors, test_authors
    """
    authors = [
        f[:-4]  # Remove .txt for all authors in the directory
        for f in os.listdir(authorship_dir)
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

    with open(os.path.join(dataset_dir, "train_authors.pkl"), "wb") as f:
        pickle.dump(train_authors, f)

    with open(os.path.join(dataset_dir, "val_authors.pkl"), "wb") as f:
        pickle.dump(val_authors, f)

    with open(os.path.join(dataset_dir, "test_authors.pkl"), "wb") as f:
        pickle.dump(test_authors, f)

    return train_authors, val_authors, test_authors


# I realize this is really ineffecient due to repeated calls
# I just don't particularly care to optimize it
def split_text(author: str, authorship_dir: str) -> list:
    """Splits a text by sentence (splits on . ! ?)

    Args:
        author (str): author to split
        authorship_dir (str): The directory containing the authorship data

    Returns:
        list: list of sentences
    """
    file_path = os.path.join(authorship_dir, f"{author}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(r"(\.|\!|\?)", text)
    sentences = [s.strip() + p for s, p in zip(sentences[0::2], sentences[1::2])]

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
    i = 1
    while (
        len(text.split()) < text_len
        and text_idx + i < len(texts)
        and len(text.split() + texts[text_idx + i].split()) <= text_len
    ):
        text += " " + texts[text_idx + i]
        i += 1
    return text


def get_random_pair(
    authorship_dir: str, pairs_dir: str, pair_number: int, authors: list
) -> None:
    """Gets a random pair of texts and writes it to a file in pairs_dir

    Args:
        authorship_dir (str): Directory containing texts by author
        pairs_dir (str): Directory to write pairs to (dataset)
        pair_number (int): The number of the pair
        authors (list): List of authors to choose from
    """

    label = random.randint(0, 1)
    # same author
    if label == 0:
        texts = split_text(random.choice(authors), authorship_dir)
        text1 = get_single_data(texts)
        text2 = get_single_data(texts)

    # different author
    else:
        author1 = random.choice(authors)
        author2 = random.choice(authors)
        while author1 == author2:
            author2 = random.choice(authors)
        texts1 = split_text(author1, authorship_dir)
        texts2 = split_text(author2, authorship_dir)
        text1 = get_single_data(texts1)
        text2 = get_single_data(texts2)

    pair = {"text1": text1, "text2": text2, "label": label}
    with open(os.path.join(pairs_dir, f"{pair_number}"), "w", encoding="latin-1") as f:
        json.dump(pair, f)


def generate_pairs(
    authorship_dir: str,
    dataset_dir: str,
    dataset_name: str,
    dataset_size: int,
    authors: list,
    num_processes: int = 16,
):
    """Generates a dataset of random pairs

    Args:
        authorship_dir (str): Directory containing texts by author
        dataset_dir (str): Directory to write datasets to
        dataset_name (str): Name of the dataset
        dataset_size (int): Number of pairs to generate
        authors (list): List of authors to choose from
        num_processes (int, optional): Number of processes to use. Defaults to 16.
    """
    pairs_dir = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(pairs_dir):
        os.makedirs(pairs_dir)

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        for i in range(dataset_size):
            executor.submit(get_random_pair, authorship_dir, pairs_dir, i, authors)


def generate_dataset(
    data_dir: str = "/home/lucasrp/compute/tmp/data",
    dataset_size: int = 100000,
    train_percent: float = 0.8,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
    num_processes: int = 16,
):
    """Generates three datasets for training, validation, and testing

    Args:
        data_dir (str, optional): Base path of data dir. Defaults to "/home/lucasrp/compute/tmp/data".
        dataset_size (int, optional): Total number of pairs for all datasets. Defaults to 100000.
        train_percent (float, optional): Percent of dataset to use for training. Defaults to 0.8.
        val_percent (float, optional): Percent of dataset to use for validation. Defaults to 0.1.
        test_percent (float, optional): Percent of dataset to use for testing. Defaults to 0.1.
        num_processes (int, optional): Number of processes to use. Defaults to 16.
    """
    assert train_percent + val_percent + test_percent == 1

    authorship_dir = os.path.join(data_dir, "by_author")
    dataset_dir = os.path.join(data_dir, "dataset")
    train_authors, val_authors, test_authors = authorship_split(
        authorship_dir, dataset_dir, train_percent, val_percent, test_percent
    )

    generate_pairs(
        authorship_dir, dataset_dir, "train", dataset_size, train_authors, num_processes
    )
    generate_pairs(
        authorship_dir, dataset_dir, "val", dataset_size, val_authors, num_processes
    )
    generate_pairs(
        authorship_dir, dataset_dir, "test", dataset_size, test_authors, num_processes
    )
