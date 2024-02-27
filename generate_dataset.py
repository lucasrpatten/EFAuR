"""
Combines all (English) books by author into one file per author
"""

import os
import random
import re
import pickle
import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import Dataset
import torch

random.seed(42)

class AuthorshipPairDataset(Dataset):
    """Authorship Pair Dataset

    Args:
        text_data1 (list[torch.Tensor]): OG Author Text
        text_data2 (list[torch.Tensor]): Other Author Text
        labels (list[int]): Same or Different Booleans (0 is same)
    """

    def __init__(
        self,
        text_data1: list[torch.Tensor],
        text_data2: list[torch.Tensor],
        labels: list[list[int]],
    ) -> None:
        self.text_data1 = text_data1
        self.text_data2 = text_data2
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        inputs1 = self.text_data1[idx]
        inputs2 = self.text_data2[idx]
        label = torch.tensor(self.labels[idx])

        return inputs1, inputs2, label


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

#pylint: disable=dangerous-default-value
def split_text(
    author: str, data_path: str = "../compute/gutenberg/data", memo={}
) -> list:
    """Splits a text by sentence (splits on . ! ?)

    Args:
        author (str): author to split
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data".
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


def get_single_data(texts: list, tokenizer: RobertaTokenizer, max_length=512):
    """ Gets a single data point (1/2 of a pair)

    Args:
        texts (list): list of texts to choose from
        tokenizer (RobertaTokenizer): Roberta Tokenizer
        max_length (int, optional): Max string length. Defaults to 512.

    Returns:
        Batch Encoding: Tokenized text
    """
    text_len = random.randint(12, max_length)
    text_idx = random.randint(0, len(texts))
    text = texts[text_idx]
    while (
        len(texts[text_idx]) < text_len
        and text_idx + 1 < len(texts)
        and len(texts[text_idx] + texts[text_idx + 1]) <= text_len
    ):
        text += texts[text_idx + 1]
    text_tokenized = tokenizer(
        text, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    return text_tokenized


def get_random_pair(
    authors: list,
    tokenizer: RobertaTokenizer,
    data_path: str = "../compute/gutenberg/data",
):
    """Gets a random pair of texts

    Args:
        authors (list): list of authors to choose from
        tokenizer (RobertaTokenizer): Roberta Tokenizer
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data".

    Returns:
        (tuple[BatchEncoding, BatchEncoding, Literal[0]] | tuple[BatchEncoding, BatchEncoding, Literal[1]]): Text1 Tokenized, Text2 Tokenized, Label
    """
    # same author
    if random.randint(0, 1) == 0:
        texts = split_text(random.choice(authors), data_path)
        text1 = get_single_data(texts, tokenizer)
        text2 = get_single_data(texts, tokenizer)
        return text1, text2, 0
    # different author
    else:
        author1 = random.choice(authors)
        author2 = random.choice(authors)
        while author1 == author2:
            author2 = random.choice(authors)
        texts1 = split_text(author1, data_path)
        texts2 = split_text(author2, data_path)
        text1 = get_single_data(texts1, tokenizer)
        text2 = get_single_data(texts2, tokenizer)
        return text1, text2, 1


def generate_dataset(
    tokenizer: RobertaTokenizer,
    dataset_size: int = 80000,
    authors: list = [None],
    data_path: str = "../compute/gutenberg/data/",
) -> AuthorshipPairDataset:
    """Generates a dataset of random pairs

    Args:
        tokenizer (RobertaTokenizer): Roberta Tokenizer
        dataset_size (int, optional): Number of pairs to generate for the dataset. Defaults to 80000.
        authors (list, required): List of authors to choose from. Defaults to [None].
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data/".

    Returns:
        AuthorshipPairDataset: Dataset of random pairs
    """
    pairs = [
        get_random_pair(authors, tokenizer, data_path) for _ in range(dataset_size)
    ]
    text_data1, text_data2, labels = zip(*pairs)

    ds = AuthorshipPairDataset(text_data1, text_data2, labels)
    return ds


def authorship_split(
    dataset_size: int = 100000,
    train_percent: float = 0.8,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
    data_path: str = "../compute/gutenberg/data/",
) -> None:
    """Generates three datasets for training, validation, and testing

    Args:
        dataset_size (int, optional): Total number of pairs. Defaults to 100000.
        train_percent (float, optional): Percent of dataset to use for training. Defaults to 0.8.
        val_percent (float, optional): Percent of dataset to use for validation. Defaults to 0.1.
        test_percent (float, optional): Percent of dataset to use for testing. Defaults to 0.1.
        data_path (str, optional): The path to the data dir. Defaults to "../compute/gutenberg/data/".
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

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    train_ds = generate_dataset(
        tokenizer, int(dataset_size * train_percent), train_authors, data_path
    )
    val_ds = generate_dataset(
        tokenizer, int(dataset_size * val_percent), val_authors, data_path
    )
    test_ds = generate_dataset(
        tokenizer, int(dataset_size * test_percent), test_authors, data_path
    )

    with open(os.path.join(data_path, "datasets", "train.pkl"), "wb") as f:
        pickle.dump(train_ds, f)
    with open(os.path.join(data_path, "datasets", "val.pkl"), "wb") as f:
        pickle.dump(val_ds, f)
    with open(os.path.join(data_path, "datasets", "test.pkl"), "wb") as f:
        pickle.dump(test_ds, f)


if __name__ == "__main__":
    # group_by_author()
    authorship_split()
