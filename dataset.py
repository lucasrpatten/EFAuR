"""
Pytorch Dataset For Authorship Pairs

Written by Lucas Patten
"""

import os
import json
import torch

from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class AuthorshipPairDataset(Dataset):
    """Authorship Pair Dataset

    Args:
        dataset_dir (str): Directory containing dataset
    """

    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.length = len(os.listdir(dataset_dir)) - 1
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        max_length = 512
        with open(
            os.path.join(self.dataset_dir, f"{idx}.json"), "r", encoding="latin-1"
        ) as f:
            data = json.load(f)
        text1 = data["text1"]
        text2 = data["text2"]
        label = int(data["label"])
        inputs1 = self.tokenizer(
            text1,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        inputs2 = self.tokenizer(
            text2,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        label = torch.tensor(label)

        return inputs1, inputs2, label
