import os
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class AuthorshipPairDataset(Dataset):
    """Authorship Pair Dataset

    Args:
        text_data1 (list[str]): OG Author Texts
        text_data2 (list[str]): Unknown Author Text
        labels (list[int]): Same or Different Booleans (1 or 0)
    """

    def __init__(
        self,
        text_dir: str,
        total_length: int,
    ) -> None:
        self.files = [x for x in os.listdir(text_dir) if x.endswith(".csv")]
        self.length = total_length
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        max_length = 170
        text1 = self.text_data1[idx]
        text2 = self.text_data2[idx]
        inputs1 = tokenizer(
            text1,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        inputs2 = tokenizer(
            text2,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        label = torch.tensor(self.labels[idx])

        return inputs1, inputs2, label