"""
Contains Models for EFAuR (Authorship Embedding, Siamese Authorship Model)

Author: Lucas Patten
"""

import torch

from torch import nn
from transformers import RobertaModel


class Swish(nn.Module):
    """Swish layer defined by x*sigmoid(x)"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns x*sigmoid(x)

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        sigmoid = 1 / (1 + torch.exp(-x))
        return x * sigmoid


class AuthorshipEmbeddingModel(nn.Module):
    """
    Authorship Embedding Model
    Generates an embedding for the authorship of the book
    """

    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.ff_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            Swish(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass

        Args:
            data (dict): Tokenized input data

        Returns:
            torch.Tensor: Authorship embedding
        """
        input_ids = data["input_ids"].squeeze(1)
        input_mask = data["attention_mask"].squeeze(1)

        cls_embed = self.roberta(input_ids=input_ids, attention_mask=input_mask)[  # type: ignore
            "last_hidden_state"
        ][
            :, 0, :
        ]

        authorship_embedding = self.ff_layers(cls_embed)
        return authorship_embedding


class SiameseAuthorshipModel(nn.Module):
    """Siamese Authorship Model For Training"""

    def __init__(self):
        super().__init__()
        self.embedding_model = AuthorshipEmbeddingModel()

    def forward(self, input1, input2) -> torch.Tensor:
        """Forward pass

        Args:
            input1 (dict): Tokenized input data 1
            input2 (dict): Tokenized input data 2

        Returns:
            torch.Tensor: Similarity score
        """
        embedding1 = self.embedding_model(input1)
        embedding2 = self.embedding_model(input2)

        return torch.cosine_similarity(embedding1, embedding2, dim=1)
