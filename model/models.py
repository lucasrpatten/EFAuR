"""
Contains Models for EFAuR (Authorship Embedding, Siamese Authorship Model)

Author: Lucas Patten
"""

from typing import Tuple
import torch

from torch import nn
from transformers import RobertaModel

num1 = None
num2 = None

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

    Args:
        pooling_method (str, optional): Pooling method. Defaults to "max".
        activation_function (str, optional): Activation function. Defaults to "relu".

    Raises:
        ValueError: If activation function or pooling method is not recognized
    """

    def __init__(self, pooling_method: str = "max", activation_function: str = "relu"):
        super().__init__()
        if activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation_function == "swish":
            self.activation = Swish()
        else:
            raise ValueError(
                f"Activation function {activation_function} not recognized. "
                + "Choose from 'relu', 'swish', 'leakyrelu'"
            )
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.ff_layers = nn.Sequential(
            nn.Linear(1024, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
        )

        self.pooling_method = pooling_method
        if pooling_method == "mean":
            self.pooling = self._mean_pooling
        elif pooling_method == "max":
            self.pooling = self._max_pooling
        elif pooling_method == "attention":
            self.attention_weights = nn.Linear(1024, 1)
            self.pooling = self._attention_pooling
        elif pooling_method == "cls":
            self.pooling = self._cls_embedding
        else:
            raise ValueError(
                f"Pooling method {pooling_method} not recognised. "
                + "Choose from 'attention', 'cls', 'mean', 'max'"
            )
        self.roberta.eval()

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass

        Args:
            data (dict): Tokenized input data

        Returns:
            torch.Tensor: Authorship embedding
        """
        input_ids = data["input_ids"].squeeze(1)
        input_mask = data["attention_mask"].squeeze(1)
        for param in self.roberta.parameters():
            param.requires_grad = True
        token_embeddings = self.roberta(
            input_ids=input_ids, attention_mask=input_mask  # type: ignore
        )["last_hidden_state"]

        pooled_embeddings = self.pooling(token_embeddings, input_mask)

        authorship_embedding = self.ff_layers(pooled_embeddings)
        return authorship_embedding

    def _cls_embedding(self, token_embeddings, _) -> torch.Tensor:
        return token_embeddings[:, 0, :]

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = attention_mask.float()
        input_mask_expand = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expand, 1)
        sum_mask = torch.clamp(input_mask_expand.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _max_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = attention_mask.float()
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    def _attention_pooling(self, token_embeddings: torch.Tensor, _) -> torch.Tensor:
        attention_scores = torch.nn.functional.softmax(
            self.attention_weights(token_embeddings), dim=1
        )
        return torch.sum(token_embeddings * attention_scores, dim=1)


class SiameseAuthorshipModel(nn.Module):
    """Siamese Authorship Model For Training

    Args:
        pooling_method (str, optional): Pooling method. Defaults to "max".
        activation_function (str, optional): Activation function. Defaults to "relu".
    """

    def __init__(self, pooling_method: str = "max", activation_function: str = "relu"):
        super().__init__()
        self.embedding_model = AuthorshipEmbeddingModel(
            pooling_method, activation_function
        )

    def forward(self, input1, input2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            input1 (dict): Tokenized input data 1
            input2 (dict): Tokenized input data 2

        Returns:
            torch.Tensor: Similarity score
        """
        self.embedding_model.eval()
        embedding1 = self.embedding_model(input1)
        embedding2 = self.embedding_model(input2)
        #print idx's where num1 and num2 are different
        return embedding1, embedding2
