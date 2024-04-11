import torch
import torch.nn as nn

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


class SiameseAuthorshipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.ff_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            Swish(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, input1, input2) -> torch.Tensor:
        input1_ids = input1["input_ids"].squeeze(1)
        input1_mask = input1["attention_mask"].squeeze(1)

        input2_ids = input2["input_ids"].squeeze(1)
        input2_mask = input2["attention_mask"].squeeze(1)

        cls_embedding_1 = self.roberta(
            input_ids=input1_ids, attention_mask=input1_mask
        )["last_hidden_state"][:, 0, :] # type: ignore
        cls_embedding_2 = self.roberta(
            input_ids=input2_ids, attention_mask=input2_mask
        )["last_hidden_state"][:, 0, :] # type: ignore

        output1 = self.ff_layers(cls_embedding_1)
        output2 = self.ff_layers(cls_embedding_2)

        return torch.cosine_similarity(output1, output2, dim=1)
