# Codice contenente la classe FeedForward che implementa il feed forward
import torch
from torch import Tensor, nn

class FeedForward(nn.Module):
    def __init__(self, dim_input: int= 512, dim_feedforward: int = 2048):
        super().__init__()
        # Definiamo due layer lineari con attivazione ReLU in mezzo
        self.linear1 = nn.Linear(dim_input, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, dim_input)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
