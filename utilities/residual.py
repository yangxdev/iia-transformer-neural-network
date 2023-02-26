# Codice contenente la classe Residual
import torch
from torch import Tensor, nn

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        # sublayer: il modulo sub-layer che viene passato
        # norm: layer di normalizzazione
        # dropout: layer di dropout
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # *tensors: è un argomento variabile, quindi può essere passato un numero variabile di tensori
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

