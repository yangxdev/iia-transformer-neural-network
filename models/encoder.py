# Codice contenente la classe EncoderLayer e Encoder che implementano rispettivamente il layer encoder e l'encoder
import torch
from torch import Tensor, nn
from models.multi_head_attention import MultiHeadAttention
from utilities.residual import Residual
from utilities.feed_forward import FeedForward
from utilities.positional_encoding import position_encoding

# Si definisce la classe EncoderLayer che implementa il layer encoder,
#  che è composto da un multi-head attention e da un feed forward


class EncoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        # Si definisce il multi-head attention
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        # Si definisce il feed forward
        self.feed_forward = Residual(
            FeedForward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        # Si passa il tensore di input attraverso il multi-head attention
        src = self.attention(src, src, src)
        return self.feed_forward(src)

# Si definisce la classe Encoder che implementa l'encoder,
# che è composto da un numero di layer encoder


class Encoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        # layers: lista di layer encoder
        self.layers = nn.ModuleList(
            [EncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
             for _ in range(num_layers)]
        )

    # Si aggiunge il positional encoding al tensore di input
    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        # Si passa il tensore di input attraverso i layer encoder
        for layer in self.layers:
            src = layer(src)

        return src