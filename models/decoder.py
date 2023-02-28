# Codice contenente la classe Decoder Layer che implementa il decoder layer
# e Decoder che implementa il decoder
from torch import Tensor, nn
from models.multi_head_attention import MultiHeadAttention
from utilities.residual import Residual
from utilities.feed_forward import FeedForward
from utilities.positional_encoding import position_encoding

# La classe DecoderLayer è molto simile a quella dell'encoder
# La differenza è che il decoder layer ha due multi-head attention invece di una,
# con la seconda che prende in input `memory` per due dei suoi input
class DecoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            FeedForward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(tgt, memory, memory)
        return self.feed_forward(tgt)


class Decoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
             for _ in range(num_layers)]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.linear(tgt)