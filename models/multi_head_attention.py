# Codice contenente la classe MultiHeadAttention
import torch
from torch import Tensor, nn
from models.attention_head import AttentionHead

class MultiHeadAttention(nn.Module):
    # num_heads rappresenta il numero di attention head che compongono la multi-head attentio, con i seguenti parametri:
    # - dim_in rappresenta la dimensione delle feature nel tensore di input passato attraverso questo modulo.
    # - dim_q e dim_k rappresentano le dimensioni, rispettivamente, della query e della key.
    # 
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        # self.heads è una lista di `num_heads` attention head.
        # ModuleList è un container di PyTorch che contiene una lista di moduli.
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            # torch.cat concatena i tensori lungo la dimensione specificata.
            torch.cat([head(query, key, value) for head in self.heads], dim=-1)
        )
