# Codice contenente la classe AttentionHead
import torch
from torch import Tensor, nn
from utilities.scaled_dot_product_attention import ScaledDotProductAttention

# Una multi-head attention è composta da varie attention head identiche.
# Ciascuna attention head contiene 3 layer lineari, seguite da una scaled dot-product attention.
class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # the following line has been changed after the modification of the scaled_dot_product_attention function
        return ScaledDotProductAttention.forward(self, self.q(query), self.k(key), self.v(value))
    
# dim_in rappresenta la dimensione delle feature nel tensore di input passato attraverso questo modulo.