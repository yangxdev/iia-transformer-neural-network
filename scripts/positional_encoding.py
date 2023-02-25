import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, dim_model: int):
        # seq_len: lunghezza della sequenza
        # dim_model: dimensione del modello
        super().__init__()
        self.pos_encoding = self._create_positional_encoding(seq_len, dim_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding[:, :x.size(1), :]
    
    def _create_positional_encoding(self, seq_len: int, dim_model: int) -> torch.Tensor:
        # reshape: trasforma il tensore in un tensore di dimensione (1, 1, dim_model)
        pos = torch.arange(seq_len, dtype=torch.float).reshape(1, -1, 1)
        dim = torch.arange(dim_model, dtype=torch.float).reshape(1, 1, -1)
        phase = pos / (1e4 ** (dim / dim_model))
        sin = torch.sin(phase)
        cos = torch.cos(phase)

        # viene usato torch.where per scegliere tra sin e cos per ciascuna posizione nella sequenza in base al valore di dim
        pos_encoding = torch.where(dim.long() % 2 == 0, sin, cos)
        pos_encoding = pos_encoding.reshape(1, seq_len, -1)
        return pos_encoding