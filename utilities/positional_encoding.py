import torch 
import torch.nn as nn
from torch import Tensor

def position_encoding(
    seq_len: int, 
    dim_model: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, -1, 1)
    print("pos: ", pos.shape)
    print("dim: ", dim.shape)
    phase = pos / (1e4 ** (dim / dim_model))
    # error: The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1
    # with pos:  torch.Size([1, 4, 1])
    #      dim:  torch.Size([1, 5, 1])
    # how do I fix this?

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))