import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        temp = query.bmm(key.transpose(1, 2))
        scale = query.size(-1) ** 0.5
        softmax = nn.functional.softmax(temp / scale, dim=-1)
        output = softmax.bmm(value)
        return output