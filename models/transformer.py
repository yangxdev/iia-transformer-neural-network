from torch import Tensor, nn
import torch
from models.encoder import Encoder
from models.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decoder(tgt, self.encoder(src))
