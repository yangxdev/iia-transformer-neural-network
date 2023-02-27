# Loading of the necessary files
from models.transformer import Transformer
from utilities.positional_encoding import position_encoding
from utilities.scaled_dot_product_attention import ScaledDotProductAttention
from models.multi_head_attention import MultiHeadAttention
from utilities.feed_forward import FeedForward
from utilities.residual import Residual
import torch
from torch import Tensor, nn

# Create an instance of the Transformer class
model = Transformer(
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_model=512,
    num_heads=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation=nn.ReLU(),
)

# Preparing test data
test_inputs = torch.rand(32, 64, 512)
test_targets = torch.rand(32, 64, 512)

# Evalutation of the model
loss, accuracy = model.evaluate(test_inputs, test_targets)

# Analazing the output
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

