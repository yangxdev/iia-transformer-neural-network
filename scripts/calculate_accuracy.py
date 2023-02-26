import torch.nn.functional as F

def calculate_accuracy(outputs, targets):
    """
    Computes the accuracy of the model's predictions.

    Args:
        outputs: Tensor of shape (batch_size, seq_len, vocab_size).
        targets: Tensor of shape (batch_size, seq_len).

    Returns:
        The accuracy.
    """
    predicted_ids = outputs.argmax(dim=-1)  # shape: (batch_size, seq_len)
    num_correct = (predicted_ids == targets).sum().item()
    num_total = targets.ne(0).sum().item()  # exclude padding tokens
    return num_correct / num_total