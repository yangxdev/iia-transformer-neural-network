import torch.nn.functional as F

def criterion(outputs, targets):
    """
    Computes the cross-entropy loss between the predicted outputs and the ground truth targets.

    Args:
        outputs: Tensor of shape (batch_size * seq_len, vocab_size).
        targets: Tensor of shape (batch_size, seq_len).

    Returns:
        The cross-entropy loss.
    """
    return F.cross_entropy(outputs, targets.reshape(-1), ignore_index=0)