import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import TranslationDataset
from models.transformer import Transformer

# Define your training and validation datasets
train_src_sentences = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
]

train_tgt_sentences = [
    [6, 7, 8, 9, 10],
    [1, 2, 3, 4, 5],
    [21, 22, 23, 24, 25],
    [16, 17, 18, 19, 20]
]

train_dataset = TranslationDataset(train_src_sentences, train_tgt_sentences)

val_src_sentences = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
]

val_tgt_sentences = [
    [6, 7, 8, 9, 10],
    [1, 2, 3, 4, 5]
]

val_dataset = TranslationDataset(val_src_sentences, val_tgt_sentences)


# Define your data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define your model and move it to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(...).to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define your TensorBoard writer
log_dir = "logs"
writer = SummaryWriter(log_dir=log_dir)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()

    # Iterate over the training dataset
    for i, (src, tgt) in enumerate(train_loader):
        # Move the batch to the appropriate device
        src = src.to(device)
        tgt = tgt.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt[:, :-1])

        # Compute the loss
        loss = criterion(output.reshape(-1, output.size(-1)),
                         tgt[:, 1:].reshape(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log the loss to TensorBoard
        writer.add_scalar("train/loss", loss.item(),
                          epoch * len(train_loader) + i)

    # Set the model to evaluation mode
    model.eval()

    # Compute the validation loss and accuracy
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for src, tgt in val_loader:
            # Move the batch to the appropriate device
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            output = model(src, tgt[:, :-1])

            # Compute the loss
            loss = criterion(output.reshape(-1, output.size(-1)),
                             tgt[:, 1:].reshape(-1))

            # Update the validation loss and accuracy
            val_loss += loss.item() * src.size(0)
            val_acc += (torch.argmax(output, dim=-1)
                        == tgt[:, 1:]).sum().item()

    # Normalize the validation loss and accuracy by the number of validation samples
    val_loss /= len(val_dataset)
    val_acc /= len(val_dataset)

    # Log the validation loss and accuracy to TensorBoard
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/acc", val_acc, epoch)

    # Print the epoch, training loss, and validation loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Close the TensorBoard writer
writer.close()