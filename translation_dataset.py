from pickle import load
from numpy.random import shuffle
from torch import Tensor, int64
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, filename, max_len=10_000, train_split=0.9):
        super().__init__()
        self.max_len = max_len
        self.train_split = train_split

        # Load a clean dataset
        with open(filename, 'rb') as f:
            clean_dataset = load(f)

        # Reduce dataset size
        dataset = clean_dataset[:self.max_len, :]

        # Include start and end of string tokens
        for i in range(dataset[:, 0].size):
            dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
            dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"

        # Random shuffle the dataset
        shuffle(dataset)

        # Split the dataset
        train = dataset[:int(self.max_len * self.train_split)]

        # Prepare tokenizer for the encoder input
        enc_tokenizer = get_tokenizer('basic_english')
        enc_vocab = set()
        for seq in train[:, 0]:
            enc_vocab.update(enc_tokenizer(seq))
        enc_vocab_size = len(enc_vocab)

        # Encode and pad the input sequences
        trainX = []
        for seq in train[:, 0]:
            trainX.append(torch.tensor([enc_vocab_size + enc_tokenizer(seq)], dtype=int64))
        trainX = pad_sequence(trainX, batch_first=True)

        # Prepare tokenizer for the decoder input
        dec_tokenizer = get_tokenizer('basic_english')
        dec_vocab = set()
        for seq in train[:, 1]:
            dec_vocab.update(dec_tokenizer(seq))
        dec_vocab_size = len(dec_vocab)

        # Encode and pad the input sequences
        trainY = []
        for seq in train[:, 1]:
            trainY.append(torch.tensor([dec_vocab_size + dec_tokenizer(seq)], dtype=int64))
        trainY = pad_sequence(trainY, batch_first=True)

        self.trainX = trainX
        self.trainY = trainY

    def __len__(self):
        return len(self.trainX)

    def __getitem__(self, idx):
        return self.trainX[idx], self.trainY[idx]


# Example usage
dataset = TranslationDataset('english-german-both.pkl')
trainX = dataset.trainX
trainY = dataset.trainY
train_orig = dataset.train_orig
enc_seq_length = trainX.shape[1]
dec_seq_length = trainY.shape[1]
enc_vocab_size = dataset.enc_vocab_size
dec_vocab_size = dataset.dec_vocab_size
print(train_orig[0, 0], '\n', trainX[0, :])
