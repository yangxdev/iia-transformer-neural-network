from pickle import load
import numpy as np
from torch.utils.data import Dataset
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


class PrepareDataset(Dataset):
    def __init__(self, filename, n_sentences=10000, train_split=0.9):
        super(PrepareDataset, self).__init__()

        self.n_sentences = n_sentences  # Number of sentences to include in the dataset
        self.train_split = train_split  # Ratio of the training data split

        # Load a clean dataset
        clean_dataset = load(open(filename, 'rb'))

        # Reduce dataset size
        self.dataset = clean_dataset[:self.n_sentences, :]

        # Include start and end of string tokens
        for i in range(self.dataset[:, 0].size):
            self.dataset[i, 0] = "<START> " + self.dataset[i, 0] + " <EOS>"
            self.dataset[i, 1] = "<START> " + self.dataset[i, 1] + " <EOS>"

        # Random shuffle the dataset
        np.random.shuffle(self.dataset)

        # Split the dataset
        self.train = self.dataset[:int(self.n_sentences * self.train_split)]

        # Prepare tokenizer for the encoder input
        self.enc_tokenizer = self.create_tokenizer(self.train[:, 0])
        self.enc_seq_length = self.find_seq_length(self.train[:, 0])
        self.enc_vocab_size = self.find_vocab_size(self.enc_tokenizer, self.train[:, 0])

        # Prepare tokenizer for the decoder input
        self.dec_tokenizer = self.create_tokenizer(self.train[:, 1])
        self.dec_seq_length = self.find_seq_length(self.train[:, 1])
        self.dec_vocab_size = self.find_vocab_size(self.dec_tokenizer, self.train[:, 1])

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        enc_input = self.enc_tokenizer.texts_to_sequences([self.train[idx, 0]])[0]
        dec_input = self.dec_tokenizer.texts_to_sequences([self.train[idx, 1]])[0]

        # Convert to tensors
        enc_input = LongTensor(enc_input)
        dec_input = LongTensor(dec_input)

        return enc_input, dec_input

    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        word_counter = Counter()

        for seq in dataset:
            word_counter.update(seq.split())

        tokenizer = {w: i + 1 for i, (w, c) in enumerate(word_counter.most_common())}

        return tokenizer

    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        return len(tokenizer) + 1

    def collate_fn(self, data):
        enc_inputs, dec_inputs = zip(*data)

        # Pad sequences
        enc_inputs = pad_sequence(enc_inputs, batch_first=True, padding_value=0)
        dec_inputs = pad_sequence(dec_inputs, batch_first=True, padding_value=0)

        return enc_inputs, dec_inputs

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