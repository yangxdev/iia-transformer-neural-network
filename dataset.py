import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_sentence = torch.tensor(self.src_sentences[index])
        tgt_sentence = torch.tensor(self.tgt_sentences[index])
        return src_sentence, tgt_sentence