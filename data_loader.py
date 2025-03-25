# data_loader.py
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

from config import (
    DATA_PATH,
    MAX_VOCAB_SIZE,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    NUM_CLASSES,
    PRETRAINED_MODEL_NAME,
    MAX_SEQ_LEN_BERT
)

# ------------- BiLSTM Dataset -------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def build_vocab(texts, max_vocab_size):
    counter = Counter()
    for text in texts:
        tokens = tokenize(clean_text(text))
        counter.update(tokens)
    most_common = counter.most_common(max_vocab_size - 2)  # Reserve <PAD> and <UNK>
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    return word2idx

def encode_text(text, word2idx, max_len):
    tokens = tokenize(clean_text(text))
    seq = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    # Truncate or pad
    if len(seq) < max_len:
        seq = seq + [word2idx["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

class SentimentDatasetLSTM(Dataset):
    def __init__(self, df, word2idx, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["sentiment"].tolist()
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = encode_text(self.texts[idx], self.word2idx, self.max_len)
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def get_data_loaders_lstm(test_size=0.2, batch_size=BATCH_SIZE):
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str)
    # If 'sentiment' is a string, create a mapping (e.g., {'positive': 0, 'negative': 1, 'neutral': 2})
    if df["sentiment"].dtype == object:
        unique_labels = df["sentiment"].unique().tolist()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df["sentiment"] = df["sentiment"].map(label_mapping)
        print("Label mapping (LSTM):", label_mapping)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    word2idx = build_vocab(train_df["text"].tolist(), MAX_VOCAB_SIZE)

    train_dataset = SentimentDatasetLSTM(train_df, word2idx, MAX_SEQ_LEN)
    test_dataset = SentimentDatasetLSTM(test_df, word2idx, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, word2idx, train_df


# ------------- Transformer Dataset -------------
from transformers import AutoTokenizer

class SentimentDatasetTransformer(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["sentiment"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # encoding["input_ids"] and encoding["attention_mask"] are both [1, max_len]
        input_ids = encoding["input_ids"].squeeze(0)         # -> [max_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # -> [max_len]
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

def get_data_loaders_transformer(test_size=0.2, batch_size=BATCH_SIZE):
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str)
    # If 'sentiment' is a string, create a mapping
    if df["sentiment"].dtype == object:
        unique_labels = df["sentiment"].unique().tolist()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df["sentiment"] = df["sentiment"].map(label_mapping)
        print("Label mapping (Transformer):", label_mapping)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    train_dataset = SentimentDatasetTransformer(train_df, tokenizer, MAX_SEQ_LEN_BERT)
    test_dataset = SentimentDatasetTransformer(test_df, tokenizer, MAX_SEQ_LEN_BERT)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, tokenizer, train_df


def compute_class_weights(train_df):
    # Compute the number of samples for each class in the training set and generate weights
    label_counts = train_df["sentiment"].value_counts().to_dict()
    total = sum(label_counts.values())
    weights = [total / label_counts.get(i, 1) for i in range(NUM_CLASSES)]
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return weights_tensor
