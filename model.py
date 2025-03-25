# model.py
import torch
import torch.nn as nn
from config import EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, PRETRAINED_MODEL_NAME
from transformers import AutoModel

# ------------------ BiLSTM Model ------------------
class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super(SentimentBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Concatenate outputs from bidirectional LSTM

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden: [2, batch_size, hidden_dim] (bidirectional, 1 layer)
        hidden_fw = hidden[-2, :, :]
        hidden_bw = hidden[-1, :, :]
        hidden_cat = torch.cat((hidden_fw, hidden_bw), dim=1)
        hidden_cat = self.dropout(hidden_cat)
        out = self.fc(hidden_cat)
        return out

# ------------------ Transformer (BERT) Model ------------------
class SentimentTransformer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained_model=PRETRAINED_MODEL_NAME):
        super(SentimentTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        # The hidden_size of BERT is typically 768 (base) or 1024 (large); can be obtained from self.bert.config.hidden_size
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler_output as the sentence vector (outputs.pooler_output: [batch_size, hidden_size])
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
