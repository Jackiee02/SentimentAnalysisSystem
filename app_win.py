import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import re
from collections import Counter
import tkinter as tk
from tkinter import ttk
import keyboard
import pyperclip

# ================================ Configuration ================================ #
BASE_DIR = r'E:\ProgramFiles\PycharmProjects\PythonProject'
MODEL_DIR = r'E:\ProgramFiles\PycharmProjects\PythonProject\Results\FinalModel'
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_final_model.pth")
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_final_model.pth")
DATA_PATH = os.path.join(BASE_DIR, "sentiment_analysis.csv")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3
MAX_SEQ_LEN = 100
MAX_SEQ_LEN_BERT = 128
PRETRAINED_MODEL_NAME = "bert-base-uncased"
SENTIMENT_LABELS = ["Positive", "Negative", "Neutral"]
SENTIMENT_COLORS = ["#FF5252", "#64B5F6", "#66BB6A"]
HOTKEY = "ctrl+shift+space"

# ================================ Text Preprocessing ================================ #
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def build_vocab_from_csv(csv_path=DATA_PATH, max_vocab_size=1290):
    try:
        df = pd.read_csv(csv_path)
        texts = df["text"].astype(str).tolist()
        counter = Counter()
        for text in texts:
            tokens = tokenize(clean_text(text))
            counter.update(tokens)

        most_common = counter.most_common(max_vocab_size - 2)
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in most_common:
            word2idx[word] = len(word2idx)

        return word2idx
    except Exception as e:
        print(f"Error building vocabulary: {e}")
        return {"<PAD>": 0, "<UNK>": 1}

def encode_text(text, word2idx, max_len):
    tokens = tokenize(clean_text(text))
    seq = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    if len(seq) < max_len:
        seq += [word2idx["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# ================================ Model Definitions ================================ #
class SentimentBiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_classes=3):
        super(SentimentBiLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                                  bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_fw = hidden[-2, :, :]
        hidden_bw = hidden[-1, :, :]
        hidden_cat = torch.cat((hidden_fw, hidden_bw), dim=1)
        hidden_cat = self.dropout(hidden_cat)
        out = self.fc(hidden_cat)
        return out

class SentimentTransformer(torch.nn.Module):
    def __init__(self, num_classes=3, pretrained_model="bert-base-uncased"):
        super(SentimentTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = torch.nn.Dropout(0.3)
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# ================================ Model Loading ================================ #
print("Loading resources...")
word2idx = build_vocab_from_csv()
VOCAB_SIZE = len(word2idx)

# Load LSTM model
try:
    lstm_model = SentimentBiLSTM(VOCAB_SIZE, num_classes=NUM_CLASSES)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE, weights_only=False))
    lstm_model.to(DEVICE)
    lstm_model.eval()
    print("LSTM model loaded successfully")
    lstm_available = True
except Exception as e:
    print(f"Failed to load LSTM model: {e}")
    lstm_available = False

# Load Transformer model
try:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    transformer_model = SentimentTransformer(num_classes=NUM_CLASSES)
    transformer_model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE, weights_only=False))
    transformer_model.to(DEVICE)
    transformer_model.eval()
    print("Transformer model loaded successfully")
    transformer_available = True
except Exception as e:
    print(f"Failed to load Transformer model: {e}")
    transformer_available = False

# ================================ Prediction ================================ #
def analyze_sentiment(text):
    results = {}

    # LSTM prediction
    if lstm_available:
        seq = encode_text(text, word2idx, MAX_SEQ_LEN)
        inputs = torch.tensor([seq], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            lstm_outputs = lstm_model(inputs)
            probabilities = torch.nn.functional.softmax(lstm_outputs, dim=1)
            lstm_predicted = torch.argmax(probabilities, dim=1).item()
            results['lstm_sentiment'] = lstm_predicted
            results['lstm_probabilities'] = probabilities[0].cpu().numpy().tolist()
    else:
        results['lstm_sentiment'] = "Model not available"

    # Transformer prediction
    if transformer_available:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN_BERT,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        with torch.no_grad():
            transformer_outputs = transformer_model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(transformer_outputs, dim=1)
            transformer_predicted = torch.argmax(probabilities, dim=1).item()
            results['transformer_sentiment'] = transformer_predicted
            results['transformer_probabilities'] = probabilities[0].cpu().numpy().tolist()
    else:
        results['transformer_sentiment'] = "Model not available"

    return results

# ================================ GUI ================================ #
class SimpleSentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analyzer")
        self.root.geometry("500x400")
        self.root.configure(bg="#F5F5F5")

        self.setup_ui()

        keyboard.add_hotkey(HOTKEY, self.analyze_clipboard)

        tk.Label(root, text=f"Press {HOTKEY} to analyze clipboard text",
                 bg="#F5F5F5", fg="#666666").pack(pady=5)

    def setup_ui(self):
        style = ttk.Style()
        for i, color in enumerate(SENTIMENT_COLORS):
            style.configure(f'sentiment.{i}.Horizontal.TProgressbar',
                            background=color, troughcolor="#E0E0E0")

        results_frame = tk.Frame(self.root, bg="#F5F5F5")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # LSTM section
        lstm_frame = tk.LabelFrame(results_frame, text="BiLSTM Model", bg="#F5F5F5", padx=10, pady=10)
        lstm_frame.pack(fill=tk.X, pady=5)

        self.lstm_result_label = tk.Label(lstm_frame, text="No analysis yet", bg="#F5F5F5")
        self.lstm_result_label.pack(anchor=tk.W, pady=5)

        self.lstm_bars = []
        self.lstm_percentages = []

        for i, label in enumerate(SENTIMENT_LABELS):
            bar_frame = tk.Frame(lstm_frame, bg="#F5F5F5")
            bar_frame.pack(fill=tk.X, pady=2)

            tk.Label(bar_frame, text=f"{label}:", width=10, bg="#F5F5F5", anchor=tk.W).pack(side=tk.LEFT)

            bar = ttk.Progressbar(bar_frame, style=f'sentiment.{i}.Horizontal.TProgressbar',
                                  length=350, mode="determinate")
            bar.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            self.lstm_bars.append(bar)

            percentage = tk.Label(bar_frame, text="0%", width=6, bg="#F5F5F5")
            percentage.pack(side=tk.RIGHT)
            self.lstm_percentages.append(percentage)

        # Transformer section
        transformer_frame = tk.LabelFrame(results_frame, text="Transformer Model",
                                          bg="#F5F5F5", padx=10, pady=10)
        transformer_frame.pack(fill=tk.X, pady=5)

        self.transformer_result_label = tk.Label(transformer_frame, text="No analysis yet", bg="#F5F5F5")
        self.transformer_result_label.pack(anchor=tk.W, pady=5)

        self.transformer_bars = []
        self.transformer_percentages = []

        for i, label in enumerate(SENTIMENT_LABELS):
            bar_frame = tk.Frame(transformer_frame, bg="#F5F5F5")
            bar_frame.pack(fill=tk.X, pady=2)

            tk.Label(bar_frame, text=f"{label}:", width=10, bg="#F5F5F5", anchor=tk.W).pack(side=tk.LEFT)

            bar = ttk.Progressbar(bar_frame, style=f'sentiment.{i}.Horizontal.TProgressbar',
                                  length=350, mode="determinate")
            bar.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            self.transformer_bars.append(bar)

            percentage = tk.Label(bar_frame, text="0%", width=6, bg="#F5F5F5")
            percentage.pack(side=tk.RIGHT)
            self.transformer_percentages.append(percentage)

        # Text display section
        text_frame = tk.Frame(self.root, bg="#F5F5F5")
        text_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(text_frame, text="Current Text:", bg="#F5F5F5").pack(anchor=tk.W)
        self.text_display = tk.Label(text_frame, text="No text analyzed yet",
                                     bg="#FFFFFF", wraplength=450,
                                     justify=tk.LEFT, relief=tk.SUNKEN,
                                     padx=5, pady=5, height=3)
        self.text_display.pack(fill=tk.X, pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def analyze_clipboard(self):
        """Analyze text from clipboard"""
        self.root.focus_force()

        text = pyperclip.paste()
        if not text:
            self.status_var.set("Clipboard is empty")
            return

        self.text_display.config(text=text if len(text) < 100 else text[:97] + "...")
        self.status_var.set("Analyzing...")
        self.root.update()

        results = analyze_sentiment(text)
        self.update_results(results)

    def update_results(self, results):
        """Update analysis results on the UI"""
        if 'lstm_sentiment' in results and 'lstm_probabilities' in results:
            if isinstance(results['lstm_sentiment'], int):
                idx = results['lstm_sentiment']
                label = SENTIMENT_LABELS[idx]
                color = SENTIMENT_COLORS[idx]

                self.lstm_result_label.config(text=f"Sentiment: {label}", fg=color)
                for i, prob in enumerate(results['lstm_probabilities']):
                    pct = int(prob * 100)
                    self.lstm_bars[i]["value"] = pct
                    self.lstm_percentages[i].config(text=f"{pct}%")
            else:
                self.lstm_result_label.config(text=f"LSTM: {results['lstm_sentiment']}")

        if 'transformer_sentiment' in results and 'transformer_probabilities' in results:
            if isinstance(results['transformer_sentiment'], int):
                idx = results['transformer_sentiment']
                label = SENTIMENT_LABELS[idx]
                color = SENTIMENT_COLORS[idx]

                self.transformer_result_label.config(text=f"Sentiment: {label}", fg=color)
                for i, prob in enumerate(results['transformer_probabilities']):
                    pct = int(prob * 100)
                    self.transformer_bars[i]["value"] = pct
                    self.transformer_percentages[i].config(text=f"{pct}%")
            else:
                self.transformer_result_label.config(text=f"Transformer: {results['transformer_sentiment']}")

        self.status_var.set("Analysis complete")


# ================================ Main ================================ #
if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleSentimentApp(root)
    root.mainloop()
