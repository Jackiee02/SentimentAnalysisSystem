# config.py
import os

# Project base directory, modify according to your environment
BASE_DIR = r"E:\ProgramFiles\PycharmProjects\PythonProject"

# Dataset path (should include columns "text" and "sentiment")
DATA_PATH = os.path.join(BASE_DIR, "sentiment_analysis.csv")

# Model save path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_model.pth")

# Basic hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3   # Assuming three classes: "positive", "negative", "neutral"
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 100  # For BiLSTM
DEVICE = "cuda"    # Change to "cuda" if a GPU is available

# Settings for BERT
PRETRAINED_MODEL_NAME = "bert-base-uncased"  # or another available model
MAX_SEQ_LEN_BERT = 128  # Input sequence length for BERT (adjustable)

# Whether to use class weights (to address data imbalance)
USE_CLASS_WEIGHTS = False
