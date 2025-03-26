import os
import sys
import torch
import pandas as pd
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel
import re
from collections import Counter

# Set paths
BASE_DIR = 'C:\\Users\\Daniel ZHAO\\AIBA\\Deep_Learning\\dlgp'
MODEL_DIR = os.path.join(BASE_DIR, 'code___', 'Results', 'FinalModel')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_final_model.pth')
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, 'transformer_final_model.pth')
DATA_PATH = os.path.join(BASE_DIR, 'code___', 'sentiment_analysis.csv')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants from the error message
NUM_CLASSES = 3
MAX_SEQ_LEN = 100
MAX_SEQ_LEN_BERT = 128
PRETRAINED_MODEL_NAME = "bert-base-uncased"


# Text processing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def tokenize(text):
    return text.split()


def build_vocab_from_csv(csv_path=DATA_PATH, max_vocab_size=1290):
    """Build the vocabulary from the original CSV file with the correct size."""
    try:
        print(f"Building vocabulary from: {csv_path}")
        df = pd.read_csv(csv_path)
        texts = df["text"].astype(str).tolist()
        counter = Counter()
        for text in texts:
            tokens = tokenize(clean_text(text))
            counter.update(tokens)

        # Take only the top words to match the vocabulary size
        most_common = counter.most_common(max_vocab_size - 2)
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in most_common:
            word2idx[word] = len(word2idx)

        print(f"Vocabulary built with {len(word2idx)} words")
        return word2idx
    except Exception as e:
        print(f"Error building vocabulary: {e}")
        # Return a minimal vocabulary if there's an error
        return {"<PAD>": 0, "<UNK>": 1}


def encode_text(text, word2idx, max_len):
    """Encode text using the vocabulary."""
    tokens = tokenize(clean_text(text))
    seq = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    if len(seq) < max_len:
        seq = seq + [word2idx["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


# Model definitions
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


# Get sentiment labels from the CSV file
def get_sentiment_labels(csv_path=DATA_PATH):
    """Determine the sentiment labels from the original CSV."""
    try:
        df = pd.read_csv(csv_path)
        if "sentiment" in df.columns:
            # Get unique sentiment values
            unique_sentiments = df["sentiment"].unique()

            # Map them to indices
            if len(unique_sentiments) == NUM_CLASSES:
                sentiment_labels = {i: str(label) for i, label in enumerate(unique_sentiments)}
                print(f"Sentiment labels from CSV: {sentiment_labels}")
                return sentiment_labels
    except Exception as e:
        print(f"Error getting sentiment labels: {e}")

    # Default sentiment labels if we can't get them from the CSV
    default_labels = {0: "Positive", 1: "Negative", 2: "Neutral"}
    print(f"Using default sentiment labels: {default_labels}")
    return default_labels


# Load resources
print("Loading resources...")
word2idx = build_vocab_from_csv()
VOCAB_SIZE = len(word2idx)
print(f"Vocabulary size: {VOCAB_SIZE}")

# Get sentiment labels
SENTIMENT_LABELS = get_sentiment_labels()

# Load models
print("Loading models...")

# Load LSTM model
try:
    lstm_model = SentimentBiLSTM(VOCAB_SIZE, num_classes=NUM_CLASSES)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    lstm_model.to(DEVICE)
    lstm_model.eval()
    print("LSTM model loaded successfully")
    lstm_available = True
except Exception as e:
    print(f"Error loading LSTM model: {e}")
    lstm_available = False

# Load Transformer model
try:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    transformer_model = SentimentTransformer(num_classes=NUM_CLASSES)
    transformer_model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE))
    transformer_model.to(DEVICE)
    transformer_model.eval()
    print("Transformer model loaded successfully")
    transformer_available = True
except Exception as e:
    print(f"Error loading Transformer model: {e}")
    transformer_available = False

# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        input_text = request.form['text']

        if not input_text.strip():
            return jsonify({'error': 'Please enter text to analyze'})

        results = {'input_text': input_text}

        # LSTM prediction
        if lstm_available:
            try:
                seq = encode_text(input_text, word2idx, MAX_SEQ_LEN)
                inputs = torch.tensor([seq], dtype=torch.long).to(DEVICE)

                with torch.no_grad():
                    lstm_outputs = lstm_model(inputs)
                    _, lstm_predicted = torch.max(lstm_outputs, 1)
                    lstm_sentiment_idx = lstm_predicted.item()

                    # Get probability distribution
                    probabilities = torch.nn.functional.softmax(lstm_outputs, dim=1)
                    lstm_probs = probabilities[0].cpu().numpy().tolist()

                    # Get sentiment label
                    lstm_sentiment = SENTIMENT_LABELS.get(lstm_sentiment_idx, f"Class {lstm_sentiment_idx}")

                    results['lstm_sentiment'] = lstm_sentiment
                    results['lstm_confidence'] = float(max(lstm_probs) * 100)
                    results['lstm_probabilities'] = {SENTIMENT_LABELS.get(i, f"Class {i}"): float(prob * 100)
                                                     for i, prob in enumerate(lstm_probs)}
            except Exception as e:
                print(f"LSTM prediction error: {e}")
                results['lstm_error'] = str(e)
        else:
            results['lstm_sentiment'] = "Model not available"

        # Transformer prediction
        if transformer_available:
            try:
                encoding = tokenizer(
                    input_text,
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
                    _, transformer_predicted = torch.max(transformer_outputs, 1)
                    transformer_sentiment_idx = transformer_predicted.item()

                    # Get probability distribution
                    probabilities = torch.nn.functional.softmax(transformer_outputs, dim=1)
                    transformer_probs = probabilities[0].cpu().numpy().tolist()

                    # Get sentiment label
                    transformer_sentiment = SENTIMENT_LABELS.get(transformer_sentiment_idx,
                                                                 f"Class {transformer_sentiment_idx}")

                    results['transformer_sentiment'] = transformer_sentiment
                    results['transformer_confidence'] = float(max(transformer_probs) * 100)
                    results['transformer_probabilities'] = {SENTIMENT_LABELS.get(i, f"Class {i}"): float(prob * 100)
                                                            for i, prob in enumerate(transformer_probs)}
            except Exception as e:
                print(f"Transformer prediction error: {e}")
                results['transformer_error'] = str(e)
        else:
            results['transformer_sentiment'] = "Model not available"

        return jsonify(results)


if __name__ == '__main__':
    # Create templates folder and index.html
    os.makedirs('templates', exist_ok=True)

    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Tool</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #343a40;
        }
        .container {
            max-width: 1000px;
            margin: 30px auto;
            padding: 20px;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: #343a40;
            font-size: 2.2rem;
            margin-bottom: 5px;
        }
        .subtitle {
            color: #6c757d;
            font-size: 1.1rem;
        }
        .main-content {
            display: flex;
            flex-direction: column;
            gap: 20px;
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }
        .input-section {
            flex: 1;
        }
        .output-section {
            flex: 1;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            border: 1px solid #ced4da;
            border-radius: 5px;
            font-size: 16px;
            transition: border-color 0.15s ease-in-out;
        }
        textarea:focus {
            border-color: #80bdff;
            outline: 0;
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 12px 20px;
            text-align: center;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
            margin-top: 10px;
            transition: background-color 0.15s ease-in-out;
        }
        button:hover {
            background-color: #0069d9;
        }
        .results {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
        }
        .model-results {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .model-card {
            flex: 1;
            min-width: 300px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .model-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 10px;
            color: #343a40;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 8px;
        }
        .sentiment-result {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 15px 0;
            text-align: center;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        .neutral {
            color: #6c757d;
        }
        .confidence {
            margin-bottom: 15px;
            color: #495057;
            text-align: center;
        }
        .prob-container {
            margin-top: 15px;
        }
        .prob-bar-container {
            height: 20px;
            background-color: #e9ecef;
            border-radius: 4px;
            margin-top: 5px;
            margin-bottom: 10px;
            overflow: hidden;
            position: relative;
        }
        .prob-bar {
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }
        .prob-bar-positive {
            background-color: #28a745;
        }
        .prob-bar-negative {
            background-color: #dc3545;
        }
        .prob-bar-neutral {
            background-color: #6c757d;
        }
        .prob-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #6c757d;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #007bff;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        .error-message {
            color: #dc3545;
            padding: 10px;
            background-color: #f8d7da;
            border-radius: 4px;
            margin-top: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @media (min-width: 768px) {
            .main-content {
                flex-direction: row;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Sentiment Analysis Tool</h1>
            <p class="subtitle">Analyze text sentiment using BiLSTM and Transformer models</p>
        </header>

        <div class="main-content">
            <div class="input-section">
                <h2>Input Text</h2>
                <form id="sentiment-form">
                    <textarea id="input-text" name="text" placeholder="Enter text for sentiment analysis..."></textarea>
                    <button type="submit" id="analyze-button">Analyze Sentiment</button>
                </form>
            </div>

            <div class="output-section">
                <h2>Analysis Results</h2>
                <div id="results" class="results">
                    <p>Enter text and click "Analyze Sentiment" to see results.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('sentiment-form').addEventListener('submit', function(e) {
            e.preventDefault();

            const text = document.getElementById('input-text').value;
            if (!text.trim()) {
                alert('Please enter text for analysis.');
                return;
            }

            // Disable button and show loading
            const analyzeButton = document.getElementById('analyze-button');
            analyzeButton.disabled = true;
            analyzeButton.textContent = 'Analyzing...';

            // Show loading animation
            document.getElementById('results').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing your text...</p>
                </div>
            `;

            // Send request to the server
            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'text': text
                })
            })
            .then(response => response.json())
            .then(data => {
                // Re-enable button
                analyzeButton.disabled = false;
                analyzeButton.textContent = 'Analyze Sentiment';

                // Handle errors
                if (data.error) {
                    document.getElementById('results').innerHTML = `
                        <div class="error-message">${data.error}</div>
                    `;
                    return;
                }

                // Create result cards
                let resultsHTML = `<div class="model-results">`;

                // LSTM result
                if (data.lstm_sentiment && data.lstm_sentiment !== "Model not available") {
                    resultsHTML += `
                        <div class="model-card">
                            <div class="model-title">BiLSTM Model</div>
                            <div class="sentiment-result ${getSentimentClass(data.lstm_sentiment)}">
                                ${data.lstm_sentiment}
                            </div>
                            <div class="confidence">
                                Confidence: ${data.lstm_confidence.toFixed(1)}%
                            </div>
                            <div class="prob-container">
                                <h4>Probability Distribution:</h4>
                                ${createProbabilityBars(data.lstm_probabilities)}
                            </div>
                        </div>
                    `;
                } else {
                    resultsHTML += `
                        <div class="model-card">
                            <div class="model-title">BiLSTM Model</div>
                            <div class="error-message">
                                ${data.lstm_error || "Model not available"}
                            </div>
                        </div>
                    `;
                }

                // Transformer result
                if (data.transformer_sentiment && data.transformer_sentiment !== "Model not available") {
                    resultsHTML += `
                        <div class="model-card">
                            <div class="model-title">Transformer Model</div>
                            <div class="sentiment-result ${getSentimentClass(data.transformer_sentiment)}">
                                ${data.transformer_sentiment}
                            </div>
                            <div class="confidence">
                                Confidence: ${data.transformer_confidence.toFixed(1)}%
                            </div>
                            <div class="prob-container">
                                <h4>Probability Distribution:</h4>
                                ${createProbabilityBars(data.transformer_probabilities)}
                            </div>
                        </div>
                    `;
                } else {
                    resultsHTML += `
                        <div class="model-card">
                            <div class="model-title">Transformer Model</div>
                            <div class="error-message">
                                ${data.transformer_error || "Model not available"}
                            </div>
                        </div>
                    `;
                }

                resultsHTML += `</div>`;

                document.getElementById('results').innerHTML = resultsHTML;
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('results').innerHTML = '<div class="error-message">Error analyzing text. Please try again.</div>';

                // Re-enable button
                analyzeButton.disabled = false;
                analyzeButton.textContent = 'Analyze Sentiment';
            });
        });

        // Helper function: determine CSS class based on sentiment
        function getSentimentClass(sentiment) {
            if (!sentiment) return '';

            const lowerSentiment = sentiment.toLowerCase();
            if (lowerSentiment.includes('positive')) return 'positive';
            if (lowerSentiment.includes('negative')) return 'negative';
            if (lowerSentiment.includes('neutral')) return 'neutral';

            return '';
        }

        // Helper function: create probability bars
        function createProbabilityBars(probabilities) {
            if (!probabilities || Object.keys(probabilities).length === 0) {
                return '<p>No probability data available</p>';
            }

            let barsHTML = '';
            for (const [label, prob] of Object.entries(probabilities)) {
                let barClass = '';
                const lowerLabel = label.toLowerCase();

                if (lowerLabel.includes('positive')) barClass = 'prob-bar-positive';
                else if (lowerLabel.includes('negative')) barClass = 'prob-bar-negative';
                else if (lowerLabel.includes('neutral')) barClass = 'prob-bar-neutral';

                barsHTML += `
                    <div class="prob-label">
                        <span>${label}</span>
                        <span>${prob.toFixed(1)}%</span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar ${barClass}" style="width: ${prob}%"></div>
                    </div>
                `;
            }
            return barsHTML;
        }
    </script>
</body>
</html>
''')

    print("Starting Sentiment Analysis Web Application...")
    app.run(debug=True)
