# evaluate.py
import torch

# -------- LSTM Evaluation --------
def evaluate_model_lstm(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def visualize_predictions_lstm(model, data_loader, device, word2idx, num_samples=100):
    model.eval()
    idx2word = {v: k for k, v in word2idx.items()}

    texts = []
    actual = []
    preds = []
    count = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                seq = inputs[i].cpu().numpy()
                text_str = " ".join([idx2word.get(idx, "<UNK>") for idx in seq if idx != word2idx["<PAD>"]])
                texts.append(text_str)
                actual.append(labels[i].item())
                preds.append(predicted[i].item())
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
    import pandas as pd
    df = pd.DataFrame({"Input Text": texts, "Actual": actual, "Predicted": preds})
    print("Prediction results for the first 100 test samples (LSTM):")
    print(df.head(100))

# -------- Transformer Evaluation --------
def evaluate_model_transformer(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def visualize_predictions_transformer(model, data_loader, device, tokenizer, num_samples=100):
    model.eval()
    texts = []
    actual = []
    preds = []
    count = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            for i in range(input_ids.size(0)):
                # Convert input_ids back to text (simple demonstration, may include special tokens)
                seq_ids = input_ids[i].cpu().numpy()
                text_str = tokenizer.decode(seq_ids, skip_special_tokens=True)
                texts.append(text_str)
                actual.append(labels[i].item())
                preds.append(predicted[i].item())
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
    import pandas as pd
    df = pd.DataFrame({"Input Text": texts, "Actual": actual, "Predicted": preds})
    print("Prediction results for the first 100 test samples (Transformer):")
    print(df.head(100))
