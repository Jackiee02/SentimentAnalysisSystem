# train.py
import torch
import torch.nn as nn
import time
from evaluate import evaluate_model_lstm, evaluate_model_transformer
from config import DEVICE, NUM_CLASSES

def train_model_lstm(model, train_loader, test_loader, criterion, optimizer, num_epochs, loss_type="crossentropy"):
    """
    LSTM training function:
    If loss_type is "nllloss", apply log_softmax to the model output first.
    """
    model.to(DEVICE)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_size, num_classes]
            if loss_type == "nllloss":
                outputs = torch.log_softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate_model_lstm(model, test_loader, DEVICE)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        elapsed = time.time() - start_time
        print(f"[LSTM] Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {elapsed:.2f}s")
    return history

def train_model_transformer(model, train_loader, test_loader, criterion, optimizer, num_epochs, loss_type="crossentropy"):
    """
    Transformer training function:
    Supports two loss types:
      - "crossentropy": directly use nn.CrossEntropyLoss() with integer labels.
      - "bce": use nn.BCEWithLogitsLoss(), converting labels to one-hot encoding.
    """
    model.to(DEVICE)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)  # [batch_size, num_classes]
            if loss_type == "bce":
                # Convert labels to one-hot encoding; note that one_hot returns a tensor of type long, so convert to float
                onehot_labels = nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()
                loss = criterion(outputs, onehot_labels)
            else:  # crossentropy
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate_model_transformer(model, test_loader, DEVICE)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        elapsed = time.time() - start_time
        print(f"[Transformer] Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {elapsed:.2f}s")
    return history
