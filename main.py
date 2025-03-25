# main.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, BATCH_SIZE, USE_CLASS_WEIGHTS, NUM_CLASSES
from data_loader import (
    get_data_loaders_lstm,
    get_data_loaders_transformer,
    compute_class_weights
)
from model import SentimentBiLSTM, SentimentTransformer
from train import train_model_lstm, train_model_transformer
from evaluate import (
    visualize_predictions_lstm,
    visualize_predictions_transformer,
    evaluate_model_lstm,
    evaluate_model_transformer
)
from utils import plot_history
import matplotlib.pyplot as plt
import pandas as pd

# Global lists to record candidate results during horizontal experiments.
# Each candidate is a tuple: (config_dict, final_test_accuracy, history, state_dict)
lstm_candidate_results = []
transformer_candidate_results = []

# Use the current working directory as the base for saving results
RESULTS_DIR = os.path.join(os.getcwd(), "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FINAL_MODEL_DIR = os.path.join(RESULTS_DIR, "FinalModel")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def get_result_subdir(model_type, subfolder=""):
    sub_dir = os.path.join(RESULTS_DIR, model_type)
    if subfolder:
        sub_dir = os.path.join(sub_dir, subfolder)
    ensure_dir(sub_dir)
    return sub_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Analysis with LSTM & Transformer")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "experiments"],
                        help="Mode: 'train' or 'experiments'")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"],
                        help="Select model: 'lstm' or 'transformer'")
    parser.add_argument("--loss", type=str, default="crossentropy", choices=["crossentropy", "nllloss", "bce"],
                        help="Loss type; for LSTM: crossentropy or nllloss; for Transformer: crossentropy or bce")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    return parser.parse_args()

# ------------------ Base Training Functions ------------------
def run_training_lstm(loss_type, lr, batch_size, epochs):
    train_loader, test_loader, word2idx, train_df = get_data_loaders_lstm(batch_size=batch_size)
    vocab_size = len(word2idx)
    model = SentimentBiLSTM(vocab_size)
    if loss_type == "crossentropy":
        if USE_CLASS_WEIGHTS:
            class_weights = compute_class_weights(train_df).to(DEVICE)
            print("Class weights:", class_weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif loss_type == "nllloss":
        criterion = nn.NLLLoss()
    else:
        raise ValueError("LSTM does not support loss type: " + loss_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = train_model_lstm(model, train_loader, test_loader, criterion, optimizer, epochs, loss_type=loss_type)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[LSTM] Model saved to {MODEL_SAVE_PATH}")
    lstm_dir = get_result_subdir("lstm")
    plot_path = os.path.join(lstm_dir, "training_curve_lstm.png")
    plot_history(history, title=f"[LSTM] Loss: {loss_type}, LR: {lr}, BS: {batch_size}", save_path=plot_path)
    plt.close()
    visualize_predictions_lstm(model, test_loader, DEVICE, word2idx, num_samples=100)
    return model, history

def run_training_transformer(loss_type, lr, batch_size, epochs):
    train_loader, test_loader, tokenizer, train_df = get_data_loaders_transformer(batch_size=batch_size)
    model = SentimentTransformer()
    if loss_type == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Transformer does not support loss type: " + loss_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = train_model_transformer(model, train_loader, test_loader, criterion, optimizer, epochs, loss_type=loss_type)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[Transformer] Model saved to {MODEL_SAVE_PATH}")
    transformer_dir = get_result_subdir("transformer")
    plot_path = os.path.join(transformer_dir, "training_curve_transformer.png")
    plot_history(history, title=f"[Transformer] Loss: {loss_type}, LR: {lr}, BS: {batch_size}", save_path=plot_path)
    plt.close()
    visualize_predictions_transformer(model, test_loader, DEVICE, tokenizer, num_samples=100)
    return model, history

# ------------------ Horizontal Experiment Functions ------------------
def experiment_loss_functions_lstm():
    loss_types = ["crossentropy", "nllloss"]
    candidate_histories = {}
    for lt in loss_types:
        try:
            print(f"Running LSTM with loss function: {lt}")
            train_loader, test_loader, word2idx, train_df = get_data_loaders_lstm(batch_size=BATCH_SIZE)
            vocab_size = len(word2idx)
            model = SentimentBiLSTM(vocab_size)
            if lt == "crossentropy":
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            history = train_model_lstm(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, loss_type=lt)
            candidate_histories[lt] = history
            lstm_candidate_results.append(({"loss_type": lt, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE},
                                           history["test_acc"][-1], history, model.state_dict()))
        except torch.cuda.OutOfMemoryError as e:
            print(f"Error with loss function {lt}: {e}")
            torch.cuda.empty_cache()
    # Plot the actual histories
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for lt, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_loss"], label=f"{lt} Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("LSTM Loss Function Comparison: Train Loss")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.subplot(1, 2, 2)
    for lt, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_acc"], label=f"{lt} Train Acc")
        plt.plot(epochs_range, hist["test_acc"], label=f"{lt} Test Acc", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Loss Function Comparison: Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    folder = get_result_subdir("lstm", "loss_experiments")
    plt.savefig(os.path.join(folder, "experiment_loss_function_lstm.png"), bbox_inches="tight")
    plt.close()

def experiment_loss_functions_transformer():
    loss_types = ["crossentropy", "bce"]
    candidate_histories = {}
    for lt in loss_types:
        try:
            print(f"Running Transformer with loss function: {lt}")
            train_loader, test_loader, tokenizer, train_df = get_data_loaders_transformer(batch_size=BATCH_SIZE)
            model = SentimentTransformer()
            criterion = nn.CrossEntropyLoss() if lt == "crossentropy" else nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            history = train_model_transformer(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, loss_type=lt)
            candidate_histories[lt] = history
            transformer_candidate_results.append(({"loss_type": lt, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE},
                                                  history["test_acc"][-1], history, model.state_dict()))
        except torch.cuda.OutOfMemoryError as e:
            print(f"Error with transformer loss function {lt}: {e}")
            torch.cuda.empty_cache()
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for lt, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_loss"], label=f"{lt} Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Transformer Loss Function Comparison: Train Loss")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.subplot(1, 2, 2)
    for lt, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_acc"], label=f"{lt} Train Acc")
        plt.plot(epochs_range, hist["test_acc"], label=f"{lt} Test Acc", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Transformer Loss Function Comparison: Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    folder = get_result_subdir("transformer", "loss_experiments")
    plt.savefig(os.path.join(folder, "experiment_loss_function_transformer.png"), bbox_inches="tight")
    plt.close()

def experiment_learning_rates(model_type):
    lrs = [0.1, 0.01, 0.001, 0.0001]
    candidate_histories = {}
    if model_type == "lstm":
        for lr in lrs:
            try:
                print(f"Running LSTM with LR={lr}")
                train_loader, test_loader, word2idx, _ = get_data_loaders_lstm(batch_size=BATCH_SIZE)
                vocab_size = len(word2idx)
                model = SentimentBiLSTM(vocab_size)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                history = train_model_lstm(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS)
                candidate_histories[lr] = history
                lstm_candidate_results.append(({"loss_type": "crossentropy", "lr": lr, "batch_size": BATCH_SIZE},
                                               history["test_acc"][-1], history, model.state_dict()))
            except torch.cuda.OutOfMemoryError as e:
                print(f"LR={lr} encountered OOM error: {e}")
                torch.cuda.empty_cache()
    else:
        for lr in lrs:
            try:
                print(f"Running Transformer with LR={lr}")
                train_loader, test_loader, tokenizer, _ = get_data_loaders_transformer(batch_size=BATCH_SIZE)
                model = SentimentTransformer()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                history = train_model_transformer(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS)
                candidate_histories[lr] = history
                transformer_candidate_results.append(({"loss_type": "crossentropy", "lr": lr, "batch_size": BATCH_SIZE},
                                                      history["test_acc"][-1], history, model.state_dict()))
            except torch.cuda.OutOfMemoryError as e:
                print(f"LR={lr} encountered OOM error: {e}")
                torch.cuda.empty_cache()
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure()
    for lr, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_loss"], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"{model_type.upper()}: Train Loss for Different Learning Rates")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    folder = get_result_subdir(model_type, "lr_experiments")
    plt.savefig(os.path.join(folder, f"experiment_learning_rate_loss_{model_type}.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    for lr, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_acc"], label=f"Train Acc, LR={lr}")
        plt.plot(epochs_range, hist["test_acc"], label=f"Test Acc, LR={lr}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_type.upper()}: Accuracy for Different Learning Rates")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"experiment_learning_rate_acc_{model_type}.png"), bbox_inches="tight")
    plt.close()

def experiment_batch_sizes(model_type):
    batch_sizes = [8, 16, 32, 64, 128]
    candidate_histories = {}
    if model_type == "lstm":
        for bs in batch_sizes:
            try:
                print(f"Running LSTM with Batch Size={bs}")
                train_loader, test_loader, word2idx, _ = get_data_loaders_lstm(batch_size=bs)
                vocab_size = len(word2idx)
                model = SentimentBiLSTM(vocab_size)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                history = train_model_lstm(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS)
                candidate_histories[bs] = history
                lstm_candidate_results.append(({"loss_type": "crossentropy", "lr": LEARNING_RATE, "batch_size": bs},
                                               history["test_acc"][-1], history, model.state_dict()))
            except torch.cuda.OutOfMemoryError as e:
                print(f"Batch Size={bs} encountered OOM error: {e}")
                torch.cuda.empty_cache()
    else:
        for bs in batch_sizes:
            try:
                print(f"Running Transformer with Batch Size={bs}")
                train_loader, test_loader, tokenizer, _ = get_data_loaders_transformer(batch_size=bs)
                model = SentimentTransformer()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                history = train_model_transformer(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS)
                candidate_histories[bs] = history
                transformer_candidate_results.append(({"loss_type": "crossentropy", "lr": LEARNING_RATE, "batch_size": bs},
                                                      history["test_acc"][-1], history, model.state_dict()))
            except torch.cuda.OutOfMemoryError as e:
                print(f"Batch Size={bs} encountered OOM error: {e}")
                torch.cuda.empty_cache()

    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure()
    for bs, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_loss"], label=f"BS={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"{model_type.upper()}: Train Loss for Different Batch Sizes")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    folder = get_result_subdir(model_type, "batch_experiments")
    plt.savefig(os.path.join(folder, f"experiment_batch_size_loss_{model_type}.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    for bs, hist in candidate_histories.items():
        plt.plot(epochs_range, hist["train_acc"], label=f"Train Acc, BS={bs}")
        plt.plot(epochs_range, hist["test_acc"], label=f"Test Acc, BS={bs}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_type.upper()}: Accuracy for Different Batch Sizes")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"experiment_batch_size_acc_{model_type}.png"), bbox_inches="tight")
    plt.close()

# ------------------ Candidate Model Selection ------------------
def select_best_candidate(candidates):
    if not candidates:
        return None, None
    best = max(candidates, key=lambda x: x[1])  # x[1] is test_acc
    return best[0], best[3]  # return config and state dict

def select_best_model_lstm():
    config, state = select_best_candidate(lstm_candidate_results)
    best_acc = max([x[1] for x in lstm_candidate_results]) if lstm_candidate_results else None
    print(f"Selected LSTM model: {config} with Test Accuracy: {best_acc}")
    return state

def select_best_model_transformer():
    config, state = select_best_candidate(transformer_candidate_results)
    best_acc = max([x[1] for x in transformer_candidate_results]) if transformer_candidate_results else None
    print(f"Selected Transformer model: {config} with Test Accuracy: {best_acc}")
    return state

# ------------------ Save Final Predictions ------------------
def save_final_predictions_csv_lstm(state):
    if state is None:
        print("Unable to save LSTM final predictions CSV: no model state found.")
        return
    train_loader, test_loader, word2idx, _ = get_data_loaders_lstm(batch_size=BATCH_SIZE)
    vocab_size = len(word2idx)
    model = SentimentBiLSTM(vocab_size)
    model.load_state_dict(state)
    model.to(DEVICE)
    idx2word = {v: k for k, v in word2idx.items()}
    texts, actual, preds = [], [], []
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                seq = inputs[i].cpu().numpy()
                text_str = " ".join([idx2word.get(idx, "<UNK>") for idx in seq if idx != word2idx["<PAD>"]])
                texts.append(text_str)
                actual.append(labels[i].item())
                preds.append(predicted[i].item())
                count += 1
                if count >= 100:
                    break
            if count >= 100:
                break
    df = pd.DataFrame({"Text": texts, "Actual": actual, "Predicted": preds})
    out_file = os.path.join(FINAL_MODEL_DIR, "lstm_final_predictions.csv")
    df.to_csv(out_file, index=False)
    print(f"LSTM final predictions saved to {out_file}")

def save_final_predictions_csv_transformer(state):
    if state is None:
        print("Unable to save Transformer final predictions CSV: no model state found.")
        return
    train_loader, test_loader, tokenizer, _ = get_data_loaders_transformer(batch_size=BATCH_SIZE)
    model = SentimentTransformer()
    model.load_state_dict(state)
    model.to(DEVICE)
    texts, actual, preds = [], [], []
    count = 0
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            for i in range(input_ids.size(0)):
                seq_ids = input_ids[i].cpu().numpy()
                text_str = tokenizer.decode(seq_ids, skip_special_tokens=True)
                texts.append(text_str)
                actual.append(labels[i].item())
                preds.append(predicted[i].item())
                count += 1
                if count >= 100:
                    break
            if count >= 100:
                break
    df = pd.DataFrame({"Text": texts, "Actual": actual, "Predicted": preds})
    out_file = os.path.join(FINAL_MODEL_DIR, "transformer_final_predictions.csv")
    df.to_csv(out_file, index=False)
    print(f"Transformer final predictions saved to {out_file}")

# ------------------ Run Experiments ------------------
def run_experiments():
    final_log = []  # Collect final output messages
    baseline_lstm_state = None
    baseline_transformer_state = None

    # Run baseline experiments and record baseline states
    for model_type in ["lstm", "transformer"]:
        print(f"=== Baseline Experiment: {model_type.upper()} ===")
        final_log.append(f"=== Baseline Experiment: {model_type.upper()} ===")
        if model_type == "lstm":
            train_loader, test_loader, word2idx, _ = get_data_loaders_lstm(batch_size=BATCH_SIZE)
            vocab_size = len(word2idx)
            model = SentimentBiLSTM(vocab_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            history = train_model_lstm(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS)
            baseline_lstm_state = model.state_dict()
            folder = get_result_subdir(model_type)
            plot_history(history, title=f"[Baseline {model_type.upper()}] Training Curves",
                         save_path=os.path.join(folder, f"baseline_{model_type}.png"))
            plt.close()
            visualize_predictions_lstm(model, test_loader, DEVICE, word2idx, num_samples=100)
        else:
            train_loader, test_loader, tokenizer, _ = get_data_loaders_transformer(batch_size=BATCH_SIZE)
            model = SentimentTransformer()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            history = train_model_transformer(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS)
            baseline_transformer_state = model.state_dict()
            folder = get_result_subdir(model_type)
            plot_history(history, title=f"[Baseline {model_type.upper()}] Training Curves",
                         save_path=os.path.join(folder, f"baseline_{model_type}.png"))
            plt.close()
            visualize_predictions_transformer(model, test_loader, DEVICE, tokenizer, num_samples=100)

        # Run horizontal experiments for each model type
        if model_type == "lstm":
            experiment_loss_functions_lstm()
        else:
            experiment_loss_functions_transformer()
        experiment_learning_rates(model_type)
        experiment_batch_sizes(model_type)

    # Candidate Model Selection (using recorded candidate results)
    print("=== Final Model Selection ===")
    final_log.append("=== Final Model Selection ===")
    best_lstm_state = select_best_model_lstm()
    best_transformer_state = select_best_model_transformer()

    # Save final LSTM model state
    if best_lstm_state is None:
        print("No candidate LSTM model found; using baseline model as final model.")
        final_log.append("No candidate LSTM model found; using baseline model as final model.")
        best_lstm_state = baseline_lstm_state
    lstm_final_path = os.path.join(FINAL_MODEL_DIR, "lstm_final_model.pth")
    torch.save(best_lstm_state, lstm_final_path)
    msg = f"Final LSTM model saved to '{lstm_final_path}'"
    print(msg)
    final_log.append(msg)

    # Save final Transformer model state
    if best_transformer_state is None:
        print("No candidate Transformer model found; using baseline model as final model.")
        final_log.append("No candidate Transformer model found; using baseline model as final model.")
        best_transformer_state = baseline_transformer_state
    transformer_final_path = os.path.join(FINAL_MODEL_DIR, "transformer_final_model.pth")
    torch.save(best_transformer_state, transformer_final_path)
    msg = f"Final Transformer model saved to '{transformer_final_path}'"
    print(msg)
    final_log.append(msg)

    # Save final predictions CSV
    save_final_predictions_csv_lstm(best_lstm_state)
    save_final_predictions_csv_transformer(best_transformer_state)
    final_log.append("LSTM and Transformer final predictions CSV files saved.")
    msg = "All experiments completed. Please check the 'Results' folder for images and CSV files."
    print(msg)
    final_log.append(msg)

    # Write final log to a text file
    final_log_path = os.path.join(RESULTS_DIR, "Final_Model_Selection.txt")
    with open(final_log_path, "w") as f:
        for line in final_log:
            f.write(line + "\n")
    print(f"Final experiment log saved to '{final_log_path}'")

def main():
    args = parse_args()
    if args.mode == "train":
        if args.model_type == "lstm":
            run_training_lstm(args.loss, args.lr, args.batch_size, args.epochs)
        else:
            run_training_transformer(args.loss, args.lr, args.batch_size, args.epochs)
    elif args.mode == "experiments":
        run_experiments()
    else:
        raise ValueError("Unknown mode.")

if __name__ == "__main__":
    main()
