"""
Trains ONLY the LSTM + Bahdanau attention variant. Fully independent of
train_rnn.py / train_gru.py — run this on its own, stop it (Ctrl+C) whenever
you want, and it won't affect the other two. The checkpoint is saved after
every epoch, so stopping early still leaves you a usable model.

Usage:
    python train_lstm.py
"""

from model_lstm import Seq2Seq
from data import get_dataloaders
from train_utils import run_training

if __name__ == "__main__":
    print("Loading CNN/DailyMail and vocab (reuses vocab.json if it already exists)...")
    train_loader, val_loader, vocab = get_dataloaders(batch_size=32)
    print(f"Vocab size: {len(vocab)} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print(f"\n{'='*60}\nTraining variant: LSTM\n{'='*60}")
    result = run_training("lstm", Seq2Seq, train_loader, val_loader, vocab)
    print(f"\nDone: {result}")
