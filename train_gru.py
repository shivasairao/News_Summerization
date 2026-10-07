"""
Trains ONLY the GRU + Bahdanau attention variant. Fully independent of
train_rnn.py / train_lstm.py — run this on its own, stop it (Ctrl+C) whenever
you want, and it won't affect the other two. The checkpoint is saved after
every epoch, so stopping early still leaves you a usable model.

Usage:
    python train_gru.py
"""

from model_gru import Seq2Seq
from data import get_dataloaders
from train_utils import run_training

if __name__ == "__main__":
    print("Loading CNN/DailyMail and vocab (reuses vocab.json if it already exists)...")
    train_loader, val_loader, vocab = get_dataloaders(batch_size=32)
    print(f"Vocab size: {len(vocab)} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print(f"\n{'='*60}\nTraining variant: GRU\n{'='*60}")
    result = run_training("gru", Seq2Seq, train_loader, val_loader, vocab)
    print(f"\nDone: {result}")
