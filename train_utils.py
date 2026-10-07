"""
Shared helpers for train_rnn.py / train_gru.py / train_lstm.py.
Not meant to be run directly — each of those three scripts is fully
independent at the point of execution (you can Ctrl+C any one of them
without touching the others), they just share this identical logic so
training/eval/ROUGE code isn't triplicated three times.
"""

import time
import torch
import torch.nn as nn
from tqdm import tqdm

EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, ATTN_DIM = 128, 128, 256, 128
EPOCHS = 15
LR = 1e-3
CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc="  train", leave=False)
    for src, src_lengths, trg in progress:
        src, src_lengths, trg = src.to(DEVICE), src_lengths.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, src_lengths, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for src, src_lengths, trg in loader:
        src, src_lengths, trg = src.to(DEVICE), src_lengths.to(DEVICE), trg.to(DEVICE)
        output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def ids_to_text(ids, vocab):
    words = []
    for i in ids:
        w = vocab.itos[i]
        if w == "<eos>":
            break
        if w not in ("<pad>", "<sos>"):
            words.append(w)
    return " ".join(words)


@torch.no_grad()
def compute_rouge(model, loader, vocab, sos_idx, eos_idx, unk_idx, n_batches=20):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    n = 0
    model.eval()
    for i, (src, src_lengths, trg) in enumerate(loader):
        if i >= n_batches:
            break
        src, src_lengths = src.to(DEVICE), src_lengths.to(DEVICE)
        gen = model.generate(src, src_lengths, sos_idx, eos_idx, max_len=30,
                              block_token_ids=[unk_idx], no_repeat_ngram_size=3)
        for g, t in zip(gen.tolist(), trg.tolist()):
            pred = ids_to_text(g, vocab)
            ref = ids_to_text(t, vocab)
            if not pred.strip() or not ref.strip():
                continue
            r = scorer.score(ref, pred)
            for k in scores:
                scores[k] += r[k].fmeasure
            n += 1
    return {k: v / max(n, 1) for k, v in scores.items()}


def run_training(cell_type, model_cls, train_loader, val_loader, vocab):
    print(f"Device: {DEVICE}" +
          (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda"
           else " — CPU only, this will be slow."))

    model = model_cls(len(vocab), EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, ATTN_DIM,
                       pad_idx=vocab.stoi["<pad>"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss {train_loss:.3f} | val_loss {val_loss:.3f}")

        # save after every epoch, not just at the end — so a Ctrl+C mid-run
        # still leaves you with the latest usable checkpoint, not nothing
        torch.save(model.state_dict(), f"{cell_type}_summarizer.pt")

    train_time = time.time() - start

    rouge = compute_rouge(model, val_loader, vocab, vocab.stoi["<sos>"],
                           vocab.stoi["<eos>"], vocab.stoi["<unk>"])
    print(f"ROUGE: {rouge}")
    print(f"Total train time: {train_time:.1f}s")
    print(f"Saved {cell_type}_summarizer.pt")

    return {
        "cell_type": cell_type,
        "params": n_params,
        "train_time_sec": round(train_time, 1),
        "final_val_loss": round(val_loss, 3),
        **{k: round(v, 4) for k, v in rouge.items()},
    }
