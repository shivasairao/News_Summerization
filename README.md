# News Summarizer — RNN vs GRU vs LSTM with Bahdanau Attention

An ablation study: three encoder-decoder architectures (vanilla RNN, GRU, LSTM),
each with Bahdanau attention, trained on CNN/DailyMail under an identical
parameter budget. The question this answers: **at a fixed capacity, how much
does the gating mechanism itself matter for abstractive summarization?**

## Architecture

- Tied embeddings (encoder input = decoder input = output projection weight)
- Bidirectional encoder (single layer)
- Unidirectional decoder with Bahdanau attention over encoder outputs
- Same vocab, same hidden sizes, same training schedule across all 3 variants —
  only the recurrent cell type changes
- Decode-time safeguards: `<unk>` is blocked from generation, and no-repeat
  trigram blocking prevents the model looping on the same phrase

Each architecture lives in its own self-contained file (no shared imports
between them) — verified parameter counts at vocab=8000, emb=128, enc_hidden=128,
dec_hidden=256, attn_dim=128:

| Variant | File | Params |
|---|---|---|
| RNN + Attention | `model_rnn.py` | 1,393,728 |
| GRU + Attention | `model_gru.py` | 1,854,528 |
| LSTM + Attention | `model_lstm.py` | 2,084,928 |

## Files

- `model_rnn.py`, `model_gru.py`, `model_lstm.py` — self-contained Encoder,
  BahdanauAttention, Decoder, Seq2Seq per architecture
- `data.py` — CNN/DailyMail loading, vocab building, tokenization, DataLoader
- `train_utils.py` — shared training/eval/ROUGE logic used by the three
  scripts below (not run directly)
- `train_rnn.py`, `train_gru.py`, `train_lstm.py` — **independent** training
  scripts, one per architecture. Run any one on its own, stop it (Ctrl+C)
  without affecting the others — each saves its checkpoint after every epoch,
  so stopping early still leaves a usable model. All three read/write the
  same `vocab.json`, so run in any order and they'll stay consistent.
- `param_count.py` — sanity-checks parameter counts and forward/generate shapes
  (run this before a real training run — no dataset download needed)
- `check_gpu.py` — confirms CUDA is actually usable (not just "available")
  before committing to a training run
- `app.py` — Streamlit demo: single-model or side-by-side comparison mode
- `requirements.txt`

## Setup — fresh start, in order

```bash
# 1. new venv
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # macOS/Linux

# 2. CUDA-enabled torch FIRST, before anything else
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. rest of the deps
pip install datasets rouge-score streamlit tqdm --break-system-packages

# 4. confirm the GPU is actually usable (not just "available")
python check_gpu.py         # must print ALL CHECKS PASSED before continuing

# 5. sanity-check the architecture (~10 sec, no download)
python param_count.py

# 6. train — run these separately, in any order, whenever you want.
#    Each is independent: stop one with Ctrl+C without affecting the others.
python train_rnn.py
python train_gru.py
python train_lstm.py

# 7. demo (needs all three checkpoints + vocab.json to exist)
streamlit run app.py
```

## Design notes / known limitations (worth stating explicitly in your writeup)

- Source truncated to 150 tokens, target to 30 tokens, to keep sequences
  tractable for the small-to-mid parameter range this project targets.
- Word-level tokenization with an 8,000-word vocab (not subword/BPE) — simpler
  to reason about, but still has a real UNK rate on news-domain proper nouns.
  `<unk>` is blocked at generation time so it can't dominate output, but the
  underlying OOV rate is still a real quality ceiling — a follow-up experiment
  worth trying is swapping in a small BPE vocab.
- No-repeat trigram blocking prevents the model looping on a phrase, but
  doesn't fix ungrounded/hallucinated content (e.g. mentioning "the U.S." for
  an article about India) — that's a sign the model needs more training
  and/or capacity to properly learn to attend to the source, not something a
  decode-time trick can fix.
- No coverage mechanism or pointer-generator copying (unlike production
  summarizers) — flagged here as explicit "future work," a well-known fix for
  repetition and for correctly copying names/numbers from source.

## Next steps to extend this into a bigger multi-domain project

1. Add ROUGE tracking per-epoch (not just final) to see if variants converge
   at different rates, not just different final quality
2. Add a pointer-generator copy mechanism for names/numbers/OOV terms
3. Swap in BPE tokenization to reduce the OOV rate
4. Reuse these three model files as the smallest tier in a multi-domain
   (reviews + news + stories) system, scaling hidden sizes up per domain
