# News Summarizer: RNN vs GRU vs LSTM with Bahdanau Attention

An ablation study of three encoder-decoder architectures (vanilla RNN, GRU, LSTM), each with Bahdanau attention. All three are trained from scratch on CNN/DailyMail at a modest, fixed capacity. A Streamlit app lets you paste an article and compare the generated summaries side by side.

**Question:** at a fixed capacity, how much does the recurrent cell type (and its gating mechanism) matter for abstractive news summarization?

---

## Features

- Three self-contained models: RNN, GRU and LSTM, each with Bahdanau attention
- Shared vocabulary, hidden sizes and training schedule, so only the recurrent cell differs
- Beam search decoding with:
  - `<unk>` / `<sos>` blocking
  - no-repeat n-gram blocking
  - length-normalized scoring
- Streamlit demo with a single-model mode and a compare-all-three mode
- Independent training scripts: stop any one with `Ctrl+C` and the latest checkpoint is kept
- Sanity-check scripts for the GPU and the architecture, so you can verify things before a long run

---

## Architecture

- Tied embeddings (encoder input, decoder input and output projection share weights)
- Single-layer bidirectional encoder
- Unidirectional decoder with Bahdanau (additive) attention over the encoder outputs
- Encoder final state (forward and backward concatenated) initializes the decoder

| Setting | Value |
|---|---|
| Vocabulary | 8,000 words (word-level) |
| Embedding dim | 128 |
| Encoder hidden (per direction) | 128 |
| Decoder hidden | 256 |
| Attention dim | 128 |
| Source length | 150 tokens (training) |
| Target length | 30 tokens |
| Optimizer | Adam, lr 1e-3, grad-clip 1.0 |
| Epochs | 15 |
| Teacher forcing | 0.5 |

| Variant | File | Parameters |
|---|---|---|
| RNN + Attention | `model_rnn.py` | 1,393,728 |
| GRU + Attention | `model_gru.py` | 1,854,528 |
| LSTM + Attention | `model_lstm.py` | 2,084,928 |

Parameter counts differ because gated cells have more weights per unit (GRU has 3 gates' worth, LSTM has 4). The hidden sizes are identical across variants.

---

## Project structure

```
.
├── app.py               # Streamlit demo
├── data.py              # CNN/DailyMail loading, tokenizer, vocab, DataLoaders
├── model_rnn.py         # RNN encoder-decoder + attention (self-contained)
├── model_gru.py         # GRU encoder-decoder + attention (self-contained)
├── model_lstm.py        # LSTM encoder-decoder + attention (self-contained)
├── train_utils.py       # shared train / eval / ROUGE logic
├── train_rnn.py         # train the RNN variant
├── train_gru.py         # train the GRU variant
├── train_lstm.py        # train the LSTM variant
├── param_count.py       # parameter counts + forward/generate sanity check
├── check_gpu.py         # confirms CUDA is usable by PyTorch
├── requirements.txt
├── vocab.json           # vocabulary (created by the first training run)
└── *_summarizer.pt      # trained checkpoints (created by training)
```

Each `model_*.py` file contains its own Encoder, Attention, Decoder and Seq2Seq classes, with no shared imports between them. It exposes `generate()` (greedy) and `generate_beam()` (beam search).

---

## Setup

```bash
# 1. create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate            # Windows
# source myenv/bin/activate       # macOS / Linux

# 2. install a CUDA-enabled PyTorch FIRST (adjust cu128 to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. install the remaining dependencies
pip install datasets rouge-score streamlit tqdm
# or: pip install -r requirements.txt   (after step 2)
```

CPU-only works too, but training will be slow.

### Verify before training

```bash
python check_gpu.py      # should end with "ALL CHECKS PASSED"
python param_count.py    # ~10 s, no dataset download needed
```

---

## Training

Each variant trains independently and can be run in any order:

```bash
python train_rnn.py
python train_gru.py
python train_lstm.py
```

- The first run downloads CNN/DailyMail through Hugging Face `datasets` and builds `vocab.json`. Later runs reuse it, so all three variants share the exact same vocabulary.
- A checkpoint (`rnn_summarizer.pt`, `gru_summarizer.pt` or `lstm_summarizer.pt`) is saved after every epoch, so stopping early still leaves a usable model.
- When training finishes, the script prints validation loss and ROUGE-1/2/L on 20 validation batches.

---

## Running the demo

```bash
streamlit run app.py
```

Requires `vocab.json` and the checkpoint files in the same folder as `app.py`.

- **Single model:** pick RNN, GRU or LSTM and summarize.
- **Compare all three:** see the three summaries side by side.
- **Controls:**
  - maximum summary length
  - repetition strictness (n-gram block size)
  - beam width (1 means greedy decoding)

Inputs longer than 150 tokens work, up to 700 tokens at inference. Quality on the later part of a long article may be weaker, since the models were trained on 150-token sources.

---

## Results

Fill in after training (printed at the end of each `train_*.py` run):

| Variant | Params | Val loss | ROUGE-1 | ROUGE-2 | ROUGE-L | Train time |
|---|---|---|---|---|---|---|
| RNN  | 1.39M | | | | | |
| GRU  | 1.85M | | | | | |
| LSTM | 2.08M | | | | | |

---

## Limitations

- Word-level tokenization with an 8,000-word vocabulary leaves a real out-of-vocabulary rate on news proper nouns. `<unk>` is blocked at generation time, but the OOV rate still limits quality.
- There is no coverage mechanism and no pointer-generator copying, so names and numbers are not reliably copied from the source and the model can still repeat itself or hallucinate. Decoding tricks such as beam search and n-gram blocking reduce repetition but do not fix ungrounded content.
- The models are deliberately small, to isolate the effect of the cell type. They are not intended to match state-of-the-art summarizers.
- ROUGE is computed on a small subset of the validation set during training, so treat it as an indicative figure rather than a benchmark.

---

## Future work

1. Track ROUGE per epoch to compare convergence speed across variants
2. Add a pointer-generator copy mechanism and a coverage penalty
3. Replace word-level tokens with BPE to reduce OOV
4. Use these models as the smallest tier of a multi-domain summarizer (reviews, news, stories)

---

## Dataset

[CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail), loaded through Hugging Face `datasets`.

## License

Add a license of your choice (for example MIT) as a `LICENSE` file.
