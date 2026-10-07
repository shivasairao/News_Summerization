"""
Streamlit demo — paste a news article, pick RNN / GRU / LSTM, see the generated
summary from each. Requires trained checkpoints (rnn_summarizer.pt,
gru_summarizer.pt, lstm_summarizer.pt) and vocab.json, all produced by train.py.

Run:
    streamlit run app.py
"""

import json
import torch
import streamlit as st

from model_rnn import Seq2Seq as Seq2SeqRNN
from model_gru import Seq2Seq as Seq2SeqGRU
from model_lstm import Seq2Seq as Seq2SeqLSTM
from data import Vocab, simple_tokenize, MAX_SRC_LEN, PAD

EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, ATTN_DIM = 128, 128, 256, 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model was TRAINED on articles truncated to MAX_SRC_LEN (150 tokens, see
# data.py). This app-level constant lets you feed longer input at inference
# time anyway — mechanically it works (RNN/GRU/LSTM handle variable-length
# sequences), but quality on the later part of a long article may be weaker
# than on short articles, since the model never saw sequences this long
# during training. ~500 words is roughly 550-650 tokens once punctuation is
# counted, so this gives real headroom above that.
INFERENCE_MAX_SRC_LEN = 700

CHECKPOINTS = {
    "RNN + Attention": "rnn_summarizer.pt",
    "GRU + Attention": "gru_summarizer.pt",
    "LSTM + Attention": "lstm_summarizer.pt",
}
MODEL_CLASSES = {
    "RNN + Attention": Seq2SeqRNN,
    "GRU + Attention": Seq2SeqGRU,
    "LSTM + Attention": Seq2SeqLSTM,
}


@st.cache_resource
def load_vocab():
    return Vocab.load("vocab.json")


@st.cache_resource
def load_model(cell_type_label, _vocab):
    model_cls = MODEL_CLASSES[cell_type_label]
    model = model_cls(len(_vocab), EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, ATTN_DIM,
                       pad_idx=_vocab.stoi[PAD]).to(DEVICE)
    state = torch.load(CHECKPOINTS[cell_type_label], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def summarize(model, vocab, text, max_len=30, no_repeat_ngram_size=2, beam_width=5):
    tokens = simple_tokenize(text)
    ids = vocab.encode(tokens, INFERENCE_MAX_SRC_LEN, add_eos=False)
    src = torch.tensor([ids], device=DEVICE)
    src_lengths = torch.tensor([len(ids)], device=DEVICE)

    # Beam search: explores several candidate continuations at once and keeps
    # the most likely overall sequence, instead of greedily taking the single
    # most-likely next word at each step (which compounds small mistakes into
    # incoherent output). Same <unk>/<sos> blocking and no-repeat n-gram
    # blocking as before, just a better search strategy over the same weights.
    gen = model.generate_beam(
        src, src_lengths, vocab.stoi["<sos>"], vocab.stoi["<eos>"], max_len=max_len,
        beam_width=beam_width,
        block_token_ids=[vocab.stoi["<unk>"], vocab.stoi["<sos>"]],
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    words = []
    for i in gen[0].tolist():
        w = vocab.itos[i]
        if w == "<eos>":
            break
        if w not in ("<pad>", "<sos>"):
            words.append(w)
    return " ".join(words) if words else "(empty output — try a longer article or a different variant)"


def main():
    st.set_page_config(page_title="News Summarizer — RNN vs GRU vs LSTM", layout="centered")
    st.title("News Summarizer")
    st.caption("RNN vs GRU vs LSTM encoder-decoder, each with Bahdanau attention, "
               "trained from scratch on CNN/DailyMail (~1.4M-2.1M params depending "
               "on cell type). Decoded with beam search.")

    try:
        vocab = load_vocab()
    except FileNotFoundError:
        st.error(
            "vocab.json not found. Run `python train.py` first — it saves vocab.json "
            "plus one checkpoint per variant (rnn_summarizer.pt, gru_summarizer.pt, "
            "lstm_summarizer.pt) in the same directory as this app."
        )
        st.stop()

    mode = st.radio("Mode", ["Single model", "Compare all three"], horizontal=True)

    article = st.text_area(
        "Paste a news article",
        height=220,
        placeholder="Paste article text here (long articles are truncated to the first "
                    f"{INFERENCE_MAX_SRC_LEN} tokens \u2014 about 500-600 words)...",
    )
    max_len = st.slider("Max summary length (tokens)", 10, 50, 30)
    no_repeat_n = st.slider(
        "Repetition strictness (lower = stricter, blocks repeats sooner)", 2, 4, 2,
        help="Forbids the model from regenerating any word-sequence of this length that "
             "already appeared. 2 = no word can repeat more than twice in a row. "
             "3 = allows up to 3 repeats before blocking. Lower is stricter.",
    )
    beam_width = st.slider(
        "Beam width (higher = more fluent, slower)", 1, 8, 5,
        help="Number of candidate sequences explored in parallel during generation. "
             "1 = greedy decoding (fast, more prone to incoherent output). "
             "Higher values generally produce more fluent, coherent text at the cost "
             "of more compute per summary.",
    )

    if mode == "Single model":
        variant = st.selectbox("Model variant", list(CHECKPOINTS.keys()))
        if st.button("Summarize", type="primary") and article.strip():
            try:
                model = load_model(variant, vocab)
            except FileNotFoundError:
                st.error(f"{CHECKPOINTS[variant]} not found. Train that variant first with train.py.")
                st.stop()
            with st.spinner(f"Generating with {variant}..."):
                summary = summarize(model, vocab, article, max_len, no_repeat_n, beam_width)
            st.subheader("Summary")
            st.write(summary)

    else:
        if st.button("Compare all three", type="primary") and article.strip():
            cols = st.columns(3)
            for col, variant in zip(cols, CHECKPOINTS.keys()):
                with col:
                    st.markdown(f"**{variant}**")
                    try:
                        model = load_model(variant, vocab)
                    except FileNotFoundError:
                        st.warning(f"{CHECKPOINTS[variant]} not found — skip")
                        continue
                    with st.spinner("..."):
                        summary = summarize(model, vocab, article, max_len, no_repeat_n, beam_width)
                    st.write(summary)

    with st.expander("About this project"):
        st.markdown(
            "- Tied embeddings, bidirectional encoder, Bahdanau attention decoder\n"
            "- Same vocab (8,000 words), same hidden sizes, same training schedule "
            "across all three cell types — only the recurrent gating mechanism differs\n"
            "- Trained on CNN/DailyMail, source truncated to 150 tokens, summary to 30\n"
            "- ~1.4M-2.1M params depending on cell type, to isolate the effect of the "
            "cell type itself under a controlled, modest capacity — not to chase "
            "state-of-the-art summary quality\n"
            "- Decoded with beam search (not greedy) for more coherent output, plus "
            "<unk>-blocking and no-repeat n-gram blocking"
        )


if __name__ == "__main__":
    main()