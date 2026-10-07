import torch
from model_rnn import Seq2Seq as Seq2SeqRNN
from model_gru import Seq2Seq as Seq2SeqGRU
from model_lstm import Seq2Seq as Seq2SeqLSTM

VOCAB_SIZE = 8000
EMB_DIM = 128
ENC_HIDDEN = 128
DEC_HIDDEN = 256   # must equal ENC_HIDDEN*2
ATTN_DIM = 128

PAD, SOS, EOS = 0, 1, 2

MODEL_CLASSES = {"rnn": Seq2SeqRNN, "gru": Seq2SeqGRU, "lstm": Seq2SeqLSTM}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanity_check(cell_type):
    model_cls = MODEL_CLASSES[cell_type]
    model = model_cls(VOCAB_SIZE, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, ATTN_DIM, pad_idx=PAD)
    n_params = count_params(model)

    # dummy batch: batch_size=4, src_len=40, trg_len=10
    batch_size, src_len, trg_len = 4, 40, 10
    src = torch.randint(3, VOCAB_SIZE, (batch_size, src_len))
    src_lengths = torch.full((batch_size,), src_len)
    trg = torch.randint(3, VOCAB_SIZE, (batch_size, trg_len))
    trg[:, 0] = SOS

    # forward (training mode, teacher forcing)
    out = model(src, src_lengths, trg, teacher_forcing_ratio=0.5)
    assert out.shape == (batch_size, trg_len, VOCAB_SIZE), f"bad shape {out.shape}"

    # generate (inference mode)
    gen = model.generate(src, src_lengths, sos_idx=SOS, eos_idx=EOS, max_len=15)
    assert gen.shape[0] == batch_size

    print(f"{cell_type.upper():5s} | params: {n_params:>10,} | "
          f"forward OK {tuple(out.shape)} | generate OK {tuple(gen.shape)}")
    return n_params


if __name__ == "__main__":
    print(f"Config: vocab={VOCAB_SIZE}, emb={EMB_DIM}, enc_hidden={ENC_HIDDEN}, "
          f"dec_hidden={DEC_HIDDEN}, attn_dim={ATTN_DIM}\n")
    for ct in ["rnn", "gru", "lstm"]:
        sanity_check(ct)
