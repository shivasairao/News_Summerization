"""
GRU encoder-decoder with Bahdanau attention, for news summarization.
Self-contained file — no dependency on rnn/lstm model files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size):
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Collapse bidirectional final hidden state -> (1, batch, hidden*2),
        # used directly as the unidirectional decoder's initial hidden state.
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(0)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask):
        src_len = enc_outputs.size(1)
        dec_hidden_exp = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)
        energy = self.v(torch.tanh(self.W1(enc_outputs) + self.W2(dec_hidden_exp))).squeeze(2)
        energy = energy.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, embedding, enc_dim, hidden_size, attn_dim, vocab_size):
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        emb_dim = embedding.embedding_dim

        self.attention = BahdanauAttention(enc_dim, hidden_size, attn_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim + enc_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.pre_out = nn.Linear(hidden_size + enc_dim, emb_dim)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_token, hidden, enc_outputs, mask):
        embedded = self.embedding(input_token).unsqueeze(1)
        dec_hidden_for_attn = hidden.squeeze(0)
        context, attn_weights = self.attention(dec_hidden_for_attn, enc_outputs, mask)

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)

        pre_logits = self.pre_out(torch.cat([output, context], dim=1))
        logits = F.linear(pre_logits, self.embedding.weight, self.out_bias)
        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, enc_hidden=128, dec_hidden=256,
                 attn_dim=128, pad_idx=0):
        super().__init__()
        assert dec_hidden == enc_hidden * 2, (
            "dec_hidden must equal enc_hidden*2 since the encoder's bidirectional "
            "final state is fed directly in as the decoder's initial hidden state."
        )
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder = Encoder(self.embedding, enc_hidden)
        self.decoder = Decoder(self.embedding, enc_dim=enc_hidden * 2,
                                hidden_size=dec_hidden, attn_dim=attn_dim,
                                vocab_size=vocab_size)
        self.pad_idx = pad_idx

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.embedding.num_embeddings
        device = src.device

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=device)
        enc_outputs, hidden = self.encoder(src, src_lengths)
        mask = (src != self.pad_idx)

        input_token = trg[:, 0]
        for t in range(1, trg_len):
            logits, hidden, _ = self.decoder(input_token, hidden, enc_outputs, mask)
            outputs[:, t] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs

    @torch.no_grad()
    def generate(self, src, src_lengths, sos_idx, eos_idx, max_len=30,
                 block_token_ids=None, no_repeat_ngram_size=3):
        """
        block_token_ids: vocab ids to forbid emitting (e.g. <unk>) — prevents
        collapse into always predicting the most frequent training token.
        no_repeat_ngram_size: forbid regenerating an already-seen n-gram —
        prevents the repetition-looping failure mode. Set 0 to disable.
        """
        self.eval()
        enc_outputs, hidden = self.encoder(src, src_lengths)
        mask = (src != self.pad_idx)
        batch_size = src.size(0)
        device = src.device

        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        generated = []
        history = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            logits, hidden, _ = self.decoder(input_token, hidden, enc_outputs, mask)
            if block_token_ids:
                logits[:, block_token_ids] = float("-inf")

            if no_repeat_ngram_size > 0:
                n = no_repeat_ngram_size
                for b in range(batch_size):
                    seq = history[b]
                    if len(seq) < n - 1:
                        continue
                    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
                    banned = {
                        seq[i + n - 1]
                        for i in range(len(seq) - n + 1)
                        if tuple(seq[i:i + n - 1]) == prefix
                    }
                    if banned:
                        logits[b, list(banned)] = float("-inf")

            top1 = logits.argmax(1)
            for b in range(batch_size):
                history[b].append(top1[b].item())
            generated.append(top1)
            input_token = top1
            if (top1 == eos_idx).all():
                break

        return torch.stack(generated, dim=1)

    @torch.no_grad()
    def generate_beam(self, src, src_lengths, sos_idx, eos_idx, max_len=30,
                      beam_width=5, block_token_ids=None, no_repeat_ngram_size=3,
                      length_penalty=0.7, min_len=5):
        """
        Beam search decoding. Same interface as generate() plus beam_width.
        Returns a (batch, L) LongTensor of token ids (padded with pad_idx).
        Each source in the batch is decoded independently, with the beam
        dimension used as the decoder batch dimension.
        length_penalty: finished score = sum_logprob / len ** length_penalty
        min_len: <eos> is blocked for the first min_len steps (avoids empty output).
        """
        self.eval()
        results = [
            self._beam_single(src[b:b + 1], src_lengths[b:b + 1], sos_idx, eos_idx,
                              max_len, max(1, beam_width), block_token_ids,
                              no_repeat_ngram_size, length_penalty, min_len)
            for b in range(src.size(0))
        ]
        width = max(len(r) for r in results)
        out = torch.full((len(results), width), self.pad_idx, dtype=torch.long,
                         device=src.device)
        for i, r in enumerate(results):
            out[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=src.device)
        return out

    def _beam_single(self, src, src_lengths, sos_idx, eos_idx, max_len, K,
                     block_token_ids, n, length_penalty, min_len):
        device = src.device
        enc_outputs, hidden = self.encoder(src, src_lengths)
        mask = (src != self.pad_idx)

        # replicate encoder results / initial state across the K beams
        enc_outputs = enc_outputs.expand(K, -1, -1).contiguous()
        mask = mask.expand(K, -1).contiguous()
        if isinstance(hidden, tuple):                      # LSTM: (h, c)
            hidden = tuple(h.expand(-1, K, -1).contiguous() for h in hidden)
        else:                                              # RNN / GRU
            hidden = hidden.expand(-1, K, -1).contiguous()

        seqs = [[] for _ in range(K)]                      # generated ids per live beam
        scores = torch.full((K,), float("-inf"), device=device)
        scores[0] = 0.0                                    # only beam 0 alive at step 0
        input_token = torch.full((K,), sos_idx, dtype=torch.long, device=device)
        finished = []                                      # (normalized_score, ids)
        min_len = min(min_len, max_len - 1)

        for t in range(max_len):
            logits, hidden, _ = self.decoder(input_token, hidden, enc_outputs, mask)
            logp = F.log_softmax(logits, dim=-1)
            if block_token_ids:
                logp[:, block_token_ids] = float("-inf")
            if t < min_len:
                logp[:, eos_idx] = float("-inf")

            if n > 0:
                for k in range(K):
                    seq = seqs[k]
                    if len(seq) < n - 1:
                        continue
                    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
                    banned = {
                        seq[i + n - 1]
                        for i in range(len(seq) - n + 1)
                        if tuple(seq[i:i + n - 1]) == prefix
                    }
                    if banned:
                        logp[k, list(banned)] = float("-inf")

            cand = (scores.unsqueeze(1) + logp).view(-1)
            top_scores, top_idx = cand.topk(min(2 * K, cand.numel()))
            V = logp.size(1)

            new_seqs, new_scores, src_beams, new_tokens = [], [], [], []
            for sc, idx in zip(top_scores.tolist(), top_idx.tolist()):
                if sc == float("-inf"):
                    continue
                k, tok = divmod(idx, V)
                if tok == eos_idx:
                    ids = seqs[k] + [eos_idx]
                    finished.append((sc / (len(ids) ** length_penalty), ids))
                else:
                    if len(new_seqs) < K:
                        new_seqs.append(seqs[k] + [tok])
                        new_scores.append(sc)
                        src_beams.append(k)
                        new_tokens.append(tok)

            # stop when enough hypotheses finished, or no live beams remain
            if len(finished) >= K or not new_seqs:
                break

            # pad up to K live beams if some candidates were -inf (rare)
            while len(new_seqs) < K:
                new_seqs.append(new_seqs[0]); new_scores.append(float("-inf"))
                src_beams.append(src_beams[0]); new_tokens.append(new_tokens[0])

            seqs = new_seqs
            scores = torch.tensor(new_scores, device=device)
            input_token = torch.tensor(new_tokens, dtype=torch.long, device=device)
            order = torch.tensor(src_beams, dtype=torch.long, device=device)
            if isinstance(hidden, tuple):
                hidden = tuple(h.index_select(1, order).contiguous() for h in hidden)
            else:
                hidden = hidden.index_select(1, order).contiguous()

        # hit max_len with unfinished beams: consider them as candidates too
        for k in range(K):
            if seqs[k] and scores[k].item() != float("-inf"):
                finished.append((scores[k].item() / (len(seqs[k]) ** length_penalty), seqs[k]))

        if not finished:
            return [eos_idx]
        return max(finished, key=lambda x: x[0])[1]