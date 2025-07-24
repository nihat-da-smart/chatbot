# Install needed packages (uncomment if needed)
# pip install torch tokenizers convokit numpy

from convokit import Corpus, download
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import sys

# ------------------ PARAMETERS ------------------
NUM_LAYERS = 2
HIDDEN_DIM = 512
EMBED_DIM = 512
BLOCK_SIZE = 20      # max tokens for prompt or reply (truncate/clip)
BATCH_SIZE = 64      # Larger batch, trains faster if RAM allows
NUM_EPOCHS = 30      # You may need 20+ for good output
VOCAB_SIZE = 3000    # for BPE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- 1. Load and Prepare Data ----------
print("Loading Cornell Movie Corpus...")
corpus = Corpus(filename=download("movie-corpus"))
pairs = []
for conv in corpus.iter_conversations():
    utts = list(conv.iter_utterances())
    for i in range(len(utts) - 1):
        prompt = utts[i].text.strip()
        reply = utts[i+1].text.strip()
        if prompt and reply:
            pairs.append((prompt, reply))
print(f"Loaded {len(pairs)} (prompt, reply) pairs.")

# ----------- 2. Subword Tokenizer (BPE) ------------
print("Training subword tokenizer...")
all_text = [p for pair in pairs for p in pair]
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True, special_tokens=["<PAD>", "<BOS>", "<EOS>"])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.train_from_iterator(all_text, trainer)
tokenizer.save("slm_bpe_tokenizer.json")
print("Tokenizer trained.")

pad_token = "<PAD>"
bos_token = "<BOS>"
eos_token = "<EOS>"
pad_id = tokenizer.token_to_id(pad_token)
bos_id = tokenizer.token_to_id(bos_token)
eos_id = tokenizer.token_to_id(eos_token)

def encode_with_eos(text, max_len):
    ids = tokenizer.encode(text).ids[:max_len-1]
    return [bos_id] + ids + [eos_id]

def pad_seq(seq, max_len):
    return seq + [pad_id] * (max_len - len(seq))

vocab_size = tokenizer.get_vocab_size()

# ----------- 3. Training Data Loader -------------
inputs, targets = [], []
dropped = 0
MAX_EXAMPLES = 10000  # Or 5000 for much faster startup!
for prompt, reply in random.sample(pairs, min(MAX_EXAMPLES, len(pairs))):
    enc = encode_with_eos(prompt, BLOCK_SIZE)
    dec = encode_with_eos(reply, BLOCK_SIZE)
    if len(enc) <= BLOCK_SIZE+1 and len(dec) <= BLOCK_SIZE+1:
        inputs.append(pad_seq(enc, BLOCK_SIZE+1))
        targets.append(pad_seq(dec, BLOCK_SIZE+1))
    else:
        dropped += 1
print(f"Training on {len(inputs)} examples. Dropped {dropped} pairs due to length.")

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

def get_batch(batch_size=BATCH_SIZE):
    idx = torch.randint(0, inputs.size(0), (batch_size,))
    return inputs[idx].to(DEVICE), targets[idx].to(DEVICE)

class DotAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        # decoder_hidden: [batch, hidden]
        # encoder_outputs: [batch, seq_len, hidden]
        # Calculate attention weights
        attn_scores = torch.bmm(
            encoder_outputs, decoder_hidden.unsqueeze(2)
        ).squeeze(2)  # [batch, seq_len]
        if src_mask is not None:
            attn_scores = attn_scores.masked_fill(~src_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, seq_len]
        # Weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
        context = context.squeeze(1)  # [batch, hidden]
        return context, attn_weights  # return weights if you want visualization


class Seq2SeqLSTM_Attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)  # decoder gets context!
        self.attn = DotAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, src, tgt):
        # src, tgt: [batch, seq_len]
        batch_size = src.size(0)
        enc_emb = self.embed(src)
        encoder_outputs, (h, c) = self.encoder(enc_emb)  # encoder_outputs: [batch, seq_len, hidden]

        # Decoder initial state
        dec_input = self.embed(tgt[:, 0])  # [batch, embed_dim], start with <BOS>
        h_t = h[-1]  # [batch, hidden]
        c_t = c[-1]  # [batch, hidden]

        outputs = []
        for t in range(1, tgt.size(1)):
            # Attention over encoder outputs
            context, _ = self.attn(h_t, encoder_outputs)
            dec_in = torch.cat([dec_input, context], dim=1)  # [batch, embed+hidden]
            h_t, c_t = self.decoder(dec_in, (h_t, c_t))
            out = self.fc(h_t)  # [batch, vocab]
            outputs.append(out)
            dec_input = self.embed(tgt[:, t])  # Teacher forcing

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len-1, vocab]
        return outputs

    def encode(self, src):
        enc_emb = self.embed(src)
        encoder_outputs, (h, c) = self.encoder(enc_emb)
        return encoder_outputs, h[-1], c[-1]


model = Seq2SeqLSTM_Attn(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# ----------- 5. Training Loop ---------------------
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    total_loss = 0
    num_batches = max(1, inputs.size(0) // BATCH_SIZE)
    for batch_idx in range(num_batches):
        x, y = get_batch()
        optimizer.zero_grad()
        logits = model(x, y)
        loss = criterion(logits.reshape(-1, vocab_size), y[:,1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            print(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}", end='\r')
            sys.stdout.flush()
    avg_loss = total_loss / num_batches
    print(f"\nEpoch {epoch}, Avg Loss: {avg_loss:.4f}")
print('Training finished.')
model.eval()
total_loss = 0

# ----------- 6. Inference (Random Sampling) -------
def generate_reply(model, prompt, max_len=40, temp=1.0):
    model.eval()
    with torch.no_grad():
        src = encode_with_eos(prompt, BLOCK_SIZE)
        src = pad_seq(src, BLOCK_SIZE+1)
        src_tensor = torch.tensor([src], dtype=torch.long).to(DEVICE)
        encoder_outputs, h_t, c_t = model.encode(src_tensor)
        generated = [bos_id]
        dec_input = model.embed(torch.tensor([bos_id], dtype=torch.long).to(DEVICE)).squeeze(0)
        for _ in range(max_len):
            context, _ = model.attn(h_t, encoder_outputs)
            dec_in = torch.cat([dec_input, context], dim=-1).unsqueeze(0)  # [1, embed+hidden]
            h_t, c_t = model.decoder(dec_in, (h_t, c_t))
            out = model.fc(h_t)
            probs = torch.softmax(out / temp, dim=-1).cpu().numpy().flatten()
            next_id = int(np.random.choice(len(probs), p=probs / probs.sum()))
            if next_id == eos_id:
                break
            generated.append(next_id)
            dec_input = model.embed(torch.tensor([next_id], dtype=torch.long).to(DEVICE)).squeeze(0)
        return tokenizer.decode([i for i in generated[1:]])  # skip <BOS>

# ----------- 7. Chatbot Demo Loop -----------------
print("Chatbot ready! (type 'quit' to exit)\n")
while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"quit", "exit"}:
        break
    reply = generate_reply(model, prompt, max_len=40, temp=0.8)  # Lower temp = safer, higher = more creative
    print(f"Bot: {reply}")
