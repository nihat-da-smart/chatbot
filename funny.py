# pip install torch tokenizers convokit numpy tqdm

from convokit import Corpus, download
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random, os, sys, re
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# PARAMETERS
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
NUM_LAYERS = 3
HIDDEN_DIM = 512
EMBED_DIM = 25
BLOCK_SIZE = 20
BATCH_SIZE = 32
NUM_EPOCHS = 50
VOCAB_SIZE = 6000
MAX_EXAMPLES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- CLEANING FUNCTION ---------
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Collapse spaces
    return text.strip()

# ---------- DATA ----------
print("Loading Cornell Movie Corpus...")
corpus = Corpus(filename=download("movie-corpus"))
pairs = []
for conv in corpus.iter_conversations():
    utts = list(conv.iter_utterances())
    for i in range(len(utts) - 1):
        prompt = clean_text(utts[i].text)     # <-- changed
        reply  = clean_text(utts[i+1].text)   # <-- changed
        if prompt and reply:
            pairs.append((prompt, reply))
print(f"Loaded {len(pairs)} (prompt, reply) pairs.")

# --------- TOKENIZER ---------
tokenizer_path = "slm_bpe_tokenizer.json"
if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("Loaded existing tokenizer.")
else:
    print("Training subword tokenizer...")
    all_text = [p for pair in pairs for p in pair]
    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True,
                                  special_tokens=["<PAD>", "<BOS>", "<EOS>"])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(all_text, trainer)
    tokenizer.save(tokenizer_path)
    print("Tokenizer trained and saved.")

pad_token, bos_token, eos_token = "<PAD>", "<BOS>", "<EOS>"
pad_id = tokenizer.token_to_id(pad_token)
bos_id = tokenizer.token_to_id(bos_token)
eos_id = tokenizer.token_to_id(eos_token)
vocab_size = tokenizer.get_vocab_size()

def encode_with_eos(text, max_len):
    ids = tokenizer.encode(text).ids[:max_len-1]
    return [bos_id] + ids + [eos_id]

def pad_seq(seq, max_len):
    return seq + [pad_id] * (max_len - len(seq))

# --------- VECTORIZE DATA ----------
random.shuffle(pairs)
inputs, targets = [], []
for prompt, reply in tqdm(pairs[:MAX_EXAMPLES], desc="Encoding pairs"):
    enc, dec = encode_with_eos(prompt, BLOCK_SIZE), encode_with_eos(reply, BLOCK_SIZE)
    if len(enc) <= BLOCK_SIZE+1 and len(dec) <= BLOCK_SIZE+1:
        inputs.append(pad_seq(enc, BLOCK_SIZE+1))
        targets.append(pad_seq(dec, BLOCK_SIZE+1))
inputs, targets = torch.tensor(inputs), torch.tensor(targets)
print(f"Training on {len(inputs)} examples.")

# PyTorch DataLoader
dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# --------- MODEL ----------
class DotAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        attn_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        if src_mask is not None:
            attn_scores = attn_scores.masked_fill(~src_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Seq2SeqLSTM_Attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.attn = DotAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.pad_id = pad_id

    def forward(self, src, tgt):
        batch_size = src.size(0)
        enc_emb = self.embed(src)
        encoder_outputs, (h, c) = self.encoder(enc_emb)

        dec_input = self.embed(tgt[:, 0])
        h_t, c_t = h[-1], c[-1]

        outputs = []
        for t in range(1, tgt.size(1)):
            context, _ = self.attn(h_t, encoder_outputs)
            dec_in = torch.cat([dec_input, context], dim=1)
            h_t, c_t = self.decoder(dec_in, (h_t, c_t))
            out = self.fc(h_t)
            outputs.append(out)
            dec_input = self.embed(tgt[:, t])
        return torch.stack(outputs, dim=1)

    def encode(self, src):
        enc_emb = self.embed(src)
        encoder_outputs, (h, c) = self.encoder(enc_emb)
        return encoder_outputs, h[-1], c[-1]

model = Seq2SeqLSTM_Attn(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# --------- TRAINING LOOP ----------
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, y)
        loss = criterion(logits.reshape(-1, vocab_size), y[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
print("Training finished.")

# --------- INFERENCE ----------
def generate_reply(model, prompt, max_len=40, temp=1.0):
    model.eval()
    with torch.no_grad():
        prompt = clean_text(prompt)   # <-- changed: clean user input too!
        src = encode_with_eos(prompt, BLOCK_SIZE)
        src = pad_seq(src, BLOCK_SIZE+1)
        src_tensor = torch.tensor([src], dtype=torch.long).to(DEVICE)
        encoder_outputs, h_t, c_t = model.encode(src_tensor)
        generated = [bos_id]
        dec_input = model.embed(torch.tensor([bos_id], dtype=torch.long).to(DEVICE))
        for _ in range(max_len):
            context, _ = model.attn(h_t, encoder_outputs)
            dec_in = torch.cat([dec_input, context], dim=1)
            h_t, c_t = model.decoder(dec_in, (h_t, c_t))
            out = model.fc(h_t)
            probs = torch.softmax(out / temp, dim=-1).squeeze().cpu().numpy()
            next_id = int(np.random.choice(len(probs), p=probs / probs.sum()))
            if next_id == eos_id:
                break
            generated.append(next_id)
            dec_input = model.embed(torch.tensor([next_id], dtype=torch.long).to(DEVICE))
        return tokenizer.decode([i for i in generated[1:]])

# --------- CHAT LOOP ----------
print("\nChatbot ready! (type 'quit' to exit)\n")
while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"quit", "exit"}:
        break
    reply = generate_reply(model, prompt, max_len=40, temp=0.4)
    print(f"Bot: {reply}")
