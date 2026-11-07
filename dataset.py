import urllib.request
import torch

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")

text = open("input.txt", encoding="utf-8").read()

chars = sorted(set(text))
n_vocab = len(chars)                          
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]