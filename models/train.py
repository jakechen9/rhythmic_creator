import torch
from preprocess.batch import Batch
from preprocess.preprocessing import MIDIProcessor
from models.transformerdecoder import DecoderModel
torch.manual_seed(1337)

path = '/content/rhythmic_creator/training_1.txt'
MIDIPreprocess = MIDIProcessor(path)
midi_flatten, unique_notes = MIDIPreprocess.split_text()
data = torch.tensor(MIDIPreprocess.encode_with_mapping(midi_flatten), dtype=torch.long)

# input size
batch_size = 64  # b
block_size = 256  # t
vocab_size = len(unique_notes)  # c

# attention mechanism params
n_embd = 192
num_heads = 6
n_layer = 6

# optimizer params
dropout = 0.2
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# eval params
max_iters = 5000
evaluation_interval = 500
evaluation_iters = 200

get_batch = Batch(data, block_size, batch_size)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iters)
        for k in range(evaluation_iters):
            X, Y = get_batch.create_batch(split, device)
            logits, loss = model(device, X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = DecoderModel(block_size, vocab_size, n_embd, num_heads, n_layer, dropout)
m = model.to(device)

print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iteration in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iteration % evaluation_interval == 0 or iteration == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch.create_batch('train', device)

    # evaluate the loss
    logits, loss = model(device, xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

