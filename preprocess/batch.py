import torch


class Batch:
    def __init__(self, indata, block_size, batch_size):
        self.indata = indata
        self.block_size = block_size
        self.batch_size = batch_size

    def create_batch(self, split, device):
        n = int(0.9 * len(self.indata))
        data = self.indata[:n] if split == 'train' else self.indata[n:]
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + self.block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
