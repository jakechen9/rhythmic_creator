import torch
from preprocess.batch import Batch


class Train(Batch):
    def __init__(self, model, optimizer, evaluation_iters, indata, block_size, batch_size):
        super().__init__(indata, block_size, batch_size)
        self.evaluation_iters = evaluation_iters
        self.model = model
        self.optimizer = optimizer

    @torch.no_grad()
    def estimate_loss(self, device):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.evaluation_iters)
            for k in range(self.evaluation_iters):
                X, Y = self.create_batch(split, device)
                logits, loss = self.model(device, X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self, max_iters, evaluation_interval, device):
        for iteration in range(max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iteration % evaluation_interval == 0 or iteration == max_iters - 1:
                losses = self.estimate_loss(self.model, device)
                print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.create_batch('train', device)

            # evaluate the loss
            logits, loss = self.model(device, xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
