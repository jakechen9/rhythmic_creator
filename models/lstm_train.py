import torch
from preprocess.batch import Batch


class LSTMTrain(Batch):
    def __init__(self, model, optimizer, evaluation_iters, indata, block_size, batch_size):
        super().__init__(indata, block_size, batch_size)
        self.evaluation_iters = evaluation_iters
        self.model = model
        self.optimizer = optimizer

    @torch.no_grad()
    def estimate_loss(self, device):
        out = {}
        self.model.eval()
        hidden = self.model.init_hidden(self.batch_size, device)
        for split in ['train', 'val']:
            losses = torch.zeros(self.evaluation_iters)
            for k in range(self.evaluation_iters):
                hidden = self.model.detach_hidden(hidden)
                X, Y = self.create_batch(split, device)
                logits, loss, hidden = self.model(device, X, hidden, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self, max_iters, evaluation_interval, device):
        for iteration in range(max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iteration % evaluation_interval == 0 or iteration == max_iters - 1:
                losses = self.estimate_loss(device)
                print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            hidden = self.model.init_hidden(self.batch_size, device)
            # sample a batch of data
            xb, yb = self.create_batch('train', device)

            # evaluate the loss
            logits, loss, hidden = self.model(device, xb, hidden, yb)
            self.optimizer.zero_grad(set_to_none=True)
            # hidden = self.model.detach_hidden(hidden)
            loss.backward()
            self.optimizer.step()
