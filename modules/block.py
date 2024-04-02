import torch.nn as nn
from modules.sublayers import MultiHeadAttention
from modules.feedforward import MlPFeedForward, LSTMFeedForward, MlPFeedForward2


class AttentionBlock(nn.Module):

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        self.sa2 = MultiHeadAttention(num_heads // 2, head_size * 2, n_embd, block_size, dropout)
        self.ffwd = MlPFeedForward(n_embd, dropout)
        self.ffwd_2 = MlPFeedForward2(n_embd, dropout)
        self.lyr_norm1 = nn.LayerNorm(n_embd)
        self.lyr_norm2 = nn.LayerNorm(n_embd)
        self.lyr_norm4 = nn.LayerNorm(n_embd)
        self.lyr_norm5 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.lyr_norm1(x))
        x = x + self.ffwd(self.lyr_norm2(x))  # skip connection, residual connection.
        x = x + self.sa2(self.lyr_norm4(x))
        x = x + self.ffwd_2(self.lyr_norm5(x))

        return x


class BlockTwo(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout, n_hidden, lstm_layers):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        self.lstm_lyr = LSTMFeedForward(n_embd, n_hidden, lstm_layers, dropout)
        self.ffwd = MlPFeedForward(n_embd, dropout)
        self.lyr_norm1 = nn.LayerNorm(n_embd)
        self.lyr_norm2 = nn.LayerNorm(n_embd)
        self.lyr_norm4 = nn.LayerNorm(n_embd)

    def forward(self, inputs):
        x, hidden = inputs
        x = x + self.sa(self.lyr_norm1(x))
        x_lstm, h = self.lstm_lyr(self.lyr_norm4(x), hidden)
        x = x + x_lstm

        # x = x + self.ffwd(self.lyr_norm2(x))  # skip connection, residual connection.
        # x = x + self.sa(self.lyr_norm4(x))
        return x, h


class LSTMffBlock(nn.Module):
    def __init__(self, n_embd, n_hidden, lstm_layers, dropout):
        super().__init__()
        self.lstmffwd = LSTMFeedForward(n_embd, n_hidden, lstm_layers, dropout)
        # self.lyr_norm3 = nn.LayerNorm(n_embd)

    def forward(self, x, hidden):
        x, h = self.lstmffwd(x, hidden)
        return x, h
