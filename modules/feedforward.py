import torch.nn as nn


class MlPFeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LSTMFeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden, lstm_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_embd, n_hidden, num_layers=lstm_layers, dropout=dropout,
                            batch_first=True)
        self.dl = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hidden, n_embd)

    def forward(self, x, hidden):
        x, h = self.lstm(x, hidden)
        x = self.dl(x)
        x = self.fc(x)
        return x, h
