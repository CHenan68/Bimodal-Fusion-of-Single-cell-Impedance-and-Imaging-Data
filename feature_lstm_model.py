import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=1):
        super(Lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=input_size,
        #               out_channels=input_size,
        #               kernel_size=conv_size),
        #     nn.BatchNorm1d(input_size),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=pool_size)
        # )

        self.Lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=False, batch_first=True)  # ps.有embed的话input= hidden
        self.classify = nn.Sequential(
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # x = self.conv(x)
        # x = x.permute(0, 2, 1)

        output, (hidden, cn) = self.Lstm(x)
        fea = output[:, -1, :]   # 128*128
        x = self.classify(fea)

        return x, fea
