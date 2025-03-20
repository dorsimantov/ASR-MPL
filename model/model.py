import numpy as np
import torch
import math
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ASRModel(nn.Module):
    def __init__(self, d_model=256, d_head=4, d_ff=2048, N_enc=12, max_seq_len=10000):
        super().__init__()
        self.MAX_SEQ_LEN = max_seq_len

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 256, 3, stride=2)  # might be (4, 256, 3, stride=2), check
        self.conv2 = nn.Conv2d(256, 256, 3, stride=2)

        # dimension adapter
        self.adapter = nn.Linear(256*19, d_model)
        
        # positional encoder
        self.posenc = PositionalEncoder(d_model, self.MAX_SEQ_LEN)

        # transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, d_head, dim_feedforward=d_ff, norm_first=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, N_enc)

        self.projection = nn.Linear(d_model, 1001)

        self.output_temporal_len = math.floor((math.floor((self.MAX_SEQ_LEN - 3) / 2) + 1 - 3) / 2) + 1

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)

        x_conv2 = torch.permute(x_conv2, (0, 2, 3, 1))
        x_conv2 = torch.flatten(x_conv2, start_dim=2)
        x_adapter = self.adapter(x_conv2)

        x_posenc = self.posenc(x_adapter)
        x_tf = self.transformer(x_posenc)
        x_out = self.projection(x_tf)
        # print(x_out.shape)

        return x_out