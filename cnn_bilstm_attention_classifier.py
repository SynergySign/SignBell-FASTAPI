import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (batch_first)"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class CNN_BiLSTM_Attention(nn.Module):
    """CNN-BiLSTM-Attention 모델 - 저장된 가중치와 일치하는 구조"""
    def __init__(self, input_size=147, num_classes=7, cnn_channels=None, lstm_hidden=128, dropout=0.5):
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        super().__init__()

        # CNN layers - ModuleList로 구성 (저장된 구조와 일치)
        self.conv_layers = nn.ModuleList()
        in_channels = input_size
        for out_channels in cnn_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ))
            in_channels = out_channels

        # LSTM layer - 2 layers로 설정 (저장된 구조와 일치)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=lstm_hidden * 2, max_len=500)

        # Temporal weight parameter
        self.temporal_weight = nn.Parameter(torch.tensor(0.1))

        # Change detector
        self.change_detector = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Classifier - Sequential로 구성 (저장된 구조와 일치)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        # CNN processing
        x_cnn = x.transpose(1, 2)  # (batch, features, seq_len)
        for conv_layer in self.conv_layers:
            x_cnn = conv_layer(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, seq_len, features)

        # LSTM processing
        lstm_out, _ = self.lstm(x_cnn)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.pos_encoding(lstm_out)

        # Change detection
        change_scores = self.change_detector(lstm_out).squeeze(-1)

        # Multi-head attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Apply temporal weighting
        change_weights = change_scores.unsqueeze(-1) * self.temporal_weight
        weighted_attn = attn_out * (1 + change_weights)

        # Global average pooling
        context_vector = weighted_attn.mean(dim=1)

        # Classification
        output = self.classifier(context_vector)

        return output, attn_weights.mean(dim=1), change_scores
