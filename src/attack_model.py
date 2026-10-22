"""
Acoustic Keystroke Attack Model: CNN Encoder + Bi-LSTM + CTC Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 48-character target set: a-z, 0-9, space, enter, backspace, specials
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'\"-_/\\()")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = CTC blank
VOCAB_SIZE = len(CHARS) + 1  # +1 for CTC blank


class CNNEncoder(nn.Module):
    """4-layer spectrogram encoder."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, 1, freq_bins, time_frames)
        out = self.conv(x)  # (B, 256, f', t')
        return out.mean(dim=2).permute(0, 2, 1)  # (B, t', 256)


class KeystrokeAttackModel(nn.Module):
    """
    Full acoustic keystroke transcription model.
    Input:  STFT spectrogram (B, 1, n_freq, n_time)
    Output: character sequence via CTC
    """

    def __init__(self, lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=1)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.ctc_head = nn.Linear(lstm_hidden * 2, VOCAB_SIZE)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities of shape (time, batch, vocab)."""
        enc = self.encoder(spectrogram)          # (B, T, 256)
        lstm_out, _ = self.lstm(enc)             # (B, T, 512)
        logits = self.ctc_head(lstm_out)         # (B, T, VOCAB)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)         # (T, B, VOCAB) — CTC format

    @torch.no_grad()
    def transcribe(self, spectrogram: torch.Tensor) -> str:
        """Greedy CTC decoding of a single spectrogram."""
        self.eval()
        log_probs = self.forward(spectrogram)   # (T, 1, VOCAB)
        preds = log_probs.argmax(dim=-1).squeeze(-1).tolist()  # (T,)

        # Collapse repeats and remove blanks
        chars = []
        prev = -1
        for p in preds:
            if p != prev and p != 0:
                idx = p - 1
                if idx < len(CHARS):
                    chars.append(CHARS[idx])
            prev = p
        return "".join(chars)
