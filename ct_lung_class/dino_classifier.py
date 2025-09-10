import torch.nn as nn
import torch

class VolumeClassifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, D, H, W)
        B, _, D, H, W = x.shape
        x = x.squeeze(1)  # (B, D, H, W)

        # Reshape to (B * D, 3, H, W)
        x = x.view(B * D, H, W)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # → (B*D, 3, H, W)
        x = self.embedder(x)  # → (B*D, embed_dim)
        x = x.view(B, D, -1)  # → (B, D, embed_dim)

        lstm_out, _ = self.lstm(x)  # → (B, D, hidden_dim)
        final_out = lstm_out[:, -1, :]  # use last LSTM output
        return self.classifier(final_out)
    