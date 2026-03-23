"""
Model architectures for emotion recognition.
- StaticEmotionModel: ResNet-18 for single-image classification.
- DynamicEmotionModel: ResNet-18 (feature extractor) + LSTM for sequence classification.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, NUM_CLASSES, RESNET_FEATURE_DIM


class StaticEmotionModel(nn.Module):
    """
    ResNet-18 for static (single image) emotion recognition.
    Pretrained on ImageNet, with final FC layer replaced for NUM_CLASSES.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def get_feature_extractor(self):
        """Return ResNet without the final FC layer (for use in dynamic model)."""
        modules = list(self.resnet.children())[:-1]  # Remove FC
        return nn.Sequential(*modules)


class DynamicEmotionModel(nn.Module):
    """
    ResNet-18 + LSTM for dynamic (video sequence) emotion recognition.
    ResNet-18 extracts spatial features from each frame.
    LSTM models temporal dependencies across the sequence.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        pretrained=True,
    ):
        super().__init__()

        # Feature extractor: ResNet-18 without FC
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Temporal model: LSTM
        self.lstm = nn.LSTM(
            input_size=RESNET_FEATURE_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, C, H, W) — sequence of frames
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, C, H, W = x.shape

        # Extract features from each frame
        # Reshape to (batch * seq_len, C, H, W) for batch processing
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.feature_extractor(x)  # (batch * seq_len, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, 512)

        # LSTM: process sequence
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden_size)

        # Use output of last time step
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Classify
        logits = self.fc(last_output)  # (batch, num_classes)
        return logits


def get_model(model_type="static", pretrained=True):
    """
    Factory function to create model.

    Args:
        model_type: 'static' or 'dynamic'
        pretrained: Whether to use ImageNet pretrained weights.
    """
    if model_type == "static":
        return StaticEmotionModel(pretrained=pretrained)
    elif model_type == "dynamic":
        return DynamicEmotionModel(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'static' or 'dynamic'.")
