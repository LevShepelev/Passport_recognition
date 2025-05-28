from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision import models, transforms as T
import pytesseract
from transformers import AutoModel, AutoTokenizer


class PassportCountryModel(nn.Module):
    """Hybrid model combining visual and text features."""

    def __init__(
        self, num_classes: int, text_model_name: str = "distilbert-base-uncased"
    ) -> None:
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        cnn_out = (
            self.cnn.fc.in_features if hasattr(self.cnn.fc, "in_features") else 512
        )
        text_out = self.text_model.config.hidden_size
        self.classifier = nn.Linear(cnn_out + text_out, num_classes)

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            visual_feat = self.cnn(images)
            encoded = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(images.device)
            text_feat = self.text_model(**encoded).last_hidden_state[:, 0, :]
        features = torch.cat([visual_feat, text_feat], dim=1)
        return self.classifier(features)

    def extract_text(self, images: torch.Tensor) -> list[str]:
        """Extract text from a batch of images using Tesseract OCR."""
        to_pil = T.ToPILImage()
        texts = []
        for img in images:
            pil_img = to_pil(img.cpu())
            text = pytesseract.image_to_string(pil_img, lang="eng")
            texts.append(text)
        return texts
