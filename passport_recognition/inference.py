from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms as T
from PIL import Image

from .model import PassportCountryModel


def load_model(model_path: str, num_classes: int) -> PassportCountryModel:
    model = PassportCountryModel(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict(
    model: PassportCountryModel, image_paths: Iterable[Path], device: str = "cpu"
) -> list[str]:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    results = []
    model.to(device)
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        text = model.extract_text(tensor)
        with torch.no_grad():
            logits = model(tensor, text)
            pred = logits.argmax(dim=1).item()
        results.append(pred)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on passport images")
    parser.add_argument("model", type=Path, help="Trained model path")
    parser.add_argument("images", type=Path, nargs="+", help="Image files")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_model(str(args.model), num_classes=256)  # adjust as needed
    preds = predict(model, args.images, device=args.device)
    for path, label in zip(args.images, preds):
        print(path, label)
