from pathlib import Path

import torch
import torchvision.transforms as T
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from .dataset import PassportDataset
from .model import PassportCountryModel


def evaluate(model_path: str, data_dir: str) -> None:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = PassportDataset(data_dir, transform=transform)
    model = PassportCountryModel(num_classes=len(dataset.label_to_idx))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loader = DataLoader(dataset, batch_size=1)
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        texts = model.extract_text(imgs)
        with torch.no_grad():
            logits = model(imgs, texts)
        pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(labels.item())
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data")
    args = parser.parse_args()
    evaluate(args.model, args.data)
