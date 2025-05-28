import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .dataset import PassportDataset
from .model import PassportCountryModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train passport country model")
    parser.add_argument("data_dir", type=Path, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=Path("model.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = PassportDataset(str(args.data_dir), transform=transform)
    model = PassportCountryModel(num_classes=len(dataset.label_to_idx))
    model.to(args.device)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        all_preds = []
        all_labels = []
        for images, labels in loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            texts = model.extract_text(images)
            logits = model(images, texts)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{args.epochs} accuracy: {acc:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
