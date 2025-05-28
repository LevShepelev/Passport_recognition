# Passport Recognition

This project demonstrates a PyTorch pipeline for detecting the issuing country of passport images. It uses a hybrid model that combines visual features from images and textual features extracted from the document.

## Project Structure

- `passport_recognition/` – Python package with dataset and model code.
- `train.py` – script for training the model.
- `inference.py` – example inference script.
- `evaluate.py` – evaluation helper.

## Setup

1. Install [Poetry](https://python-poetry.org/docs/#installation).
2. Install dependencies:

```bash
poetry install
 pre-commit install
```

### OCR

Text extraction relies on [Tesseract](https://github.com/tesseract-ocr/tesseract).
Install it using your system package manager, e.g. on Ubuntu:

```bash
sudo apt-get install tesseract-ocr
```

pytesseract will be installed automatically with the Python dependencies.

## Data

Download the dataset archive and extract it so that the directory structure looks like:

```
data/
  USA/
    img1.jpg
    img2.jpg
  CAN/
    img3.jpg
    ...
```

Each folder represents a country and contains JPEG images of synthetic passports.

## Training

Run the training script specifying the dataset directory:

```bash
poetry run python -m passport_recognition.train data/ --epochs 20
```

Model weights will be saved to `model.pth` by default.

## Inference

```bash
poetry run python -m passport_recognition.inference --help
```

## Notes

The repository does not include the dataset or trained weights. You must download them separately using the provided link.

