# Handwritten Character Recognition (A–Z)

A desktop-based handwritten character recognition system using a CNN trained on self-collected data. The project includes a zipped self-drawn dataset for immediate training and testing.

## Features

- Draw characters using mouse
- Real-time prediction with confidence score
- Save new samples for training
- Retrain model directly from UI
- Undo strokes (Backspace / Ctrl+Z)
- Keyboard-based prediction (Enter key)
- Single unified desktop interface

## Tech Stack

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- Tkinter

## Project Structure

```
handwriting_recognition/
│
├── app/
│   └── unified_whiteboard.py
│
├── model/
│   ├── cnn.py
│   └── train.py
│
├── data/
│   ├── self_data.zip
│   └── self_data/
│       ├── A/
│       ├── B/
│       └── ...
│
├── inference/
│   ├── preprocess.py
│   └── predict.py
│
├── utils/
│   ├── config.py
│   └── label_map.py
│
└── README.md
```

## Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install torch torchvision numpy opencv-python
```

## Dataset Setup (Required)

The repository includes a zipped self-collected dataset.

### Extract Dataset

```powershell
Expand-Archive data\self_data.zip data\
```

After extraction, the structure should be:
```
data/self_data/A
data/self_data/B
...
```

## Training

Train the model using the provided self-data:

```bash
python -m model.train
```

The trained model is saved to:
```
model/model.pth
```

## Run Application

Start the unified desktop UI:

```bash
python -m app.unified_whiteboard
```

## Usage

1. Draw a character on the board
2. Press Enter or click Predict
3. Press Backspace or Ctrl+Z to undo
4. Enter a label and click Save Sample to add new training data
5. Click Retrain Model to update the model

## Model Architecture

- Input: 1 × 28 × 28 grayscale image
- Convolutional layers with ReLU and max pooling
- Fully connected layers
- Output: 26 classes (A–Z)

## Notes

- The included dataset matches the whiteboard input format
- Adding more samples and retraining improves accuracy
- Public datasets were intentionally avoided due to input mismatch
