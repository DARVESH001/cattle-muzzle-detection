# Cattle Muzzle Identification System

A deep learning-based cattle identification system that uses muzzle print patterns (similar to human fingerprints) to uniquely identify cattle. Built with a Siamese Neural Network using ResNet-50 backbone and a Streamlit web interface.

## Features

- **Muzzle Detection** — Verify if an uploaded image is a cattle muzzle
- **Cattle Identification** — Match a muzzle image against registered cattle
- **Image Comparison** — Compare two muzzle images to check if they belong to the same cattle
- **Cattle Registration** — Register new cattle with multiple muzzle images
- **Registry Management** — View and manage all registered cattle

## Project Structure

```
cattle_muzzle_detection/
├── app.py                  # Streamlit web application
├── config.py               # Configuration (paths, hyperparameters, thresholds)
├── requirements.txt        # Python dependencies
├── start.bat               # Windows launcher script
├── train_colab.ipynb       # Training notebook for Google Colab
├── train_kaggle.ipynb      # Training notebook for Kaggle
├── model/
│   ├── __init__.py
│   ├── network.py          # Siamese Network & Contrastive Loss
│   ├── dataset.py          # Dataset classes & data augmentation
│   ├── train.py            # Training script
│   └── inference.py        # Inference engine (detection, comparison, registration)
├── saved_models/           # Trained model weights (not tracked in git)
├── data/
│   ├── registry/           # Registered cattle embeddings
│   └── uploads/            # Uploaded images
└── .gitignore
```

## How It Works

The system uses a **Siamese Neural Network** with a **ResNet-50** backbone to extract 128-dimensional embedding vectors from muzzle images. These embeddings capture the unique texture patterns of each cattle's muzzle.

- **Training**: Pairs of muzzle images are fed into the network. The model learns to produce similar embeddings for the same cattle and different embeddings for different cattle using **Contrastive Loss**.
- **Identification**: A new muzzle image is compared against registered cattle embeddings using **cosine similarity**. If the similarity exceeds the threshold (default: 0.70), it's considered a match.

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DARVESH001/cattle-muzzle-detection.git
   cd cattle-muzzle-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

### Option 1: Local Training

1. Download the [Cattle Muzzle Dataset](https://www.kaggle.com/) and extract it.

2. Update the dataset path in `config.py`:
   ```python
   DATASET_PATH = r"path/to/your/dataset/Original"
   ```

3. Run the training script:
   ```bash
   python model/train.py
   ```

   Optional arguments:
   ```bash
   python model/train.py --epochs 30 --batch-size 32 --lr 0.0001 --dataset-path /path/to/dataset
   ```

### Option 2: Google Colab / Kaggle

Use the provided notebooks for cloud-based training with free GPU access:
- `train_colab.ipynb` — For Google Colab
- `train_kaggle.ipynb` — For Kaggle

After training, download the `saved_models/` folder and place it in the project root.

### Training Details

| Parameter | Default Value |
|-----------|--------------|
| Backbone | ResNet-50 (pretrained on ImageNet) |
| Embedding Dimension | 128 |
| Image Size | 224 x 224 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Epochs | 30 |
| Backbone Freeze Epochs | 5 |
| Loss Function | Contrastive Loss (margin=1.0) |
| Data Split | 70% train / 15% val / 15% test |

## Running the App

```bash
streamlit run app.py
```

Or on Windows, double-click `start.bat`.

The app will open at **http://localhost:8501**.

### App Tabs

| Tab | Description |
|-----|-------------|
| **Check Image** | Upload an image to detect if it's a muzzle and find matches in the registry |
| **Compare Two Images** | Upload two images to check if they belong to the same cattle |
| **Register Cattle** | Register a new cattle with multiple muzzle images (3+ recommended) |
| **Registry** | View and delete registered cattle |

## Configuration

Edit `config.py` to adjust:

```python
# Inference thresholds
SIMILARITY_THRESHOLD = 0.70        # Min similarity for a match
MUZZLE_DETECTION_THRESHOLD = 0.40  # Min similarity for muzzle detection

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
EMBEDDING_DIM = 128
```

## Requirements

- torch >= 2.0.0
- torchvision >= 0.15.0
- streamlit >= 1.30.0
- Pillow >= 10.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

## License

This project is for educational and research purposes.
