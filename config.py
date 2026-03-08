import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\darve\Downloads\archive (52)\Cattle Muzzle - DB\Original"
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
REGISTRY_DIR = os.path.join(BASE_DIR, "data", "registry")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")

# Model
EMBEDDING_DIM = 128
IMAGE_SIZE = 224
MODEL_FILENAME = "siamese_resnet50.pth"
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, MODEL_FILENAME)

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 30
FREEZE_EPOCHS = 5
MARGIN = 1.0
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_WORKERS = 0  # Windows compatibility
PAIRS_PER_EPOCH = 5000

# Inference
SIMILARITY_THRESHOLD = 0.70
MUZZLE_DETECTION_THRESHOLD = 0.40

# Ensure directories exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
