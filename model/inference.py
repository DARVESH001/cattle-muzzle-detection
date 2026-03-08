import os
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.network import SiameseNetwork
from model.dataset import get_transforms


class MuzzleIdentifier:
    """Handles all inference tasks: muzzle detection, comparison, identification."""

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or config.MODEL_PATH
        self.transform = get_transforms(is_training=False)
        self.model = None
        self.reference_embedding = None
        self.registry = {}

        self._load_model()
        self._load_reference()
        self._load_registry()

    def _load_model(self):
        """Load trained Siamese network."""
        if not os.path.exists(self.model_path):
            print(f"Warning: No trained model found at {self.model_path}")
            return

        self.model = SiameseNetwork(embedding_dim=config.EMBEDDING_DIM).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {self.model_path} (epoch {checkpoint.get('epoch', '?')})")

    def _load_reference(self):
        """Load reference embedding for muzzle detection."""
        ref_path = os.path.join(config.SAVED_MODELS_DIR, "reference_embeddings.npy")
        if os.path.exists(ref_path):
            self.reference_embedding = np.load(ref_path)
            print("Reference embedding loaded for muzzle detection")

    def _load_registry(self):
        """Load all registered cattle embeddings."""
        self.registry = {}
        if not os.path.exists(config.REGISTRY_DIR):
            return

        for fname in os.listdir(config.REGISTRY_DIR):
            if fname.endswith('.npy'):
                name = fname[:-4]
                self.registry[name] = np.load(os.path.join(config.REGISTRY_DIR, fname))

        if self.registry:
            print(f"Loaded {len(self.registry)} registered cattle")

    def is_ready(self):
        """Check if model is loaded and ready."""
        return self.model is not None

    def extract_embedding(self, image):
        """Extract embedding from a PIL Image or file path.

        Args:
            image: PIL Image, file path string, or file-like object

        Returns:
            numpy array of shape (128,)
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Train the model first.")

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif hasattr(image, 'read'):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Expected PIL Image, file path, or file-like object")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.get_embedding(img_tensor)

        return embedding.cpu().numpy().flatten()

    def is_cattle_muzzle(self, image):
        """Check if the image is a cattle muzzle.

        Uses cosine similarity with the reference muzzle embedding.

        Returns:
            dict with keys: is_muzzle (bool), confidence (float), confidence_label (str)
        """
        if self.reference_embedding is None:
            return {"is_muzzle": True, "confidence": 0.0, "confidence_label": "Unknown (no reference)"}

        embedding = self.extract_embedding(image)
        similarity = float(np.dot(embedding, self.reference_embedding) /
                          (np.linalg.norm(embedding) * np.linalg.norm(self.reference_embedding)))

        is_muzzle = similarity > config.MUZZLE_DETECTION_THRESHOLD

        return {
            "is_muzzle": is_muzzle,
            "confidence": round(similarity, 4),
            "confidence_label": self._confidence_label(similarity),
        }

    def compare_images(self, image1, image2):
        """Compare two images and determine if they're the same cattle.

        Returns:
            dict with keys: similarity (float), same_cattle (bool), confidence_label (str)
        """
        emb1 = self.extract_embedding(image1)
        emb2 = self.extract_embedding(image2)

        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        same_cattle = similarity >= config.SIMILARITY_THRESHOLD

        return {
            "similarity": round(similarity, 4),
            "same_cattle": same_cattle,
            "confidence_label": self._confidence_label(similarity),
        }

    def register_cattle(self, name, images):
        """Register a new cattle with multiple muzzle images.

        Args:
            name: Name/ID for the cattle
            images: List of PIL Images or file paths

        Returns:
            dict with registration info
        """
        if len(images) < 1:
            raise ValueError("At least 1 image required for registration")

        embeddings = []
        for img in images:
            emb = self.extract_embedding(img)
            embeddings.append(emb)

        # Average embedding as representative vector
        avg_embedding = np.mean(embeddings, axis=0)
        # Re-normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        # Save to registry
        save_path = os.path.join(config.REGISTRY_DIR, f"{name}.npy")
        np.save(save_path, avg_embedding)
        self.registry[name] = avg_embedding

        return {
            "name": name,
            "num_images": len(images),
            "status": "registered",
        }

    def identify_cattle(self, image):
        """Identify which registered cattle matches the given image.

        Returns:
            list of dicts sorted by similarity (highest first),
            each with keys: name, similarity, confidence_label
        """
        if not self.registry:
            return []

        query_emb = self.extract_embedding(image)
        matches = []

        for name, reg_emb in self.registry.items():
            similarity = float(np.dot(query_emb, reg_emb) /
                              (np.linalg.norm(query_emb) * np.linalg.norm(reg_emb)))
            matches.append({
                "name": name,
                "similarity": round(similarity, 4),
                "confidence_label": self._confidence_label(similarity),
                "is_match": similarity >= config.SIMILARITY_THRESHOLD,
            })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    def delete_cattle(self, name):
        """Remove a cattle from the registry."""
        file_path = os.path.join(config.REGISTRY_DIR, f"{name}.npy")
        if os.path.exists(file_path):
            os.remove(file_path)
        self.registry.pop(name, None)
        return {"name": name, "status": "deleted"}

    def get_registry_list(self):
        """Get list of all registered cattle."""
        return list(self.registry.keys())

    @staticmethod
    def _confidence_label(similarity):
        if similarity >= 0.85:
            return "Very High"
        elif similarity >= 0.75:
            return "High"
        elif similarity >= 0.65:
            return "Medium"
        elif similarity >= 0.50:
            return "Low"
        else:
            return "Very Low"
