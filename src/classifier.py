import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "animal_classifier.keras"
LABELS_PATH = BASE_DIR / "models" / "class_indices.json"

class AnimalClassifier:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.load_resources()

    def load_resources(self):
        if self.model is None:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        
        if self.class_names is None:
            with open(LABELS_PATH, 'r') as f:
                self.class_names = json.load(f)

    def preprocess_image(self, image):
        # 1. Resize
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        # 2. Convert to Array
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        # 3. Batch Dimension
        return tf.expand_dims(img_array, 0)

    def predict(self, image):
        processed_img = self.preprocess_image(image)
        
        predictions = self.model.predict(processed_img)

        score = predictions[0] 
        
        class_idx = np.argmax(score)
        confidence = 100 * np.max(score)
        
        label_name = self.class_names[str(class_idx)]
        return label_name, confidence