import os
from PIL import Image
import numpy as np

def load_and_preprocess_data(directory):
    images = []
    labels = []

    for category in os.listdir(directory):
        category_dir = os.path.join(directory,category)
        for filename in os.listdir(category_dir):
            image_path = os.path.join(category_dir, filename)
            image = Image.open(image_path)
            image = image.resize((224,224))
            image = np.array(image)/255.0
            images.append(image)
            labels.append(1 if category == 'R' else 0)

    return np.array(images), np.array(labels)