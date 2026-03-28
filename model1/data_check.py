import matplotlib.pyplot as plt
import os
import random
from PIL import Image

DATASET_PATH = "../dataset/Training"

classes = os.listdir(DATASET_PATH)

plt.figure(figsize=(10, 10))

for i, cls in enumerate(classes):
    img_path = os.path.join(DATASET_PATH, cls)
    img_name = random.choice(os.listdir(img_path))
    img_full_path = os.path.join(img_path, img_name)

    # Load image using PIL (PyTorch standard)
    img = Image.open(img_full_path).convert("RGB")
    img = img.resize((224, 224))

    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.show()