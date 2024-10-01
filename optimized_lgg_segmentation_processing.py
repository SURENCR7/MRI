
import os
import numpy as np
import pandas as pd

# Checking if the necessary library for handling images is installed, if not, install it.
try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

# Step 1: Load the Dataset
dataset_dir = r"C:\Users\surendran.m\Downloads\Data"

# Initialize empty lists to store image and mask paths
image_paths = []
mask_paths = []

# Walk through the dataset directory and its subdirectories to get image and mask paths
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".tif") and "_mask" not in file:  # Collect images excluding masks
            image_path = os.path.join(root, file)
            image_paths.append(image_path)
        elif file.endswith("_mask.tif"):  # Collect masks
            mask_path = os.path.join(root, file)
            mask_paths.append(mask_path)

# Step 2: Process Images
processed_images = []
processed_masks = []

# Function to preprocess image (convert to numpy array, normalize, etc.)
def preprocess_image(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)  # Convert image to numpy array
    # Normalize the image
    image_np = image_np / 255.0
    return image_np

# Function to preprocess mask (convert to numpy array)
def preprocess_mask(mask_path):
    mask = Image.open(mask_path)
    mask_np = np.array(mask)  # Convert mask to numpy array
    return mask_np

# Preprocess all images and masks
for img_path, mask_path in zip(image_paths, mask_paths):
    processed_images.append(preprocess_image(img_path))
    processed_masks.append(preprocess_mask(mask_path))

# Step 3: Convert data to numpy arrays for further use (e.g., model training)
X = np.array(processed_images)
y = np.array(processed_masks)

# Step 4: Load Tumor Genomic Clusters and Patient Data from CSV
csv_path = os.path.join(dataset_dir, 'data.csv')
patient_data = pd.read_csv(csv_path)

# Print out basic information
print(f"Total Images: {len(image_paths)}")
print(f"Total Masks: {len(mask_paths)}")
print(f"Patient Data Shape: {patient_data.shape}")

# Save the processed data if needed
np.save('processed_images.npy', X)
np.save('processed_masks.npy', y)

print("Data processing complete and saved to files.")
