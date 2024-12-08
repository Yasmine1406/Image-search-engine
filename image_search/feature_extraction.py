# vgg16_lbp_feature_extraction.py
from tensorflow import keras
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Preprocess input image and extract VGG16 features
def extract_vgg16_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()

# Extract LBP features from an image
def extract_lbp_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  
    return hist

# Combine VGG16 and LBP features
def combine_features(vgg_features, lbp_features):
    return np.concatenate((vgg_features, lbp_features), axis=None)

# Get all image files from a directory
def get_image_files(data_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

# Compute Euclidean distance
def compute_euclidean_distance(query_vector, dataset_vectors):
    distances = np.linalg.norm(dataset_vectors - query_vector, axis=1)
    return distances

# Compute Cosine similarity (optional alternative)
def compute_cosine_similarity(query_vector, dataset_vectors):
    query_vector = query_vector.reshape(1, -1)
    similarity = cosine_similarity(query_vector, dataset_vectors)
    return similarity.flatten()

# Save feature vectors to a CSV
def save_features_to_csv(feature_vectors, image_paths, output_csv):
    feature_df = pd.DataFrame(feature_vectors)
    feature_df['image_path'] = image_paths
    feature_df.to_csv(output_csv, index=False)

# Load features from a CSV
def load_features_from_csv(csv_file):
    dataset = pd.read_csv(csv_file)
    feature_vectors = dataset.iloc[:, :-1].values
    image_paths = dataset['image_path'].tolist()
    return feature_vectors, image_paths

# Display images using Matplotlib
def display_images(image_paths):
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# Directory paths
root_directory = r"D:\Tbourbi\ImageSearchEngine\Data"
output_csv = "feature_vectors.csv"

# Feature extraction for the dataset
print("Extracting features for dataset images...")
image_files = get_image_files(root_directory)
final_vector_list = []

for image_path in image_files:
    vgg_features = extract_vgg16_features(image_path)
    lbp_features = extract_lbp_features(image_path)
    final_vector = combine_features(vgg_features, lbp_features)
    final_vector_list.append(final_vector)

# Save feature vectors to CSV
print("Saving feature vectors to CSV...")
save_features_to_csv(final_vector_list, image_files, output_csv)

# Load features for similarity search
print("Loading feature vectors from CSV...")
dataset_features, image_paths = load_features_from_csv(output_csv)

# Process the query image
new_image_path = r"D:\Tbourbi\ImageSearchEngine\Data\image1.png"
print("Processing query image...")
query_vgg_features = extract_vgg16_features(new_image_path)
query_lbp_features = extract_lbp_features(new_image_path)
query_combined_features = combine_features(query_vgg_features, query_lbp_features)

# Compute similarity scores (Euclidean Distance or Cosine Similarity)
print("Computing similarity scores...")
similarity_scores = compute_euclidean_distance(query_combined_features, dataset_features)
# similarity_scores = compute_cosine_similarity(query_combined_features, dataset_features)  # Optional

# Retrieve top N most similar images
top_N = 10
sorted_indices = np.argsort(similarity_scores)[:top_N]
most_similar_images = [image_paths[i] for i in sorted_indices]

# Display results
print("Most similar images:")
for img_path in most_similar_images:
    print(img_path)

print("Displaying similar images...")
display_images(most_similar_images)
