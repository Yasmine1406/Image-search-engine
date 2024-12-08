# Image Search Engine with AI

This project provides an image search engine that allows users to upload an image, extract its visual features using **VGG16** and **Local Binary Pattern (LBP)**, and then search for similar images from an **Elasticsearch** index. The application utilizes **Streamlit** for the user interface and displays the results visually.

## Features

- **VGG16 Feature Extraction**: Extracts deep features from images using the pre-trained **VGG16** model.
- **Local Binary Pattern (LBP)**: Extracts texture features from images using the **LBP** algorithm.
- **Elasticsearch**: Uses **Elasticsearch** to index images and search for similar ones based on cosine similarity between feature vectors.
- **Streamlit Interface**: A user-friendly web interface built with **Streamlit** for uploading images and viewing search results.

## Prerequisites

To run this project, you will need:

- **Python 3.x**
- **Elasticsearch**: Ensure you have an Elasticsearch server running locally on `http://localhost:9200`.

## Installation

### Step 1: Install Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/Yasmine1406/Image_search_engine.git
   cd imagesearchengine
   
### Step 2: Install Required Python Libraries

1. Install the required libraries via pip:
   ```bash
   pip install elasticsearch streamlit pillow numpy opencv-python tensorflow scikit-image
   
### Step 3: Install Elasticsearch

1. Download and install Elasticsearch
2. Start the Elasticsearch service:
   ```bash
   bin\elasticsearch.bat

### Step 4: Index Images in Elasticsearch

1. To use the search functionality, you need to index images and their feature vectors into Elasticsearch.
2. Use the provided indexing script to add your images and their corresponding features to Elasticsearch. Ensure that your images are stored locally or provide valid paths in the dataset.
3. The indexing script will create an index called images_data in Elasticsearch with fields for storing image paths and feature vectors.

### Step 5: Run the Streamlit Application

1. Run the Streamlit application with the following command:
   ```bash
   streamlit run interface.py

This will start the Streamlit server, and you can access the web interface in your browser at http://localhost:8501.

## Usage

### Upload an Image

1. On the sidebar, use the "Upload an image" button to upload an image file in .jpg, .jpeg, .png, or .webp format.
2. The uploaded image will be displayed in the main area.

### Search for Similar Images

1. Once the image is uploaded, the system will automatically extract its features using VGG16 and Local Binary Pattern (LBP) methods.
2. After feature extraction, click the Search button to find similar images in the Elasticsearch index.
3. The system will perform a cosine similarity search and display the most similar images based on the extracted feature vectors.

### View Search Results
1. The search results will be displayed in a grid layout with images shown side by side.
2. The images that are the most visually similar to the uploaded image will be displayed first.

## Key Components

* VGG16 Model: Used for extracting deep features from the images.
* Local Binary Pattern (LBP): A method for extracting texture features, often used in computer vision tasks.
* Elasticsearch: Used for indexing images and performing efficient searches based on feature vectors.
* Streamlit: Provides an interactive web interface for the user.
