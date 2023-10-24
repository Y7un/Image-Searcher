import glob
import os
import pickle

import cv2
from model.descriptor import Histogram

# Function to extract features from a dataset of images
def extract(dataset, descriptor, output_path):
    features = {}

    # Initialize the descriptor with the desired number of bins
    descriptor = Histogram(bins=[8, 8, 8])

    for filename in glob.glob(os.path.join(dataset, '*.jpg|*.png')):
        # Extract the image name from the filename (excluding extension)
        img_name = os.path.splitext(os.path.basename(filename))[0]

        # Read the image using OpenCV
        image = cv2.imread(filename)

        # Calculate the image's feature vector using the descriptor
        feature = descriptor.describe(image)

        # Store the feature vector in the dictionary with the image name as the key
        features[img_name] = feature

    # Save the features to a file using pickle
    save(features, output_path)

# Function to save data to a file using pickle
def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Create the directory if it doesn't exist
    with open(path, 'wb') as f:  # Use 'wb' for writing in binary mode
        pickle.dump(obj, f)

if __name__ == "__main__":
    dataset_dir = 'C:\\Users\\yjun0\\PycharmProjects\\ImageSearcher\\Photos'  # Specify the path to your image dataset
    output_file = 'C:\\Users\\yjun0\\PycharmProjects\\ImageSearcher\\SavedPhotos'  # Specify the path for saving the feature data

    # Create a Histogram descriptor with the desired number of bins
    descriptor = Histogram(bins=[8, 8, 8])

    # Extract features from the dataset and save them to a file
    extract(dataset_dir, descriptor, output_file)