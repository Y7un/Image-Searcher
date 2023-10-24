import argparse
import glob
import os
import pickle

import args
import cv2
from scipy.spatial.distance import euclidean

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 1: Image Descriptor
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################

class RGBHistogram:
    """
    Image descriptor using color histogram.

    :param bins: list
        Histogram size. 1-D list containing ideal values
        between 8 and 128; but you can go up till 0 - 256.

    Example:
        >>> histogram = RGBHistogram(bins=[32, 32, 32])
        >>> feature_vector = histogram.describe(image='folder/image.jpg')
        >>> print(feature_vector.shape)
    """

    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        """
        Color description of a given image

        compute a 3D histogram in the RGB color space,
        then normalize the histogram so that images
        with the same content, but either scaled larger
        or smaller will have (roughly) the same histogram

        :param image:
            Image to be described.
        :return: flattened 3-D histogram
            Flattened descriptor [feature vector].
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=self.bins, ranges=[0, 256] * 3)
        hist = cv2.normalize(hist, dst=hist.shape)
        return hist.flatten()


################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 2: Indexing
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################

# key - image file name, value - computed feature vector/descriptor
def feature_extraction(dataset):
    features = {}
    descriptor = RGBHistogram(bins=[8, 8, 8])

    for filename in glob.glob(os.path.join(dataset, '*.jpg|png$')):
        # Extract the image name without extension, e.g., "places/eiffel_tower.jpg" becomes "eiffel_tower"
        img_name = os.path.basename(filename).split('.')[0]

        image = cv2.imread(filename)
        feature = descriptor.describe(image)
        # Store the feature vector in the dictionary with the image name as the key.
        features[img_name] = feature
    return features


# Writing the index to disk
def save(obj, path):
    if not os.path.isfile(path):
        # Ensure the directory structure for the output file is created if it doesn't exist.
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:  # Use 'wb' for writing in binary mode
        pickle.dump(obj, f)

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 3: Searching
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################

class Searcher:
    def __init__(self, features):
        self.features = features

    def search(self, query):
        results = {}

        for name, feature in self.features.items():
            # Calculate the Euclidean distance between the query feature and each stored feature.
            dist = euclidean(query, feature)
            results[name] = dist

        # Sort the results by distance in ascending order and return them as a list of (distance, image_name) tuples.
        results = sorted([(d, n) for n, d in results.items()])
        return results

    ################################################################################################
    # +———————————————————————————————————————————————————————————————————————————————————————————+
    # | Step 4: Image Search
    # +———————————————————————————————————————————————————————————————————————————————————————————+
    ################################################################################################

    def perform_image_search(query_image_path, features, descriptor):
        # Load the query image
        query_image = cv2.imread(query_image_path)
        query_feature = descriptor.describe(query_image)  # Describe the query image

        # Initialize the searcher
        searcher = Searcher(features)

        # Perform the search
        search_results = searcher.search(query_feature)

        # Print the search results
        print("Search Results:")
        for distance, image_name in search_results:
            print(f"Image: {image_name}, Distance: {distance}")

    if __name__ == '__main__':
        # ... (previous code sections)

        # Extracting features
        features = feature_extraction(args.dataset)
        # Saving extracted features
        save(features, args.features)

        # Added section: Image Search
        # You can use this part to test the image search functionality.
        query_image_path = 'path_to_query_image.jpg'  # Specify the path to your query image
        perform_image_search(query_image_path, features, RGBHistogram(bins=[8, 8, 8]))

if __name__ == '__main__':
    ################################################################################################
    # +———————————————————————————————————————————————————————————————————————————————————————————+
    # | Command line argument
    # +———————————————————————————————————————————————————————————————————————————————————————————+
    ################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='../images/lord-of-the-rings/',
                        help='Path to dataset.')
    parser.add_argument('-i', '--features', default='../saved/features.pkl',
                        help='Path to the features file.')
    args = parser.parse_args()

    # Extracting features
    features = feature_extraction(args.dataset)
    # Saving extracted features
    save(features, args.features)