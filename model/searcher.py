from scipy.spatial.distance import euclidean  # Import the Euclidean distance function from SciPy

class Searcher:

    def __init__(self, features):
        # Constructor for the Searcher class.
        # Initialize the class with a dictionary of features where keys are image names and values are feature vectors.
        self.features = features

    def search(self, query):
        # This method performs a similarity search based on the Euclidean distance between the query feature and stored features.
        results = {}  # Initialize an empty dictionary to store results.

        for name, feature in self.features.items():
            # Calculate the Euclidean distance between the query feature and each stored feature.
            dist = euclidean(query, feature)
            results[name] = dist  # Store the distance in the results dictionary with the image name as the key.

        # Sort the results by distance in ascending order and return them as a list of (distance, image_name) tuples.
        results = sorted([(d, n) for n, d in results.items()])

        return results  # Return the sorted results.