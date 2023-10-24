import cv2  # Import the OpenCV library

class Histogram:
    def __init__(self, bins):
        # Constructor for the Histogram class.
        # Initialize the class with the number of bins for the histogram.
        self.bins = bins

    def describe(self, filename):
        # This method calculates and describes the color histogram of an image.

        # 1. Read the image from the given filename using OpenCV.
        image = cv2.imread(filename)

        # 2. Convert the image from BGR color space (default for OpenCV) to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Calculate the color histogram of the image using OpenCV's calcHist function.
        #    - images: List of images (in this case, only one image).
        #    - channels: Channels for which the histogram is calculated (0: Red, 1: Green, 2: Blue).
        #    - mask: Optional mask for the histogram (None means no mask).
        #    - histSize: Number of bins for each channel.
        #    - ranges: Range of pixel values (0-255 for each channel).
        hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=self.bins, ranges=[0, 256] * 3)

        # 4. Normalize the histogram values to have values in the range [0, 1].
        hist = cv2.normalize(hist, dst=hist.shape)

        # 5. Flatten the histogram to a 1D array and return it.
        return hist.flatten()
