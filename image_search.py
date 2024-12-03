import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Load the dataset of images
def load_images_from_folder(folder):
    images = {}
    for filename in glob.glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if img is not None:
            images[filename] = img
    return images


# Compute histogram for an image
def compute_histogram(image, bins=256, normalize=True):
    color = ('b', 'g', 'r')
    histograms = []
    for channel, col in enumerate(color):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        if normalize:
            hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_L1)
        histograms.append(hist)

    histograms = np.concatenate(histograms, axis=0)
    return histograms


# Assuming hist is a 3D histogram with shape (bins, bins, bins)
def plot_histogram(hist):
    # Split the histogram into B, G, R channels
    hist_b = hist[0]
    hist_g = hist[1]
    hist_r = hist[2]

    # Plot the histograms
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    # plt.xticks(range(len(hist_b)))
    plt.ylabel("# of Pixels")
    plt.plot(hist_b, color='b', label='Blue')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_r, color='r', label='Red')
    plt.legend()
    plt.show()


# Compare histograms using all available metrics in OpenCV
def compare_histograms(hist1, hist2):
    methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
    results = {}
    for method in methods:
        results[method] = cv2.compareHist(hist1, hist2, method)
    return results


# Normalize the results
def normalize_results(results):
    normalized_results = {}
    for method, value in results.items():
        if method == cv2.HISTCMP_CORREL:
            # Correlation: -1 to 1 -> 0 to 1
            normalized_results[method] = (value + 1) / 2

        elif method == cv2.HISTCMP_CHISQR:
            # Chi-Square: 0 to inf -> 1 to 0 (exponential decay)
            normalized_results[method] = np.exp(-value / 8)  # /8 controls decay rate

        elif method == cv2.HISTCMP_INTERSECT:
            # Intersection: needs to be normalized by total sum
            # Assuming histograms are normalized, max intersection is 1
            normalized_results[method] = value

        elif method == cv2.HISTCMP_BHATTACHARYYA:
            # Bhattacharyya: 0 to 1 -> 1 to 0
            normalized_results[method] = 1 - value
    return normalized_results


def name_results(results):
    methods = {
        'Correlation': cv2.HISTCMP_CORREL,
        'Chi-Square': cv2.HISTCMP_CHISQR,
        'Intersection': cv2.HISTCMP_INTERSECT,
        'Bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
    }
    named_results = {}
    for name, method in methods.items():
        named_results[name] = results[method]

    return named_results


def search_images(query_image, images, metric=cv2.HISTCMP_CORREL, bins=256):
    histograms = {filename: compute_histogram(img, bins=bins) for filename, img in images.items()}
    query_histogram = compute_histogram(query_image)

    results = {}
    for filename, hist in histograms.items():
        comparison = compare_histograms(query_histogram, hist)
        results[filename] = comparison

    sorted_results = dict(sorted(results.items(), key=lambda item: item[metric], reverse=True))
    return sorted_results


# Main function
def main():
    folder = 'images/search'
    images = load_images_from_folder(folder)
    query_image = cv2.imread('images/query.jpg')

    results = search_images(query_image, images)
    print(results)


if __name__ == "__main__":
    main()
