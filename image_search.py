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
        if img is not None:
            images[filename] = img
    return images


# Compute histogram for an image
def compute_histogram(image, bins=256, normalize=False):
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
def compare_histograms(hist1, hist2, normalize=True):
    methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
    results = {}
    for method in methods:
        results[method] = cv2.compareHist(hist1, hist2, method)
    if normalize:
        results = normalize_results(results)
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


def sort_results(results, method):
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1][method], reverse=True))
    return sorted_results


def search_images(query_image, images, bins=256, sorting_metric=cv2.HISTCMP_CORREL):
    histograms = {filename: compute_histogram(img, bins=bins, normalize=False) for filename, img in images.items()}
    query_histogram = compute_histogram(query_image, bins=bins, normalize=False)

    results = {}
    for filename, hist in histograms.items():
        comparison = compare_histograms(query_histogram, hist)
        results[filename] = comparison

    sorted_results = sort_results(results, sorting_metric)
    return sorted_results


def print_results(reference_comparison, results):
    print("Query image:")
    reference_comparison = name_results(reference_comparison)
    print(f"{reference_comparison}")

    print("Search results:")
    for filename, comparison in results.items():
        named_comparison = name_results(comparison)
        print(f"{filename}: {named_comparison}")

# Main function
def main():
    folder = 'images/search'
    images = load_images_from_folder(folder)
    query_image = cv2.imread('images/query.jpg')

    bins = 32
    reference_histogram = compute_histogram(query_image, bins=bins, normalize=False)
    reference_comparison = compare_histograms(reference_histogram, reference_histogram)

    results = search_images(query_image, images, bins=bins)
    print_results(reference_comparison, results)

if __name__ == "__main__":
    image = list(load_images_from_folder('images/search').values())[2]
    hist_32 = compute_histogram(image, bins=32)
    plot_histogram(hist_32)
    # hist_64 = compute_histogram(image, bins=64)
    # hist_128 = compute_histogram(image, bins=128)
    # hist_256 = compute_histogram(image, bins=256)
    # plot_histogram(hist_64)
    # plot_histogram(hist_128)
    # plot_histogram(hist_256)
