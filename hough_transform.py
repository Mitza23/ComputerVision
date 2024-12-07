# Python program to illustrate HoughLine
# method for line detection
import math

import cv2
import numpy as np


# The Hough transform is a technique for detecting lines, circles, or other shapes in an image.
# It works by representing the shapes in the image as points in a multi-dimensional space,
# and then using voting to identify the points that belong to the same shape.
#
# For example, to detect lines in an image, the Hough transform represents each point in the
# image as a pair of polar coordinates (rho, theta) in a two-dimensional space, where rho is
# the distance from the point to the origin and theta is the angle between the point and the
# positive x-axis. The line through the point can then be represented as a sinusoidal curve in
# this space. By identifying clusters of points that fall on the same curve, the
# Hough transform can detect lines in the image.

def hough_transform(file_name, peak_vicinity=1, theta_precision=5, pixel_intensity_threshold=200):
    # Load the desired image
    img = cv2.imread(file_name)

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image = np.asarray(gray)  # (height, width)
    img_h, img_w = image.shape
    diag = int(math.sqrt(img_w ** 2 + img_h ** 2))

    # The precision of the search is of 1 degree and of one pixel
    accumulator = np.zeros((diag, 180))

    # Mark down in the accumulator the lines that can go through each point from the image
    for i, line in enumerate(image):
        for j, val in enumerate(line):
            if val > pixel_intensity_threshold:
                for theta in range(0, 180, theta_precision):
                    # x*cos(theta) + y*sin(theta) = r
                    r = int(j * math.cos(math.pi * theta / 180) + i * math.sin(math.pi * theta / 180))
                    accumulator[r][theta] = accumulator[r][theta] + 1

    # Find out the biggest frequency in the accumulator
    max = 0
    for i in range(diag):
        for j in range(180):
            if accumulator[i][j] > max:
                max = accumulator[i][j]

    lines = []
    # y = (r - x * cos(theta)) / sin(theta)
    for i in range(diag):
        for j in range(180):
            if max - accumulator[i][j] <= peak_vicinity:
                lines.append({"theta": j, "r": i})
                r = i
                theta = math.pi * j / 180
                # Stores the value of cos(theta) in a
                cos = np.cos(theta)

                # Stores the value of sin(theta) in sin
                sin = np.sin(theta)

                # x0 stores the value rcos(theta)
                x0 = cos * r

                # y0 stores the value rsin(theta)
                y0 = sin * r

                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1000 * (-sin))

                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1000 * (cos))

                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1000 * (-sin))

                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1000 * (cos))

                # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                # (0,0,255) denotes the colour of the line to be
                # drawn. In this case, it is red.
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite('linesDetected_' + file_name, img)
    cv2.imshow("linesDetected", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for i in range(diag):
    #     for j in range(180):
    #         print(int(accumulator[i][j]), end=' ')
    #     print('')

    # Normalize values in the Hough space for better visibility
    for i in range(diag):
        for j in range(180):
            accumulator[i][j] = (255 // max) * accumulator[i][j]

    cv2.imwrite('houghSpace_' + file_name, accumulator)
    # cv2.imshow("HoughSpace", accumulator)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return lines


def hough_transform_segments(file_name, peak_vicinity=1, theta_precision=5, pixel_intensity_threshold=200):
    # Load the desired image
    img = cv2.imread(file_name)

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image = np.asarray(gray)  # (height, width)
    img_h, img_w = image.shape
    diag = int(math.sqrt(img_w ** 2 + img_h ** 2))

    # The precision of the search is of 1 degree and of one pixel
    accumulator = [[[] for _ in range(180)] for _ in range(diag)]

    # Mark down in the accumulator the lines that can go through each point from the image
    for i, line in enumerate(image):
        for j, val in enumerate(line):
            if val > pixel_intensity_threshold:
                for theta in range(0, 180, theta_precision):
                    # x*cos(theta) + y*sin(theta) = r
                    r = int(j * math.cos(math.pi * theta / 180) + i * math.sin(math.pi * theta / 180))
                    accumulator[r][theta].append((i, j))

    # Find out the biggest frequency in the accumulator
    max = 0
    for i in range(diag):
        for j in range(180):
            if len(accumulator[i][j]) > max:
                max = len(accumulator[i][j])

    segments = []
    # y = (r - x * cos(theta)) / sin(theta)
    for i in range(diag):
        for j in range(180):
            if max - len(accumulator[i][j]) <= peak_vicinity:
                segments.append(accumulator[i][j])
                for point in accumulator[i][j]:
                    cv2.circle(img, (point[1], point[0]), 1, (0, 0, 255), 1)

    cv2.imwrite('segments_' + file_name, img)

    # for i in range(diag):
    #     for j in range(180):
    #         print(int(accumulator[i][j]), end=' ')
    #     print('')

    # Normalize values in the Hough space for better visibility
    hough_space = np.zeros((diag, 180))
    for i in range(diag):
        for j in range(180):
            hough_space[i][j] = (255 // max) * len(accumulator[i][j])

    cv2.imwrite('houghSpace_' + file_name, hough_space)

    return segments

def hough_transform_circles(file_name, radius=20, peak_vicinity=1, pixel_intensity_threshold=200):
    # Load the desired image
    img = cv2.imread(file_name)

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image = np.asarray(gray)  # (height, width)
    img_h, img_w = image.shape

    # Create an accumulator for the Hough space
    accumulator = np.zeros((img_h, img_w))

    # Mark down in the accumulator the circles that can go through each point from the image

    accumulate_circles(accumulator, image, img_h, img_w, pixel_intensity_threshold, radius)

    # Find out the biggest frequency in the accumulator
    max_acc = np.max(accumulator)

    circles = []
    for i in range(img_h):
        for j in range(img_w):
            if max_acc - accumulator[i][j] <= peak_vicinity:
                circles.append({"x": j, "y": i, "radius": radius})
                # Draw the circle
                cv2.circle(img, (j, i), radius, (0, 0, 255), 1)

    cv2.imwrite('circlesDetected_' + file_name, img)

    return circles


def accumulate_circles(accumulator, image, img_h, img_w, pixel_intensity_threshold, radius):
    for i in range(img_h):
        for j in range(img_w):
            if image[i][j] > pixel_intensity_threshold:
                for theta in range(0, 360):
                    a = int(j - radius * math.cos(math.pi * theta / 180))
                    b = int(i - radius * math.sin(math.pi * theta / 180))
                    if 0 <= a < img_w and 0 <= b < img_h:
                        accumulator[b][a] += 1


if __name__ == '__main__':
    # hough_transform("5.png", peak_vicinity=1, theta_precision=5, pixel_intensity_threshold=200)
    # hough_transform_circles("circles.png", radius=20, peak_vicinity=1, pixel_intensity_threshold=200)
    hough_transform_segments("5.png", peak_vicinity=1, theta_precision=5, pixel_intensity_threshold=200)
