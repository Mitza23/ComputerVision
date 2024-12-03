import cv2
import matplotlib.pyplot as plt

# Load the image in color
image = cv2.imread('images/hist/red_hist.png')

# Split the image into BGR channels
b, g, r = cv2.split(image)

print(b)

# Compute the histogram for each channel
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

# Plot the histograms
plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist_b, color='b')
plt.plot(hist_g, color='g')
plt.plot(hist_r, color='r')
plt.xlim([0, 256])
plt.show()
