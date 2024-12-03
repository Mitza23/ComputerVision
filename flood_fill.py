import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread('images/misc/flood.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# print(gray)
# plt.imshow(gray, cmap='gray')
# plt.title('Grayscale Image')
# plt.axis('off')
# plt.show()


# Apply binary threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Invert the grayscale image
inverted_image = cv2.bitwise_not(binary)

height, width = image.shape[:2]
seed = (height // 2, width // 2)
kernel = np.ones((3, 3), np.uint8)

mask = np.zeros((height, width), np.uint8)

mask[seed[1], seed[0]] = 255
prev = None
mask_images = []
while not np.array_equal(prev, mask):
    prev = np.copy(mask)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.bitwise_and(mask, binary)
    mask_images.append(np.copy(mask))

mask = cv2.bitwise_not(mask)
result = cv2.bitwise_and(binary, mask)

# Calculate the number of rows needed for the grid
num_images = len(mask_images) + 4  # Initial, Binary, Inverted, and Result images
num_cols = 3
num_rows = (num_images + num_cols - 1) // num_cols

# Plot the images in a grid
plt.figure(figsize=(15, 5 * num_rows))

# Initial image
plt.subplot(num_rows, num_cols, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Initial Image')
plt.axis('off')

# Binary image
plt.subplot(num_rows, num_cols, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# Inverted image
plt.subplot(num_rows, num_cols, 3)
plt.imshow(inverted_image, cmap='gray')
plt.title('Inverted Image')
plt.axis('off')

# Mask images
for i, mask_img in enumerate(mask_images, start=4):
    plt.subplot(num_rows, num_cols, i)
    plt.imshow(mask_img, cmap='gray')
    plt.title(f'Mask Image {i - 3}')
    plt.axis('off')

# Result image
plt.subplot(num_rows, num_cols, num_images)
plt.imshow(result, cmap='gray')
plt.title('Result Image')
plt.axis('off')

plt.tight_layout()
plt.show()
