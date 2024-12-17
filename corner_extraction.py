import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread('images/misc/building.png', cv2.IMREAD_GRAYSCALE)

# Define the structuring elements
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
diamond = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
x_shape = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
diamond = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],
                   dtype=np.uint8)
x_shape = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],
                   dtype=np.uint8)
square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

print("cross:\n", cross)
print("diamond:\n", diamond)
print("x_shape:\n", x_shape)
print("square:\n", square)

# Step 1: R1 = Dilate(Img, cross)
R1 = cv2.dilate(image, cross)

# Step 2: R1 = Erode(R1, diamond)
R1 = cv2.erode(R1, diamond)

# Step 3: R2 = Dilate(Img, Xshape)
R2 = cv2.dilate(image, x_shape)

# Step 4: R2 = Erode(R2, square)
R2 = cv2.erode(R2, square)

# Step 5: R = absdiff(R2, R1)
R = cv2.absdiff(R2, R1)

# Step 6: Display(R)
plt.imshow(R, cmap='gray')
plt.title('Extracted Corners')
plt.axis('off')
plt.show()
