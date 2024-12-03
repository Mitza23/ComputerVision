import cv2
import matplotlib.pyplot as plt

# Load the image in BGR format
image_bgr = cv2.imread('images/misc/melon.png')

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Convert to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Convert to LAB
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

# Convert to YCrCb
image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

# Convert to RGBA
image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)

# Create a list of images and their titles
images = [image_rgb, image_gray, image_hsv, image_lab, image_ycrcb, image_rgba]
titles = ['RGB', 'Grayscale', 'HSV', 'LAB', 'YCrCb', 'RGBA']

# Here is a brief explanation of the different image formats:
#
# 1. **RGB (Red, Green, Blue)**:
#    - A common color space for images.
#    - Each pixel is represented by three values corresponding to the intensities of red, green, and blue.
#
# 2. **Grayscale**:
#    - Single channel image.
#    - Each pixel represents the intensity of light, ranging from black to white.
#
# 3. **HSV (Hue, Saturation, Value)**:
#    - Hue represents the color type.
#    - Saturation represents the vibrancy of the color.
#    - Value represents the brightness of the color.
#    - Useful for color-based segmentation and image analysis.
#
# 4. **LAB (CIELAB)**:
#    - L* represents lightness.
#    - a* and b* represent color-opponent dimensions.
#    - Designed to be perceptually uniform, meaning changes in color are consistent with human vision.
#
# 5. **YCrCb (Luminance, Chrominance)**:
#    - Y represents the brightness (luminance).
#    - Cr and Cb represent the chrominance (color information).
#    - Commonly used in video compression and broadcasting.
#
# 6. **RGBA (Red, Green, Blue, Alpha)**:
#    - Similar to RGB but includes an alpha channel.
#    - The alpha channel represents transparency, allowing for blending and overlaying images.
#
# These formats are used for different purposes depending on the application, such as image processing, computer vision, and video compression.

# # Display the image using matplotlib
# plt.imshow(image_hsv)
# plt.axis('off')  # Hide the axis
# plt.show()

# Plot the images in a grid
plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    if len(images[i].shape) == 2:  # Grayscale image
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')  # Hide the axis

plt.tight_layout()
plt.show()
