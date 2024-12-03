import cv2

# Load the logo image
logo = cv2.imread('images/misc/logo.png', cv2.IMREAD_UNCHANGED)

# Convert the logo to grayscale
gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_logo", gray_logo)
cv2.waitKey(0)

# Threshold the grayscale image to create a binary mask
_, mask = cv2.threshold(gray_logo, 230, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("mask", mask)
cv2.waitKey(0)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("inv mask", mask_inv)
cv2.waitKey(0)

# Extract the logo region
logo_fg = cv2.bitwise_and(logo, logo, mask=mask)
cv2.imshow("logo fg", logo_fg)
cv2.waitKey(0)

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the region of interest (ROI) in the frame where the logo will be placed
    rows, cols, _ = logo.shape
    roi = frame[0:rows, 0:cols]

    # Black-out the area of the logo in the ROI
    frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Add the logo to the ROI
    combined = cv2.add(frame_bg, logo_fg)

    # Place the combined image back into the frame
    frame[0:rows, 0:cols] = combined

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
