# References:
# - Mediapipe Hair segmentation model (Not supported in Python due to the custom operations): https://google.github.io/mediapipe/solutions/hair_segmentation
# - Model used in this example: https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP
# - Read gif from url: https://stackoverflow.com/questions/48163539/how-to-read-gif-from-url-using-opencv-python

import cv2
import numpy as np

from utils.HairSegmentation import HairSegmentation

# Initialize webcam
cap = cv2.VideoCapture(0)
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
windowName = 'RT5-61B Pakalo A. S. Hair Segmentation'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

# Initialize trackbars.
def empty(a):
    pass
# Color channels.
cv2.createTrackbar('Blue', windowName, 0, 255, empty)
cv2.createTrackbar('Green', windowName, 0, 255, empty)
cv2.createTrackbar('Red', windowName, 0, 255, empty)
# Dye Weight (kind of alpha channel).
cv2.createTrackbar('Dye Weight', windowName, 0, 100, empty)


# Inialize hair segmentation model
hair_segmentation = HairSegmentation(webcam_width, webcam_height)

# Dye image.
def dye_bgr(image, b, g, r):
    dyed_image = np.zeros_like(image)

    dyed_image[:] = b, g, r
    return dyed_image


while cap.isOpened():

    # Read frame
    success, frame = cap.read()

    img_height, img_width, _ = frame.shape

    if not success:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)
    hair_mask = hair_segmentation(frame)

    # Get color values from corresponding trackbars.
    b = cv2.getTrackbarPos('Blue', windowName)
    g = cv2.getTrackbarPos('Green', windowName)
    r = cv2.getTrackbarPos('Red', windowName)

    # Get dyed frame.
    dyed_frame = dye_bgr(frame, b, g, r)

    # Mask our dyed frame (pixels out of mask are black).
    dyed_hair = cv2.bitwise_or(frame, dyed_frame, mask=hair_mask)

    dye_weight = cv2.getTrackbarPos('Dye Weight', windowName) / 100
    # Overlay initial frame with masked (dyed hair) by formula:
    #   dst = scr1*alpha + scr2*beta + gamma,
    #   where dst = cv.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    frame_with_dyed_hair = cv2.addWeighted(frame, 1, dyed_hair, dye_weight, 0.0)

    cv2.imshow(windowName, frame_with_dyed_hair)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
