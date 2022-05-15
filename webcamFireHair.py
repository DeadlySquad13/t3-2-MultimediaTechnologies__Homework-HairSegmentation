# References:
# - Mediapipe Hair segmentation model (Not supported in Python due to the custom operations): https://google.github.io/mediapipe/solutions/hair_segmentation
# - Model used in this example: https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP
# - Read gif from url: https://stackoverflow.com/questions/48163539/how-to-read-gif-from-url-using-opencv-python

import cv2
import numpy as np

from utils.fire_hair_utils import HairSegmentation, get_fire_gif

# Initialize webcam
cap = cv2.VideoCapture(0)
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
windowName = 'РТ5-61Б Пакало А. С. Hair Segmentation'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

# Inialize hair segmentation model
hair_segmentation = HairSegmentation(webcam_width, webcam_height)


def empty(a):
    pass


# cv2.resizeWindow('BGR', 640, 240)
cv2.createTrackbar('Blue', windowName, 0, 255, empty)
cv2.createTrackbar('Green', windowName, 0, 255, empty)
cv2.createTrackbar('Red', windowName, 0, 255, empty)

cv2.createTrackbar('Dye Weight', windowName, 0, 100, empty)

# Dye hair.
def dye_hsv(hair): 
    # extract bgr channels
    bgr = hair[:,:,0:3]

    # convert to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    # purple is 276 in range 0 to 360; so half in OpenCV
    # green is 120 in range 0 to 360; so half in OpenCV
    purple = 138
    green = 60

    # diff_color = green - purple
    # diff_color = 100

    h_diff = cv2.getTrackbarPos('Blue', windowName)
    s_diff = cv2.getTrackbarPos('Green', windowName)
    v_diff = cv2.getTrackbarPos('Red', windowName)

    # modify hue channel by adding difference and modulo 180
    hnew = np.mod(h + h_diff, 180).astype(np.uint8)
    snew = np.mod(s + s_diff, 180).astype(np.uint8)
    vnew = np.mod(v + v_diff, 180).astype(np.uint8)

    # recombine channels
    hsv_new = cv2.merge([hnew,snew,vnew])

    # convert back to bgr.
    dyed_hair = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    return dyed_hair


def dye_bgr(image):
    dyed_image = np.zeros_like(image)
    b = cv2.getTrackbarPos('Blue', windowName)
    g = cv2.getTrackbarPos('Green', windowName)
    r = cv2.getTrackbarPos('Red', windowName)

    dyed_image[:] = b, g, r
    # dyed_hair[:] = np.array([b, g, r])[]
    # dyed_image = cv2.bitwise_or(image, dyed_image)
    # dyed_hair = cv2.GaussianBlur(dyed_hair,(7,7),10)

    #color_image
    # imgColorLips = cv2.addWeighted(imgOriginal,1,imgColorLips,0.4,0)
    return dyed_image


while cap.isOpened():

    # Read frame
    ret, frame = cap.read()

    img_height, img_width, _ = frame.shape

    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)
    hair_mask = hair_segmentation(frame)
    dyed_frame = dye_bgr(frame)
    dyed_hair = cv2.bitwise_or(frame, dyed_frame, mask=hair_mask)
    # frame_with_dyed_hair = cv2.bitwise_or(frame, dyed_hair)
    dye_weight = cv2.getTrackbarPos('Dye Weight', windowName) / 100
    frame_with_dyed_hair = cv2.addWeighted(frame, 1, dyed_hair, dye_weight, 0.0)

    # frame_with_dyed_hair = cv2.bitwise_or(frame, dyed_frame)

    cv2.imshow(windowName, frame_with_dyed_hair)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
