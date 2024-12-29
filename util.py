import pickle
from skimage.transform import resize
import numpy as np
import cv2

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))  # Load the pre-trained model


def empty_or_not(spot_bgr):
    # Resize the image to the model's input size
    img_resized = resize(spot_bgr, (15, 15, 3), anti_aliasing=True)
    flat_data = img_resized.flatten().reshape(1, -1)
    # Predict and return the result
    y_output = MODEL.predict(flat_data)
    return y_output == 0  # Return boolean directly


def get_parking_spots_bboxes(connected_components):
    (totalLabels, _, values, _) = connected_components
    slots = []

    for i in range(1, totalLabels):  # Skip background
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        slots.append([x1, y1, w, h])

    return slots
