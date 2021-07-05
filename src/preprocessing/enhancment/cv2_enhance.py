import cv2
import numpy as np


def enhance(img, clip_limit=2, nogray=False):
    clahe_cv2 = cv2.createCLAHE(clipLimit=clip_limit)
    if(nogray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = clahe_cv2.apply(img)
    if(nogray):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
