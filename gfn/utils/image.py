import cv2
import numpy as np


def crops(image, bbox):
  h, w = image.shape[:2]
  x1, y1, x2, y2 = bbox
  x1 = int(max(0, x1))
  y1 = int(max(0, y1))
  x2 = int(min(w, x2))
  y2 = int(min(h, y2))
  return image[y1:y2, x1:x2]


