import numpy as np
from .stain_normalizer import StainNormalizer
import cv2


class Normalizers:
    def __init__(self, method: str, fit="V1"):
        self.norm = StainNormalizer()
        if fit == "V1":
            temp_images = np.asarray(cv2.imread("./norm/template/V1.png"))
        elif fit == "V2":
            temp_images = np.asarray(cv2.imread("./norm/template/V2.png"))
        else:
            raise ValueError(f"Unrecognized normalizer fit {fit}")
        self.norm.fit(temp_images[:, :, [2, 1, 0]])

    def transform(self, image: np.array):
        return self.norm.transform(image)
