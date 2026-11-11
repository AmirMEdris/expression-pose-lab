"""An explicit image prior: bound Dream changes around the source luminance."""
import cv2
import numpy as np


def guided_step(pixels,gradient,reference,strength,budget=.14):
    gradient=cv2.GaussianBlur(np.asarray(gradient,dtype=np.float32),(5,5),.8)
    scale=max(float(gradient.std()),float(np.sqrt(np.mean(gradient**2)))*.05,1e-8)
    candidate=pixels+float(strength)*gradient/scale
    return np.clip(np.clip(candidate,reference-budget,reference+budget),0.,1.).astype(np.float32)
