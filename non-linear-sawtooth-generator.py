import numpy as np
from PIL import Image


x_max = 17
y_min = 0
y_max = 255
phi = 45
height, width = (1080, 1920)

# 0: exponential, 1: power, 2: log, else: linear
function_type = 1
alpha = 0

def exponential_sawtooth(phase, alpha):
    return (np.exp(alpha * phase) - 1) / (np.exp(alpha) - 1)

def power_sawtooth(phase, alpha):
    assert alpha >= 0
    return (phase ** alpha)/(1 ** alpha)

def log_sawtooth(phase, alpha):
    assert alpha > 0
    return (np.log(1+phase*(np.e-1)*alpha))/np.log(1+alpha*(np.e-1))

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

sidelength = int(np.sqrt(height**2 + width**2))

x = np.arange(sidelength)
phase = (x % x_max) / (x_max - 1)

if function_type == 0:
    sawtooth_func = lambda: exponential_sawtooth(phase, alpha)
elif function_type == 1:
    sawtooth_func = lambda: power_sawtooth(phase, alpha)
elif function_type == 2:
    sawtooth_func = lambda: log_sawtooth(phase, alpha)
else:
    sawtooth_func = lambda: phase

values = y_min + sawtooth_func() * (y_max - y_min)

image_matrix = np.tile(values.astype(np.uint8), (sidelength, 1))
if phi == 0 or phi == 180:
    img = crop_center(Image.fromarray(image_matrix), width, height)
else:
    img = crop_center(Image.fromarray(image_matrix).rotate(phi), width, height)

img.show()
