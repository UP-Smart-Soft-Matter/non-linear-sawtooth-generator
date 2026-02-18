import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import screeninfo


x_max = 17
y_min = 0
y_max = 255
phi = 45
height, width = (1080, 1920)

# 0: exponential, 1: power, 2: log, else: linear
function_type = 0
alpha = 2

class App(tk.Tk):
    def __init__(self, monitor: int):
        super().__init__()
        self.image_display = ImageDisplay(monitor)

        self.protocol("WM_DELETE_WINDOW")

        img = generate_sawtooth(x_max, y_min, y_max, phi, height, width, function_type, alpha)

        self.image_display.show_image(img)

        self.mainloop()

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


def generate_sawtooth(x_max, y_min, y_max, phi, height, width, function_type, alpha):
    sidelength = int(np.sqrt(height ** 2 + width ** 2))
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

    return img

class ImageDisplay(tk.Toplevel):
    def __init__(self, monitor: int):
        assert isinstance(monitor, int) and monitor >= 0, "Monitor must be a non-negative integer!"

        super().__init__()

        monitors = screeninfo.get_monitors()


        if len(monitors) <= monitor:
            raise Exception(f"Monitor index {monitor} is out of range. Found {len(monitors)} monitors.")

        # Select the specified monitor
        selected_monitor = monitors[monitor]
        self.width, self.height = selected_monitor.width, selected_monitor.height

        self.geometry(f"{self.width}x{self.height}+{selected_monitor.x}+{selected_monitor.y}")
        self.configure(background='black')

        self.overrideredirect(True)

        # Initialize the label to None
        self.label = None

    def show_image(self, image_object):
        assert isinstance(image_object, Image.Image), "Image must be a PIL Image object"

        photo = ImageTk.PhotoImage(image_object)

        if self.label is None:
            # Create a label to hold the image
            self.label = tk.Label(self, image=photo)
            self.label.image = photo  # Keep a reference to avoid garbage collection
            self.label.pack()
        else:
            self.__update_image(photo)

    def __update_image(self, photo):
        assert isinstance(photo, ImageTk.PhotoImage), "Image must be a PhotoImage object"

        # Update the image in the existing label
        self.label.configure(image=photo)
        self.label.image = photo  # Update the reference to avoid garbage collection

    class NoSecondMonitorError(Exception):
        pass

App(0)
