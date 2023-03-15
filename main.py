import os
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilenames

from PIL import Image, ImageTk
from matplotlib import pyplot as plt

from filter import *


class Main:
    def __init__(self):
        self.CONVOLUTION = 1
        self.FOURIER = 2
        self.window = tk.Tk()
        self.window.title("Image Processing")

        self.window.rowconfigure(0, minsize=350, weight=1)
        self.window.columnconfigure(1, minsize=350, weight=1)

        self.lbl1 = tk.Label(self.window)
        self.lbl2 = tk.Label(self.window)
        frm_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)

        btn_open = tk.Button(frm_buttons, text="Open", command=self.open_image)
        btn_hpf = tk.Button(frm_buttons, text="High Pass Filter", command=self.high_pass)
        btn_lpf = tk.Button(frm_buttons, text="Low Pass Filter", command=self.low_pass)
        btn_hf = tk.Button(frm_buttons, text="Hybrid Filter", command=self.hybrid_image_filter)
        btn_lf = tk.Button(frm_buttons, text="Laplacian Filter", command=self.laplacian_filter)
        btn_bila = tk.Button(frm_buttons, text="Bilateral Filter", command=self.bilateral_filter)
        btn_rank = tk.Button(frm_buttons, text="Rank Filter", command=self.rank_filter)
        btn_cf = tk.Button(frm_buttons, text="Canny Filter", command=self.canny_filter)

        btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        btn_hpf.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        btn_lpf.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        btn_hf.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        btn_lf.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        btn_cf.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        btn_bila.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        btn_rank.grid(row=7, column=0, sticky="ew", padx=5, pady=5)

        frm_buttons.grid(row=0, column=0, sticky="ns")
        self.lbl1.grid(row=0, column=1, sticky="nsew")
        self.lbl2.grid(row=0, column=2, sticky="nsew")

        self.window.mainloop()

    def set_image(self, filepath, lbl):
        if not filepath:
            return

        img = Image.open(filepath)
        img = img.convert("L")
        img.thumbnail((350, 350))
        img = ImageTk.PhotoImage(img)

        if lbl == self.lbl1:
            global image1
            image1 = cv2.imread(filepath, 0)
        else:
            global image2
            image2 = cv2.imread(filepath, 0)

        lbl.configure(image=img)
        lbl.image = img

    def open_image(self):
        filepaths = askopenfilenames(
            initialdir=os,
            title="SelectImage File",
            filetypes=[("JPG Files", "*.jpg"), ("JPEG Files", "*.jpeg"), ("PNG Files", "*.png")]
        )

        if len(filepaths) == 1:
            self.set_image(filepaths[0], self.lbl1)
            self.lbl2.configure(image=None)
            self.lbl2.image = None
        elif len(filepaths) == 2:
            self.set_image(filepaths[0], self.lbl1)
            self.set_image(filepaths[1], self.lbl2)

    def high_pass(self):
        result = high_pass(image1)
        plt.imshow(np.abs(result), cmap='gray')
        plt.show()

    def low_pass(self):
        result = low_pass(image1)
        plt.imshow(np.abs(result), cmap='gray')
        plt.show()

    def hybrid_image_filter(self):
        # hybrid_image_filter(image1, image2, 15, 15)
        resultImg = hybrid_filter(high_pass_image=image1, low_pass_image=image2)
        plt.imshow(np.real(resultImg), cmap='gray')
        plt.show()

    def laplacian_filter(self):
        result = laplace_filter(image=image1, sigma=50, threshold=10)
        plt.imshow(np.real(result), cmap='gray')
        plt.show()

    def canny_filter(self):
        result = canny_filter(image=image1)
        plt.imshow(np.real(result), cmap='gray')
        plt.show()

    def bilateral_filter(self):
        result = low_pass(image=image1, method_type=4, sigma=50, sigma_color=100)
        plt.imshow(np.real(result), cmap='gray')
        plt.show()

    def rank_filter(self):
        result = low_pass(image=image1, method_type=3)
        plt.imshow(np.real(result), cmap='gray')
        plt.show()


if __name__ == '__main__':
    main = Main()
