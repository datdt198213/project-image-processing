import os
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilenames

from PIL import Image, ImageTk
from matplotlib import pyplot as plt

from filter import *


class Main:
    def __init__(self):

        # create a tkinter window
        self.window = tk.Tk()
        self.window.title("Image Processing")
                
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}")
        # create the top frame and add it to the window
        top_frame = tk.Frame(self.window, relief=RAISED, borderwidth=1)
        top_frame.pack(side=TOP, pady=10)

        # Create a menu button and add it to the window
        menu_button = tk.Menubutton(top_frame, text="File", relief=FLAT, borderwidth=0, highlightthickness=0)
        menu_button.pack(side=LEFT, padx=10)
        # Create a menu and associate it with the menu button
        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.config(menu=menu)
        # Create button open and add to the menu
        menu.add_command(label="Open", command=self.open_image)

        menu_button2 = tk.Menubutton(top_frame, text="Filter", relief=FLAT, borderwidth=0, highlightthickness=0)
        menu_button2.pack(side=LEFT)
        # Create a menu and associate it with the menu button
        menu2 = tk.Menu(menu_button2, tearoff=0)
        menu_button2.config(menu=menu2)
        # Create 7 buttons and add them to the menu
        menu2.add_command(label="High Pass Filter", command=self.high_pass)
        menu2.add_command(label="Low Pass Filter", command=self.low_pass)
        menu2.add_command(label="Hybrid Filter", command=self.hybrid_image_filter)
        menu2.add_command(label="Laplacian Filter", command=self.laplacian_filter)
        menu2.add_command(label="Bilateral Filter", command=self.bilateral_filter)
        menu2.add_command(label="Rank Filter", command=self.rank_filter)
        menu2.add_command(label="Canny Filter", command=self.canny_filter)
        # add widgets to the top frame

        # create the bottom frames and add them to the window
        bottom_frame1 = tk.Frame(self.window, relief=RAISED, borderwidth=1)
        bottom_frame1.pack(side=LEFT, padx=10, pady=10)

        menu_button2 = tk.Menubutton(bottom_frame1, text="File", relief=FLAT, borderwidth=0, highlightthickness=0)
        menu_button2.pack(side=LEFT, padx=10)
        # Create a menu and associate it with the menu button
        menu3 = tk.Menu(menu_button2, tearoff=0)
        menu_button2.config(menu=menu3)
        # Create button open and add to the menu
        menu3.add_command(label="Open", command=self.open_image)

        bottom_frame2 = Frame(self.window, relief=RAISED, borderwidth=1)
        bottom_frame2.pack(side=RIGHT, padx=10, pady=10)

        # add widgets to the bottom frames
        # e.g. add images for old and new

        # start the tkinter event loop
        self.window.mainloop()
    
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