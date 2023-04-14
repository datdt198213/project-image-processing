import os
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilenames

import PIL
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from filter import *

class Main:
    global result
    def __init__(self):
        
        # Create a tkinter window
        self.window = tk.Tk()
        self.window.title("Image Processing")
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}")

        # Create and configure the top frame
        top_frame = tk.Frame(self.window, bd=0, relief=SOLID)
        top_frame.pack(side=TOP, fill=BOTH, expand=False)
        
        # Create a menu button and add it to the window
        menu_button = tk.Menubutton(top_frame, text="File", relief=FLAT, borderwidth=0, highlightthickness=0)
        menu_button.pack(side=LEFT, padx=10)
        # Create a menu and associate it with the menu button
        menu = tk.Menu(menu_button, tearoff=0)
        menu_button.config(menu=menu)
        # Create button open and add to the menu
        menu.add_command(label="Open", command=self.open_image)
        menu.add_command(label="Save", command=self.save_image)
        menu.add_separator()
        menu.add_command(label="Exit", command=self.exit_program)

        menu_button2 = tk.Menubutton(top_frame, text="Filter", relief=FLAT, borderwidth=0, highlightthickness=0)
        menu_button2.pack(side=LEFT)
        # Create a menu and associate it with the menu button
        filter_menu = tk.Menu(menu_button2, tearoff=0)
        menu_button2.config(menu=filter_menu)
        # Create 7 buttons and add them to the menu
        filter_menu.add_command(label="High Pass Filter", command=self.high_pass)
        filter_menu.add_command(label="Low Pass Filter", command=self.low_pass)
        filter_menu.add_command(label="Laplacian Filter", command=self.laplacian_filter)
        filter_menu.add_command(label="Bilateral Filter", command=self.bilateral_filter)
        filter_menu.add_command(label="Rank Filter", command=self.rank_filter)
        filter_menu.add_command(label="Canny Filter", command=self.canny_filter)
        filter_menu.add_separator()
        filter_menu.add_command(label="Hybrid Filter", command=self.hybrid_image_filter)

        # Create the main frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill='both', expand=True)

        # Create the text and image subframes
        text_frame = tk.Frame(main_frame)
        image_frame = tk.Frame(main_frame)

        # Create the left and right columns inside the text frame
        left_column = tk.Frame(text_frame)
        right_column = tk.Frame(text_frame)

        # Create the left and right columns inside the image frame
        left_column2 = tk.Frame(image_frame)
        right_column2 = tk.Frame(image_frame)

        # Create the text and image widgets
        self.text_widget = tk.Label(left_column, font=('Arial', 18), text="Input Image")
        self.text_widget.pack()
        self.text_widget2 = tk.Label(right_column, font=('Arial', 18), text="Output Image")
        self.text_widget2.pack()
        self.lbl_input = tk.Label(left_column2, font=('Arial', 18), text="Image goes here")
        self.lbl_input.pack()
        self.lbl_result = tk.Label(right_column2, font=('Arial', 18), text="Result goes here")
        self.lbl_result.pack()

        # Center the text and image subframes on the main frame
        main_frame.grid_rowconfigure(0, weight=0)
        main_frame.grid_columnconfigure(0, weight=1)

        text_frame.grid_rowconfigure(0, weight=0)
        text_frame.grid_columnconfigure(0, weight=0)
        text_frame.grid_columnconfigure(1, weight=0)

        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)

        # Pack the subframes and widgets into the main frame
        text_frame.pack(fill='x', pady=10)
        left_column.pack(side='left', expand=True)
        right_column.pack(side='left', expand=True)

        image_frame.pack(fill='both', expand=True)
        left_column2.pack(side='left', fill='both', expand=True)
        right_column2.pack(side='left', fill='both', expand=True)

        self.text_widget.pack(side='left', fill='both', expand=True)
        self.text_widget2.pack(side='left', fill='both', expand=True)
        self.lbl_input.pack(side='left', fill='both', expand=True)
        self.lbl_result.pack(side='left', fill='both', expand=True)
        
        # start the tkinter event loop
        self.window.mainloop()

    def set_image(self, filepath, lbl):
        if not filepath:
            return

        img = Image.open(filepath)
        img = img.convert("L")
        img.thumbnail((1000, 1000))
        img = ImageTk.PhotoImage(img)

        if lbl == self.lbl_input:
            global image1
            image1 = cv2.imread(filepath, 0)
        else:
            global image2
            image2 = cv2.imread(filepath, 0)

        lbl.configure(image=img)
        lbl.image = img
    
    def set_result(self, lbl):
        img_pil = PIL.Image.fromarray(result)
        img_tk = ImageTk.PhotoImage(img_pil)
        lbl.config(image=img_tk)
        lbl.image = img_tk
    
    def open_image(self):
        filepaths = askopenfilenames(
            initialdir=os,
            title="SelectImage File",
            filetypes=[("JPG Files", "*.jpg"), ("JPEG Files", "*.jpeg"), ("PNG Files", "*.png")]
        )

        if len(filepaths) == 1:
            self.set_image(filepath=filepaths[0], lbl=self.lbl_input)
            self.lbl_result.configure(image=None)
            self.lbl_result.image = None
        elif len(filepaths) == 2:
            self.set_image(filepaths[0], self.lbl_input)
            self.set_image(filepaths[1], self.lbl_result)

    def save_image(self):
        cv2.imwrite('saved_image.png', result)

    def exit_program(self):
        self.window.destroy()

    def high_pass(self):
        global result
        result = high_pass(image1)
        self.set_result(lbl=self.lbl_result)

    def low_pass(self):
        global result
        result = low_pass(image1)
        plt.imshow(np.abs(result), cmap='gray')
        self.set_result(lbl=self.lbl_result)

    def hybrid_image_filter(self):
        global result
        # hybrid_image_filter(image1, image2, 15, 15)
        result = hybrid_filter(high_pass_image=image1, low_pass_image=image2)
        self.set_result(lbl=self.lbl_result)

    def laplacian_filter(self):
        global result
        result = laplace_filter(image=image1, sigma=50, threshold=10)
        self.set_result(lbl=self.lbl_result)

    def canny_filter(self):
        global result
        result = canny_filter(image=image1)
        self.set_result(lbl=self.lbl_result)

    def bilateral_filter(self):
        global result
        result = low_pass(image=image1, method_type=4, sigma=50, sigma_color=100)
        self.set_result(lbl=self.lbl_result)

    def rank_filter(self):
        global result
        result = low_pass(image=image1, method_type=3)
        self.set_result(lbl=self.lbl_result)


if __name__ == '__main__':
    main = Main()