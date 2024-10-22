import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

class ImageProcessingTool:
    def __init__(self, root):
        # Initialize the main application
        self.root = root
        self.root.title("Image Processing Tool")
        self.root.geometry("1200x1200")
        self.image = None  # Original loaded image
        self.display_image = None  # Image to display on canvas
        self.history = []  # Stack to keep track of image history for undo
        self.redo_stack = []  # Stack for redo functionality
        self.initial = True
        # Set up GUI Elements
        self.setup_gui()

    def setup_gui(self):

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=4)
        
        for i in range(12):  # Adjust this number based on the number of columns you have
            button_frame.columnconfigure(i, weight=1)

        # Button to open an image
        open_button = ttk.Button(button_frame, text="Open Image", command=self.open_image)
        open_button.grid(row=0, column=0, columnspan=6, padx=12, pady=4, sticky="ew")

        # Button to save the image
        save_button = ttk.Button(button_frame, text="Save Image", command=self.save_image)
        save_button.grid(row=0, column=6, columnspan=6, padx=12, pady=4, sticky="ew")

        # Button to adjust contrast/brightness
        adjust_button = ttk.Button(button_frame, text="Adjust Contrast/Brightness", command=self.adjust_contrast_brightness)
        adjust_button.grid(row=1, column=0, columnspan=3, padx=12, pady=4, sticky="ew")

        # Button for gray-level slicing
        slice_button = ttk.Button(button_frame, text="Gray-Level Slicing", command=self.gray_level_slicing)
        slice_button.grid(row=1, column=3, columnspan=3, padx=12, pady=4, sticky="ew")
        
        # Button to display the histogram
        histogram_button = ttk.Button(button_frame, text="Display Histogram", command=self.display_histogram)
        histogram_button.grid(row=1, column=6, columnspan=3, padx=12, pady=4, sticky="ew")

        # Button for bit-plane slicing
        bit_plane_button = ttk.Button(button_frame, text="Bit-plane Slicing", command=self.bit_plane_slicing)
        bit_plane_button.grid(row=1, column=9, columnspan=3, padx=12, pady=4, sticky="ew")
        
        # Button to zoom in or out of the image
        zoom_button = ttk.Button(button_frame, text="Zoom In/Out", command=self.zoom_image)
        zoom_button.grid(row=2, column=0, columnspan=4, padx=12, pady=4, sticky="ew")

        # Button to rotate the image
        rotate_button = ttk.Button(button_frame, text="Rotate Image", command=self.rotate_image)
        rotate_button.grid(row=2, column=4, columnspan=4, padx=12, pady=4, sticky="ew")

        # Button for smoothing and sharpening the image
        smooth_sharp_button = ttk.Button(button_frame, text="Smooth/Sharpen", command=self.smooth_sharpen)
        smooth_sharp_button.grid(row=2, column=8, columnspan=4, padx=12, pady=4, sticky="ew")

        # Button to undo the last action
        undo_button = ttk.Button(button_frame, text="Undo", command=self.undo)
        undo_button.grid(row=3, column=0, columnspan=4, padx=12, pady=4, sticky="ew")

        # Button to redo the last undone action
        redo_button = ttk.Button(button_frame, text="Redo", command=self.redo)
        redo_button.grid(row=3, column=4, columnspan=4, padx=12, pady=4, sticky="ew")

        # Button to undo all actions
        undo_all_button = ttk.Button(button_frame, text="Undo All", command=self.undo_all)
        undo_all_button.grid(row=3, column=8, columnspan=4, padx=12, pady=4, sticky="ew")

        # Canvas to display the image
        self.canvas = tk.Canvas(self.root, width=800, height=800)
        self.canvas.pack(side=tk.BOTTOM, pady=12)

    def clean_widget(self):
        try:
            self.widge_frame.destroy()
        except:
            pass
        self.widge_frame = tk.Frame(self.root)
        self.widge_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=5)
        for i in range(12):  # Adjust this number based on the number of columns you have
            self.widge_frame.columnconfigure(i, weight=1)
    
    def open_image(self):
        print("Opening image...")
        # Load image using file dialog
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.tif")])
        if file_path:
            print(f"Image loaded from: {file_path}")
            self.original_image = Image.open(file_path).convert("L") # Convert image to grayscale
            self.adjusted_image = self.original_image.copy()
            self.image = self.original_image.copy()
            self.save_state()  # Save the initial state
            self.update_display()
        else:
            print("No file selected.")

    def save_image(self):
        print("Saving image...")
        # Save the current image to a file
        if not self.image:
            messagebox.showerror("Error", "No image to save.")
            print("Error: No image to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("TIFF files", "*.tif")])
        if file_path:
            self.image.save(file_path)
            print(f"Image saved to: {file_path}")
        else:
            print("Save operation cancelled.")

    def save_state(self):
        # Save the current state of the image to history for undo functionality
        time.sleep(0.05)
        if self.image:
            self.history.append(self.image.copy())
            self.redo_stack.clear()  # Clear the redo stack whenever a new action is performed

    def undo(self):
        print("Undoing last action...")
        if self.history:
            self.redo_stack.append(self.image.copy())  # Save the current state to redo stack
            self.image = self.history.pop()  # Restore the last saved state
            self.update_display()
        else:
            print("No actions to undo.")

    def redo(self):
        print("Redoing last undone action...")
        if self.redo_stack:
            self.history.append(self.image.copy())  # Save the current state to history
            self.image = self.redo_stack.pop()  # Restore the last undone state
            self.update_display()
        else:
            print("No actions to redo.")

    def undo_all(self):
        print("Undoing all actions...")
        if self.history:
            self.image = self.history[0]  # Restore the initial state
            self.history = []  # Clear the history stack
            self.redo_stack = []  # Clear the redo stack
            self.update_display()
            print("All actions undone.")
        else:
            print("No actions to undo.")
        self.clean_widget()
        
    def update_display(self):
        time.sleep(0.05)
        print("Updating display...")
        # Update the display canvas with the current image
        self.display_image = ImageTk.PhotoImage(self.image)
        # Calculate the coordinates to center the image on the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = self.display_image.width()
        image_height = self.display_image.height()
        x = (canvas_width - image_width) // 2
        y = (canvas_height - image_height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.display_image)
        print("Display updated.")
        
# adjust contrast/brightness
    def adjust_contrast_brightness(self):
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
            
        self.clean_widget()
        # Row 1: Radio buttons to select method
        method_label = tk.Label(self.widge_frame, text="Select Adjustment Method:")
        method_label.grid(row=0, column=0, columnspan=6, padx=12, pady=5, sticky="w")

        # Radio buttons for selecting method (linear, exponential, logarithmic)
        self.method_var = tk.StringVar(value="linear")
        methods = [("Linear (Y = aX+b)", "linear"), ("Exponential (Y = exp(aX+b))", "exp"), ("Logarithmic (Y = ln(aX+b), b > 1)", "log")]
        for idx, (text, method) in enumerate(methods):
            method_button = ttk.Radiobutton(self.widge_frame, text=text, variable=self.method_var, value=method)
            if idx == 0:
                method_button.grid(row=0, column=2 * (idx + 3), columnspan=2, padx=12, pady=5, sticky='w')
            elif idx == 1:
                method_button.grid(row=0, column=2 * (idx + 3), columnspan=2, padx=12, pady=5, sticky='we')
            else:
                method_button.grid(row=0, column=2 * (idx + 3), columnspan=2, padx=12, pady=5, sticky='e')

        # Row 2: Slider for contrast 'a'
        contrast_label = tk.Label(self.widge_frame, text="Adjust Contrast (a):")
        contrast_label.grid(row=1, column=0, columnspan=3, padx=12, pady=5, sticky="w")
        self.contrast_value_label = tk.Label(self.widge_frame, text="1.0")  # Initialize with the default value
        self.contrast_value_label.grid(row=1, column=3, padx=12, pady=3, sticky="w")
        self.contrast_slider = ttk.Scale(self.widge_frame, from_=0.1, to_=3.0, orient=tk.HORIZONTAL, length=700, command=self.update_contrast)
        self.contrast_slider.set(1.0)  # Default contrast
        self.contrast_slider.grid(row=1, column=6, columnspan=6, padx=12, pady=5, sticky='we')
        
        # Row 3: Slider for brightness 'b'
        brightness_label = tk.Label(self.widge_frame, text="Adjust Brightness (b):")
        brightness_label.grid(row=2, column=0, columnspan=3, padx=12, pady=5, sticky="w")
        self.brightness_value_label = tk.Label(self.widge_frame, text="1.0")  # Initialize with the default value
        self.brightness_value_label.grid(row=2, column=3, padx=12, pady=3, sticky="w")
        self.brightness_slider = ttk.Scale(self.widge_frame, from_=-100, to_=100, orient=tk.HORIZONTAL, length=700, command=self.update_brightness)
        self.brightness_slider.set(0)  # Default brightness
        self.brightness_slider.grid(row=2, column=6, columnspan=6, padx=12, pady=5, sticky='we')
       
    def update_contrast(self, value):
        self.contrast_value_label.config(text=f"{float(value):.2f}")
        self.apply_adjustments()
        
    def update_brightness(self, value):
        self.brightness_value_label.config(text=f"{float(value):.2f}")
        self.apply_adjustments()
        
    def apply_adjustments(self):
        method = self.method_var.get()
        a = self.contrast_slider.get()
        b = self.brightness_slider.get()
        
        img_array = np.array(self.original_image.copy())

        if method == "linear":
            img_array = a * img_array + b
        elif method == "exp":
            img_array = np.exp(a * img_array + b)
        elif method == "log":
            img_array = np.log(a * img_array + b)

        # Clip the values to be within valid grayscale range
        img_array = np.clip(img_array, 0, 255)
        self.save_state()  # Save the current state before making changes
        self.adjusted_image = Image.fromarray(img_array.astype(np.uint8))
        self.image = self.adjusted_image.copy()
        self.update_display()

# zoom   
    def zoom_image(self):
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        self.clean_widget()
        # Row 1: Label to indicate zooming factor adjustment
        zoom_label = tk.Label(self.widge_frame, text="Adjust Zooming Factor:")
        zoom_label.grid(row=0, column=0, columnspan=3, padx=12, pady=3, sticky="w")

        # Label to display the zoom factor dynamically
        self.zoom_value_label = tk.Label(self.widge_frame, text="1.0")  # Initialize with the default value
        self.zoom_value_label.grid(row=0, column=9, padx=12, pady=3, sticky="e")
        
        # Row 1: Slider for zooming factor
        self.zoom_slider = ttk.Scale(self.widge_frame, from_=0.1, to_=10.0, orient=tk.HORIZONTAL, length=600, command=self.apply_zoom)
        self.zoom_slider.set(1.0)  # Default zooming factor
        self.zoom_slider.grid(row=0, column=3, columnspan=6, padx=12, pady=3, sticky="we")

    def apply_zoom(self, value):
        # Update the label next to the slider with the current zoom factor
        self.zoom_value_label.config(text=f"{float(value):.2f}")
        factor = self.zoom_slider.get()
        # Zoom in or out of the image
        new_size = (int(self.original_image.width * factor), int(self.original_image.height * factor))
        self.save_state()  # Save the current state before making changes
        self.zoomed_image = self.image.copy().resize(new_size, Image.BILINEAR)
        self.image = self.zoomed_image.copy()
        self.update_display()


# rotate
    def rotate_image(self):
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        self.clean_widget()
        
        # Row 1: Label to indicate rotating angle adjustment
        rotate_label = tk.Label(self.widge_frame, text="Adjust Rotating Angle (degree):")
        rotate_label.grid(row=0, column=0, columnspan=3, padx=12, pady=3, sticky="w")

        # Label to display the rotating angle dynamically
        self.rotate_value_label = tk.Label(self.widge_frame, text="0.0")  # Initialize with the default value
        self.rotate_value_label.grid(row=0, column=9, padx=12, pady=3, sticky="e")
        
        # Row 1: Slider for rotating angle
        self.rotate_slider = ttk.Scale(self.widge_frame, from_=-180, to_=180, orient=tk.HORIZONTAL, length=600, command=self.apply_rotate)
        self.rotate_slider.set(0.0)  # Default rotating angle
        self.rotate_slider.grid(row=0, column=3, columnspan=6, padx=12, pady=3, sticky="we")

    def apply_rotate(self, value):
        # Update the label next to the slider with the current zoom factor
        self.rotate_value_label.config(text=f"{float(value):.2f}")
        angle = self.rotate_slider.get()
        self.save_state()  # Save the current state before making changes
        self.rotated_image = self.original_image.copy().rotate(angle, expand=True)  # Expand to fit the entire rotated image
        # self.image = self.rotated_image.resize((self.original_image.width, self.original_image.height))
        self.image = self.rotated_image.copy()
        self.update_display()

# gray level slicing    
    def gray_level_slicing(self):

        # Gray-level slicing to highlight specific gray levels in the image
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        
        self.clean_widget()
        # Row 1: Radio buttons to select method
        preserve_label = tk.Label(self.widge_frame, text="Preserve unselected areas:")
        preserve_label.grid(row=0, column=0, columnspan=6, padx=12, pady=5, sticky="w")

        # Radio buttons for checking if preserved
        self.preserve_var = tk.StringVar(value="no")
        preserve = [("Preserve", "yes"), ("Don't Preserve", "no")]
        for idx, (text, method) in enumerate(preserve):
            preserve_button = ttk.Radiobutton(self.widge_frame, text=text, variable=self.preserve_var, value=method, command=self.apply_grey_level)
            if idx == 0:
                preserve_button.grid(row=0, column=6, columnspan=3, padx=12, pady=5, sticky='we')
            else:
                preserve_button.grid(row=0, column=9, columnspan=3, padx=12, pady=5, sticky='e')
        # Row 2: Slider for Lower level
        lower_label = tk.Label(self.widge_frame, text="Lower level:")
        lower_label.grid(row=1, column=0, columnspan=3, padx=12, pady=5, sticky="w")
        self.lower_value_label = tk.Label(self.widge_frame, text="0")  # Initialize with the default value
        self.lower_value_label.grid(row=1, column=3, padx=12, pady=3, sticky="w")
        self.lower_slider = ttk.Scale(self.widge_frame, from_=0, to_=255, orient=tk.HORIZONTAL, length=700, command=self.update_lower)
        self.lower_slider.set(1.0)  # Default contrast
        self.lower_slider.grid(row=1, column=6, columnspan=6, padx=12, pady=5, sticky='we')
        
        # Row 3: Slider for Upper level
        upper_label = tk.Label(self.widge_frame, text="Upper level:")
        upper_label.grid(row=2, column=0, columnspan=3, padx=12, pady=5, sticky="w")
        self.upper_value_label = tk.Label(self.widge_frame, text="255")  # Initialize with the default value
        self.upper_value_label.grid(row=2, column=3, padx=12, pady=3, sticky="w")
        self.upper_slider = ttk.Scale(self.widge_frame, from_=0, to_=255, orient=tk.HORIZONTAL, length=700, command=self.update_upper)
        self.upper_slider.set(255)  # Default brightness
        self.upper_slider.grid(row=2, column=6, columnspan=6, padx=12, pady=5, sticky='we')
    
    def update_lower(self, value):
        self.lower_value_label.config(text=f"{int(float(value))}")
        self.apply_grey_level()
        
    def update_upper(self, value):
        self.upper_value_label.config(text=f"{int(float(value))}")
        self.apply_grey_level()
        
    def apply_grey_level(self):
        
        preserve = self.preserve_var.get()
        try:
            lower = int(self.lower_slider.get())
        except:
            lower = int(0)
        try:
            upper = int(self.upper_slider.get())
        except:
            upper = int(255)

        img_array = np.array(self.original_image)
        # Create a mask to select specific gray levels
        mask = (img_array >= lower) & (img_array <= upper)
        if preserve == 'yes':
            img_array[~mask] = img_array[~mask]  # Preserve unselected areas
        else:
            img_array[~mask] = 0  # Set unselected areas to black

        self.save_state()  # Save the current state before making changes
        self.sliced_image = Image.fromarray(img_array)
        self.image = self.sliced_image.copy()
        self.update_display()
        
# display histogram
    def display_histogram(self):

        # Display the histogram of the current image
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        # Plot histogram using matplotlib
        plt.hist(np.array(self.image).flatten(), bins=256, range=[0, 256], color='black')
        plt.xlabel('Gray Level')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.show()

# bit plane slicing
    def bit_plane_slicing(self):
        # Bit-plane slicing to extract specific bit levels of the image
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        
        self.clean_widget()
        # Row 1: Radio buttons to select bit-plane
        level_label = tk.Label(self.widge_frame, text="Bit-plane Level:")
        level_label.grid(row=0, column=0, columnspan=3, padx=12, pady=5, sticky="w")

        self.level_var = tk.StringVar(value="Original")
        level = list(range(8)) + ['Original']
        for idx, l in enumerate(level):
            level_button = ttk.Radiobutton(self.widge_frame, text=str(l), variable=self.level_var, value=str(l), command=self.apply_bit_plane)
            level_button.grid(row=0, column=3 + idx, padx=12, pady=5, sticky='we')
   
    def apply_bit_plane(self):
        bit = self.level_var.get()
        if bit != 'Original':
            img_array = np.array(self.original_image).astype(np.uint8)
            # Extract the selected bit-plane
            bit_plane_img = (img_array >> int(float(bit))) & 1
            bit_plane_img *= 255  # Scale values to 0 or 255 for visualization
            self.save_state()  # Save the current state before making changes
            self.bit_plane_image = Image.fromarray(bit_plane_img.astype(np.uint8))
            self.image = self.bit_plane_image.copy()
        else:
            self.image = self.original_image.copy()
        self.update_display()


    def smooth_sharpen(self):
        # Apply smoothing or sharpening to the image using spatial filters
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        
        
        level = simpledialog.askfloat("Input", "Enter level of smoothing/sharpening")
        method = simpledialog.askstring("Input", "Enter method (smooth or sharpen)")

        print(f"Method: {method}, Level: {level}")
        img_array = np.array(self.image)
        # Apply appropriate filter based on user input
        if method == "smooth":
            kernel = np.ones((3, 3), np.float32) / (9 * level)  # Smoothing kernel
            img_array = cv2.filter2D(img_array, -1, kernel)
        elif method == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9 + level, -1], [-1, -1, -1]])  # Sharpening kernel
            img_array = cv2.filter2D(img_array, -1, kernel)
        else:
            print("Invalid method selected.")
            return

        self.save_state()  # Save the current state before making changes
        self.image = Image.fromarray(img_array)
        self.update_display()

if __name__ == "__main__":
    print("Starting Image Processing Tool...")
    root = tk.Tk()
    app = ImageProcessingTool(root)
    root.mainloop()
    print("Image Processing Tool closed.")
