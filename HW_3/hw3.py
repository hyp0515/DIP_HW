import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.ndimage import median_filter
from scipy.fftpack import fft2 , ifft2, fftshift, fftfreq
import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)


class ImageProcessingTool:
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("Image Processing Tool")
        self.root.geometry("1000x700")
        
        # Attributes to store image data
        self.image = None  # Original loaded image
        self.display_image = None  # Image to display on canvas
        self.history = []  # Stack to keep track of image history for undo functionality
        self.redo_stack = []  # Stack for redo functionality
        self.initial = True
        
        # Set up GUI Elements
        self.setup_gui()
        self.zoom_factor = 1  # Current zoom factor
        self.method = 'original'  # Current filter method
        self.click = 0  # Click count for laplacian filter toggle
        self.adjusted = False  # Flag to indicate if image is adjusted
    
    def setup_gui(self):
        # Set up GUI buttons and canvas
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=3)
        
        for i in range(12):  # Configure columns in button frame
            button_frame.columnconfigure(i, weight=1)

        # Button to open an image
        open_button = ttk.Button(button_frame, text="Open Image", command=self.open_image)
        open_button.grid(row=0, column=0, columnspan=3, padx=12, pady=3, sticky="ew")

        # Button to save the image
        save_button = ttk.Button(button_frame, text="Save Image", command=self.save_image)
        save_button.grid(row=0, column=3, columnspan=3, padx=12, pady=3, sticky="ew")

        # Button to undo the last action
        undo_button = ttk.Button(button_frame, text="Undo", command=self.undo)
        undo_button.grid(row=0, column=6, columnspan=2, padx=12, pady=3, sticky="ew")

        # Button to redo the last undone action
        redo_button = ttk.Button(button_frame, text="Redo", command=self.redo)
        redo_button.grid(row=0, column=8, columnspan=2, padx=12, pady=3, sticky="ew")

        # Button to undo all actions
        undo_all_button = ttk.Button(button_frame, text="Undo All", command=self.undo_all)
        undo_all_button.grid(row=0, column=10, columnspan=2, padx=12, pady=3, sticky="ew")
        
        # Button to zoom in or out of the image
        zoom_button = ttk.Button(button_frame, text="Zoom In/Out", command=self.zoom_image)
        zoom_button.grid(row=1, column=0, columnspan=6, padx=12, pady=3, sticky="ew")

        # Button to apply filtering to the image
        filter_button = ttk.Button(button_frame, text="Filtering", command=self.filtering)
        filter_button.grid(row=1, column=6, columnspan=6, padx=12, pady=3, sticky="ew")
        
        # Button to perform 2D FFT on the image
        fft_button = ttk.Button(button_frame, text="2D-FFT", command=self.FFT_2D)
        fft_button.grid(row=2, column=0, columnspan=6, padx=12, pady=3, sticky="ew")
        
        # Button to perform DFT on the image
        dft_button = ttk.Button(button_frame, text="DFT", command=self.DFT)
        dft_button.grid(row=2, column=6, columnspan=6, padx=12, pady=3, sticky="ew")

        # Canvas to display the image
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack(side=tk.BOTTOM, pady=12)

    def clean_widget(self):
        # Clean up the widget frame if it exists
        try:
            self.widget_frame.destroy()
            self.adjusted_image = self.image.copy()
            self.adjusted = True
        except:
            pass
        # Create a new widget frame
        self.widget_frame = tk.Frame(self.root)
        self.widget_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=3)
        for i in range(12):  # Adjust this number based on the number of columns you have
            self.widget_frame.columnconfigure(i, weight=1)
    
    def open_image(self):
        print("Opening image...")
        # Load image using file dialog
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.tif *.raw")],
                                               initialdir='./images/')
        if file_path:
            self.undo_all()  # Reset all actions before loading a new image
            print(f"Image loaded from: {file_path}")
            if file_path.endswith(".raw"):
                # Load raw image data
                with open(file_path, "rb") as file:
                    raw_data = file.read()
                self.original_image = Image.frombytes('L', (512, 512), raw_data)
            else:
                # Open image and convert to grayscale
                self.original_image = Image.open(file_path).convert("L")
            # Store copies for manipulation and display
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
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("TIFF files", "*.tif")],
                                                 initialdir='./TestImages')
        if file_path:
            self.image.save(file_path)
            print(f"Image saved to: {file_path}")
        else:
            print("Save operation cancelled.")

    def save_state(self):
        # Save the current state of the image to history for undo functionality
        time.sleep(0.05)  # Brief pause to ensure state is correctly saved
        if self.image:
            self.history.append(self.image.copy())
            self.redo_stack.clear()  # Clear the redo stack whenever a new action is performed

    def undo(self):
        print("Undoing last action...")
        # Restore the previous state of the image
        if self.history:
            self.redo_stack.append(self.image.copy())  # Save the current state to redo stack
            self.image = self.history.pop()  # Restore the last saved state
            self.update_display()
        else:
            print("No actions to undo.")

    def redo(self):
        print("Redoing last undone action...")
        # Redo the last undone action
        if self.redo_stack:
            self.history.append(self.image.copy())  # Save the current state to history
            self.image = self.redo_stack.pop()  # Restore the last undone state
            self.update_display()
        else:
            print("No actions to redo.")

    def undo_all(self):
        print("Undoing all actions...")
        # Restore the initial state of the image
        if self.history:
            self.image = self.history[0]  # Restore the initial state
            self.history = []  # Clear the history stack
            self.redo_stack = []  # Clear the redo stack
            self.update_display()
            # Reset other attributes to their default values
            self.zoom_factor = 1
            self.click = 0
            self.method = 'original'
            self.adjusted = False
            del self.dft_img
            print("All actions undone.")
        else:
            print("No actions to undo.")
        self.clean_widget()
        
    def update_display(self):
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
        # Display the image at the calculated coordinates
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.display_image)
        print("Display updated.")
 
    # Zoom image
    def zoom_image(self):
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        self.clean_widget()
        # Row 1: Label to indicate zooming factor adjustment
        zoom_label = tk.Label(self.widget_frame, text="Adjust Zooming Factor:")
        zoom_label.grid(row=0, column=0, columnspan=3, padx=12, pady=3, sticky="w")

        # Label to display the zoom factor dynamically
        self.zoom_value_label = tk.Label(self.widget_frame, text="1.0")  # Initialize with the default value
        self.zoom_value_label.grid(row=0, column=9, padx=12, pady=3, sticky="e")
        
        # Row 1: Slider for zooming factor
        self.zoom_slider = ttk.Scale(self.widget_frame, from_=0.1, to_=10.0, orient=tk.HORIZONTAL, length=600, command=self.apply_zoom)
        self.zoom_slider.set(self.zoom_factor)  # Default zooming factor
        self.zoom_slider.grid(row=0, column=3, columnspan=6, padx=12, pady=3, sticky="we")

    def apply_zoom(self, value):
        # Apply zoom based on the slider value
        self.zoom_value_label.config(text=f"{float(value):.2f}")
        self.zoom_factor = self.zoom_slider.get()
        # Calculate new size based on zoom factor
        new_size = (int(self.original_image.width * self.zoom_factor), int(self.original_image.height * self.zoom_factor))
        
        self.save_state()  # Save the current state before making changes
        self.image = self.adjusted_image.copy().resize(new_size, Image.BILINEAR)  # Resize the image
        self.update_display()  # Update the display with the resized image

    # Smoothing and sharpening functionality
    def filtering(self):
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        
        self.clean_widget()
        method_label = tk.Label(self.widget_frame, text="Filter:")
        method_label.grid(row=0, column=0, columnspan=1, padx=12, pady=3, sticky="w")
        
        # Radio buttons to select between different filtering methods
        self.method_var = tk.StringVar(value=self.method)
        methods = [("Original", "original"),
                   ("3x3 averaging", "33average"),
                   ("7x7 averaging", "77average"),
                   ("3x3 median", "33median"),
                   ("7x7 median", "77median")]
        for idx, (text, method) in enumerate(methods):
            method_button = ttk.Radiobutton(self.widget_frame, text=text, variable=self.method_var, value=method, command=self.apply_filter)
            method_button.grid(row=0, column=2 * (idx+1), columnspan=1, padx=12, pady=3, sticky='w')

        # Button to apply Laplacian filter
        laplacian_button = ttk.Button(self.widget_frame, text="Laplacian", command=self.laplacian)
        laplacian_button.grid(row=0, column=11, columnspan=1, padx=12, pady=3, sticky="e")

    def apply_filter(self):
        # Apply smoothing or sharpening to the image based on user settings
        self.click = 0  # Reset laplacian click count
        method = self.method_var.get()
        img_array = np.array(self.original_image.copy())
        if self.adjusted is True:
            self.adjusted = False
            self.update_display()

        # Apply appropriate filter based on user input
        if method == 'original':
            img_array = img_array  # No filter, use original image
        elif method == "33average":
            kernel = np.ones((3, 3), np.float32) / 9  # 3x3 averaging filter kernel
            img_array = cv2.filter2D(img_array, -1, kernel)
        elif method == "77average":
            kernel = np.ones((7, 7), np.float32) / 49  # 7x7 averaging filter kernel
            img_array = cv2.filter2D(img_array, -1, kernel)
        elif method == "33median":
            img_array = median_filter(img_array, size=3)  # 3x3 median filter
        elif method == "77median":
            img_array = median_filter(img_array, size=7)  # 7x7 median filter
        
        self.save_state()  # Save the current state before making changes
        self.image = Image.fromarray(img_array)  # Convert filtered array back to image
        self.method = method  # Update current method
        self.update_display()  # Update the display with the filtered image

    def laplacian(self):
        # Apply or remove Laplacian filter based on click count
        self.click += 1
        if self.click % 2 == 1:
            # Apply Laplacian filter
            img_array = np.array(self.adjusted_image.copy())
            kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
            img_array = cv2.filter2D(img_array, -1, kernel)
        else:
            # Revert to the original image
            img_array = np.array(self.original_image.copy())
        
        self.save_state()  # Save the current state before making changes
        self.image = Image.fromarray(img_array)  # Convert array back to image
        self.update_display()  # Update the display
        
    def FFT_2D(self):
        # Set up GUI for 2D FFT operations
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        self.clean_widget()

        # Button to display the spectrum of the FFT
        spectrum_button = ttk.Button(self.widget_frame, text="Spectrum image", command=self.spectrum)
        spectrum_button.grid(row=0, column=0, columnspan=4, padx=12, pady=3, sticky="ew")
        
        # Button to display the magnitude-only image
        mag_buttun = ttk.Button(self.widget_frame, text="Magnitude-only image", command=self.mag_image)
        mag_buttun.grid(row=0, column=4, columnspan=4, padx=12, pady=3, sticky="ew")
        
        # Button to display the phase-only image
        phase_button = ttk.Button(self.widget_frame, text="Phasor-only image", command=self.phase_image)
        phase_button.grid(row=0, column=8, columnspan=4, padx=12, pady=3, sticky="ew")

    def spectrum(self):
        # Calculate and display the magnitude spectrum of the FFT
        F_uv = fft2(self.image)  # Perform FFT on the image
        F_uv_shifted = fftshift(F_uv)  # Shift zero frequency to center
        magnitude_spectrum = np.log10(np.abs(F_uv_shifted))  # Calculate log magnitude spectrum
        height, width = np.array(self.original_image).shape
        u = fftshift(fftfreq(width, d=1.0))  # Frequency range along width
        v = fftshift(fftfreq(height, d=1.0))  # Frequency range along height
        
        # Plot the magnitude spectrum
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(magnitude_spectrum, cmap='magma', vmin=0, extent=(u.min(), u.max(), v.min(), v.max()))
        ax.set_title("Spectrum of log(|F(u, v)|)", fontsize=16)
        ax.set_xlabel('u', fontsize=16)
        ax.set_ylabel('v', fontsize=16)
        
        # Add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)  # Adjust 'size' and 'pad' as needed
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label('log(|F(u, v)|)', fontsize=16)
        plt.show()
        
    def mag_image(self):
        # Calculate and display magnitude-only image
        F_uv = fft2(self.image)  # Perform FFT on the image
        F_uv_shifted = fftshift(F_uv)  # Shift zero frequency to center
        magnitude = np.abs(F_uv_shifted)  # Calculate magnitude
        magnitude_only = ifft2(fftshift(magnitude)).real  # Inverse FFT of magnitude to obtain image
        
        # Plot magnitude-only image
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(np.log10(magnitude_only), cmap='magma', vmin=0, vmax=4)
        ax.set_title("Magnitude-only Image", fontsize=16)
        
        # Add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)  # Adjust 'size' and 'pad' as needed
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label('log(Magnitude)', fontsize=16)
        ax.set_axis_off()
        plt.show()
        
    def phase_image(self):
        # Calculate and display phase-only image
        F_uv = fft2(self.image)  # Perform FFT on the image
        F_uv_shifted = fftshift(F_uv)  # Shift zero frequency to center
        phase = np.angle(F_uv_shifted)  # Calculate phase
        phase_only = ifft2(fftshift(np.exp(1j * phase))).real  # Inverse FFT of phase to obtain image
        
        # Plot phase-only image
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(phase_only, cmap='magma', vmin=-0.03, vmax=0.03)
        ax.set_title("Phase-only Image", fontsize=16)
        
        # Add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)  # Adjust 'size' and 'pad' as needed
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label('Phase', fontsize=16)
        ax.set_axis_off()
        plt.show()
    
    def DFT(self):
        # Set up GUI for DFT operations
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            self.open_image()
        self.clean_widget()  

        # Button to multiply the image by a checkerboard pattern
        checkboard_button = ttk.Button(self.widget_frame, text="Multiply checkboard", command=self.checkboard)
        checkboard_button.grid(row=0, column=0, columnspan=3, padx=12, pady=3, sticky="ew")
        
        # Button to apply DFT
        dft_button = ttk.Button(self.widget_frame, text="DFT", command=self.dft)
        dft_button.grid(row=0, column=3, columnspan=3, padx=12, pady=3, sticky="ew")
        
        # Button to apply conjugate operation
        conjugate_button = ttk.Button(self.widget_frame, text="Conjugate", command=self.conjugate)
        conjugate_button.grid(row=0, column=6, columnspan=3, padx=12, pady=3, sticky="ew")
        
        # Button to apply inverse DFT
        idft_button = ttk.Button(self.widget_frame, text="Inverse DFT", command=self.idft)
        idft_button.grid(row=0, column=9, columnspan=3, padx=12, pady=3, sticky="ew")
        
    def checkboard(self):
        # Multiply image by a checkerboard pattern to center the frequency content
        try:
            img_array = np.array(self.dft_img.real)
        except:
            img_array = np.array(self.adjusted_image)
        img_temp = np.zeros(img_array.shape)
        height, width = img_array.shape
        
        # Apply checkerboard pattern
        for x in range(height):
            for y in range(width):
                img_temp[x, y] = img_array[x, y] * np.power(-1, x + y)
        
        self.dft_img = img_temp  # Update image with checkerboard pattern
        self.save_state()  # Save the current state before making changes
        self.image = Image.fromarray(img_temp.astype(np.uint8))
        self.update_display()
    
    def dft(self):
        # Perform Discrete Fourier Transform (DFT)
        F_uv = fft2(np.array(self.dft_img))  # Apply FFT to the image
        self.dft_img = F_uv  # Store the transformed image
        magnitude_spectrum = F_uv

        self.save_state()  # Save the current state before making changes
        self.image = Image.fromarray(magnitude_spectrum.astype(np.uint8))
        self.update_display()
        
    def conjugate(self):
        # Compute the complex conjugate of the DFT
        conjugate_image = np.conjugate(self.dft_img)
        self.dft_img = conjugate_image  # Store the conjugate image
        magnitude_spectrum = conjugate_image
             
        self.save_state()  # Save the current state before making changes
        self.image = Image.fromarray(magnitude_spectrum.astype(np.uint8))
        self.update_display()

    def idft(self):
        # Perform Inverse Discrete Fourier Transform (IDFT)
        inverse_dft_image = ifft2(self.dft_img)  # Apply inverse FFT
        self.dft_img = inverse_dft_image  # Store the inverse transformed image
        magnitude_spectrum = inverse_dft_image
             
        self.save_state()  # Save the current state before making changes
        self.image = Image.fromarray(magnitude_spectrum.astype(np.uint8))
        self.update_display()
        
    
if __name__ == "__main__":
    print("Starting Image Processing Tool...")
    root = tk.Tk()
    app = ImageProcessingTool(root)
    root.mainloop()
    print("Image Processing Tool closed.")
