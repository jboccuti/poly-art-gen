from sqlite3 import enable_shared_cache
from tkinter.font import NORMAL
from turtle import width
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from process import process_image
from tkinter import ACTIVE, DISABLED, filedialog, messagebox
from PIL import ImageTk, Image
from pathlib import Path


class Application(ttk.Frame):
    def __init__(self):
        super().__init__()
        self.setup()

    def setup(self):
        self.master.title("PolyArtGen")
        self.pack(fill=tk.BOTH, expand=1)
        self.button_width = 25
        self.file_entry_width = 52
        self.slider_length = 250
        self.thresh_entry_width = 3
        self.preview_size = (600, 600)
        self.selected_file_name = tk.StringVar()
        self.selected_file_name.set("")
        self.output_file_name = tk.StringVar()
        self.output_file_name.set("")
        self.lower_thresh = tk.StringVar()
        self.lower_thresh.set("255")
        self.upper_thresh = tk.StringVar()
        self.upper_thresh.set("255")
        self.auto_threshold = tk.BooleanVar()
        self.auto_threshold.set(False)
        self.processed_img = None

        # Configure window style
        style = ttk.Style()
        style.configure("TLabel", padding=(5, 5, 5, 5))
        style.configure("TButton", padding=(5, 5, 5, 5))

        # Default Image Setup
        d_image = Image.open(Path(__file__).parent.parent / "resources/default.png")
        self.default_image = ImageTk.PhotoImage(d_image)
        self.browsed_image = self.default_image
        self.generated_image = self.default_image

        # Create frames for application and add to outer grid
        options_frame = ttk.Frame(self, style="options_frame.TFrame")
        preview_frame = ttk.Frame(self, style="preview_frame.TFrame")

        options_frame.grid(row=0, column=0)
        preview_frame.grid(row=1, column=0)

        # Create options components and add to inner, options grid
        welcome_label = ttk.Label(
            options_frame,
            text="Welcome to PolyArtGen - Generate interesting polygon art from photos and images.",
            font=(None, 20),
        )
        # Control Buttons
        browse_button = tk.Button(
            options_frame,
            text="1 - Select an image from files",
            command=self.browse_files,
            width=self.button_width,
            bg="#44bd32",
        )
        generate_button = tk.Button(
            options_frame,
            text="2 - Generate",
            command=self.generate,
            width=self.button_width,
            bg="#44bd32",
        )
        download_button = tk.Button(
            options_frame,
            text="3 - Save Output",
            command=self.download,
            width=self.button_width,
            bg="#44bd32",
        )
        reset_button = tk.Button(
            options_frame,
            text="4 - Reset",
            command=self.reset,
            width=self.button_width,
            bg="#7f8fa6",
        )
        # File Location Components
        file_location_frame = ttk.Frame(
            options_frame, style="file_location_frame.TFrame"
        )

        selected_file_label = ttk.Label(file_location_frame, text="Selected File")
        selected_file_field = tk.Entry(
            file_location_frame,
            textvariable=self.selected_file_name,
            width=self.file_entry_width,
            disabledbackground="#FFFFFF",
            disabledforeground="#000000",
        )
        output_file_label = ttk.Label(file_location_frame, text="Output File")
        output_file_field = tk.Entry(
            file_location_frame,
            textvariable=self.output_file_name,
            width=self.file_entry_width,
            disabledbackground="#FFFFFF",
            disabledforeground="#000000",
        )
        # Add the file components to their frame grid
        selected_file_label.grid(row=0, column=0)
        selected_file_field.grid(row=0, column=1)
        output_file_label.grid(row=1, column=0)
        output_file_field.grid(row=1, column=1)

        # Threshold Components
        threshold_frame = ttk.Frame(options_frame, style="threshold_frame.TFrame")

        lower_thresh_label = ttk.Label(threshold_frame, text="Lower Threshold")
        self.lower_thresh_slider = ttk.Scale(
            threshold_frame,
            from_=0,
            to=255,
            value=255,
            length=self.slider_length,
            command=self.slider_changed,
        )
        self.lower_thresh_value_field = tk.Entry(
            threshold_frame,
            textvariable=self.lower_thresh,
            width=self.thresh_entry_width,
            disabledbackground="#FFFFFF",
            disabledforeground="#000000",
        )
        upper_thresh_label = ttk.Label(threshold_frame, text="Upper Threshold")
        self.upper_thresh_slider = ttk.Scale(
            threshold_frame,
            from_=0,
            to=255,
            value=255,
            length=self.slider_length,
            command=self.slider_changed,
        )
        self.upper_thresh_value_field = tk.Entry(
            threshold_frame,
            textvariable=self.upper_thresh,
            width=self.thresh_entry_width,
            disabledbackground="#FFFFFF",
            disabledforeground="#000000",
        )
        self.auto_threshold_checkbox = tk.Checkbutton(
            threshold_frame,
            text="Use Automatic\nCanny Thresholds",
            variable=self.auto_threshold,
            command=self.toggle_sliders,
        )

        # Add the threshold components to their frame grid
        lower_thresh_label.grid(row=0, column=0, padx=(10, 10))
        self.lower_thresh_slider.grid(row=0, column=1, sticky="ew")
        self.lower_thresh_value_field.grid(row=0, column=2, padx=(10, 10))
        upper_thresh_label.grid(row=1, column=0, padx=(10, 10))
        self.upper_thresh_slider.grid(row=1, column=1, sticky="ew")
        self.upper_thresh_value_field.grid(row=1, column=2, padx=(10, 10))
        self.auto_threshold_checkbox.grid(row=0, column=3, rowspan=2, padx=(10, 10))

        # Configure the Options Frame Grid
        welcome_label.grid(row=0, column=0, columnspan=2, padx=(10, 10), pady=(10, 10))

        file_location_frame.grid(row=1, column=0, rowspan=2, padx=(10, 10))
        threshold_frame.grid(row=3, column=0, rowspan=2, padx=(10, 10))

        browse_button.grid(row=1, column=1, padx=(10, 10))
        generate_button.grid(row=2, column=1, padx=(10, 10))
        download_button.grid(row=3, column=1, padx=(10, 10))
        reset_button.grid(row=4, column=1, padx=(10, 10))

        # Create preview components and add to inner, preview grid
        self.browsed_image_label = ttk.Label(preview_frame, image=self.browsed_image)
        self.generated_image_label = ttk.Label(
            preview_frame, image=self.generated_image
        )
        self.browsed_image_label.image = self.browsed_image
        self.generated_image_label.image = self.generated_image

        self.browsed_image_label.grid(row=1, column=0)
        self.generated_image_label.grid(row=1, column=1)

        # Make entries non-editable fields
        selected_file_field.config(state="disabled")
        output_file_field.config(state="disabled")
        self.lower_thresh_value_field.config(state="disabled")
        self.upper_thresh_value_field.config(state="disabled")

    def reset(self):
        self.img = None
        self.processed_img = None
        self.selected_file_name.set("")
        self.output_file_name.set("")
        self.browsed_image_label.configure(image=self.default_image)
        self.browsed_image_label.image = self.default_image
        self.generated_image_label.configure(image=self.default_image)
        self.generated_image_label.image = self.default_image

    def browse_files(self):
        # Create file dialog
        browsed_file_name = filedialog.askopenfilename(
            initialdir="/",
            title="Select a File",
            filetypes=(("Image", "*.png"), ("Image", "*.jpg"), ("Image", "*.jpeg")),
        )

        if browsed_file_name:
            # Update selected file entry with path to file selected
            self.selected_file_name.set(browsed_file_name)

            # Create and resize the image
            img = Image.open(browsed_file_name)
            # Resize to display on the thumbnail
            resized_browsed_img = self.resize_to_square(img)
            self.browsed_image = ImageTk.PhotoImage(resized_browsed_img)

            # Update the selected image label
            self.browsed_image_label.configure(image=self.browsed_image)
            self.browsed_image_label.image = self.browsed_image
        else:
            browsed_file_name = None

    def generate(self):
        if len(self.selected_file_name.get()) > 0:
            # Open the selected image
            img = Image.open(self.selected_file_name.get())
            # Run the selected image through the processing algorithm
            self.processed_img = process_image(
                img,
                int(self.lower_thresh.get()),
                int(self.upper_thresh.get()),
                self.auto_threshold.get(),
            )
            # Resize to display on the thumbnail
            generated_img = self.processed_img
            generated_img = self.resize_to_square(generated_img)
            # Update the generated image label to show the output
            self.generated_image = ImageTk.PhotoImage(generated_img)
            self.generated_image_label.configure(image=self.generated_image)
            self.generated_image_label.image = self.generated_image
        else:
            tk.messagebox.showwarning(
                title="Generate Error", message="Please browse for a file first."
            )

    def download(self):
        if self.processed_img is not None:
            # Create file dialog
            download_loc = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialdir="/",
                title="Select download name",
                filetypes=(("Image", "*.png"),),
            )

            if download_loc:
                self.output_file_name.set(download_loc)
                self.processed_img.save(download_loc)
        else:
            tk.messagebox.showwarning(
                title="Download Error", message="Please generate an image first."
            )

    def resize_to_square(self, image: Image, fill=(0, 0, 0, 0)):
        w, h = image.size
        resize_ratio = min(self.preview_size[0] / w, self.preview_size[0] / h)
        resized_resolution = (int(w * resize_ratio), int(h * resize_ratio))

        return image.resize(resized_resolution, Image.ANTIALIAS)

    def slider_changed(self, event):
        self.lower_thresh.set(int(self.lower_thresh_slider.get()))
        self.upper_thresh.set(int(self.upper_thresh_slider.get()))

    def toggle_sliders(self):
        if self.auto_threshold.get():
            self.lower_thresh_slider.config(state=DISABLED)
            self.upper_thresh_slider.config(state=DISABLED)
        else:
            self.lower_thresh_slider.config(state=NORMAL)
            self.upper_thresh_slider.config(state=NORMAL)


def main():
    root = tk.Tk()
    root.resizable(False, False)
    app = Application()
    root.mainloop()


if __name__ == "__main__":
    main()
