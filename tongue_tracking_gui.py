#!/usr/bin/env python3
"""
Simple GUI for 3D Tongue Tip Tracking

This provides a user-friendly interface for:
- Processing video files
- Real-time webcam tracking
- Configuring parameters
- Viewing results
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import os
import sys
from pathlib import Path


class TongueTrackingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Tongue Tip Tracking")
        self.root.geometry("800x600")

        # Variables
        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="./output")
        self.skip_frames = tk.IntVar(value=1)
        self.no_display = tk.BooleanVar(value=False)
        self.export_csv = tk.BooleanVar(value=True)
        self.export_json = tk.BooleanVar(value=True)
        self.export_video = tk.BooleanVar(value=False)
        self.camera_index = tk.IntVar(value=0)

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Video Processing
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Video Processing")
        self.setup_video_tab(video_frame)

        # Tab 2: Webcam
        webcam_frame = ttk.Frame(notebook)
        notebook.add(webcam_frame, text="Webcam (Live)")
        self.setup_webcam_tab(webcam_frame)

        # Tab 3: Settings
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        self.setup_settings_tab(settings_frame)

        # Tab 4: About
        about_frame = ttk.Frame(notebook)
        notebook.add(about_frame, text="About")
        self.setup_about_tab(about_frame)

        # Status bar
        self.status_bar = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_video_tab(self, parent):
        """Setup video processing tab"""
        # Model file selection
        model_frame = ttk.LabelFrame(parent, text="Model File", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)

        ttk.Entry(model_frame, textvariable=self.model_path, width=60).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            model_frame, text="Browse...", command=self.browse_model
        ).pack(side=tk.LEFT)

        # Video file selection
        video_frame = ttk.LabelFrame(parent, text="Input Video", padding=10)
        video_frame.pack(fill='x', padx=10, pady=5)

        ttk.Entry(video_frame, textvariable=self.video_path, width=60).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            video_frame, text="Browse...", command=self.browse_video
        ).pack(side=tk.LEFT)

        # Output directory
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding=10)
        output_frame.pack(fill='x', padx=10, pady=5)

        ttk.Entry(output_frame, textvariable=self.output_dir, width=60).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            output_frame, text="Browse...", command=self.browse_output
        ).pack(side=tk.LEFT)

        # Processing options
        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding=10)
        options_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(options_frame, text="Skip Frames:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        ttk.Spinbox(
            options_frame, from_=1, to=10, textvariable=self.skip_frames, width=10
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(options_frame, text="(1 = process all frames)").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=2
        )

        ttk.Checkbutton(
            options_frame, text="No Display (faster)", variable=self.no_display
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(
            options_frame, text="Export CSV", variable=self.export_csv
        ).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(
            options_frame, text="Export JSON", variable=self.export_json
        ).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(
            options_frame, text="Export Annotated Video", variable=self.export_video
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        # Process button
        ttk.Button(
            parent,
            text="Start Processing",
            command=self.process_video,
            style='Accent.TButton'
        ).pack(pady=20)

        # Output log
        log_frame = ttk.LabelFrame(parent, text="Output Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.video_log = scrolledtext.ScrolledText(
            log_frame, height=10, state='disabled'
        )
        self.video_log.pack(fill='both', expand=True)

    def setup_webcam_tab(self, parent):
        """Setup webcam tab"""
        # Model file
        model_frame = ttk.LabelFrame(parent, text="Model File", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)

        ttk.Entry(model_frame, textvariable=self.model_path, width=60).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            model_frame, text="Browse...", command=self.browse_model
        ).pack(side=tk.LEFT)

        # Camera settings
        camera_frame = ttk.LabelFrame(parent, text="Camera Settings", padding=10)
        camera_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(camera_frame, text="Camera Index:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        ttk.Spinbox(
            camera_frame, from_=0, to=5, textvariable=self.camera_index, width=10
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(camera_frame, text="(0 = default camera)").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=2
        )

        # Instructions
        instructions = """
        Controls:
        - Press 'q' to quit
        - Press 'r' to start/stop recording
        - Press 'c' to clear recorded data

        The webcam window will show real-time tracking with:
        - Face bounding boxes
        - Facial landmarks (68 points)
        - Highlighted mouth position
        - Live FPS counter
        - Recording status
        """

        info_frame = ttk.LabelFrame(parent, text="Instructions", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)

        ttk.Label(info_frame, text=instructions, justify=tk.LEFT).pack(
            anchor=tk.W, padx=5, pady=5
        )

        # Start button
        ttk.Button(
            parent,
            text="Start Webcam Tracking",
            command=self.start_webcam,
            style='Accent.TButton'
        ).pack(pady=20)

    def setup_settings_tab(self, parent):
        """Setup settings tab"""
        settings_text = """
        Default Settings:

        • Model Path: Select the shape_predictor_68_face_landmarks_finetuned.dat file
        • Skip Frames: 1 (process all), increase for faster processing
        • No Display: Enable for headless/batch processing
        • Export Options: Choose CSV, JSON, or annotated video output

        Performance Tips:
        • Use skip_frames=2 for 2x faster processing
        • Enable "No Display" for batch processing
        • Lower video resolution for faster processing
        • Process shorter segments for testing

        Model Download:
        Download the facial landmark model from:
        https://drive.google.com/file/d/1kEOn0SsyToOCGr45UDygxnkDo4uxlWeh/view
        """

        ttk.Label(parent, text=settings_text, justify=tk.LEFT).pack(
            anchor=tk.NW, padx=20, pady=20
        )

    def setup_about_tab(self, parent):
        """Setup about tab"""
        about_text = """
        3D Tongue Tip Tracking

        A novel method for tracking tongue tip movements in 3D for medical applications.
        Uses optical flow and facial landmark detection for accurate tracking.

        Features:
        • Video file processing with data export
        • Real-time webcam tracking
        • Multiple export formats (CSV, JSON, video)
        • Performance optimizations (3-4x faster)
        • Comprehensive error handling

        GitHub: https://github.com/dronefreak/3D-tongue-tip-tracking

        Citation:
        If you use this software in your research, please cite:
        Kumar, Navaneeth. (2019). Optical Flow based Tongue Tip Tracking in 3D.
        GitHub repository: https://github.com/dronefreak/3D-tongue-tip-tracking

        License: MIT
        """

        ttk.Label(parent, text=about_text, justify=tk.LEFT).pack(
            anchor=tk.NW, padx=20, pady=20
        )

    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Shape Predictor Model",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)

    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.avi *.mp4 *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)

    def browse_output(self):
        """Browse for output directory"""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)

    def log_message(self, message, log_widget):
        """Add message to log widget"""
        log_widget.configure(state='normal')
        log_widget.insert(tk.END, message + '\n')
        log_widget.see(tk.END)
        log_widget.configure(state='disabled')

    def process_video(self):
        """Process video file"""
        # Validate inputs
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file")
            return

        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return

        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Model file does not exist")
            return

        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Video file does not exist")
            return

        # Create output directory
        os.makedirs(self.output_dir.get(), exist_ok=True)

        # Build command
        video_name = Path(self.video_path.get()).stem
        cmd = [
            sys.executable, "facial_landmarks_video.py",
            "--shape-predictor", self.model_path.get(),
            "--video", self.video_path.get(),
            "--skip-frames", str(self.skip_frames.get())
        ]

        if self.no_display.get():
            cmd.append("--no-display")

        if self.export_csv.get():
            csv_path = os.path.join(self.output_dir.get(), f"{video_name}.csv")
            cmd.extend(["--export-csv", csv_path])

        if self.export_json.get():
            json_path = os.path.join(self.output_dir.get(), f"{video_name}.json")
            cmd.extend(["--export-json", json_path])

        if self.export_video.get():
            video_path = os.path.join(
                self.output_dir.get(), f"{video_name}_annotated.avi"
            )
            cmd.extend(["--output-video", video_path])

        # Run in thread
        self.status_bar.config(text="Processing...")
        self.video_log.configure(state='normal')
        self.video_log.delete('1.0', tk.END)
        self.video_log.configure(state='disabled')

        thread = threading.Thread(
            target=self.run_command, args=(cmd, self.video_log)
        )
        thread.daemon = True
        thread.start()

    def start_webcam(self):
        """Start webcam tracking"""
        # Validate model
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file")
            return

        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Model file does not exist")
            return

        # Build command
        cmd = [
            sys.executable, "facial_landmarks_webcam.py",
            "--shape-predictor", self.model_path.get(),
            "--camera", str(self.camera_index.get())
        ]

        # Run webcam (blocking)
        self.status_bar.config(text="Webcam running...")
        try:
            subprocess.run(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {e}")
        finally:
            self.status_bar.config(text="Ready")

    def run_command(self, cmd, log_widget):
        """Run command and capture output"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                self.root.after(0, lambda l=line: self.log_message(l.strip(), log_widget))

            process.wait()

            if process.returncode == 0:
                self.root.after(0, lambda: self.status_bar.config(text="Processing complete!"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success", "Processing completed successfully!"
                ))
            else:
                self.root.after(0, lambda: self.status_bar.config(text="Processing failed"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "Processing failed. Check the log for details."
                ))

        except Exception as e:
            self.root.after(0, lambda: self.status_bar.config(text="Error"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))


def main():
    root = tk.Tk()
    app = TongueTrackingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
