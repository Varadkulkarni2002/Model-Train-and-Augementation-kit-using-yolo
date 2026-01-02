import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import re
import yaml
import shutil
import torch
from ultralytics import YOLO



def check_gpu():
    """Checks if a CUDA-enabled GPU is available and returns its name."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return True, f"GPU Detected: {gpu_name}"
    else:
        return False, "No GPU Detected."

def find_latest_run_dir(base_dir="."):
    """Finds the latest 'train' experiment directory within the specified base directory."""
    if not os.path.isdir(base_dir):
        return None

    
    exp_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('train')]
    if not exp_dirs:
        return None

    exp_dirs.sort(key=lambda x: int(re.search(r'(\d*)$', x).group(1) or 0), reverse=True)
    
    return os.path.join(base_dir, exp_dirs[0])



class StreamRedirector:
    """A helper class to redirect stdout to a Tkinter widget."""
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        
        self.widget.after(0, self.widget.insert, tk.END, text)
        self.widget.after(0, self.widget.see, tk.END)

    def flush(self):
        pass

def run_training(params, log_widget):
    """
    Runs the YOLOv8 training process with the given parameters.
    This function is intended to be run in a separate thread.
    """
    original_stdout = sys.stdout
    sys.stdout = StreamRedirector(log_widget)
    
    model_path = None
    try:
        log_widget.after(0, log_widget.delete, '1.0', tk.END)
        print("🚀 Starting YOLOv8 training...\n")
        
        model = YOLO(params['model'])

        
        model.train(
            data=params['data'],
            epochs=params['epochs'],
            batch=params['batch'],
            imgsz=params['imgsz'],
            cache=params['cache'],
            half=params['half'],
            device=params['device'],
            project=params['project'],
            name=params['name']
        )
        
        
        run_dir = find_latest_run_dir(base_dir=params['project'])
        if run_dir:
            model_path = os.path.join(run_dir, 'weights', 'best.pt')
            print(f"\n✅ Training complete. Best model saved at: {model_path}")
        else:
            print(f"\n⚠️ Training finished, but couldn't locate the saved model in '{params['project']}'.")

    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
    finally:
        sys.stdout = original_stdout
        return model_path

def export_model(model_path, export_format):
    """Exports a trained model to ONNX or TorchScript format."""
    if not model_path or not os.path.exists(model_path):
        return f"Error: Model path '{model_path}' not found."

    try:
        model = YOLO(model_path)
        # For TorchScript, the output file extension is .torchscript.pt
        # We'll handle renaming for clarity if needed.
        exported_path = model.export(format=export_format.lower())
        return f"✅ Model successfully exported to: {exported_path}"
    except Exception as e:
        return f"❌ Export failed: {e}"


class TrainingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 Training Dashboard")
        self.geometry("900x750")

        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # --- Variables for UI elements ---
        self.yaml_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.model_var = tk.StringVar(value='yolov8n.pt')
        self.epochs_var = tk.StringVar(value='100')
        self.batch_var = tk.StringVar(value='16')
        self.imgsz_var = tk.StringVar(value='640')
        self.device_var = tk.StringVar(value='CPU')
        self.cache_var = tk.BooleanVar()
        self.half_var = tk.BooleanVar()
        self.trained_model_path = None

        self._create_widgets()
        self._check_gpu_status()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Configuration Frame ---
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, expand=False, pady=5)
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="Data YAML File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.yaml_path_var, state='readonly').grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(config_frame, text="Browse...", command=self._select_yaml).grid(row=0, column=2, padx=5)
        
        ttk.Label(config_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.output_path_var, state='readonly').grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Button(config_frame, text="Browse...", command=self._select_output_folder).grid(row=1, column=2, padx=5)

        ttk.Label(config_frame, text="YOLOv8 Model:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        ttk.Combobox(config_frame, textvariable=self.model_var, values=model_options, state='readonly').grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5)

        # --- Parameters & Optimization ---
        params_frame = ttk.LabelFrame(main_frame, text="Parameters & Performance", padding="10")
        params_frame.pack(fill=tk.X, expand=False, pady=5)
        params_frame.columnconfigure(1, weight=1); params_frame.columnconfigure(3, weight=1); params_frame.columnconfigure(5, weight=1)
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=8).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.batch_var, width=8).grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)

        ttk.Label(params_frame, text="Image Size:").grid(row=0, column=4, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.imgsz_var, width=8).grid(row=0, column=5, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(params_frame, text="Device:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.device_menu = ttk.Combobox(params_frame, textvariable=self.device_var, state='readonly', width=10)
        self.device_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Checkbutton(params_frame, text="Cache Images", variable=self.cache_var).grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Checkbutton(params_frame, text="FP16/Half Precision", variable=self.half_var).grid(row=1, column=3, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.gpu_status_label = ttk.Label(params_frame, text="Checking GPU...")
        self.gpu_status_label.grid(row=1, column=5, padx=5, pady=5, sticky=tk.E)

        # --- Actions Frame ---
        action_frame = ttk.Frame(main_frame, padding="5")
        action_frame.pack(fill=tk.X, expand=False, pady=5)
        self.start_btn = ttk.Button(action_frame, text="Start Training", command=self._start_training, style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_pt_btn = ttk.Button(action_frame, text="Save Model As... (.pt)", state=tk.DISABLED, command=self._save_model_as)
        self.save_pt_btn.pack(side=tk.LEFT, padx=5)
        self.export_onnx_btn = ttk.Button(action_frame, text="Export to ONNX", state=tk.DISABLED, command=lambda: self._export_model_handler('onnx'))
        self.export_onnx_btn.pack(side=tk.LEFT, padx=5)

        # --- Log Frame ---
        log_frame = ttk.LabelFrame(main_frame, text="Live Training Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state=tk.NORMAL, bg="#2E2E2E", fg="white")
        self.log_widget.pack(fill=tk.BOTH, expand=True)
        self.style.configure("Accent.TButton", foreground="white", background="#007ACC", font=('Helvetica', 10, 'bold'))

    def _check_gpu_status(self):
        has_gpu, status_text = check_gpu()
        self.gpu_status_label.config(text=status_text)
        self.device_menu['values'] = ('CPU', 'GPU') if has_gpu else ('CPU',)
        self.device_var.set('CPU')

    def _select_yaml(self):
        path = filedialog.askopenfilename(title="Select YAML File", filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
        if path:
            self.yaml_path_var.set(path)

    def _select_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_path_var.set(path)

    def _start_training(self):
        if not self.yaml_path_var.get() or not os.path.exists(self.yaml_path_var.get()):
            messagebox.showerror("Error", "Please select a valid Data YAML file.")
            return
        if not self.output_path_var.get():
            messagebox.showerror("Error", "Please select an Output Folder.")
            return

        try:
            params = {
                'model': self.model_var.get(),
                'data': self.yaml_path_var.get(),
                'epochs': int(self.epochs_var.get()),
                'batch': int(self.batch_var.get()),
                'imgsz': int(self.imgsz_var.get()),
                'cache': self.cache_var.get(),
                'half': self.half_var.get(),
                'device': '0' if self.device_var.get() == 'GPU' else 'cpu',
                'project': self.output_path_var.get(),
                'name': 'train' # YOLO will auto-increment this (train2, train3, etc.)
            }
        except ValueError:
            messagebox.showerror("Error", "Epochs, Batch Size, and Image Size must be integers.")
            return

        self.start_btn.config(state=tk.DISABLED, text="Training...")
        self.save_pt_btn.config(state=tk.DISABLED)
        self.export_onnx_btn.config(state=tk.DISABLED)

        threading.Thread(target=self._training_thread_worker, args=(params,), daemon=True).start()
        
    def _training_thread_worker(self, params):
        model_path = run_training(params, self.log_widget)
        self.after(0, self._on_training_complete, model_path)

    def _on_training_complete(self, model_path):
        self.start_btn.config(state=tk.NORMAL, text="Start Training")
        self.trained_model_path = model_path
        if self.trained_model_path and os.path.exists(self.trained_model_path):
            self.save_pt_btn.config(state=tk.NORMAL)
            self.export_onnx_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Training Complete", f"Training finished! Best model is at:\n{self.trained_model_path}")
        else:
            messagebox.showerror("Training Failed", "Training did not complete successfully or model path could not be found. Check logs for details.")

    def _save_model_as(self):
        if not self.trained_model_path:
            messagebox.showerror("Error", "No trained model available to save.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Trained Model As...",
            defaultextension=".pt",
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if save_path:
            try:
                shutil.copy(self.trained_model_path, save_path)
                messagebox.showinfo("Success", f"Model saved successfully to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")

    def _export_model_handler(self, export_format):
        if not self.trained_model_path:
            messagebox.showerror("Error", "No trained model available to export.")
            return
        
        # Note: 'torchscript' is a valid format for export
        if export_format.lower() == 'torchscript':
            self.log_widget.insert(tk.END, "\nℹ️ Exporting to TorchScript format...\n")
        
        result = export_model(self.trained_model_path, export_format)
        self.log_widget.insert(tk.END, f"\n{result}\n")
        messagebox.showinfo("Export Status", result)

if __name__ == "__main__":
    app = TrainingApp()
    app.mainloop()