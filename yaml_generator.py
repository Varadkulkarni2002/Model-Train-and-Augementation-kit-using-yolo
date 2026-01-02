import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import yaml
from pathlib import Path

class YOLOv8YAMLGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 YAML Generator")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables to store file paths
        self.classes_file_path = tk.StringVar()
        self.labels_folder_path = tk.StringVar()
        self.images_folder_path = tk.StringVar()
        self.output_yaml_path = tk.StringVar()
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="YOLOv8 Dataset YAML Generator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Classes file selection
        ttk.Label(main_frame, text="Classes File (.txt):").grid(row=1, column=0, sticky=tk.W, pady=5)
        classes_entry = ttk.Entry(main_frame, textvariable=self.classes_file_path, width=50)
        classes_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", 
                  command=self.browse_classes_file).grid(row=1, column=2, pady=5)
        
        # Labels folder selection
        ttk.Label(main_frame, text="Labels Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        labels_entry = ttk.Entry(main_frame, textvariable=self.labels_folder_path, width=50)
        labels_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", 
                  command=self.browse_labels_folder).grid(row=2, column=2, pady=5)
        
        # Images folder selection
        ttk.Label(main_frame, text="Images Folder:").grid(row=3, column=0, sticky=tk.W, pady=5)
        images_entry = ttk.Entry(main_frame, textvariable=self.images_folder_path, width=50)
        images_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", 
                  command=self.browse_images_folder).grid(row=3, column=2, pady=5)
        
        # Output YAML file selection
        ttk.Label(main_frame, text="Output YAML File:").grid(row=4, column=0, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(main_frame, textvariable=self.output_yaml_path, width=50)
        output_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", 
                  command=self.browse_output_file).grid(row=4, column=2, pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=5, column=0, columnspan=3, 
                                                           sticky=(tk.W, tk.E), pady=20)
        
        # Dataset split options
        split_frame = ttk.LabelFrame(main_frame, text="Dataset Split Options", padding="10")
        split_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        split_frame.columnconfigure(1, weight=1)
        
        # Train/Val/Test split percentages
        ttk.Label(split_frame, text="Train %:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_percent = tk.StringVar(value="80")
        ttk.Entry(split_frame, textvariable=self.train_percent, width=10).grid(row=0, column=1, 
                                                                               sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(split_frame, text="Validation %:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.val_percent = tk.StringVar(value="15")
        ttk.Entry(split_frame, textvariable=self.val_percent, width=10).grid(row=0, column=3, 
                                                                             sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(split_frame, text="Test %:").grid(row=0, column=4, sticky=tk.W, pady=2, padx=(20, 0))
        self.test_percent = tk.StringVar(value="5")
        ttk.Entry(split_frame, textvariable=self.test_percent, width=10).grid(row=0, column=5, 
                                                                              sticky=tk.W, padx=5, pady=2)
        
        # Auto-split checkbox
        self.auto_split = tk.BooleanVar(value=True)
        ttk.Checkbutton(split_frame, text="Auto-split dataset (creates train/val/test folders)", 
                       variable=self.auto_split).grid(row=1, column=0, columnspan=6, 
                                                      sticky=tk.W, pady=10)
        
        # Generate button
        generate_btn = ttk.Button(main_frame, text="Generate YAML", command=self.generate_yaml,
                                 style="Accent.TButton")
        generate_btn.grid(row=7, column=0, columnspan=3, pady=20)
        
        # Preview text area
        preview_frame = ttk.LabelFrame(main_frame, text="YAML Preview", padding="10")
        preview_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        self.preview_text = tk.Text(preview_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=scrollbar.set)
        
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def browse_classes_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Classes File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.classes_file_path.set(file_path)
            self.update_preview()
    
    def browse_labels_folder(self):
        folder_path = filedialog.askdirectory(title="Select Labels Folder")
        if folder_path:
            self.labels_folder_path.set(folder_path)
            self.update_preview()
    
    def browse_images_folder(self):
        folder_path = filedialog.askdirectory(title="Select Images Folder")
        if folder_path:
            self.images_folder_path.set(folder_path)
            self.update_preview()
    
    def browse_output_file(self):
        file_path = filedialog.asksaveasfilename(
            title="Save YAML File",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")]
        )
        if file_path:
            self.output_yaml_path.set(file_path)
    
    def read_classes(self, classes_file_path):
        """Read class names from the classes file"""
        try:
            with open(classes_file_path, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            return classes
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read classes file: {str(e)}")
            return None
    
    def count_files_in_folder(self, folder_path, extensions):
        """Count files with specific extensions in a folder"""
        if not os.path.exists(folder_path):
            return 0
        
        count = 0
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                count += 1
        return count
    
    def create_dataset_structure(self, base_path, images_folder, labels_folder):
        """Create train/val/test folder structure if auto-split is enabled"""
        if not self.auto_split.get():
            return images_folder, labels_folder, None
        
        try:
            # Create base dataset directory
            dataset_dir = Path(base_path).parent / "yolo_dataset"
            dataset_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            for split in ['train', 'val', 'test']:
                (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
                (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Get split percentages
            train_pct = float(self.train_percent.get()) / 100
            val_pct = float(self.val_percent.get()) / 100
            test_pct = float(self.test_percent.get()) / 100
            
            # Validate percentages
            if abs(train_pct + val_pct + test_pct - 1.0) > 0.01:
                messagebox.showwarning("Warning", "Split percentages don't add up to 100%. Adjusting...")
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(images_folder).glob(f'*{ext}'))
                image_files.extend(Path(images_folder).glob(f'*{ext.upper()}'))
            
            image_files = list(set(image_files))  # Remove duplicates
            total_images = len(image_files)
            
            if total_images == 0:
                messagebox.showerror("Error", "No image files found in the images folder!")
                return None, None, None
            
            # Calculate split indices
            train_count = int(total_images * train_pct)
            val_count = int(total_images * val_pct)
            
            # Split files
            train_files = image_files[:train_count]
            val_files = image_files[train_count:train_count + val_count]
            test_files = image_files[train_count + val_count:]
            
            # Copy files to respective directories
            import shutil
            
            def copy_file_pair(src_img, split_name):
                # Copy image
                dst_img = dataset_dir / split_name / 'images' / src_img.name
                shutil.copy2(src_img, dst_img)
                
                # Copy corresponding label file
                label_name = src_img.stem + '.txt'
                src_label = Path(labels_folder) / label_name
                if src_label.exists():
                    dst_label = dataset_dir / split_name / 'labels' / label_name
                    shutil.copy2(src_label, dst_label)
            
            # Copy train files
            for img_file in train_files:
                copy_file_pair(img_file, 'train')
            
            # Copy val files
            for img_file in val_files:
                copy_file_pair(img_file, 'val')
            
            # Copy test files
            for img_file in test_files:
                copy_file_pair(img_file, 'test')
            
            self.status_var.set(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
            
            return (str(dataset_dir / 'train' / 'images'),
                    str(dataset_dir / 'val' / 'images'),
                    str(dataset_dir / 'test' / 'images') if test_files else None)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset structure: {str(e)}")
            return None, None, None
    
    def generate_yaml(self):
        """Generate the YAML configuration file"""
        # Validate inputs
        if not self.classes_file_path.get():
            messagebox.showerror("Error", "Please select a classes file!")
            return
        
        if not self.labels_folder_path.get():
            messagebox.showerror("Error", "Please select a labels folder!")
            return
        
        if not self.images_folder_path.get():
            messagebox.showerror("Error", "Please select an images folder!")
            return
        
        if not self.output_yaml_path.get():
            messagebox.showerror("Error", "Please specify output YAML file location!")
            return
        
        try:
            # Read classes
            classes = self.read_classes(self.classes_file_path.get())
            if classes is None:
                return
            
            # Handle dataset structure
            if self.auto_split.get():
                train_path, val_path, test_path = self.create_dataset_structure(
                    self.output_yaml_path.get(),
                    self.images_folder_path.get(),
                    self.labels_folder_path.get()
                )
                if train_path is None:
                    return
            else:
                train_path = self.images_folder_path.get()
                val_path = self.images_folder_path.get()
                test_path = None
            
            # Create YAML structure
            yaml_data = {
                'path': str(Path(self.output_yaml_path.get()).parent),
                'train': os.path.relpath(train_path, Path(self.output_yaml_path.get()).parent) if train_path else 'images',
                'val': os.path.relpath(val_path, Path(self.output_yaml_path.get()).parent) if val_path else 'images',
                'nc': len(classes),
                'names': classes
            }
            
            if test_path:
                yaml_data['test'] = os.path.relpath(test_path, Path(self.output_yaml_path.get()).parent)
            
            # Write YAML file
            with open(self.output_yaml_path.get(), 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            self.update_preview()
            self.status_var.set(f"YAML file generated successfully: {self.output_yaml_path.get()}")
            messagebox.showinfo("Success", "YAML file generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate YAML: {str(e)}")
            self.status_var.set("Error generating YAML file")
    
    def update_preview(self):
        """Update the preview text area"""
        if not all([self.classes_file_path.get(), self.labels_folder_path.get(), 
                   self.images_folder_path.get()]):
            return
        
        try:
            classes = self.read_classes(self.classes_file_path.get())
            if classes is None:
                return
            
            # Count files for preview
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_count = self.count_files_in_folder(self.images_folder_path.get(), image_extensions)
            label_count = self.count_files_in_folder(self.labels_folder_path.get(), ['.txt'])
            
            # Create preview YAML
            if self.auto_split.get():
                train_pct = float(self.train_percent.get()) / 100
                val_pct = float(self.val_percent.get()) / 100
                test_pct = float(self.test_percent.get()) / 100
                
                preview_data = f"""# YOLOv8 Dataset Configuration
# Generated by YOLOv8 YAML Generator

path: ./yolo_dataset  # Dataset root directory
train: train/images   # Train images (relative to 'path') - {int(image_count * train_pct)} images
val: val/images       # Validation images (relative to 'path') - {int(image_count * val_pct)} images
test: test/images     # Test images (relative to 'path') - {int(image_count * test_pct)} images

# Number of classes
nc: {len(classes)}

# Class names
names:"""
                for i, class_name in enumerate(classes):
                    preview_data += f"\n  {i}: '{class_name}'"
            else:
                preview_data = f"""# YOLOv8 Dataset Configuration
# Generated by YOLOv8 YAML Generator

path: .  # Dataset root directory
train: {os.path.basename(self.images_folder_path.get())}  # Train images - {image_count} images
val: {os.path.basename(self.images_folder_path.get())}    # Validation images - {image_count} images

# Number of classes
nc: {len(classes)}

# Class names
names:"""
                for i, class_name in enumerate(classes):
                    preview_data += f"\n  {i}: '{class_name}'"
            
            preview_data += f"\n\n# Dataset Statistics:\n# Total Images: {image_count}\n# Total Labels: {label_count}\n# Classes: {len(classes)}"
            
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, preview_data)
            
        except Exception as e:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, f"Error generating preview: {str(e)}")

def main():
    root = tk.Tk()
    app = YOLOv8YAMLGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    main()