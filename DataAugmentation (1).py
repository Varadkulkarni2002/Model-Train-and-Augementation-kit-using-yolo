import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import multiprocessing
import threading
import queue
import time
import xml.etree.ElementTree as ET
from PIL import Image, ImageTk

# ########################################################################### #
#  CORE LOGIC: ROBUST MATH & SAFETY (NO CRASHES)
# ########################################################################### #

class FormatHandler:
    """Handles safe conversion between bbox formats."""
    
    @staticmethod
    def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
        x1 = (x_center - w / 2) * img_w
        y1 = (y_center - h / 2) * img_h
        x2 = (x_center + w / 2) * img_w
        y2 = (y_center + h / 2) * img_h
        return [x1, y1, x2, y2]

    @staticmethod
    def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
        # Safe clipping
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        
        if x2 <= x1 or y2 <= y1: return None # Invalid box
        
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        x_center = (x1 + x2) / (2 * img_w)
        y_center = (y1 + y2) / (2 * img_h)
        return [x_center, y_center, w, h]

    @staticmethod
    def read_xml_annotation(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            boxes = []
            classes = []
            for obj in root.findall('object'):
                boxes.append([
                    float(obj.find('bndbox/xmin').text),
                    float(obj.find('bndbox/ymin').text),
                    float(obj.find('bndbox/xmax').text),
                    float(obj.find('bndbox/ymax').text)
                ])
                classes.append(obj.find('name').text)
            return boxes, classes
        except Exception:
            return [], []

    @staticmethod
    def save_xml_annotation(output_path, boxes, classes, filename, img_w, img_h):
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = filename
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)
        
        for box, cls in zip(boxes, classes):
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(cls)
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(box[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(box[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(box[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(box[3]))
        
        tree = ET.ElementTree(root)
        tree.write(output_path)


class AugmentationEngine:
    """
    Industrial-grade engine. 
    - Supports Probability (p) per transform.
    - Supports Ranges (min/max) per transform.
    - Never crashes (catches internal math errors).
    """

    @staticmethod
    def check_prob(prob_percentage):
        """Returns True if we should apply this augmentation based on probability."""
        return random.random() < (prob_percentage / 100.0)

    @staticmethod
    def safe_warp_affine(img, M, dsize):
        try:
            return cv2.warpAffine(img, M, dsize, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        except:
            return img

    @staticmethod
    def transform_boxes(boxes, matrix, w, h, is_perspective=False):
        new_boxes = []
        if not boxes: return []
        
        try:
            for b in boxes:
                points = np.array([[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]], dtype=np.float32)
                
                if is_perspective:
                    t_points = cv2.perspectiveTransform(points, matrix)
                else:
                    t_points = cv2.transform(points, matrix)
                
                x_coords = t_points[0, :, 0]
                y_coords = t_points[0, :, 1]
                
                x1 = max(0, min(np.min(x_coords), w-1))
                y1 = max(0, min(np.min(y_coords), h-1))
                x2 = max(0, min(np.max(x_coords), w-1))
                y2 = max(0, min(np.max(y_coords), h-1))
                
                if x2 > x1 + 1 and y2 > y1 + 1: # Ignore tiny boxes
                    new_boxes.append([x1, y1, x2, y2])
        except:
            return boxes # Fallback to original boxes if math fails
            
        return new_boxes

    @staticmethod
    def apply(image, bboxes, configs):
        """
        configs: dict containing { 'aug_name': {'enabled': bool, 'prob': int, 'params': dict} }
        """
        if image is None: return None, []
        h, w = image.shape[:2]
        
        # Make a copy to avoid modifying original
        aug_img = image.copy()
        aug_bboxes = list(bboxes)

        # --- Geometric Transforms ---
        
        # 1. Rotation
        c = configs.get('rotate', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            angle = random.uniform(c['params']['min'], c['params']['max'])
            cx, cy = w // 2, h // 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            # Adjust bounds
            abs_cos, abs_sin = abs(M[0,0]), abs(M[0,1])
            nw = int(h * abs_sin + w * abs_cos)
            nh = int(h * abs_cos + w * abs_sin)
            M[0, 2] += nw/2 - cx
            M[1, 2] += nh/2 - cy
            
            aug_img = AugmentationEngine.safe_warp_affine(aug_img, M, (nw, nh))
            aug_bboxes = AugmentationEngine.transform_boxes(aug_bboxes, M, nw, nh)
            h, w = nh, nw # Update dims

        # 2. Shift (Translation)
        c = configs.get('shift', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            limit = c['params']['factor'] # 0.1 etc
            tx = w * random.uniform(-limit, limit)
            ty = h * random.uniform(-limit, limit)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = AugmentationEngine.safe_warp_affine(aug_img, M, (w, h))
            aug_bboxes = AugmentationEngine.transform_boxes(aug_bboxes, M, w, h)

        # 3. Shear
        c = configs.get('shear', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            limit = c['params']['factor']
            sx = random.uniform(-limit, limit)
            sy = random.uniform(-limit, limit)
            M = np.float32([[1, sx, 0], [sy, 1, 0]])
            aug_img = AugmentationEngine.safe_warp_affine(aug_img, M, (w, h))
            aug_bboxes = AugmentationEngine.transform_boxes(aug_bboxes, M, w, h)

        # 4. Zoom
        c = configs.get('zoom', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            scale = random.uniform(c['params']['min'], c['params']['max'])
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            aug_img = AugmentationEngine.safe_warp_affine(aug_img, M, (w, h))
            aug_bboxes = AugmentationEngine.transform_boxes(aug_bboxes, M, w, h)

        # 5. Perspective
        c = configs.get('perspective', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            limit = c['params']['factor']
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            off = min(w, h) * limit
            pts2 = pts1 + np.random.uniform(-off, off, size=pts1.shape).astype(np.float32)
            try:
                M = cv2.getPerspectiveTransform(pts1, pts2)
                aug_img = cv2.warpPerspective(aug_img, M, (w, h))
                aug_bboxes = AugmentationEngine.transform_boxes(aug_bboxes, M, w, h, is_perspective=True)
            except: pass

        # 6. Flips
        c = configs.get('hflip', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            aug_img = cv2.flip(aug_img, 1)
            aug_bboxes = [[w - b[2], b[1], w - b[0], b[3]] for b in aug_bboxes]

        c = configs.get('vflip', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            aug_img = cv2.flip(aug_img, 0)
            aug_bboxes = [[b[0], h - b[3], b[2], h - b[1]] for b in aug_bboxes]

        # --- Pixel Transforms ---

        # Noise
        c = configs.get('noise', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            sigma = random.uniform(c['params']['min'], c['params']['max'])
            noise = np.random.normal(0, sigma, aug_img.shape).astype('uint8')
            aug_img = cv2.add(aug_img, noise)

        # Blur
        c = configs.get('blur', {})
        if c.get('enabled') and AugmentationEngine.check_prob(c['prob']):
            k = random.choice([3, 5, 7])
            aug_img = cv2.GaussianBlur(aug_img, (k, k), 0)

        # Color (Brightness/Contrast/Gamma)
        c_br = configs.get('brightness', {})
        c_cn = configs.get('contrast', {})
        c_gm = configs.get('gamma', {})
        
        if (c_br.get('enabled') or c_cn.get('enabled') or c_gm.get('enabled')):
            img_f = aug_img.astype(np.float32)
            
            if c_br.get('enabled') and AugmentationEngine.check_prob(c_br['prob']):
                factor = random.uniform(c_br['params']['min'], c_br['params']['max'])
                img_f *= factor
                
            if c_cn.get('enabled') and AugmentationEngine.check_prob(c_cn['prob']):
                factor = random.uniform(c_cn['params']['min'], c_cn['params']['max'])
                img_f = (img_f - 127.5) * factor + 127.5

            if c_gm.get('enabled') and AugmentationEngine.check_prob(c_gm['prob']):
                gamma = random.uniform(c_gm['params']['min'], c_gm['params']['max'])
                img_f = np.power(np.maximum(img_f, 0) / 255.0, gamma) * 255.0

            aug_img = np.clip(img_f, 0, 255).astype(np.uint8)

        return aug_img, aug_bboxes




class IndustrialAugmentor:
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial Grade Data Augmentation Tool v3.1")
        self.root.geometry("1280x850")
        
        # Theme
        self.BG = "#1e1e1e"
        self.FG = "#f0f0f0"
        self.ACCENT = "#007acc"
        self.PANEL = "#252526"
        
        self.root.configure(bg=self.BG)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.setup_styles()
        
        self.init_variables()
        self.create_layout()
        
        self.log_queue = queue.Queue()
        self.root.after(200, self.monitor_logs)

    def setup_styles(self):
        self.style.configure("TFrame", background=self.BG)
        self.style.configure("TLabel", background=self.BG, foreground=self.FG, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=self.ACCENT)
        self.style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"), foreground="#aaaaaa")
        
        self.style.configure("TButton", padding=6, background=self.PANEL, foreground=self.FG, borderwidth=0)
        self.style.map("TButton", background=[('active', self.ACCENT)])
        
        self.style.configure("Accent.TButton", background=self.ACCENT, foreground="white", font=("Segoe UI", 10, "bold"))
        self.style.map("Accent.TButton", background=[('active', '#005a9e')])
        
        self.style.configure("TLabelframe", background=self.BG, bordercolor="#444")
        self.style.configure("TLabelframe.Label", background=self.BG, foreground=self.ACCENT)
        
        self.style.configure("TNotebook", background=self.BG, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=self.PANEL, foreground=self.FG, padding=[15, 5])
        self.style.map("TNotebook.Tab", background=[("selected", self.ACCENT)])

    def init_variables(self):
        # Paths
        self.p_img = tk.StringVar()
        self.p_lbl = tk.StringVar()
        self.p_out = tk.StringVar()
        self.format_var = tk.StringVar(value="YOLO")
        self.aug_count = tk.IntVar(value=3)
        
        # Configuration Dictionary (Stores State for all Augmentations)
        # Structure: Key -> {var_enable, var_prob, var_p1, var_p2}
        self.cfg = {
            # Geometric
            "rotate":      {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=50), "min": tk.DoubleVar(value=-15), "max": tk.DoubleVar(value=15)},
            "shift":       {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=30), "val": tk.DoubleVar(value=0.1)},
            "shear":       {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=30), "val": tk.DoubleVar(value=0.1)},
            "zoom":        {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=50), "min": tk.DoubleVar(value=0.8), "max": tk.DoubleVar(value=1.2)},
            "perspective": {"e": tk.BooleanVar(value=False),"p": tk.IntVar(value=20), "val": tk.DoubleVar(value=0.05)},
            "hflip":       {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=50)},
            "vflip":       {"e": tk.BooleanVar(value=False),"p": tk.IntVar(value=50)},
            
            # Pixel
            "noise":       {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=30), "min": tk.DoubleVar(value=5), "max": tk.DoubleVar(value=20)},
            "blur":        {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=20)},
            "brightness":  {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=40), "min": tk.DoubleVar(value=0.7), "max": tk.DoubleVar(value=1.3)},
            "contrast":    {"e": tk.BooleanVar(value=True), "p": tk.IntVar(value=40), "min": tk.DoubleVar(value=0.8), "max": tk.DoubleVar(value=1.2)},
            "gamma":       {"e": tk.BooleanVar(value=False),"p": tk.IntVar(value=20), "min": tk.DoubleVar(value=0.8), "max": tk.DoubleVar(value=1.2)},
        }

    def create_layout(self):
        # Left Panel (Controls)
        left_panel = ttk.Frame(self.root, padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Right Panel (Preview & Logs)
        right_panel = ttk.Frame(self.root, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- LEFT PANEL CONTENT ---
        
        # 1. I/O Section
        io_frame = ttk.LabelFrame(left_panel, text="Dataset Configuration", padding=10)
        io_frame.pack(fill=tk.X, pady=(0, 10))
        self.build_io_row(io_frame, "Images:", self.p_img)
        self.build_io_row(io_frame, "Labels:", self.p_lbl)
        self.build_io_row(io_frame, "Output:", self.p_out)
        
        opts = ttk.Frame(io_frame)
        opts.pack(fill=tk.X, pady=5)
        ttk.Label(opts, text="Format:").pack(side=tk.LEFT)
        ttk.Radiobutton(opts, text="YOLO", variable=self.format_var, value="YOLO").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(opts, text="XML", variable=self.format_var, value="XML").pack(side=tk.LEFT)
        ttk.Label(opts, text="| Copies per Img:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(opts, textvariable=self.aug_count, width=5).pack(side=tk.LEFT)

        # 2. Tabs
        tabs = ttk.Notebook(left_panel, width=450)
        tabs.pack(fill=tk.BOTH, expand=True)
        
        tab_geo = ttk.Frame(tabs, padding=10)
        tab_pix = ttk.Frame(tabs, padding=10)
        tabs.add(tab_geo, text="Geometric")
        tabs.add(tab_pix, text="Color & Noise")
        
        # Populate Geometric
        self.add_control(tab_geo, "Rotation", "rotate", "Range (deg)", has_range=True)
        self.add_control(tab_geo, "Shift/Translate", "shift", "Factor (0-1)", has_val=True)
        self.add_control(tab_geo, "Shear", "shear", "Factor (0-1)", has_val=True)
        self.add_control(tab_geo, "Zoom/Scale", "zoom", "Range (0.8-1.2)", has_range=True)
        self.add_control(tab_geo, "Perspective", "perspective", "Factor", has_val=True)
        self.add_control(tab_geo, "Horiz. Flip", "hflip", "")
        self.add_control(tab_geo, "Vert. Flip", "vflip", "")

        # Populate Pixel
        self.add_control(tab_pix, "Add Noise", "noise", "Sigma Range", has_range=True)
        self.add_control(tab_pix, "Blur", "blur", "")
        self.add_control(tab_pix, "Brightness", "brightness", "Factor Range", has_range=True)
        self.add_control(tab_pix, "Contrast", "contrast", "Factor Range", has_range=True)
        self.add_control(tab_pix, "Gamma", "gamma", "Gamma Range", has_range=True)

        # --- RIGHT PANEL CONTENT ---
        
        # Preview Area
        prev_frame = ttk.LabelFrame(right_panel, text="Live Preview (No Crash)", padding=10)
        prev_frame.pack(fill=tk.BOTH, expand=True)
        
        btn_bar = ttk.Frame(prev_frame)
        btn_bar.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_bar, text="🎲 Generate Random Preview", command=self.generate_preview).pack(side=tk.LEFT)
        
        # Canvas for Image
        self.canvas_frame = tk.Frame(prev_frame, bg="black")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_lbl = tk.Label(self.canvas_frame, bg="black", text="Load directory and click Generate Preview", fg="#666")
        self.preview_lbl.pack(expand=True)

        # Logs & Action
        btm_frame = ttk.Frame(right_panel)
        btm_frame.pack(fill=tk.X, pady=10)
        
        self.log_box = tk.Text(btm_frame, height=6, bg="#111", fg="#0f0", font=("Consolas", 9))
        self.log_box.pack(fill=tk.X, pady=(0, 5))
        
        self.progress = ttk.Progressbar(btm_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_run = ttk.Button(btm_frame, text="START BATCH PROCESSING", style="Accent.TButton", command=self.start_processing)
        self.btn_run.pack(fill=tk.X, ipady=5)

    def build_io_row(self, parent, label, var):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label, width=8).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(f, text="...", width=3, command=lambda: var.set(filedialog.askdirectory())).pack(side=tk.LEFT)

    def add_control(self, parent, display_name, key, param_label, has_range=False, has_val=False):
        """Generates a standardized control row with Enable, Probability, and Parameters."""
        frame = ttk.Frame(parent, style="TFrame")
        frame.pack(fill=tk.X, pady=5)
        
        # Checkbox Enable
        # FIXED: width is a Checkbutton property, not a pack() property
        chk = ttk.Checkbutton(frame, text=display_name, variable=self.cfg[key]["e"], width=15) 
        chk.pack(side=tk.LEFT) 
        
        # Probability Slider
        ttk.Label(frame, text="Prob %:").pack(side=tk.LEFT, padx=(5, 0))
        scl = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                       variable=self.cfg[key]["p"], showvalue=0, length=60, 
                       bg=self.BG, fg=self.FG, highlightthickness=0, troughcolor=self.PANEL)
        scl.pack(side=tk.LEFT, padx=2)
        lbl_p = ttk.Label(frame, textvariable=self.cfg[key]["p"], width=3)
        lbl_p.pack(side=tk.LEFT)

        # Params
        if has_range:
            ttk.Label(frame, text=param_label).pack(side=tk.LEFT, padx=(10, 2))
            ttk.Entry(frame, textvariable=self.cfg[key]["min"], width=4).pack(side=tk.LEFT)
            ttk.Label(frame, text="-").pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=self.cfg[key]["max"], width=4).pack(side=tk.LEFT)
        elif has_val:
            ttk.Label(frame, text=param_label).pack(side=tk.LEFT, padx=(10, 2))
            ttk.Entry(frame, textvariable=self.cfg[key]["val"], width=5).pack(side=tk.LEFT)

    def get_config_dict(self):
        """Extracts current UI values into a clean dict for the worker."""
        d = {}
        for k, v in self.cfg.items():
            d[k] = {
                'enabled': v["e"].get(),
                'prob': v["p"].get(),
                'params': {}
            }
            if "min" in v: d[k]['params']['min'] = v["min"].get()
            if "max" in v: d[k]['params']['max'] = v["max"].get()
            if "val" in v: d[k]['params']['factor'] = v["val"].get()
        return d

    # --- LOGIC & WORKERS ---

    def log(self, msg):
        self.log_queue.put(msg)

    def monitor_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_box.insert(tk.END, f"> {msg}\n")
                self.log_box.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(200, self.monitor_logs)

    def generate_preview(self):
        """Generates preview internally and displays on Tkinter Label (Zero Crash Risk)."""
        try:
            d = Path(self.p_img.get())
            if not d.exists(): return
            files = list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg"))
            if not files: return
            
            # Load Random
            f = random.choice(files)
            img = cv2.imread(str(f))
            if img is None: return
            h, w = img.shape[:2]
            
            # Load Labels (if any)
            bboxes = []
            l_path = Path(self.p_lbl.get()) / f"{f.stem}.{'txt' if self.format_var.get()=='YOLO' else 'xml'}"
            if l_path.exists():
                if self.format_var.get() == "YOLO":
                    with open(l_path) as lf:
                        for line in lf:
                            p = list(map(float, line.strip().split()))
                            bboxes.append(FormatHandler.yolo_to_xyxy(p[1], p[2], p[3], p[4], w, h))
                else:
                    bboxes, _ = FormatHandler.read_xml_annotation(str(l_path))

            # Apply Augmentation
            configs = self.get_config_dict()
            
            # Force probability 100% for enabled items during preview so user sees effect?
            # Or keep random to show reality? Let's keep random but maybe ensure at least one applies.
            aug_img, aug_boxes = AugmentationEngine.apply(img, bboxes, configs)

            # Draw Boxes
            def draw(im, boxes, color=(0, 255, 0)):
                for b in boxes:
                    cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                return im

            img_disp = draw(img.copy(), bboxes, (0, 0, 255)) # Red for original
            aug_disp = draw(aug_img.copy(), aug_boxes, (0, 255, 0)) # Green for aug
            
            # Combine Side-by-Side
            # Resize for display if too big
            max_h = 400
            scale = max_h / h if h > max_h else 1.0
            new_size = (int(w * scale), int(h * scale))
            
            img_disp = cv2.resize(img_disp, new_size)
            aug_disp = cv2.resize(aug_disp, new_size)
            
            combined = np.hstack((img_disp, aug_disp))
            
            # Convert to Tkinter
            b, g, r = cv2.split(combined)
            combined = cv2.merge((r, g, b))
            im_pil = Image.fromarray(combined)
            im_tk = ImageTk.PhotoImage(image=im_pil)
            
            self.preview_lbl.config(image=im_tk, text="")
            self.preview_lbl.image = im_tk # Keep ref
            
        except Exception as e:
            self.log(f"Preview Error: {e}")

    def start_processing(self):
        if not self.p_out.get(): return
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.run_batch, daemon=True).start()

    def run_batch(self):
        try:
            inp = Path(self.p_img.get())
            lbl = Path(self.p_lbl.get())
            out = Path(self.p_out.get())
            (out/"images").mkdir(parents=True, exist_ok=True)
            (out/"labels").mkdir(parents=True, exist_ok=True)
            
            imgs = list(inp.glob("*.jpg")) + list(inp.glob("*.png"))
            self.log(f"Found {len(imgs)} images. Starting...")
            
            cfg = self.get_config_dict()
            count = self.aug_count.get()
            fmt = self.format_var.get()
            
            tasks = []
            for i in imgs:
                l = lbl / f"{i.stem}.{'txt' if fmt == 'YOLO' else 'xml'}"
                tasks.append((i, l, out, count, cfg, fmt))
            
            processed = 0
            with multiprocessing.Pool() as pool:
                for res in pool.imap_unordered(worker_task, tasks):
                    processed += 1
                    if processed % 5 == 0:
                        prog = (processed / len(imgs)) * 100
                        self.progress['value'] = prog
                        self.log(f"Processed {processed}/{len(imgs)}")
            
            self.log("Batch Complete.")
            self.progress['value'] = 100
        except Exception as e:
            self.log(f"Fatal Error: {e}")
        finally:
            self.root.after(0, lambda: self.btn_run.config(state="normal"))

# Separate worker function must be outside class for pickling
def worker_task(args):
    img_path, lbl_path, out_dir, count, cfg, fmt = args
    try:
        img = cv2.imread(str(img_path))
        if img is None: return
        h, w = img.shape[:2]
        
        # Read Labels
        bboxes = []
        classes = []
        
        if lbl_path.exists():
            if fmt == "YOLO":
                with open(lbl_path) as f:
                    for line in f:
                        p = line.strip().split()
                        classes.append(p[0]) # Keep as string/id
                        bboxes.append(FormatHandler.yolo_to_xyxy(float(p[1]), float(p[2]), float(p[3]), float(p[4]), w, h))
            else:
                bboxes, classes = FormatHandler.read_xml_annotation(str(lbl_path))
        
        # Augment N times
        for i in range(count):
            aug_img, aug_boxes = AugmentationEngine.apply(img, bboxes, cfg)
            
            if aug_img is None: continue
            
            h_new, w_new = aug_img.shape[:2]
            name = f"{img_path.stem}_aug_{i}"
            
            # Save Image
            cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), aug_img)
            
            # Save Labels
            if fmt == "YOLO":
                lines = []
                for b, c in zip(aug_boxes, classes):
                    yb = FormatHandler.xyxy_to_yolo(b[0], b[1], b[2], b[3], w_new, h_new)
                    if yb:
                        lines.append(f"{c} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")
                with open(out_dir / "labels" / f"{name}.txt", 'w') as f:
                    f.write("\n".join(lines))
            else:
                FormatHandler.save_xml_annotation(str(out_dir / "labels" / f"{name}.xml"), aug_boxes, classes, f"{name}.jpg", w_new, h_new)
                
    except Exception:
        pass # Fail silently on single file to keep batch running

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = IndustrialAugmentor(root)
    root.mainloop()