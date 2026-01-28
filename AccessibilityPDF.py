import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import fitz  # PyMuPDF

# configuration
class AppConfig:
    def __init__(self):
        # Default Mapping, nothing changes
        self.gesture_actions = {
            "look_right": "next_page",
            "look_left": "prev_page",
            "long_blink": "toggle_fullscreen",
            "look_up": "scroll_up",
            "look_down": "scroll_down",
            "smile": "zoom_in",
            "open_mouth": "zoom_reset",
            "raise_eyebrows": "zoom_out",
            "right_wink": "none",
            "left_wink": "none"
        }
        
        # default sensitivity and refractory period
        self.thresholds = {
            "YAW_THRESHOLD_RIGHT": -30,    # Degrees (Negative = Right)
            "YAW_THRESHOLD_LEFT": 30,      # Degrees (Positive = Left)
            "PITCH_THRESHOLD_UP": -20,     # Degrees (Negative = Up)
            "PITCH_THRESHOLD_DOWN": 20,    # Degrees (Positive = Down)
            
            # duration to trigger
            "LONG_BLINK_DURATION": 1.5,
            "RIGHT_WINK_DURATION": 1.0,
            "LEFT_WINK_DURATION": 1.0,
            "SMILE_DURATION": 0.5,
            "OPEN_MOUTH_DURATION": 1.0,
            "RAISED_EYEBROWS_DURATION": 1.0,
            
            # Refractory Periods/ action potential style
            "ACTION_COOLDOWN": 2.0,             # Time to wait after turning a page
            "CONTINUOUS_ACTION_INTERVAL": 0.15, # Speed for scroll/zoom
            
            #  Sensitivity
            "EYE_AR_THRESH": 0.25,
            "MOUTH_AR_THRESH": 0.6,
            "SMILE_THRESHOLD": 0.02
        }
        
        self.available_actions = [
            "next_page", "prev_page", "zoom_in", "zoom_out", 
            "zoom_reset", "first_page", "last_page", "toggle_fullscreen", 
            "scroll_up", "scroll_down", "none"
        ]
        self.config_path = "accessibility_pdf_config.json"

    def save_config(self):
        data = {"gesture_actions": self.gesture_actions, "thresholds": self.thresholds}
        try:
            with open(self.config_path, 'w') as f: json.dump(data, f, indent=4)
            return True
        except: return False

    def load_config(self):
        if not os.path.exists(self.config_path): return False
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            self.gesture_actions.update(data.get("gesture_actions", {}))
            self.thresholds.update(data.get("thresholds", {}))
            return True
        except: return False

config = AppConfig()

# mediapipe logic
def get_aspect_ratio(p1, p2, p3, p4, p5, p6):
    A = np.linalg.norm(p2 - p6); B = np.linalg.norm(p3 - p5); C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C) if C != 0 else 0

def get_mouth_ratio(top, bottom, left, right):
    v = np.linalg.norm(top - bottom); h = np.linalg.norm(left - right)
    return v / h if h != 0 else 0

def get_head_pose(landmarks, w, h):
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ]) / 4.5
    idx = [1, 152, 33, 263, 61, 291]
    image_points = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in idx], dtype="double")
    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)
    if not success: return None
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))[6]
    return angles[0,0], angles[1,0], angles[2,0]

# --- Professional Configuration UI ---
class ConfigurationUI:
    def __init__(self, master):
        self.master = master
        self.config = config
        self.master.title("AccessibilityPDF by Michael Li - Configuration")
        self.master.geometry("950x750")
        self.config.load_config()
        
        # user interface
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # interface
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#333")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground="#555")
        self.style.configure("TButton", font=("Segoe UI", 10), padding=6)
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background="#0078D7", foreground="white")
        self.style.map("Accent.TButton", background=[("active", "#005a9e")])
        self.style.configure("TLabelframe", background="#f0f0f0", padding=15)
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground="#0078D7", background="#f0f0f0")

        # container
        main_container = ttk.Frame(master, padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)

        #  Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="AccessibilityPDF", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header_frame, text=" | by Michael Li", style="SubHeader.TLabel", foreground="#888").pack(side=tk.LEFT, padx=5, pady=(4,0))

        # Notebook 
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_map = ttk.Frame(notebook, padding=20)
        notebook.add(self.tab_map, text="   Gesture Mappings   ")
        self.setup_mappings_tab()
        
        self.tab_sense = ttk.Frame(notebook, padding=20)
        notebook.add(self.tab_sense, text="   Sensitivities & Timing   ")
        self.setup_sensitivity_tab()
        
        # Footer 
        btn_frame = ttk.Frame(main_container, padding=(0, 20, 0, 0))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Save & Launch Reader", style="Accent.TButton", command=self.launch).pack(side=tk.RIGHT)

    def setup_mappings_tab(self):
        ttk.Label(self.tab_map, text="Map your facial gestures to PDF actions", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 15))
        
        scroll_container = ttk.Frame(self.tab_map)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        # layout for mappings
        left_col = ttk.Frame(scroll_container)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        right_col = ttk.Frame(scroll_container)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.mapping_vars = {}
        gesture_names = {
            "look_right": "Look Right", "look_left": "Look Left", 
            "look_up": "Look Up", "look_down": "Look Down",
            "smile": "Smile", "open_mouth": "Open Mouth",
            "raise_eyebrows": "Raise Eyebrows", "long_blink": "Long Blink (Both)",
            "right_wink": "Right Wink", "left_wink": "Left Wink"
        }

        items = list(gesture_names.items())
        mid = len(items) // 2
        
        def create_row(parent, key, name):
            row_frame = ttk.Frame(parent, padding=(10, 8), relief="flat", style="TFrame")
            row_frame.pack(fill=tk.X, pady=5)
            
            # label
            ttk.Label(row_frame, text=name, width=20, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
            
            #  Decorator
            ttk.Label(row_frame, text="→", foreground="#999").pack(side=tk.LEFT, padx=10)
            
            # dropdown
            var = tk.StringVar(value=self.config.gesture_actions.get(key, "none"))
            self.mapping_vars[key] = var
            
            # 
            display_actions = [a.replace("_", " ").title() for a in self.config.available_actions]
            current_display = self.config.gesture_actions.get(key, "none").replace("_", " ").title()
            
            cb = ttk.Combobox(row_frame, values=display_actions, state="readonly", font=("Segoe UI", 10))
            cb.set(current_display)
            cb.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            # Bind update to variable
            def on_change(event):
                # convert back to snake
                val = cb.get().lower().replace(" ", "_")
                var.set(val)
            cb.bind("<<ComboboxSelected>>", on_change)

        for i, (k, n) in enumerate(items):
            target = left_col if i < mid else right_col
            create_row(target, k, n)

    def setup_sensitivity_tab(self):
        canvas = tk.Canvas(self.tab_sense, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_sense, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="TFrame")
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=canvas.winfo_reqwidth())
        
        # resiziing
        def on_canvas_configure(event):
            canvas.itemconfig(1, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.threshold_vars = {}
        
        groups = {
            "Head Movement Angles": [
                ("YAW_THRESHOLD_RIGHT", "Turn Right Threshold (Neg Deg)", -90, -10),
                ("YAW_THRESHOLD_LEFT", "Turn Left Threshold (Pos Deg)", 10, 90),
                ("PITCH_THRESHOLD_UP", "Look Up Threshold (Neg Deg)", -90, -5),
                ("PITCH_THRESHOLD_DOWN", "Look Down Threshold (Pos Deg)", 5, 90),
            ],
            "Refractory Periods & Speed (Crucial)": [
                ("ACTION_COOLDOWN", "Page Turn Wait Time (Seconds)", 0.5, 5.0),
                ("CONTINUOUS_ACTION_INTERVAL", "Scroll/Zoom Speed (Lower is Faster)", 0.05, 1.0),
            ],
            "Gesture Hold Durations": [
                ("LONG_BLINK_DURATION", "Long Blink Hold Time", 0.5, 4.0),
                ("SMILE_DURATION", "Smile Hold Time", 0.2, 3.0),
                ("OPEN_MOUTH_DURATION", "Open Mouth Hold Time", 0.2, 3.0),
                ("RAISED_EYEBROWS_DURATION", "Eyebrows Hold Time", 0.2, 3.0),
            ],
            "Advanced Detection Sensitivity": [
                ("EYE_AR_THRESH", "Blink Threshold (0.1 - 0.4)", 0.1, 0.5),
                ("SMILE_THRESHOLD", "Smile Threshold (0.01 - 0.1)", 0.01, 0.1),
                ("MOUTH_AR_THRESH", "Mouth Open Threshold (0.3 - 1.0)", 0.3, 1.0),
            ]
        }
        
        for group_name, items in groups.items():
            gf = ttk.LabelFrame(scroll_frame, text=group_name)
            gf.pack(fill=tk.X, padx=5, pady=10)
            
            for key, label, min_v, max_v in items:
                tf = ttk.Frame(gf)
                tf.pack(fill=tk.X, pady=4, padx=5)
                
                ttk.Label(tf, text=label, width=35).pack(side=tk.LEFT)
                
                curr_val = self.config.thresholds.get(key, min_v)
                var = tk.DoubleVar(value=curr_val)
                self.threshold_vars[key] = var
                
                # Value Label
                val_lbl = ttk.Label(tf, text=f"{curr_val:.2f}", width=6, font=("Segoe UI", 10, "bold"), foreground="#0078D7")
                val_lbl.pack(side=tk.RIGHT)
                
                def update_lbl(v, l=val_lbl): l.config(text=f"{float(v):.2f}")
                
                s = ttk.Scale(tf, from_=min_v, to=max_v, variable=var, command=update_lbl)
                s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

    def reset_defaults(self):
        default = AppConfig()
        self.config.thresholds = default.thresholds
        self.config.gesture_actions = default.gesture_actions
        self.master.destroy()
        self.__init__(tk.Tk())

    def launch(self):
        for k, v in self.mapping_vars.items(): self.config.gesture_actions[k] = v.get()
        for k, v in self.threshold_vars.items(): self.config.thresholds[k] = v.get()
        self.config.save_config()
        self.master.destroy()
        root = tk.Tk()
        app = PDFReaderApp(root, self.config)
        root.mainloop()

# integrated pdf reader with professional user interface!
class PDFReaderApp:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.root.title("AccessibilityPDF by Michael Li - Reader")
        self.root.geometry("1200x900")
        
        # dark theme
        self.root.configure(bg="#2E2E2E")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Reader.TFrame", background="#3E3E3E")
        self.style.configure("Reader.TLabel", background="#3E3E3E", foreground="#FFFFFF", font=("Segoe UI", 10))
        self.style.configure("Reader.TButton", background="#555", foreground="white", borderwidth=0)
        self.style.map("Reader.TButton", background=[("active", "#777")])

        self.doc = None; self.page_num = 0; self.zoom = 1.0; self.fullscreen = False
        
        # timing State
        self.last_act_time = 0
        self.last_cont_time = 0
        
        self.gesture_active = {k: False for k in config.gesture_actions}
        self.gesture_start = {k: 0 for k in config.gesture_actions}
        self.gesture_dur = {k: 0 for k in config.gesture_actions}
        
        self.blink_count = 0; self.eyes_closed = False; self.blink_start = 0
        
        self.setup_ui()
        self.mp_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)
        
        self.load_pdf()
        self.process()

    def setup_ui(self):
        # toolbar
        bar = ttk.Frame(self.root, style="Reader.TFrame", padding=10)
        bar.pack(side=tk.TOP, fill=tk.X)
        
        btn_open = ttk.Button(bar, text="📂 Open PDF", style="Reader.TButton", command=self.load_pdf)
        btn_open.pack(side=tk.LEFT, padx=5)
        
        self.lbl_info = ttk.Label(bar, text="No PDF Loaded", style="Reader.TLabel", font=("Segoe UI", 11, "bold"))
        self.lbl_info.pack(side=tk.LEFT, padx=20)
        
        # viewer
        ttk.Label(bar, text="AccessibilityPDF by Michael Li", style="Reader.TLabel", foreground="#888").pack(side=tk.RIGHT)

        # canvas area
        self.canvas_frame = tk.Frame(self.root, bg="#2E2E2E")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#2E2E2E", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # bottom status
        status_bar = ttk.Frame(self.root, style="Reader.TFrame", padding=(10, 5))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.lbl_action = ttk.Label(status_bar, text="Last Action: None", style="Reader.TLabel", font=("Segoe UI", 10, "bold"), foreground="#00A2FF")
        self.lbl_action.pack(side=tk.LEFT)
        
        self.lbl_status = ttk.Label(status_bar, text="System: Ready", style="Reader.TLabel")
        self.lbl_status.pack(side=tk.RIGHT)
        
        # Webcam Overlay 
        self.cam_lbl = tk.Label(self.root, bg="black", borderwidth=2, relief="solid")
        self.cam_lbl.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-50)

    def load_pdf(self):
        f = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if not f: return
        self.doc = fitz.open(f)
        self.page_num = 0
        self.render()
        self.root.focus_force()
        self.lbl_status.config(text="System: Tracking Active")

    def render(self):
        if not self.doc: return
        page = self.doc.load_page(self.page_num)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 100: cw, ch = 800, 600
        self.canvas.create_image(cw//2, ch//2, image=self.tk_img)
        
        self.lbl_info.config(text=f"Page {self.page_num+1} of {len(self.doc)}  |  Zoom: {int(self.zoom*100)}%")

    def do_action(self, act):
        if act == "none": return
        
        # feedback at bottom for actions
        clean_name = act.replace("_", " ").upper()
        self.lbl_action.config(text=f"ACTION: {clean_name}")
        
        if act == "next_page":
            if self.doc and self.page_num < len(self.doc)-1:
                self.page_num += 1; self.render()
        elif act == "prev_page":
            if self.doc and self.page_num > 0:
                self.page_num -= 1; self.render()
        elif act == "first_page": self.page_num = 0; self.render()
        elif act == "last_page": 
            if self.doc: self.page_num = len(self.doc)-1; self.render()
        elif act == "zoom_in": self.zoom += 0.1; self.render()
        elif act == "zoom_out": self.zoom = max(0.2, self.zoom-0.1); self.render()
        elif act == "zoom_reset": self.zoom = 1.0; self.render()
        elif act == "scroll_down": self.canvas.yview_scroll(1, "units")
        elif act == "scroll_up": self.canvas.yview_scroll(-1, "units")
        elif act == "toggle_fullscreen":
            self.fullscreen = not self.fullscreen
            self.root.attributes("-fullscreen", self.fullscreen)
            self.root.after(100, self.render)

    def process(self):
        ret, frame = self.cap.read()
        if not ret: self.root.after(20, self.process); return
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        cur_t = time.time()
        
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0]
            pts = np.array([[int(p.x*w), int(p.y*h)] for p in lms.landmark])
            
            # detection
            state = {"eyes":"open", "mouth":"closed", "eyebrows":"normal"}
            
            # Eyes/Blink
            le = get_aspect_ratio(pts[33], pts[160], pts[158], pts[133], pts[153], pts[144])
            re = get_aspect_ratio(pts[362], pts[385], pts[387], pts[263], pts[373], pts[380])
            if le < self.config.thresholds["EYE_AR_THRESH"] and re < self.config.thresholds["EYE_AR_THRESH"]:
                self.blink_count += 1
                if self.blink_count > 2:
                    state["eyes"] = "closed"
                    if not self.eyes_closed: self.eyes_closed = True; self.blink_start = cur_t
            elif le < self.config.thresholds["EYE_AR_THRESH"]: state["eyes"] = "left_wink"; self.eyes_closed = False
            elif re < self.config.thresholds["EYE_AR_THRESH"]: state["eyes"] = "right_wink"; self.eyes_closed = False
            else: self.blink_count = 0; self.eyes_closed = False; state["eyes"] = "open"

            # Mouth
            m = pts[[13, 14, 61, 291]]
            mar = get_mouth_ratio(m[0], m[1], m[2], m[3])
            smile = ((m[0][1]+m[1][1])/2 - (m[2][1]+m[3][1])/2) / np.linalg.norm(m[2]-m[3])
            if mar > self.config.thresholds["MOUTH_AR_THRESH"]: state["mouth"] = "open"
            elif smile > self.config.thresholds["SMILE_THRESHOLD"]: state["mouth"] = "smile"
            
            # Eyebrows
            brow = (np.linalg.norm(pts[52]-pts[159]) + np.linalg.norm(pts[282]-pts[386]))/2
            eye_w = np.linalg.norm(pts[33]-pts[133])
            if (brow/eye_w) > 0.85: state["eyebrows"] = "raised"
            
            # Pose
            pitch, yaw, roll = get_head_pose(lms, w, h)
            
            # logic 
            def check(g, cond):
                # Update 
                if cond:
                    if not self.gesture_active[g]:
                        self.gesture_active[g] = True
                        self.gesture_start[g] = cur_t
                    self.gesture_dur[g] = cur_t - self.gesture_start[g]
                else:
                    self.gesture_active[g] = False
                    self.gesture_dur[g] = 0
                    return False
                
                # Checking triggers
                act = self.config.gesture_actions.get(g, "none")
                if act == "none": return False
                
                # Determinng action type
                is_continuous_act = act in ["scroll_up", "scroll_down", "zoom_in", "zoom_out"]
                
                if is_continuous_act:
                    # fast
                    if cur_t - self.last_cont_time > self.config.thresholds["CONTINUOUS_ACTION_INTERVAL"]:
                        self.do_action(act)
                        self.last_cont_time = cur_t
                        return True
                else:
                    # cooldown
                    req_dur = self.config.thresholds.get(f"{g.upper()}_DURATION", 0.0)
                    if g == "raise_eyebrows": req_dur = self.config.thresholds.get("RAISED_EYEBROWS_DURATION", 1.0)
                    
                    if self.gesture_dur[g] >= req_dur:
                        if cur_t - self.last_act_time > self.config.thresholds["ACTION_COOLDOWN"]:
                            self.do_action(act)
                            self.last_act_time = cur_t
                            if g == "long_blink": self.eyes_closed = False
                return False

            active = False
            active |= check("look_right", yaw < self.config.thresholds["YAW_THRESHOLD_RIGHT"])
            active |= check("look_left", yaw > self.config.thresholds["YAW_THRESHOLD_LEFT"])
            active |= check("look_up", pitch < self.config.thresholds["PITCH_THRESHOLD_UP"])
            active |= check("look_down", pitch > self.config.thresholds["PITCH_THRESHOLD_DOWN"])
            
            if not active:
                check("long_blink", self.eyes_closed)
                check("smile", state["mouth"] == "smile")
                check("open_mouth", state["mouth"] == "open")
                check("raise_eyebrows", state["eyebrows"] == "raised")
                check("right_wink", state["eyes"] == "right_wink")
                check("left_wink", state["eyes"] == "left_wink")
                
            mp.solutions.drawing_utils.draw_landmarks(rgb, lms, self.mp_mesh.FACEMESH_TESSELATION, None, 
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

        s_frame = cv2.resize(rgb, (240, 180))
        img = ImageTk.PhotoImage(Image.fromarray(s_frame))
        self.cam_lbl.config(image=img)
        self.cam_lbl.image = img
        self.root.after(20, self.process)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigurationUI(root)
    root.mainloop()
