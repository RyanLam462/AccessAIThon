"""
Navigational Accessibility Device
=================================================
Hardware : Intel RealSense D435i depth camera
Purpose  : Real-time object detection, distance estimation,
           bump/drop-off detection, and asynchronous audio warnings
           to assist visually-impaired users in navigating safely.
"""

# ──────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────
import time
import threading
import queue
import subprocess
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# === CONFIGURATION ===

STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FRAME_RATE    = 30

YOLO_MODEL_PATH   = "yolov8n.pt"
YOLO_CONFIDENCE   = 0.45
YOLO_IOU          = 0.50

DANGER_DISTANCE_M       = 1.5
CRITICAL_DISTANCE_M     = 0.6
MAX_VALID_DEPTH_M       = 6.0
DEPTH_ROI_SHRINK_RATIO  = 0.4

GROUND_ROI_TOP_RATIO    = 0.60
GROUND_ROI_BOTTOM_RATIO = 1.00
ELEVATION_CHANGE_THRESH = 0.12
BUMP_MIN_AREA_RATIO     = 0.005

AUDIO_COOLDOWN_S      = 20.0
PERSISTENCE_THRESHOLD = 5
PERSISTENCE_DECAY     = 3
POSITION_TOLERANCE    = 50


# === AUDIO WARNING SYSTEM ===

class AudioWarningSystem:
    """Non-blocking text-to-speech warnings with cooldown per category."""

    def __init__(self, cooldown: float = AUDIO_COOLDOWN_S):
        self._queue: queue.Queue = queue.Queue()
        self._cooldown = cooldown
        self._last_spoken: dict[str, float] = {}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def warn(self, message: str, category: str = "general") -> None:
        now = time.time()
        if now - self._last_spoken.get(category, 0.0) >= self._cooldown:
            self._last_spoken[category] = now
            self._queue.put(message)

    def _worker(self) -> None:
        import subprocess
        while True:
            msg = self._queue.get()
            try:
                cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 2; $s.Speak("{msg}")'
                subprocess.run(["powershell", "-Command", cmd], capture_output=True, timeout=10)
            except Exception as exc:
                print(f"[AudioWarning] TTS error: {exc}")


# === DEPTH HELPERS ===

def median_depth_in_bbox(depth_frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, shrink: float = DEPTH_ROI_SHRINK_RATIO, max_depth: float = MAX_VALID_DEPTH_M) -> float:
    """Get median depth (metres) in the center of a bounding box."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * shrink / 2
    half_h = (y2 - y1) * shrink / 2

    h, w = depth_frame.shape[:2]
    rx1 = max(int(cx - half_w), 0)
    ry1 = max(int(cy - half_h), 0)
    rx2 = min(int(cx + half_w), w)
    ry2 = min(int(cy + half_h), h)

    roi = depth_frame[ry1:ry2, rx1:rx2]
    valid = roi[(roi > 0) & (roi < max_depth)]
    return float(np.median(valid)) if valid.size > 0 else float("inf")


def detect_elevation_changes(depth_image: np.ndarray, thresh: float = ELEVATION_CHANGE_THRESH, min_area_ratio: float = BUMP_MIN_AREA_RATIO) -> list[dict]:
    """Detect bumps and drop-offs in the ground region of the depth frame."""
    h, w = depth_image.shape[:2]
    y_start = int(h * GROUND_ROI_TOP_RATIO)
    y_end = int(h * GROUND_ROI_BOTTOM_RATIO)
    roi = depth_image[y_start:y_end, :]

    roi_clean = roi.astype(np.float32)
    roi_clean[roi_clean <= 0] = np.nan
    roi_filled = np.nan_to_num(roi_clean, nan=0.0)
    
    grad_y = cv2.Sobel(roi_filled, cv2.CV_32F, 0, 1, ksize=5)
    abs_grad = np.abs(grad_y)
    norm = cv2.normalize(abs_grad, None, 0, 255, cv2.NORM_MINMAX)
    binary = (norm > (thresh * 255 / 0.5)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = roi.shape[0] * roi.shape[1]
    detections: list[dict] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < roi_area * min_area_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_grad = cv2.mean(grad_y, mask=mask)[0]

        patch_depth = roi_filled[y:y + bh, x:x + bw]
        valid = patch_depth[patch_depth > 0]
        distance = float(np.median(valid)) if valid.size > 0 else float("inf")

        detections.append({
            "bbox": (x, y + y_start, bw, bh),
            "kind": "drop_off" if mean_grad > 0 else "bump",
            "severity": float(np.abs(mean_grad)),
            "distance": distance,
        })

    return detections


# === DRAWING HELPERS ===

def draw_object_detection(frame: np.ndarray, label: str, distance: float, x1: int, y1: int, x2: int, y2: int) -> None:
    if distance < CRITICAL_DISTANCE_M:
        colour = (0, 0, 255)
    elif distance < DANGER_DISTANCE_M:
        colour = (0, 165, 255)
    else:
        colour = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def draw_elevation_warning(frame: np.ndarray, detection: dict) -> None:
    x, y, w, h = detection["bbox"]
    colour = (255, 0, 255) if detection["kind"] == "drop_off" else (255, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    label = f"{detection['kind'].replace('_', ' ').title()} {detection['distance']:.2f}m"
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 2)


# === DETECTION PERSISTENCE TRACKER ===

class DetectionTracker:
    """Filters out transient detections by requiring persistence across frames."""

    def __init__(self, threshold: int = PERSISTENCE_THRESHOLD, decay: int = PERSISTENCE_DECAY, tolerance: int = POSITION_TOLERANCE):
        self._threshold = threshold
        self._decay = decay
        self._tolerance = tolerance
        self._tracked: dict[tuple, dict] = {}

    def _quantize_pos(self, x: int, y: int) -> tuple[int, int]:
        return (x // self._tolerance, y // self._tolerance)

    def update(self, detections: list[dict]) -> list[dict]:
        seen_keys = set()

        for det in detections:
            qx, qy = self._quantize_pos(det["x"], det["y"])
            key = (qx, qy, det["label"])
            seen_keys.add(key)

            if key in self._tracked:
                self._tracked[key]["count"] = min(self._tracked[key]["count"] + 1, self._threshold + 50)
                self._tracked[key]["data"] = det
            else:
                self._tracked[key] = {"count": 1, "data": det}

        to_remove = []
        for key in self._tracked:
            if key not in seen_keys:
                self._tracked[key]["count"] -= self._decay
                if self._tracked[key]["count"] <= 0:
                    to_remove.append(key)
        for key in to_remove:
            del self._tracked[key]

        return [entry["data"] for entry in self._tracked.values()
                if entry["count"] >= self._threshold]

    def reset(self):
        self._tracked.clear()


# === GUI APPLICATION ===

class CameraGUI:
    """Main application window with camera feed and detection overlay."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AccessAI — Navigation Assistant")
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._running = False
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self._model = None
        self._audio = None
        self._frame_count = 0
        self._last_yolo_results = None
        self._object_tracker = DetectionTracker()
        self._hazard_tracker = DetectionTracker()

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"),
                        background="#1e1e1e", foreground="#ffffff")
        style.configure("Status.TLabel", font=("Segoe UI", 10),
                        background="#1e1e1e", foreground="#aaaaaa")
        style.configure("Info.TLabel", font=("Consolas", 10),
                        background="#2d2d2d", foreground="#00ff88")
        style.configure("Dark.TFrame", background="#1e1e1e")
        style.configure("Card.TFrame", background="#2d2d2d")
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8)

        top_frame = ttk.Frame(self.root, style="Dark.TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(top_frame, text="AccessAI — Navigation Assistant", style="Title.TLabel").pack(side=tk.LEFT)
        self._status_label = ttk.Label(top_frame, text="  ● Stopped", style="Status.TLabel")
        self._status_label.pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, style="Dark.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        cam_frame = ttk.Frame(body, style="Card.TFrame")
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cam_frame.pack_propagate(False)

        self._canvas_label = tk.Label(cam_frame, bg="#000000", text="Camera feed will appear here", fg="#555555", font=("Segoe UI", 14))
        self._canvas_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        side = ttk.Frame(body, style="Dark.TFrame", width=250)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        ttk.Label(side, text="Detections", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 5))
        log_frame = ttk.Frame(side, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(log_frame, bg="#2d2d2d", fg="#00ff88", font=("Consolas", 9), wrap=tk.WORD, borderwidth=0, highlightthickness=0, state=tk.DISABLED)
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        stats_frame = ttk.Frame(side, style="Card.TFrame")
        stats_frame.pack(fill=tk.X, pady=(8, 0))
        self._fps_label = ttk.Label(stats_frame, text="FPS: --", style="Info.TLabel")
        self._fps_label.pack(anchor=tk.W, padx=5, pady=3)
        self._hazard_label = ttk.Label(stats_frame, text="Hazards: 0", style="Info.TLabel")
        self._hazard_label.pack(anchor=tk.W, padx=5, pady=3)
        self._obj_label = ttk.Label(stats_frame, text="Objects: 0", style="Info.TLabel")
        self._obj_label.pack(anchor=tk.W, padx=5, pady=3)

        btn_frame = ttk.Frame(self.root, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self._start_btn = ttk.Button(btn_frame, text="▶  Start Camera", command=self._start)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self._stop_btn = ttk.Button(btn_frame, text="■  Stop", command=self._stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT)

        self.root.geometry("960x580")
        self.root.minsize(800, 500)

    def _start(self):
        self._status_label.config(text="  ● Initialising…", foreground="#ffaa00")
        self.root.update_idletasks()

        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16, FRAME_RATE)
            config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FRAME_RATE)
            profile = self._pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)
            self._depth_scale = depth_sensor.get_depth_scale()
            self._align = rs.align(rs.stream.color)

            if self._model is None:
                self._model = YOLO(YOLO_MODEL_PATH)
                self._model.predict(np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8), verbose=False)

            if self._audio is None:
                self._audio = AudioWarningSystem(cooldown=AUDIO_COOLDOWN_S)

        except RuntimeError as err:
            self._status_label.config(text=f"  ● Error: {err}", foreground="#ff4444")
            return

        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._status_label.config(text="  ● Running", foreground="#00ff88")
        self._last_time = time.time()
        self._frame_loop()

    def _stop(self):
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

        self._object_tracker.reset()
        self._hazard_tracker.reset()
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._status_label.config(text="  ● Stopped", foreground="#aaaaaa")
        self._canvas_label.config(image="", text="Camera feed will appear here")

    def _frame_loop(self):
        if not self._running:
            return

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=200)
        except Exception:
            self.root.after(10, self._frame_loop)
            return

        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            self.root.after(1, self._frame_loop)
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale

        # Run YOLO every 5th frame for performance
        self._frame_count += 1
        if self._frame_count % 5 == 0 or self._last_yolo_results is None:
            self._last_yolo_results = self._model.predict(color_image, conf=YOLO_CONFIDENCE, iou=YOLO_IOU, verbose=False)
        results = self._last_yolo_results

        det_lines = []
        obj_count = 0
        pending_warnings = []

        # Collect and track object detections
        raw_objects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self._model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                distance = median_depth_in_bbox(depth_image, x1, y1, x2, y2)
                raw_objects.append({
                    "x": (x1 + x2) // 2, "y": (y1 + y2) // 2,
                    "label": label, "conf": conf, "distance": distance,
                    "bbox": (x1, y1, x2, y2)
                })

        confirmed_objects = self._object_tracker.update(raw_objects)

        for obj in confirmed_objects:
            x1, y1, x2, y2 = obj["bbox"]
            draw_object_detection(color_image, f"{obj['label']} {obj['conf']:.0%}", obj["distance"], x1, y1, x2, y2)
            obj_count += 1
            det_lines.append(f"{obj['label']} {obj['conf']:.0%}  —  {obj['distance']:.2f} m")

            if obj["distance"] < CRITICAL_DISTANCE_M:
                pending_warnings.append({
                    "distance": obj["distance"], "priority": 0,
                    "message": f"Warning! {obj['label']} very close, {obj['distance']:.1f} metres.",
                    "category": f"obj_{obj['label']}"
                })
            elif obj["distance"] < DANGER_DISTANCE_M:
                pending_warnings.append({
                    "distance": obj["distance"], "priority": 1,
                    "message": f"Caution, {obj['label']} ahead, {obj['distance']:.1f} metres.",
                    "category": f"obj_{obj['label']}"
                })

        # Collect and track hazard detections
        raw_hazards = detect_elevation_changes(depth_image)
        trackable_hazards = []
        for h in raw_hazards:
            x, y, w, h_height = h["bbox"]
            trackable_hazards.append({
                "x": x + w // 2, "y": y + h_height // 2,
                "label": h["kind"], "distance": h["distance"],
                "severity": h["severity"], "bbox": h["bbox"]
            })

        confirmed_hazards = self._hazard_tracker.update(trackable_hazards)

        for h in confirmed_hazards:
            draw_elevation_warning(color_image, {"bbox": h["bbox"], "kind": h["label"], "distance": h["distance"]})
            if h["distance"] < DANGER_DISTANCE_M:
                pending_warnings.append({
                    "distance": h["distance"],
                    "priority": 0 if h["distance"] < CRITICAL_DISTANCE_M else 1,
                    "message": f"Caution, {h['label'].replace('_', ' ')} detected, {h['distance']:.1f} metres ahead.",
                    "category": f"elev_{h['label']}"
                })
            det_lines.append(f"{h['label'].replace('_',' ').title()}  —  {h['distance']:.2f} m")

        # Announce closest hazards
        pending_warnings.sort(key=lambda w: (w["priority"], w["distance"]))
        for warning in pending_warnings[:2]:
            self._audio.warn(warning["message"], category=warning["category"])

        cv2.putText(color_image, f"Hazards: {len(confirmed_hazards)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Update GUI
        now = time.time()
        fps = 1.0 / max(now - self._last_time, 1e-6)
        self._last_time = now

        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        lw, lh = self._canvas_label.winfo_width(), self._canvas_label.winfo_height()
        if lw > 10 and lh > 10:
            pil_img.thumbnail((lw, lh), Image.NEAREST)

        imgtk = ImageTk.PhotoImage(image=pil_img)
        self._canvas_label.config(image=imgtk, text="")
        self._canvas_label._photo = imgtk

        self._fps_label.config(text=f"FPS: {fps:.1f}")
        self._hazard_label.config(text=f"Hazards: {len(confirmed_hazards)}")
        self._obj_label.config(text=f"Objects: {obj_count}")

        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.insert(tk.END, "\n".join(det_lines) if det_lines else "No detections")
        self._log_text.config(state=tk.DISABLED)

        self.root.after(1, self._frame_loop)

    def _on_close(self):
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    CameraGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
