"""
Navigational Accessibility Device — Main Script
=================================================
Hardware : Intel RealSense D435i depth camera
Purpose  : Real-time object detection, distance estimation,
           bump/drop-off detection, and asynchronous audio warnings
           to assist visually-impaired users in navigating safely.

Author   : (your name)
Date     : 2026-02-22

Dependencies (install with pip — see bottom of this file for commands):
    pyrealsense2, opencv-python, numpy, ultralytics, pyttsx3
"""

# ──────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# pyttsx3 — offline text-to-speech engine (no internet required)
import pyttsx3


# ──────────────────────────────────────────────────────────────────────
#  CONFIGURATION / TUNABLES
# ──────────────────────────────────────────────────────────────────────

# ----- RealSense stream settings -----
STREAM_WIDTH  = 424
STREAM_HEIGHT = 240
FRAME_RATE    = 30           # fps

# ----- Object detection -----
YOLO_MODEL_PATH   = "yolov8n.pt"      # Nano model (fastest)
YOLO_CONFIDENCE   = 0.45               # minimum confidence threshold
YOLO_IOU          = 0.50               # NMS IoU threshold

# ----- Distance / danger thresholds -----
DANGER_DISTANCE_M       = 1.5   # metres — warn if object closer than this
CRITICAL_DISTANCE_M     = 0.6   # metres — urgent warning
MAX_VALID_DEPTH_M       = 6.0   # ignore depth values beyond this range
DEPTH_ROI_SHRINK_RATIO  = 0.4   # use central 40 % of bbox for median depth

# ----- Bump / drop-off detection (ground-plane analysis) -----
GROUND_ROI_TOP_RATIO    = 0.60  # analyse bottom 40 % of depth frame (ground)
GROUND_ROI_BOTTOM_RATIO = 1.00
ELEVATION_CHANGE_THRESH = 0.12  # metres — sudden depth jump = bump / drop-off
BUMP_MIN_AREA_RATIO     = 0.005 # fraction of ROI area that must trigger

# ----- Audio warnings -----
AUDIO_COOLDOWN_S = 2.0   # seconds – min gap between repeated warnings


# ──────────────────────────────────────────────────────────────────────
#  ASYNC AUDIO WARNING SYSTEM (non-blocking, runs in its own thread)
# ──────────────────────────────────────────────────────────────────────

class AudioWarningSystem:
    """
    Provides non-blocking spoken warnings via pyttsx3.

    A dedicated daemon thread consumes messages from a queue so the
    main camera loop is never blocked by TTS latency.  A per-category
    cooldown prevents repetitive spam.
    """

    def __init__(self, cooldown: float = AUDIO_COOLDOWN_S):
        self._queue: queue.Queue = queue.Queue()
        self._cooldown = cooldown
        self._last_spoken: dict[str, float] = {}   # category → timestamp

        # Spin up the TTS worker thread (daemon → auto-killed on exit)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---- public API (called from the main loop) ----

    def warn(self, message: str, category: str = "general") -> None:
        """Enqueue a warning if the cooldown for *category* has elapsed."""
        now = time.time()
        last = self._last_spoken.get(category, 0.0)
        if now - last >= self._cooldown:
            self._last_spoken[category] = now
            self._queue.put(message)

    # ---- private worker running in background thread ----

    def _worker(self) -> None:
        """Continuously pull messages off the queue and speak them."""
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)    # words per minute
        engine.setProperty("volume", 1.0)
        while True:
            msg = self._queue.get()        # blocks until a message arrives
            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception as exc:
                # pyttsx3 can occasionally raise if the engine is busy;
                # swallow the error so the thread stays alive.
                print(f"[AudioWarning] TTS error: {exc}")


# ──────────────────────────────────────────────────────────────────────
#  HELPER — median depth inside a bounding box
# ──────────────────────────────────────────────────────────────────────

def median_depth_in_bbox(
    depth_frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    shrink: float = DEPTH_ROI_SHRINK_RATIO,
    max_depth: float = MAX_VALID_DEPTH_M,
) -> float:
    """
    Return the **median** depth (metres) inside the central portion of a
    bounding box.  Using the centre avoids noisy readings at object edges
    where the depth sensor often produces artefacts.

    Parameters
    ----------
    depth_frame : 2-D float array of depths in metres (already scaled).
    x1, y1, x2, y2 : bounding-box corners (pixel coords).
    shrink : fraction of the bbox to keep (centred).
    max_depth : depths beyond this are treated as invalid / background.

    Returns
    -------
    Median depth in metres, or ``float('inf')`` when no valid pixels exist.
    """
    # Shrink the box to its central region
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * shrink / 2
    half_h = (y2 - y1) * shrink / 2

    # Clip to image bounds
    h, w = depth_frame.shape[:2]
    rx1 = max(int(cx - half_w), 0)
    ry1 = max(int(cy - half_h), 0)
    rx2 = min(int(cx + half_w), w)
    ry2 = min(int(cy + half_h), h)

    roi = depth_frame[ry1:ry2, rx1:rx2]
    valid = roi[(roi > 0) & (roi < max_depth)]

    if valid.size == 0:
        return float("inf")
    return float(np.median(valid))


# ──────────────────────────────────────────────────────────────────────
#  HELPER — bump / drop-off detection from raw depth
# ──────────────────────────────────────────────────────────────────────

def detect_elevation_changes(
    depth_image: np.ndarray,
    thresh: float = ELEVATION_CHANGE_THRESH,
    min_area_ratio: float = BUMP_MIN_AREA_RATIO,
) -> list[dict]:
    """
    Detect sudden depth discontinuities (bumps, curbs, potholes) in the
    bottom portion of the depth frame — the region most likely to
    represent the ground in front of the user.

    Algorithm
    ---------
    1. Extract the ground ROI (bottom ~40 % of the depth frame).
    2. Compute the vertical gradient (Sobel-Y) of the depth image.
       Large gradients correspond to sudden rises or drops in the scene.
    3. Threshold the absolute gradient to binary, then find contours.
    4. Filter by area to discard tiny noise blobs.

    Returns
    -------
    A list of dicts, each with keys:
        ``bbox``      – (x, y, w, h) in *full-frame* coordinates
        ``kind``      – ``"bump"`` | ``"drop_off"``
        ``severity``  – absolute mean gradient magnitude in the contour
        ``distance``  – median depth (metres) in that region
    """
    h, w = depth_image.shape[:2]
    y_start = int(h * GROUND_ROI_TOP_RATIO)
    y_end   = int(h * GROUND_ROI_BOTTOM_RATIO)
    roi = depth_image[y_start:y_end, :]

    # Replace zeros / invalid depths with NaN so they don't bias gradient
    roi_clean = roi.astype(np.float32)
    roi_clean[roi_clean <= 0] = np.nan

    # Vertical Sobel gradient (approximates dDepth / dy)
    # We use cv2.Sobel on a gap-filled version (NaN → 0) and then mask.
    roi_filled = np.nan_to_num(roi_clean, nan=0.0)
    grad_y = cv2.Sobel(roi_filled, cv2.CV_32F, 0, 1, ksize=5)

    # Absolute gradient — we care about magnitude, not direction (yet)
    abs_grad = np.abs(grad_y)

    # Normalise for thresholding (0-255 uint8)
    norm = cv2.normalize(abs_grad, None, 0, 255, cv2.NORM_MINMAX)
    binary = (norm > (thresh * 255 / 0.5)).astype(np.uint8) * 255
    # ↑ thresh 0.12 maps to ~61 on the 0-255 scale when max grad is 0.5 m

    # Morphological close to merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    roi_area = roi.shape[0] * roi.shape[1]
    detections: list[dict] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < roi_area * min_area_ratio:
            continue  # too small — noise

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Mean signed gradient inside contour → determines bump vs drop-off
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_grad = cv2.mean(grad_y, mask=mask)[0]

        # Median depth of that patch → how far away the hazard is
        patch_depth = roi_filled[y:y + bh, x:x + bw]
        valid = patch_depth[patch_depth > 0]
        distance = float(np.median(valid)) if valid.size > 0 else float("inf")

        detections.append({
            # Convert bbox back to full-frame coords
            "bbox": (x, y + y_start, bw, bh),
            "kind": "drop_off" if mean_grad > 0 else "bump",
            "severity": float(np.abs(mean_grad)),
            "distance": distance,
        })

    return detections


# ──────────────────────────────────────────────────────────────────────
#  VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────

def draw_object_detection(
    frame: np.ndarray,
    label: str,
    distance: float,
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    """Draw a bounding box + label + distance on the colour frame."""
    if distance < CRITICAL_DISTANCE_M:
        colour = (0, 0, 255)       # red — critical
    elif distance < DANGER_DISTANCE_M:
        colour = (0, 165, 255)     # orange — danger
    else:
        colour = (0, 255, 0)       # green — safe

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    text = f"{label} {distance:.2f}m"
    cv2.putText(frame, text, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def draw_elevation_warning(
    frame: np.ndarray,
    detection: dict,
) -> None:
    """Draw a highlighted rectangle for bump / drop-off hazards."""
    x, y, w, h = detection["bbox"]
    colour = (255, 0, 255) if detection["kind"] == "drop_off" else (255, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    label = f"{detection['kind'].replace('_', ' ').title()} {detection['distance']:.2f}m"
    cv2.putText(frame, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 2)


# ──────────────────────────────────────────────────────────────────────
#  TKINTER GUI APPLICATION
# ──────────────────────────────────────────────────────────────────────

class CameraGUI:
    """
    Tkinter-based GUI that displays the RealSense camera feed with
    real-time YOLO object detection, distance estimation, and
    bump/drop-off warnings overlaid on the video.
    """

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

        # ---- Build the UI ----
        self._build_ui()

    # ──────────────── UI LAYOUT ──────────────────────────────────────

    def _build_ui(self):
        # Style
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
        style.configure("TButton", font=("Segoe UI", 11, "bold"),
                        padding=8)

        # Top bar
        top_frame = ttk.Frame(self.root, style="Dark.TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(top_frame, text="AccessAI — Navigation Assistant",
                  style="Title.TLabel").pack(side=tk.LEFT)

        self._status_label = ttk.Label(top_frame, text="  ● Stopped",
                                       style="Status.TLabel")
        self._status_label.pack(side=tk.RIGHT)

        # Main area: camera feed + side panel
        body = ttk.Frame(self.root, style="Dark.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Camera canvas
        cam_frame = ttk.Frame(body, style="Card.TFrame")
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas_label = tk.Label(cam_frame, bg="#000000",
                                      text="Camera feed will appear here",
                                      fg="#555555",
                                      font=("Segoe UI", 14))
        self._canvas_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Side panel
        side = ttk.Frame(body, style="Dark.TFrame", width=250)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        # Detection log
        ttk.Label(side, text="Detections", style="Title.TLabel").pack(
            anchor=tk.W, pady=(0, 5))

        log_frame = ttk.Frame(side, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(log_frame, bg="#2d2d2d", fg="#00ff88",
                                 font=("Consolas", 9), wrap=tk.WORD,
                                 borderwidth=0, highlightthickness=0,
                                 state=tk.DISABLED)
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Stats row
        stats_frame = ttk.Frame(side, style="Card.TFrame")
        stats_frame.pack(fill=tk.X, pady=(8, 0))

        self._fps_label = ttk.Label(stats_frame, text="FPS: --",
                                    style="Info.TLabel")
        self._fps_label.pack(anchor=tk.W, padx=5, pady=3)

        self._hazard_label = ttk.Label(stats_frame, text="Hazards: 0",
                                       style="Info.TLabel")
        self._hazard_label.pack(anchor=tk.W, padx=5, pady=3)

        self._obj_label = ttk.Label(stats_frame, text="Objects: 0",
                                    style="Info.TLabel")
        self._obj_label.pack(anchor=tk.W, padx=5, pady=3)

        # Buttons
        btn_frame = ttk.Frame(self.root, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self._start_btn = ttk.Button(btn_frame, text="▶  Start Camera",
                                     command=self._start)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._stop_btn = ttk.Button(btn_frame, text="■  Stop",
                                    command=self._stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT)

        # Set a reasonable initial window size
        self.root.geometry("960x580")
        self.root.minsize(800, 500)

    # ──────────────── START / STOP ───────────────────────────────────

    def _start(self):
        """Initialise the camera, model, and audio — then begin the loop."""
        self._status_label.config(text="  ● Initialising…", foreground="#ffaa00")
        self.root.update_idletasks()

        try:
            # Camera
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT,
                                 rs.format.z16, FRAME_RATE)
            config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                                 rs.format.bgr8, FRAME_RATE)
            profile = self._pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)
            self._depth_scale = depth_sensor.get_depth_scale()

            self._align = rs.align(rs.stream.color)

            # YOLO
            if self._model is None:
                self._model = YOLO(YOLO_MODEL_PATH)
                _ = self._model.predict(
                    np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8),
                    verbose=False)

            # Audio
            if self._audio is None:
                self._audio = AudioWarningSystem(cooldown=AUDIO_COOLDOWN_S)

        except RuntimeError as err:
            self._status_label.config(text=f"  ● Error: {err}",
                                      foreground="#ff4444")
            return

        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._status_label.config(text="  ● Running", foreground="#00ff88")

        self._last_time = time.time()
        self._frame_loop()

    def _stop(self):
        """Stop the camera pipeline and reset UI."""
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._status_label.config(text="  ● Stopped", foreground="#aaaaaa")
        self._canvas_label.config(image="", text="Camera feed will appear here")

    # ──────────────── FRAME LOOP (runs via root.after) ───────────────

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
        depth_image = (np.asanyarray(depth_frame.get_data())
                       .astype(np.float32) * self._depth_scale)

        # ---- Object detection (Optimized) ----
        self._frame_count += 1
        
        # Only run YOLO every 5th frame (change 5 to 10 for even more speed)
        # OR if it is the very first frame (so we don't crash)
        if self._frame_count % 5 == 0 or self._last_yolo_results is None:
            # Run the heavy model
            self._last_yolo_results = self._model.predict(
                color_image, 
                conf=YOLO_CONFIDENCE, 
                iou=YOLO_IOU, 
                verbose=False
            )
        
        # Use the cached results for drawing (even on skipped frames)
        results = self._last_yolo_results

        # ---- Object detection ----
        results = self._model.predict(color_image, conf=YOLO_CONFIDENCE,
                                      iou=YOLO_IOU, verbose=False)
        det_lines = []
        obj_count = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = self._model.names[cls_id]
                conf = float(box.conf[0])
                distance = median_depth_in_bbox(depth_image, x1, y1, x2, y2)

                draw_object_detection(color_image, f"{label} {conf:.0%}",
                                      distance, x1, y1, x2, y2)
                obj_count += 1
                det_lines.append(f"{label} {conf:.0%}  —  {distance:.2f} m")

                if distance < CRITICAL_DISTANCE_M:
                    self._audio.warn(
                        f"Warning! {label} very close, {distance:.1f} metres.",
                        category=f"obj_{label}")
                elif distance < DANGER_DISTANCE_M:
                    self._audio.warn(
                        f"Caution, {label} ahead, {distance:.1f} metres.",
                        category=f"obj_{label}")

        # ---- Bump / drop-off detection ----
        hazards = detect_elevation_changes(depth_image)
        for h in hazards:
            draw_elevation_warning(color_image, h)
            if h["distance"] < DANGER_DISTANCE_M:
                kind_str = h["kind"].replace("_", " ")
                self._audio.warn(
                    f"Caution, {kind_str} detected, "
                    f"{h['distance']:.1f} metres ahead.",
                    category=f"elev_{h['kind']}")
            det_lines.append(f"{h['kind'].replace('_',' ').title()}  —  "
                             f"{h['distance']:.2f} m")

        # ---- HUD ----
        cv2.putText(color_image, f"Hazards: {len(hazards)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # ---- FPS ----
        now = time.time()
        fps = 1.0 / max(now - self._last_time, 1e-6)
        self._last_time = now

        # ---- Update GUI ----
        # Convert BGR → RGB → PIL → ImageTk
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Scale to fit the label while keeping aspect ratio
        lw = self._canvas_label.winfo_width()
        lh = self._canvas_label.winfo_height()
        if lw > 10 and lh > 10:
            pil_img.thumbnail((lw, lh), Image.NEAREST)

        imgtk = ImageTk.PhotoImage(image=pil_img)
        self._canvas_label.config(image=imgtk, text="")
        self._canvas_label._photo = imgtk  # prevent GC

        # Stats
        self._fps_label.config(text=f"FPS: {fps:.1f}")
        self._hazard_label.config(text=f"Hazards: {len(hazards)}")
        self._obj_label.config(text=f"Objects: {obj_count}")

        # Detection log
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        if det_lines:
            self._log_text.insert(tk.END, "\n".join(det_lines))
        else:
            self._log_text.insert(tk.END, "No detections")
        self._log_text.config(state=tk.DISABLED)

        # Schedule next frame
        self.root.after(1, self._frame_loop)

    # ──────────────── CLEANUP ────────────────────────────────────────

    def _on_close(self):
        """Handle window close."""
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    CameraGUI(root)
    root.mainloop()


# ──────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()


# ======================================================================
#  PIP INSTALL COMMANDS (run these in your virtual-env BEFORE launching)
# ======================================================================
#
#   pip install pyrealsense2
#   pip install opencv-python
#   pip install numpy
#   pip install ultralytics        # brings in torch + YOLOv8
#   pip install pyttsx3
#
#  Or all at once:
#
#   pip install pyrealsense2 opencv-python numpy ultralytics pyttsx3
#
# ======================================================================
