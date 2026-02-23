# AccessAI Navigation Assistant

Real-time navigation assistance for visually-impaired users using an Intel RealSense D435i depth camera and YOLOv8 object detection.

## Features

- **Object Detection**: Identifies people, furniture, vehicles, and other obstacles
- **Distance Estimation**: Uses depth camera to measure how far away objects are
- **Bump/Drop-off Detection**: Warns about curbs, steps, and potholes
- **Voice Warnings**: Speaks alerts for nearby hazards (prioritizes closest dangers)
- **Visual Display**: Shows camera feed with color-coded bounding boxes

## Requirements

- Python 3.9+
- Intel RealSense D435i camera (connected via USB 3.0)
- Windows 10/11

## Installation

1. **Create a virtual environment** (recommended):

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```
   pip install pyrealsense2 opencv-python numpy ultralytics pyttsx3 pillow
   ```

## Running the Application

1. Connect your Intel RealSense D435i camera via USB 3.0

2. Navigate to the project folder:

   ```
   cd AccessAIThon
   ```

3. Run the application:

   ```
   python main.py
   ```

4. Click **"Start Camera"** to begin detection

5. Click **"Stop"** or close the window to exit

## Troubleshooting

**"Error: Couldn't resolve requests"**

- Make sure the camera is connected via USB 3.0 (not 2.0)
- Close any other apps using the camera (Intel RealSense Viewer, etc.)
- Try unplugging and reconnecting the camera

**No audio warnings**

- Check your system volume
- Make sure no other app is blocking audio

**Low FPS**

- The app runs YOLO every 5th frame for performance
- Close other heavy applications
- Ensure you're using the `yolov8n.pt` (nano) model

## Configuration

Edit the constants at the top of `main.py` to adjust:

| Setting                 | Default | Description                             |
| ----------------------- | ------- | --------------------------------------- |
| `DANGER_DISTANCE_M`     | 1.5     | Distance (m) to trigger caution warning |
| `CRITICAL_DISTANCE_M`   | 0.6     | Distance (m) to trigger urgent warning  |
| `AUDIO_COOLDOWN_S`      | 2.0     | Seconds between repeated warnings       |
| `PERSISTENCE_THRESHOLD` | 5       | Frames needed to confirm a detection    |
