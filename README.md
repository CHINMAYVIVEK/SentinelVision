# 🛡️ SentinelVision

A high-performance **real-time person and object tracking system** built in **Rust**, using:

* OpenCV (video processing & rendering)
* ONNX Runtime (YOLO inference)
* SORT algorithm (multi-object tracking)

SentinelVision detects objects in video streams, assigns persistent tracking IDs, and renders real-time annotated output with RED bounding boxes.

---

## 🚀 Features

* 🎥 Video input support (file or webcam)
* 🧍 Real-time person/object detection using YOLO (ONNX)
* 🧠 HOG fallback detector (OpenCV-based)
* 🎯 Multi-object tracking using SORT
* 🆔 Persistent tracking IDs across frames
* 🔴 RED bounding box visualization (strict requirement)
* 📊 Real-time labels (ID | class | confidence)
* 📼 Optional video output recording
* ⚡ Optimized for real-time performance (20–30 FPS target)

---

## 📍 Status

🚧 Active development — core pipeline (OpenCV + ONNX Runtime + YOLO + SORT tracking) is being actively implemented and stabilized

---

## 🧠 System Architecture

```text id="arch"
Video Input (Webcam / File)
↓
Frame Capture (OpenCV)
↓
Preprocessing (Resize / Normalize / Letterbox)
↓
ONNX Runtime Inference (YOLO)
↓
Detection Post-processing (Confidence Filtering + NMS)
↓
SORT Tracking (Kalman Filter + IoU + Hungarian Matching)
↓
Track Management (ID lifecycle handling)
↓
Rendering (RED bounding boxes + labels)
↓
Display / Optional Video Output
```

---

## 🛠️ Tech Stack

### Core

* Rust 🦀
* OpenCV (Rust bindings)
* ONNX Runtime (ORT)

### Computer Vision

* YOLOv8 / YOLOv5 (ONNX models)
* HOG-based fallback detector (OpenCV)

### Tracking

* SORT algorithm:

  * Kalman Filter (motion prediction)
  * IoU matching (data association)
  * Hungarian assignment

---

## 📁 Project Structure

```text id="structure"
sentinelvision/
│
├── src/
│   ├── main.rs          # Pipeline orchestration
│   ├── config.rs        # Configuration & constants
│   ├── video.rs         # Video capture & frame handling
│   ├── detector.rs      # YOLO ONNX inference + HOG fallback
│   ├── tracker.rs       # Track lifecycle manager
│   ├── sort.rs          # SORT algorithm implementation
│   ├── bbox.rs          # Geometry + IoU utilities
│   ├── renderer.rs      # Visualization (RED bounding boxes)
│   ├── utils.rs         # Helper utilities
│
├── models/
│   └── yolov8.onnx
│
├── data/
│   └── input.mp4
│
├── output/
│   └── output.mp4
│
├── Cargo.toml
└── README.md
```

---

## 🧠 YOLO Model Setup (YOLOv8 → ONNX)

SentinelVision uses YOLO in **ONNX format** via ONNX Runtime.

---

### 📥 Step 1 — Install Python + Ultralytics

Create virtual environment:

```bash id="py1"
python -m venv venv
source venv/bin/activate   # Linux / macOS
# OR
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash id="py2"
pip install ultralytics
```

---

### 📦 Step 2 — Download YOLOv8 Model

```python id="py3"
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

This automatically downloads pretrained weights.

Recommended models:

* yolov8n.pt → fastest (recommended)
* yolov8s.pt → balanced
* yolov8m.pt → more accurate

---

### 🔁 Step 3 — Export to ONNX

```python id="py4"
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=640
)
```

---

### 📁 Step 4 — Move Model into Project

```bash id="py5"
mv yolov8n.onnx models/yolov8.onnx
```

Final structure:

```text id="py6"
models/
└── yolov8.onnx
```

---

### ⚙️ Step 5 — Verify Input Shape

Expected ONNX input:

```text id="py7"
[1, 3, 640, 640]
```

If different, update preprocessing in:

```text id="py8"
src/detector.rs
```

---

## ▶️ Usage

### Run with video file

```bash id="run1"
cargo run --release -- data/input.mp4
```

### Run with webcam

```bash id="run2"
cargo run --release -- 0
```

---

## 🧍 Detection Format

Each detection:

```text id="det"
[x, y, width, height, confidence, class_id]
```

---

## 🆔 Tracking Output Format

Each tracked object:

```text id="track"
ID: <id> | <class> | <confidence>
```

---

## 🔴 Visualization Rules (STRICT)

All objects MUST be drawn using:

* Color: RED (BGR = 0, 0, 255)
* Thickness: 2–3 px
* Label placed above bounding box
* Must include tracking ID

---

## 🧪 Debug Mode (Recommended)

Enable debug outputs:

* Detection-only view
* Tracking-only view
* Frame-level logging

Optional debug artifacts:

* Saved frames with overlays
* Detection counts per frame
* Track counts per frame

---

## 📈 Performance Goals

* 20–30 FPS real-time processing
* Minimal memory allocations
* Lightweight SORT tracking
* Frame skipping allowed if needed

---

## ⚠️ Known Limitations

* Occlusion may cause ID switches
* CPU-only inference may reduce FPS
* YOLO model quality affects accuracy
* Lighting conditions impact detection reliability

---

## 🔮 Future Improvements

* DeepSORT (appearance embeddings)
* Multi-camera tracking
* GPU acceleration (CUDA/OpenCL)
* REST API for analytics
* Event detection (loitering, intrusion)
* Face recognition module

---

## 🧱 Design Philosophy

SentinelVision is built with:

* Minimal dependencies
* Explicit control over each pipeline stage
* Performance-first architecture
* No unnecessary abstractions
* Production-oriented Rust design

---

## 📦 Dependency Policy

* Prefer `std` first
* Avoid unnecessary crates
* No Python runtime dependency
* No heavy ML frameworks beyond ONNX Runtime
* Keep architecture minimal and explicit

---

## 🤝 Contributing

Rules:

* Do NOT change architecture without discussion
* Do NOT add unnecessary dependencies
* Keep modules minimal and focused
* Maintain Rust idioms and performance focus
* Ensure changes do not break pipeline flow
