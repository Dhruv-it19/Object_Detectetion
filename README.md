# 🧠 Real-Time Object Detection using MobileNet SSD

This project performs **real-time object detection** using a pre-trained **MobileNet SSD (Single Shot Multibox Detector)** model with **OpenCV’s DNN module**.  
It can detect 20 different common objects (like person, car, dog, bottle, etc.) directly from your webcam feed.

---

## 🚀 Features

✅ Detects multiple object classes in real time  
✅ Uses pre-trained **MobileNetSSD** model (Caffe framework)  
✅ Displays bounding boxes with class labels and confidence scores  
✅ Adjustable confidence threshold  
✅ Runs efficiently on CPU (no GPU required)  

---

## 🧩 Model Information

The MobileNet SSD model is trained on the **COCO/VOC dataset** and detects the following 20 classes:

```
aeroplane, background, bicycle, bird, boat,
bottle, bus, car, cat, chair, cow, diningtable,
dog, horse, motorbike, person, pottedplant,
sheep, sofa, train, tvmonitor
```

---

## 🗂️ Project Structure

```
📦 Real_Time_Object_Detection
│
├── MobileNetSSD_deploy.prototxt.txt     # Model architecture
├── MobileNetSSD_deploy.caffemodel       # Pre-trained model weights
├── object_detection.py                  # Main detection script
└── README.md                            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Real_Time_Object_Detection.git
cd Real_Time_Object_Detection
```

### 2️⃣ Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# or
source venv/bin/activate     # On Linux/Mac
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:
```bash
pip install opencv-python imutils numpy
```

---

## ▶️ How to Run

1. Ensure your **webcam** is connected.  
2. Run the main script:

```bash
python object_detection.py
```

3. Press **‘q’** to quit the video window.

---

## 📸 Sample Output

Detected objects will appear in a live webcam window, like this:

```
[INFO] Loading model...
[INFO] Starting video stream...
[INFO] Approximate FPS: 25.3
```

Bounding boxes will appear around detected objects with their labels and confidence scores.

---

## ⚡ Performance Tips

- Reduce frame width (`imutils.resize(frame, width=400)`) for faster processing.  
- Increase `confidence_threshold` to filter out weak detections.  
- For GPU acceleration, use `cv2.dnn.readNetFromCaffe()` with CUDA backend (if OpenCV is compiled with CUDA support).

---

## 🧠 Dependencies

| Library | Purpose |
|----------|----------|
| OpenCV | DNN inference, video capture |
| imutils | Stream handling, resizing |
| numpy | Numerical operations |

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.

---

## 👨‍💻 Author

**Dhruvit Loliyaniya**  
💡 _“Building AI-powered intelligent systems for real-world use.”_
