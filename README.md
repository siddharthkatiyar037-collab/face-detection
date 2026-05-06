# 👤 Real-Time Face Verification System

A lightweight Python application that uses **DeepFace** and **OpenCV** to perform live face matching via webcam. The system compares the person in the camera feed against a stored `reference.jpg` and provides instant visual feedback.

---

## 🚀 Key Features
*   **Live Comparison:** Uses the **Facenet** model via the DeepFace library for high-accuracy face verification.
*   **Multithreaded Processing:** Verification runs on a separate thread every 30 frames to ensure the video feed remains smooth and lag-free.
*   **On-the-Fly Reference:** Don't have a photo? Press a key to capture your current face as the new reference image.
*   **Visual Alerts:** Displays a bold **"MATCH!"** (Green) or **"NO MATCH!"** (Red) status directly on the video window.

---

## 🛠️ How It Works

1.  **Face Detection:** The script uses a **Haar Cascade Classifier** to first ensure a face is actually present in the frame before attempting verification[cite: 3].
2.  **Image Processing:** Both the live frame and the reference image are resized to $224 \times 224$ pixels to meet the input requirements of the Facenet model[cite: 3].
3.  **Verification:** DeepFace compares the facial embeddings of both images. If the distance between them is below a certain threshold, it returns a "verified" status[cite: 3].
4.  **Thread Safety:** A threading lock is used to safely update the match status without crashing the main video loop[cite: 3].

---

## 💻 Setup & Installation

### Prerequisites
You will need Python installed along with the following libraries:
```bash
pip install opencv-python deepface tf-keras
