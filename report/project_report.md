# Project Report: Real-Time Face Verification System

**Course:** Computer Vision
**Submission Date:** September 2026

---

## 1. Title

**Real-Time Face Verification System using OpenCV and DeepFace**

---

## 2. Abstract

This project implements a real-time face verification system that continuously captures video from a standard webcam, detects human faces using the Haar Cascade classifier, and verifies the detected face against a pre-stored reference image using the FaceNet deep learning model (accessed via the DeepFace library). The system displays live feedback — "MATCH" or "NO MATCH" — overlaid on the video stream. A multithreaded design ensures that the deep learning verification step does not block the display loop. The project is implemented entirely in Python using OpenCV, DeepFace, and the standard threading library.

---

## 3. Introduction

Face recognition is one of the most widely studied and applied areas in computer vision. It encompasses two distinct tasks:

- **Face identification**: determining *who* a person is among a set of known people
- **Face verification**: determining *whether* two face images belong to the same person

This project focuses on face **verification**. Given a live webcam feed and a stored reference image, the system answers the question: "Is the person currently in front of the camera the same person as in the reference image?"

Face verification has practical applications in access control systems, device unlock mechanisms, and identity authentication systems. This project demonstrates the full pipeline — from raw video capture to a real-time visual decision — using open-source tools.

---

## 4. Problem Statement

Design and implement a real-time face verification system that:

1. Captures live video from a webcam
2. Detects whether a face is present in the current frame
3. If a face is detected, compares it to a stored reference image
4. Displays whether the current face matches the reference person
5. Allows the user to update the reference image at any time

The system must operate in real time — the display must not freeze or stutter due to the verification computation.

---

## 5. Objectives

- Understand and implement a classical face detection algorithm (Haar Cascade)
- Understand how deep learning models generate facial embeddings
- Use a pre-trained FaceNet model to perform face verification
- Design a multithreaded architecture so that the UI and the verification step run concurrently
- Build a self-contained Python application that runs from the command line

---

## 6. Background

### 6.1 Face Detection vs. Face Verification

Face detection is the task of locating faces within an image. It answers: "Where are the faces?" Face verification is a subsequent step that answers: "Do these two faces belong to the same person?"

### 6.2 Haar Cascade Classifiers

Viola and Jones (2001) proposed a real-time object detection framework based on:

1. **Haar features**: Simple rectangular features computed using integral images
2. **AdaBoost**: A boosting algorithm that selects a small subset of the most discriminative features
3. **Cascade of classifiers**: A sequence of increasingly complex classifiers applied in order; non-face regions are rejected quickly at early stages

This approach is fast enough for real-time use and is well-suited to frontal face detection.

### 6.3 Deep Learning for Face Recognition

Modern face recognition systems use deep convolutional neural networks (CNNs) to map face images to compact numerical vectors called **embeddings**. Two images of the same person produce embeddings that are close together in the embedding space; two images of different people produce embeddings that are far apart.

### 6.4 FaceNet

FaceNet (Schroff, Kalenichenko, and Philbin, 2015) trains a deep CNN using a **triplet loss function**. The loss encourages the network to produce embeddings such that:

- The embedding of an **anchor** image is close to embeddings of **positive** images (same person)
- The embedding of an anchor is far from embeddings of **negative** images (different person)

FaceNet produces 128-dimensional embeddings. Verification is performed by computing the Euclidean distance between the embeddings of two images and comparing it to a threshold.

### 6.5 DeepFace Library

DeepFace (Sefik Ilker Serengil, 2020) is an open-source Python library that provides a simple, unified API for face recognition using multiple pre-trained models including FaceNet, VGG-Face, ArcFace, and others. It handles model loading, face alignment, embedding extraction, and distance computation internally.

---

## 7. Technologies Used

| Technology | Version Used | Role |
|---|---|---|
| Python | 3.12 | Programming language |
| OpenCV (`opencv-python`) | 4.11.0 | Webcam capture, image ops, display |
| DeepFace | 0.0.93 | Face verification API |
| FaceNet | (via DeepFace) | Facial embedding model |
| TensorFlow / tf-keras | 2.19.0 | Deep learning backend for FaceNet |
| NumPy | 2.2.6 | Array and image data manipulation |
| threading | (stdlib) | Concurrent verification |

---

## 8. System Requirements

### Hardware
- Computer with a built-in or USB webcam
- Minimum 4 GB RAM (8 GB recommended for comfortable TensorFlow operation)
- Internet connection (required once to download FaceNet model weights, ~90 MB)

### Software
- Python 3.10, 3.11, or 3.12 (Python 3.14+ is not yet supported by TensorFlow)
- Dependencies listed in `requirements.txt`

---

## 9. Methodology

The system follows a producer-consumer model:

1. **Producer (main thread)**: Continuously reads frames from the webcam and displays them at the native frame rate. Every 30th frame, if no verification thread is already running, a new background thread is spawned.

2. **Consumer (background thread)**: Receives a copy of the current frame. Runs Haar Cascade to confirm a face is present. If yes, resizes the frame and the reference image to 224×224, calls `DeepFace.verify`, and updates a shared `face_match` flag.

3. **Shared state**: The `face_match` flag and access to it are protected by a `threading.Lock` to prevent race conditions between the main thread (reading the flag) and the background thread (writing it).

---

## 10. System Architecture

```
┌─────────────────────────────────────────┐
│  main.py (entry point)                  │
│  ├─ Dependency check                    │
│  └─ imports face_detection → run()      │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  face_detection.py                      │
│                                         │
│  ┌──────────────────────────────┐       │
│  │ Main Thread (display loop)   │       │
│  │  cv2.VideoCapture (webcam)   │       │
│  │  → every 30th frame:         │       │
│  │    spawn Thread(check_face)  │       │
│  │  draw bounding boxes         │       │
│  │  overlay MATCH / NO MATCH    │       │
│  │  cv2.imshow()                │       │
│  └──────────────────────────────┘       │
│                                         │
│  ┌──────────────────────────────┐       │
│  │ Background Thread            │       │
│  │  Haar Cascade detection      │       │
│  │  → if face found:            │       │
│  │    resize to 224×224         │       │
│  │    DeepFace.verify(FaceNet)  │       │
│  │    update face_match (lock)  │       │
│  └──────────────────────────────┘       │
└─────────────────────────────────────────┘
```

---

## 11. Computer Vision Techniques

### Face Detection: Haar Cascade

The project uses OpenCV's `haarcascade_frontalface_default.xml`. Detection parameters:

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)
```

- `scaleFactor=1.1`: The input image is downscaled by 10% at each pyramid level
- `minNeighbors=5`: At least 5 overlapping detections required to accept a region
- `minSize=(50, 50)`: Faces smaller than 50×50 pixels are ignored

### Image Preprocessing

Before passing images to FaceNet, both the live frame and the reference image are resized to 224×224 pixels:

```python
frame_resized = cv2.resize(frame, (224, 224))
reference_resized = cv2.resize(reference, (224, 224))
```

This ensures consistent input dimensions for the model.

---

## 12. Face Detection

The Haar Cascade is applied to a grayscale version of each frame. This is computationally cheaper than operating on the full colour image.

If `detectMultiScale` returns zero rectangles, the system immediately sets `face_match = False` and skips the DeepFace call. This avoids the overhead of running the neural network when there is clearly no face in the frame.

Detected face bounding boxes are drawn on the live display frame in green (MATCH) or red (NO MATCH).

---

## 13. Facial Representation and Verification

The core verification step is:

```python
result = DeepFace.verify(
    img1_path=frame_resized,
    img2_path=reference_resized,
    model_name="Facenet",
    enforce_detection=False
)
face_match = result["verified"]
```

`enforce_detection=False` means DeepFace will not raise an error if its internal face detector fails — the already-resized 224×224 image is used directly. The `result["verified"]` field is a boolean set by DeepFace based on whether the Euclidean distance between the two FaceNet embeddings is below the model's built-in threshold.

---

## 14. Implementation

### Key Code: Main Capture Loop

```python
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if reference_img is not None and counter % 30 == 0 and threading.active_count() < 3:
        threading.Thread(
            target=check_face,
            args=(frame.copy(), reference_img.copy()),
            daemon=True
        ).start()

    counter += 1
    frame = draw_face_boxes(frame)

    label = "MATCH!" if face_match else "NO MATCH"
    color = (0, 255, 0) if face_match else (0, 0, 255)
    cv2.putText(frame, label, (20, 465), cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 2)
    cv2.imshow("Face Verification", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite(REFERENCE_PATH, frame)
        reference_img = frame.copy()
```

### Key Code: Verification Thread

```python
def check_face(frame, reference):
    global face_match
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            with lock:
                face_match = False
            return

        frame_resized = cv2.resize(frame, (224, 224))
        reference_resized = cv2.resize(reference, (224, 224))

        result = DeepFace.verify(
            img1_path=frame_resized,
            img2_path=reference_resized,
            model_name="Facenet",
            enforce_detection=False
        )
        with lock:
            face_match = result["verified"]
    except Exception as e:
        print(f"[Verification error] {e}")
        with lock:
            face_match = False
```

---

## 15. Multithreading

The main design challenge is that `DeepFace.verify` takes approximately 0.5–2 seconds per call (depending on hardware). Running it on the main thread would make the video feed freeze for that duration, which is unacceptable for a real-time display.

The solution is to spawn a **daemon thread** for each verification call:

- `daemon=True` ensures the thread is automatically killed when the main thread exits
- `threading.active_count() < 3` prevents spawning new threads if a previous verification is still running
- `frame.copy()` and `reference_img.copy()` ensure the thread works on its own copy of the image data, avoiding data races on the pixel buffer
- `threading.Lock` protects the `face_match` flag during read and write

The result is that the video feed runs at a smooth frame rate while verification happens asynchronously in the background.

---

## 16. Experimental Setup

The system was tested on a Windows machine with a standard built-in laptop webcam in a normal indoor environment. Testing was performed with:

- A front-facing reference photo captured via the `s` key
- The author's face presented to the webcam for MATCH testing
- A different person's face and obstructed views for NO MATCH testing

No formal quantitative evaluation was performed. A quantitative evaluation would require a labelled dataset of webcam pairs with known ground-truth match/no-match labels, which was outside the scope of this project.

---

## 17. Results

The system functioned as intended during informal testing:

- The webcam capture loop ran smoothly without visible lag
- The Haar Cascade correctly detected a frontal face under normal indoor lighting
- After the FaceNet model was loaded (approximately 3–5 seconds on first run), verification results appeared within 1–2 seconds of a face being shown
- MATCH and NO MATCH labels updated correctly when the face was changed or removed
- The reference capture (`s` key) worked correctly and took effect immediately in the next verification cycle

No accuracy percentages are reported because no formal dataset or metric evaluation was performed.

---

## 18. Limitations

1. **Lighting sensitivity**: Both Haar Cascade detection and FaceNet performance degrade in poor or uneven lighting
2. **Pose sensitivity**: Haar Cascade is designed for frontal faces and fails on significant head tilts or profile views
3. **Verification lag**: The `face_match` label updates every ~30 frames; there is a brief lag between a face change and the display updating
4. **No liveness detection**: A printed photograph or a face on a phone screen could potentially trigger a MATCH
5. **Model download dependency**: FaceNet weights must be downloaded from the internet on first run
6. **Python version constraint**: TensorFlow (required by DeepFace/FaceNet) does not support Python 3.14+ as of this writing

---

## 19. Applications

- **Access control systems**: Verifying that the person attempting to enter a room or unlock a device is an authorized user
- **Attendance systems**: Automatically marking attendance when a known face is detected
- **Online exam proctoring**: Continuously verifying that the exam taker matches the registered student
- **Parental controls**: Verifying a parent's face before allowing certain actions

---

## 20. Future Scope

- **Better face detector**: Replace Haar Cascade with MTCNN or RetinaFace for robust detection under varied conditions
- **Liveness detection**: Add blink detection or depth sensing to prevent spoofing with photographs
- **Multiple reference images**: Allow multiple photos of the same person to improve verification accuracy across different lighting and pose conditions
- **Adaptive threshold**: Allow the DeepFace distance threshold to be configured via a config file
- **Logging**: Save timestamped match/no-match events to a log file
- **Edge deployment**: Optimize the pipeline for deployment on Raspberry Pi or Jetson Nano

---

## 21. Conclusion

This project demonstrates a functional real-time face verification system built with Python, OpenCV, and DeepFace. The combination of Haar Cascade detection (a classical computer vision technique) and FaceNet embeddings (a modern deep learning approach) provides a complete end-to-end verification pipeline. The multithreaded design ensures the system remains responsive even while running an expensive neural network inference in the background.

The project illustrates key concepts taught in the Computer Vision course: image acquisition, preprocessing, feature-based object detection, and deep learning-based recognition. It also demonstrates practical engineering concerns such as concurrency, error handling, and user feedback.

---

## 22. References

1. Viola, P., & Jones, M. (2001). *Rapid object detection using a boosted cascade of simple features.* Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR).

2. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

3. Serengil, S. I., & Ozpinar, A. (2020). *LightFace: A Hybrid Deep Face Recognition Framework.* 2020 Innovations in Intelligent Systems and Applications Conference (ASYU). [DeepFace library]

4. OpenCV Documentation — Cascade Classifier: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html

5. DeepFace GitHub Repository: https://github.com/serengil/deepface
