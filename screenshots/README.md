# Screenshots

This folder is for screenshots that demonstrate the system working.

Because the application requires a live webcam, screenshots cannot be auto-generated.
Please capture the following manually:

---

## Recommended Screenshots

### 1. `match.png` — Successful MATCH
- Run the application (`python main.py`)
- Face the webcam so your face matches the reference image
- Take a screenshot when the green **"MATCH!"** label appears

### 2. `no_match.png` — NO MATCH
- Run the application
- Block your face or show a different person
- Take a screenshot when the red **"NO MATCH"** label appears

### 3. `face_detection.png` — Face Detection Bounding Box
- Run the application
- Show your face to the camera
- Take a screenshot showing the green rectangle drawn around your face

### 4. `reference_capture.png` — Capturing a Reference Image
- Run the application without a reference image (rename/delete reference.jpg)
- Press **`s`** to capture the current frame as the reference
- Screenshot the moment of capture

---

## How to Take Screenshots on Windows

- **Full screen**: `Win + PrtScn` (saved to `Pictures/Screenshots`)
- **Cropped**: `Win + Shift + S` (opens Snip & Sketch)

Place the saved files in this `screenshots/` folder and rename them accordingly.
