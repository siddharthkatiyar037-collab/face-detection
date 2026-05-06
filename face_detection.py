import cv2
import threading
from deepface import DeepFace

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load reference image
reference_img = cv2.imread("reference.jpg")
if reference_img is None:
    print("Warning: reference.jpg not found. Press 's' to save a reference frame.")
face_match = False
counter = 0
lock = threading.Lock()

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def check_face(frame, reference):
    global face_match
    try:
        # Check if a face exists in the current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            with lock:
                face_match = False
            return

        # Resize both images to match DeepFace input dimensions
        frame_resized = cv2.resize(frame, (224, 224))
        reference_resized = cv2.resize(reference, (224, 224))

        result = DeepFace.verify(
            frame_resized,
            reference_resized,
            model_name="Facenet",
            enforce_detection=False
        )
        with lock:
            face_match = result["verified"]
    except Exception as e:
        print("Verification error:", e)
        with lock:
            face_match = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if reference_img is not None and counter % 30 == 0 and threading.active_count() < 3:
        threading.Thread(target=check_face, args=(frame.copy(), reference_img.copy()), daemon=True).start()

    counter += 1

    label = "MATCH!" if face_match else "NO MATCH!"
    color = (0, 255, 0) if face_match else (0, 0, 255)
    cv2.putText(frame, label, (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, color, 3)

    cv2.imshow("Face Verification", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("reference.jpg", frame)
        reference_img = frame.copy()
        print("Saved current frame as reference.jpg")

cap.release()
cv2.destroyAllWindows()
