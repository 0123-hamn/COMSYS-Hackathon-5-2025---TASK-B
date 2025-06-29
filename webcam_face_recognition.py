import cv2
import numpy as np
import pickle

# Load the trained model and label mapping
def load_model():
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('face_recognizer_model.yml')
        with open('label_mapping.pkl', 'rb') as f:
            label_dict = pickle.load(f)
        reverse_label_dict = {v: k for k, v in label_dict.items()}
        return face_recognizer, reverse_label_dict
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def main():
    face_recognizer, reverse_label_dict = load_model()
    if face_recognizer is None:
        print("Please train the model first using training_opencv.py")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            label, confidence = face_recognizer.predict(face_roi)
            person_name = reverse_label_dict.get(label, "Unknown")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{person_name} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Webcam Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 