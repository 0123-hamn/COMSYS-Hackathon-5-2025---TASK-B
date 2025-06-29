import cv2
import numpy as np
import pickle
import os

def load_model():
    """
    Load the trained face recognition model and label mapping
    """
    try:
        # Load the trained recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('face_recognizer_model.yml')
        
        # Load the label mapping
        with open('label_mapping.pkl', 'rb') as f:
            label_dict = pickle.load(f)
            
        # Create reverse mapping (label -> name)
        reverse_label_dict = {v: k for k, v in label_dict.items()}
        
        return face_recognizer, reverse_label_dict
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def recognize_face(image_path, face_recognizer, reverse_label_dict):
    """
    Recognize faces in a given image
    """
    # Load face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    print(f"Found {len(faces)} face(s) in the image")
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to match training size
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Predict
        label, confidence = face_recognizer.predict(face_roi)
        
        # Get person name
        person_name = reverse_label_dict.get(label, "Unknown")
        
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text with name and confidence
        text = f"{person_name} ({confidence:.2f})"
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        print(f"Recognized: {person_name} (confidence: {confidence:.2f})")
    
    # Display the result
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load the trained model
    face_recognizer, reverse_label_dict = load_model()
    
    if face_recognizer is None:
        print("Please train the model first using training_opencv.py")
        return
    
    # Example usage - you can modify this to test with your own images
    test_image_path = input("Enter the path to the test image: ")
    
    if os.path.exists(test_image_path):
        recognize_face(test_image_path, face_recognizer, reverse_label_dict)
    else:
        print("Image file not found!")

if __name__ == "__main__":
    main() 