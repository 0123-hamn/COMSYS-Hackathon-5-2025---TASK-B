import os
import cv2
import numpy as np
import pickle

def encode_faces_opencv(dataset_dir="train"):
    """
    Encode faces using OpenCV's face detection and recognition
    """
    # Load the face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    print("Processing training images...")
    
    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        
        if not os.path.isdir(person_folder):
            continue
            
        # Assign a numeric label to this person
        if person_name not in label_dict:
            label_dict[person_name] = current_label
            current_label += 1
            
        print(f"Processing {person_name}...")
        
        for img_file in os.listdir(person_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_folder, img_file)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Could not load image: {img_path}")
                        continue
                        
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    detected_faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    
                    for (x, y, w, h) in detected_faces:
                        # Extract face region
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Resize to standard size
                        face_roi = cv2.resize(face_roi, (100, 100))
                        
                        faces.append(face_roi)
                        labels.append(label_dict[person_name])
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Train the recognizer
    if faces and labels:
        print(f"Training recognizer with {len(faces)} faces from {len(label_dict)} people...")
        face_recognizer.train(faces, np.array(labels))
        
        # Save the trained model and label mapping
        face_recognizer.save('face_recognizer_model.yml')
        with open('label_mapping.pkl', 'wb') as f:
            pickle.dump(label_dict, f)
            
        print("Model saved successfully!")
        return face_recognizer, label_dict
    else:
        print("No faces found in the dataset!")
        return None, None

if __name__ == "__main__":
    # Use relative path to train directory
    encode_faces_opencv("train") 