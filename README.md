# Face Recognition Project
# By - Dhrubojyoti Chowdhury , Hemanta Ghosh & Iman Chandra from Narula Institute of Technology

This project implements face recognition using OpenCV instead of the `face_recognition` library to avoid installation issues on Windows.

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-contrib-python numpy Pillow matplotlib
```

## Usage

### 1. Training the Model

Run the training script to create a face recognition model from your dataset:

```bash
python training_opencv.py
```

This will:
- Process all images in the `train/` directory
- Detect faces using OpenCV's Haar Cascade
- Train a Local Binary Pattern Histogram (LBPH) face recognizer
- Save the trained model as `face_recognizer_model.yml`
- Save the label mapping as `label_mapping.pkl`

### 2. Recognizing Faces

After training, you can recognize faces in new images:

```bash
python face_recognition_opencv.py
```

The script will prompt you for the path to a test image and display the results.

### 3. Generating Model Diagrams

To create visual diagrams of the system architecture:

```bash
python model_diagram.py
```

This will generate:
- `face_recognition_model_diagram.png`: System architecture diagram
- `face_recognition_workflow.png`: Step-by-step workflow diagram

## Project Structure

- `train/` - Training dataset with person names as folder names
- `training_opencv.py` - Script to train the face recognition model
- `face_recognition_opencv.py` - Script to recognize faces in new images
- `model_diagram.py` - Script to generate system diagrams
- `MODEL_DESCRIPTION.md` - Detailed model documentation
- `requirements.txt` - Required Python packages
- `training.py` - Original script using face_recognition library (requires CMake/dlib)

## Model Architecture

The system uses a two-phase approach:

### Training Phase
1. **Data Collection**: Images organized by person in folders
2. **Face Detection**: Haar Cascade classifier locates faces
3. **Preprocessing**: Grayscale conversion and resizing to 100x100
4. **Feature Extraction**: LBPH feature computation
5. **Model Training**: LBPH recognizer training
6. **Model Storage**: Save trained model and label mapping

### Inference Phase
1. **Model Loading**: Load trained model and labels
2. **Test Image**: Input new image for recognition
3. **Face Detection**: Detect faces in test image
4. **Preprocessing**: Apply same preprocessing as training
5. **Prediction**: Predict person identity
6. **Output**: Display name and confidence score

For detailed technical information, see `MODEL_DESCRIPTION.md`.

## Alternative: Using face_recognition Library

If you want to use the original `face_recognition` library:

1. Install CMake from https://cmake.org/download/
2. Add CMake to your system PATH
3. Install the packages:
```bash
pip install face_recognition opencv-python
```

## Notes

- The OpenCV solution uses LBPH (Local Binary Pattern Histogram) for face recognition
- Face detection is done using Haar Cascade classifier
- Images are resized to 100x100 pixels for consistency
- The model works best with good quality, well-lit face images
- Lower confidence values indicate higher certainty in recognition 
