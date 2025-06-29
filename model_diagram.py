import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_model_diagram():
    """
    Create a comprehensive model diagram for the face recognition system
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E8F4FD',
        'processing': '#FFF2CC',
        'model': '#D5E8D4',
        'output': '#F8CECC',
        'storage': '#E1D5E7'
    }
    
    # Title
    ax.text(8, 11.5, 'Face Recognition System Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Training Phase (Left side)
    ax.text(4, 10.5, 'TRAINING PHASE', fontsize=14, fontweight='bold', ha='center')
    
    # Input Data
    input_box = FancyBboxPatch((0.5, 9), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 9.5, 'Training Dataset\n(train/ folder)', ha='center', va='center', fontsize=10)
    
    # Face Detection
    detect_box = FancyBboxPatch((0.5, 7.5), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['processing'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(detect_box)
    ax.text(2, 8, 'Face Detection\n(Haar Cascade)', ha='center', va='center', fontsize=10)
    
    # Feature Extraction
    feature_box = FancyBboxPatch((0.5, 6), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['processing'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(feature_box)
    ax.text(2, 6.5, 'Feature Extraction\n(Grayscale + Resize)', ha='center', va='center', fontsize=10)
    
    # Model Training
    train_box = FancyBboxPatch((0.5, 4.5), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['model'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(train_box)
    ax.text(2, 5, 'Model Training\n(LBPH Recognizer)', ha='center', va='center', fontsize=10)
    
    # Model Storage
    storage_box = FancyBboxPatch((0.5, 3), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['storage'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(storage_box)
    ax.text(2, 3.5, 'Model Storage\n(face_recognizer_model.yml\nlabel_mapping.pkl)', 
            ha='center', va='center', fontsize=9)
    
    # Inference Phase (Right side)
    ax.text(12, 10.5, 'INFERENCE PHASE', fontsize=14, fontweight='bold', ha='center')
    
    # Test Input
    test_input_box = FancyBboxPatch((12.5, 9), 3, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['input'], 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(test_input_box)
    ax.text(14, 9.5, 'Test Image', ha='center', va='center', fontsize=10)
    
    # Face Detection (Inference)
    test_detect_box = FancyBboxPatch((12.5, 7.5), 3, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['processing'], 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(test_detect_box)
    ax.text(14, 8, 'Face Detection\n(Haar Cascade)', ha='center', va='center', fontsize=10)
    
    # Feature Extraction (Inference)
    test_feature_box = FancyBboxPatch((12.5, 6), 3, 1, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['processing'], 
                                     edgecolor='black', linewidth=2)
    ax.add_patch(test_feature_box)
    ax.text(14, 6.5, 'Feature Extraction\n(Grayscale + Resize)', ha='center', va='center', fontsize=10)
    
    # Model Prediction
    predict_box = FancyBboxPatch((12.5, 4.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['model'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(predict_box)
    ax.text(14, 5, 'Model Prediction\n(LBPH Recognizer)', ha='center', va='center', fontsize=10)
    
    # Output
    output_box = FancyBboxPatch((12.5, 3), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(14, 3.5, 'Recognition Result\n(Name + Confidence)', ha='center', va='center', fontsize=10)
    
    # Arrows for Training Phase
    arrows_training = [
        ((2, 9), (2, 8.5)),      # Input to Detection
        ((2, 7.5), (2, 7)),      # Detection to Feature Extraction
        ((2, 6), (2, 5.5)),      # Feature Extraction to Training
        ((2, 4.5), (2, 4))       # Training to Storage
    ]
    
    for start, end in arrows_training:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", ec="black")
        ax.add_patch(arrow)
    
    # Arrows for Inference Phase
    arrows_inference = [
        ((14, 9), (14, 8.5)),    # Test Input to Detection
        ((14, 7.5), (14, 7)),    # Detection to Feature Extraction
        ((14, 6), (14, 5.5)),    # Feature Extraction to Prediction
        ((14, 4.5), (14, 4))     # Prediction to Output
    ]
    
    for start, end in arrows_inference:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", ec="black")
        ax.add_patch(arrow)
    
    # Model Loading Arrow
    model_arrow = ConnectionPatch((3.5, 3.5), (12.5, 5), "data", "data",
                                 arrowstyle="->", shrinkA=5, shrinkB=5,
                                 mutation_scale=20, fc="red", ec="red", linewidth=2)
    ax.add_patch(model_arrow)
    ax.text(8, 4.5, 'Load Trained Model', ha='center', va='center', 
            fontsize=10, color='red', fontweight='bold')
    
    # Technical Details Box
    details_box = FancyBboxPatch((5, 1.5), 6, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#F0F0F0', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(details_box)
    
    details_text = """Technical Implementation Details:
• Face Detection: OpenCV Haar Cascade Classifier
• Feature Extraction: Grayscale conversion + Resize to 100x100
• Recognition Algorithm: Local Binary Pattern Histogram (LBPH)
• Training Data: Person folders with face images
• Model Output: Person name + confidence score"""
    
    ax.text(8, 2.25, details_text, ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('face_recognition_model_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model diagram saved as 'face_recognition_model_diagram.png'")

def create_workflow_diagram():
    """
    Create a workflow diagram showing the step-by-step process
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Face Recognition Workflow', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Workflow steps
    steps = [
        (1, 8, "1. Data Collection", "Gather face images\norganized by person"),
        (3, 8, "2. Face Detection", "Detect faces using\nHaar Cascade"),
        (5, 8, "3. Preprocessing", "Convert to grayscale\nand resize to 100x100"),
        (7, 8, "4. Feature Extraction", "Extract LBPH\nfeatures"),
        (9, 8, "5. Model Training", "Train LBPH\nrecognizer"),
        (11, 8, "6. Model Storage", "Save model and\nlabel mapping"),
        (1, 5, "7. Load Model", "Load trained model\nfor inference"),
        (3, 5, "8. Test Image", "Input new image\nfor recognition"),
        (5, 5, "9. Face Detection", "Detect faces in\ntest image"),
        (7, 5, "10. Preprocessing", "Same preprocessing\nas training"),
        (9, 5, "11. Prediction", "Predict person\nidentity"),
        (11, 5, "12. Output", "Display name and\nconfidence score")
    ]
    
    # Colors for different phases
    colors = ['#E8F4FD', '#FFF2CC', '#FFF2CC', '#D5E8D4', '#D5E8D4', '#E1D5E7',
              '#E1D5E7', '#E8F4FD', '#FFF2CC', '#FFF2CC', '#D5E8D4', '#F8CECC']
    
    # Draw boxes and text
    for i, (x, y, title, desc) in enumerate(steps):
        box = FancyBboxPatch((x-0.4, y-0.8), 1.8, 1.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors[i], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y+0.2, title, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y-0.2, desc, ha='center', va='center', fontsize=8)
    
    # Draw arrows
    for i in range(len(steps)-1):
        if i < 5:  # Training phase arrows
            start = (steps[i][0] + 0.9, steps[i][1])
            end = (steps[i+1][0] - 0.9, steps[i+1][1])
        elif i == 5:  # Connection from training to inference
            start = (steps[i][0], steps[i][1] - 0.8)
            end = (steps[i+1][0], steps[i+1][1] + 0.8)
        else:  # Inference phase arrows
            start = (steps[i][0] + 0.9, steps[i][1])
            end = (steps[i+1][0] - 0.9, steps[i+1][1])
        
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="black", ec="black")
        ax.add_patch(arrow)
    
    # Phase labels
    ax.text(6, 7.5, "TRAINING PHASE", fontsize=12, fontweight='bold', ha='center', color='blue')
    ax.text(6, 4.5, "INFERENCE PHASE", fontsize=12, fontweight='bold', ha='center', color='red')
    
    plt.tight_layout()
    plt.savefig('face_recognition_workflow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Workflow diagram saved as 'face_recognition_workflow.png'")

if __name__ == "__main__":
    print("Creating Face Recognition Model Diagrams...")
    create_model_diagram()
    create_workflow_diagram()
    print("All diagrams created successfully!") 