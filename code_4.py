from ultralytics import YOLO
from IPython.display import Image, display  # For displaying the image inline in Jupyter
from deepface import DeepFace  # New: For gender and age estimation
import cv2  # New: For image manipulation (already installed via opencv-python)

# Load the pre-trained YOLOv8 small model
model = YOLO('yolov8s.pt')

# Path to your image
image_name_path = 'park.png'

# Perform inference with confidence threshold of 0.7
results = model(image_name_path, conf=0.7)

# Load the original image for cropping and annotation
img = cv2.imread(image_name_path)

# Extract detected objects and process only 'person' for gender/age
detected_objects = []
person_details = []  # To store gender/age info

for r in results:
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        class_name = model.names[int(cls)]
        detected_objects.append(class_name)
        
        if class_name == 'person' and conf > 0.7:  # Focus on high-conf persons
            # Crop the bounding box (convert to int for slicing)
            x1, y1, x2, y2 = map(int, box)
            person_crop = img[y1:y2, x1:x2]
            
            # Analyze with DeepFace (tries to detect face and estimate)
            try:
                analysis = DeepFace.analyze(person_crop, actions=['gender', 'age'], enforce_detection=False)
                gender = analysis[0]['dominant_gender'] if analysis else 'Unknown'
                age = analysis[0]['age'] if analysis else 'Unknown'
                
                # Classify age roughly (you can adjust thresholds)
                if age != 'Unknown':
                    if age < 18:
                        age_group = 'Child'
                    elif age > 60:
                        age_group = 'Elder'
                    else:
                        age_group = 'Adult'
                else:
                    age_group = 'Unknown'
                
                detail = f"{gender} {age_group}"
                person_details.append(detail)
                
                # Annotate the image with gender/age label (below the default label)
                label = f"Person ({detail}) {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except:
                person_details.append('Unknown')  # If no face detected or error

# Print the original detected objects
print("Detected objects in the image (conf > 0.7):")
print(detected_objects)

# Print unique objects
unique_objects = list(set(detected_objects))
print("\nUnique detected objects (conf > 0.7):")
print(unique_objects)

# Print person-specific details
print("\nPerson details (gender and age group):")
print(person_details)

# Save and display the annotated image (now with gender/age if detected)
output_name_path = 'park_detected_with_details.png'
cv2.imwrite(output_name_path, img)
display(Image(filename=output_name_path))
