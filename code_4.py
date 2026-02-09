from ultralytics import YOLO
from IPython.display import Image, display
import cv2
from deepface import DeepFace

# Load the small YOLOv8 model
model = YOLO('yolov8s.pt')

# Path to your image
image_name_path = 'park.png'

# Perform inference with confidence threshold of 0.7
results = model(image_name_path, conf=0.7)

# Load the original image for cropping and annotation
img = cv2.imread(image_name_path)

# Lists to store results
detected_objects = []
person_age_groups = []  # Only age group info

for r in results:
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        class_name = model.names[int(cls)]
        detected_objects.append(class_name)
        
        if class_name == 'person' and conf > 0.7:
            # Crop the person region
            x1, y1, x2, y2 = map(int, box)
            person_crop = img[y1:y2, x1:x2]
            
            try:
                # Only ask DeepFace for age (skip gender)
                analysis = DeepFace.analyze(person_crop, actions=['age'], enforce_detection=False)
                
                age = analysis[0]['age'] if analysis else 'Unknown'
                
                # Classify age group
                if age != 'Unknown':
                    if age < 18:
                        age_group = 'Child'
                    elif age > 60:
                        age_group = 'Elder'
                    else:
                        age_group = 'Adult'
                else:
                    age_group = 'Unknown'
                
                person_age_groups.append(age_group)
                
                # Annotate only with age group
                label = f"Person ({age_group}) {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            except Exception:
                person_age_groups.append('Unknown')
                # Optional: you can still draw a box even if analysis failed
                label = f"Person (Unknown) {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

print("Detected objects in the image (conf > 0.7):")
print(detected_objects)

print("\nUnique detected objects (conf > 0.7):")
print(list(set(detected_objects)))

print("\nPerson age groups:")
print(person_age_groups)

# Save and display the annotated image
output_name_path = 'park_detected_age_group.png'
cv2.imwrite(output_name_path, img)
display(Image(filename=output_name_path))
