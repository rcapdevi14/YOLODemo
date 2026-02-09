from ultralytics import YOLO
from IPython.display import Image, display  # For displaying the image inline in Jupyter

# Load the pre-trained YOLOv8 small model (downloads automatically if not present)
model = YOLO('yolov8s.pt')  # Change to 'yolov8n.pt' for faster/lighter, 'yolov8m.pt' for more accurate, etc.

# Path to your image
image_name_path = 'park.png'

# Perform inference with confidence threshold of 0.7 (filters out detections below this score)
results = model(image_name_path, conf=0.7)

# Extract detected object names (only those above the threshold, with possible duplicates if multiple instances)
detected_objects = [model.names[int(cls)] for r in results for cls in r.boxes.cls]

# Print the list
print("Detected objects in the image (conf > 0.7):")
print(detected_objects)

# Optional: Unique objects only
unique_objects = list(set(detected_objects))
print("\nUnique detected objects (conf > 0.7):")
print(unique_objects)

# Save the annotated image with bounding boxes and labels (only for detections above 0.7)
output_name_path = 'park_detected.png'  # Change this filename if desired
results[0].save(output_name_path)  # Saves the image with filtered detections overlaid

# Display the annotated image inline in the Jupyter Notebook
display(Image(filename=output_name_path))
