from ultralytics import YOLO
from IPython.display import Image, display  # New: For displaying the image inline in Jupyter

# Load the pre-trained YOLOv8 small model (downloads automatically if not present)
model = YOLO('yolov8s.pt')  # Change to 'yolov8n.pt' for faster/lighter, 'yolov8m.pt' for more accurate, etc.

# Path to your image
image_name_path = 'park.png'

# Perform inference
results = model(image_name_path)

# Extract detected object names (with possible duplicates if multiple instances)
detected_objects = [model.names[int(cls)] for r in results for cls in r.boxes.cls]

# Print the list
print("Detected objects in the image:")
print(detected_objects)

# Optional: Unique objects only
unique_objects = list(set(detected_objects))
print("\nUnique detected objects:")
print(unique_objects)

# New: Save the annotated image with bounding boxes and labels
output_name_path = 'park_detected_all.png'  # Change this filename if desired
results[0].save(output_name_path)  # Saves the image with detections overlaid

# New: Display the annotated image inline in the Jupyter Notebook
display(Image(filename=output_name_path))
