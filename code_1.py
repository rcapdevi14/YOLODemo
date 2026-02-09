from ultralytics import YOLO

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
