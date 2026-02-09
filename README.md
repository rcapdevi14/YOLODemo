# YOLODemo
Simple YOLO-based image recognition python script.

STEPS:

1) Open Google Colab (https://colab.google/) and create a "New Notebook."
2) Install ultralytics, the repository that contains YOLO:
   >>> pip install ultralytics pandas opencv-python ipykernel.

4) Upload the image "park.png" to Colab.
5) Run the script "code_1.py" (update "image_name_path" in the script!)

Note how the script identifies a few objects: "Unique detected objects: ['person', 'chair', 'bench', 'dog']"

7) To obtain more details, run the script "code_2.py"

Note that each identified object has a number associated to it. It measures the probability of the identification.  
We can now modify this probability, and only output identified objects above certain threshold.

8) Run the script "code_3.py" multiple times adjusting the confidence threshold "conf=0.7" variable.

Explore a limit case e.g. "conf=0.1" and look at all the noise.

We now want to classify the persons in an image according to their age group into three simple categories:  
Age < 18 = Minor  
Age > 65 = Elder  
Else, Adult

9) Install Deepface:
   >>> pip install deepface

10) Run the script "code_4.py"
11) Repeat the whole process with other images.
