# Railway-Obstacle-Detection-Using-YOLO
Railway Obstacle Detection using YOLOv8 is a deep learning project designed to detect obstacles on railway tracks in real-time. It uses the YOLOv3 object detection algorithm to identify objects like humans, animals, etc. The system is trained on a custom dataset annotated in YOLO format implemented with python, OpenCV.

1.YOLOv8 Object Detection
We used the YOLOv8 (You Only Look Once version 8) model from Ultralytics for real-time object detection. It enables high-speed and accurate identification of obstacles like cows, people, or vehicles on railway tracks.

2.Custom Dataset Labeling with Roboflow
To improve detection accuracy, a custom dataset was created and labeled using Roboflow. Roboflow also helped with data augmentation and exporting the dataset in YOLO-compatible format.

3.Training on Google Colab
The model was trained on Google Colab using GPU support to accelerate the training process. After experimenting with different hyperparameters, the best-performing weights were selected for deployment.

4.Video Masking for Region of Interest
A binary mask was applied to focus detection only on the area of the railway track. This helps reduce false positives and improves model performance by ignoring irrelevant regions.

5.Alert System
When an obstacle is detected with high confidence, the system captures the frame and sends a WhatsApp message along with the image to a predefined number using the pywhatkit library.

6.Sound Alert System
A beep sound is triggered using the winsound module whenever an obstacle is detected to alert the user in real-time.
