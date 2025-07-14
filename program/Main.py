from ultralytics import YOLO
import cv2
import cvzone
import math
import winsound
import datetime
import pywhatkit as kit
import os

# Initialize YOLO Model
model = YOLO(r"D:\MAJOR PROJECT SC\Best weights\yolov8n.pt")

# Load Video and Mask
cap = cv2.VideoCapture(r"D:\MAJOR PROJECT SC\videos\cow.mp4")
mask = cv2.imread(r"D:\MAJOR PROJECT SC\masks\cow_mask.png")

# Validate video file
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Class Names
classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"]

def send_whatsapp_notification(receiver_number, detected_object, confidence, img):
    """
    Sends a WhatsApp message with an image when an object is detected.
    """
    message = f"🚨 Alert! Object Detected: {detected_object} with confidence {confidence*100:.2f}%."

    now = datetime.datetime.now()
    hours, minutes = now.hour, now.minute + 1  

    # Fix: If minutes exceed 59, move to next hour
    if minutes == 60:
        hours = (hours + 1) % 24  # Ensure 24-hour format
        minutes = 0

    # Save detected object image
    save_path = "detected_object.jpg"
    cv2.imwrite(save_path, img)  # Save the frame

    try:
        # Send image with a caption
        kit.sendwhats_image(receiver_number, save_path, message, wait_time=10, tab_close=True)
        print(f"✅ WhatsApp notification with image sent to {receiver_number}: {message}")
        
        # Optional: Remove the image file after sending
        os.remove(save_path)

    except Exception as e:
        print(f"❌ Failed to send WhatsApp notification: {e}")

# Video Processing Loop
while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to read the frame. Ending video processing.")
        break

    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Convert mask to grayscale and apply
    mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    masked_img = cv2.bitwise_and(img, img, mask=mask_binary)

    results = model(masked_img, stream=True)
    alert_triggered = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(0, y1 - 35)), scale=1)

            if not alert_triggered:
                print(f"🚨 Alert! Detected: {classNames[cls]} with confidence {conf}")
                winsound.Beep(1000, 300)  # Play a beep sound

                # Send WhatsApp notification
                send_whatsapp_notification("+91*********", classNames[cls], conf, img)

                alert_triggered = True

    resized_masked_img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow("Image", resized_masked_img)

    if cv2.waitKey(1) & 0xFF == ord( 
        'q'):
        break  # Exit loop if 'q' is  pressed

# Cleanup
cap.release()
cv2.destroyAllWindows()
