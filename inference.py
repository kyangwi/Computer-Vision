import cv2
import os
import torch
import easyocr
from ultralytics import YOLO

reader = easyocr.Reader(['en'])

def inference(model_path):
    # Load a model
    model = YOLO(model_path)  # load a custom model

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not os.path.exists('detected'):
        os.makedirs('detected')
        
    count = 0
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(frame)

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the confidence and label
                conf = box.conf[0]
                label = model.names[int(box.cls[0])]

                region = frame[y1:y2:,x1:x2]

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                rgb_region = cv2.cvtColor(region,cv2.COLOR_BGR2RGB)
                results = reader.readtext(rgb_region)

                print(results)

                cv2.imwrite(f"./detected/img{count}.jpg",rgb_region)


                # Draw the label and confidence on the frame
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                count = count + 1


        # Convert the frame back to BGR for displaying
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow('Webcam with Bounding Boxes', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Perform inference using the webcam
inference('./model2.pt')
