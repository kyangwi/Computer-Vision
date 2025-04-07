import cv2
import os
import numpy as np
import time
import easyocr as ocr
from ultralytics import YOLO
from google import generativeai as genai
from PIL import Image
import string

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

def get_number_plate(plate_composite):
    shortest_length = float('inf')
    shortest_index = -1
    for i, text in enumerate(plate_composite):
        if len(text) < shortest_length:
            shortest_length = len(text)
            shortest_index = i
    first_part = plate_composite[shortest_index]
    second_part = ''.join(plate_composite[:shortest_index] + plate_composite[shortest_index+1:])
    plate = first_part + second_part

    return correct_ocr_output(plate.upper().replace(' ',''))

def correct_ocr_output(text):
    # Correct the first part of the number plate
    first_part = text[:2]
    corrected_first_part = ''.join([dict_int_to_char.get(c, c) for c in first_part])

    # Correct the second part of the number plate
    second_part = text[2:]
    corrected_second_part = ''.join([dict_char_to_int.get(c, c) if i < 3 else dict_int_to_char.get(c, c) for i, c in enumerate(second_part)])

    # Return the corrected number plate
    return corrected_first_part + corrected_second_part

def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase) and \
       (text[1] in string.ascii_uppercase) and \
       (text[2] in string.ascii_uppercase) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
       (text[6] in string.ascii_uppercase):
        return True
    else:
        return False


# def filter_text(rgb_region,ocr_result,region_treshold):
#     rectangle_size = rgb_region.shape[0] * rgb_region.shape[1]

#     plate = []

#     for result in ocr_result:
#         length = np.sum(np.subtract(result[0][1],result[0][0]))
#         height = np.sum(np.subtract(result[0][2],result[0][1]))

#         if length*height / rectangle_size > region_treshold:
#             plate.append(result[1])

    
#     return plate



# def generate_text(img) -> str:

#     try:
#         #load the model
#         model = genai.GenerativeModel('gemini-1.5-flash')

#         image = Image.open(img)

#         #provide image and prompt to extract text
#         response = model.generate_content(
#             [ image,
#             f'''
#                 You are an expert in number plate recognition:
#                 return the number plate only:
#             '''
#             ]
#         )

#         return response.text
    
#     except Exception as e:
#         # img = cv2.imread(img)
#         # reader = ocr.Reader(['en'], gpu=False)
#         # detections = reader.readtext(img)


#         # if len(detections) <= 1:
#         #     # For one line scripture number plate
#         #     for detection in detections:
#         #         bbox,text,score = detection
#         #     text = text.upper().replace(' ','')
#         #     number_plate = correct_ocr_output(text)
#         #     return number_plate
#         # else:
#         #     plate_composite = filter_text(img, detections, region_threshold=0.2)
#         #     number_plate = get_number_plate(plate_composite)
#         #     return number_plate
#         return None
        
GOOGLE_API_KEY="AIzaSyAxnVOuLEjDev8Zy-Oz_H5l-yXVDKq7Dm0"
genai.configure(api_key=GOOGLE_API_KEY)

def generate_text(img) -> str:

    try:
        #load the model
        model = genai.GenerativeModel('gemini-1.5-flash')

        image = Image.open(img)

        #provide image and prompt to extract text
        response = model.generate_content(
            [ image,
            f'''
                You are an expert in number plate recognition:
                return the number plate(s)
            '''
            ]
        )

        return response.text  
    except Exception as e:
        return 'Processing'


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

                region = frame[y1:y2, x1:x2]

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"./detected/img{count}.jpg", rgb_region)
  
                # Use Google's Gemini model for OCR
                license_plate = generate_text(f"./detected/img{count}.jpg")
                print(f"License Plate: {license_plate}")

                # Draw the label, confidence, and license plate on the frame
                cv2.putText(frame, f'{label} {license_plate}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # {conf:.2f} 
                count = count + 1

        # Convert the frame back to BGR for displaying
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow('Webcam with Bounding Boxes', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Pause the loop for 2 seconds
        time.sleep(1)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Perform inference using the webcam
inference('./model2.pt')
