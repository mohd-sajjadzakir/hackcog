import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
import pickle
import os
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Define a variable to hold the model
model = None

# Define the absolute file path to your trained model
model_file_path = r"C:\Users\sajja\PycharmProjects\new\TrafficSigns/model_trained.p"  # Replace with the actual file path

# Check if the trained model file exists
if os.path.exists(model_file_path):
    # Try to open and load the trained model
    with open(model_file_path, "rb") as pickle_in:
        model = pickle.load(pickle_in)
        print("Trained model loaded successfully.")
else:
    print(f"Trained model file '{model_file_path}' not found. Check the file path.")

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    # Define your class names here
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
        'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
        'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve',
        'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
        'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
        'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
        'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo]

while True:
    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # PREDICT IMAGE
    if model is not None:
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.max(predictions)

        if probabilityValue > threshold:
            class_name = getClassName(classIndex)
            cv2.putText(imgOrignal, f"{classIndex} {class_name}", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, f"{round(probabilityValue * 100, 2)}%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
