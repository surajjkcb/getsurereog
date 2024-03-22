import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

#base_model=load_model('keras_model.h5')
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf 
print(tf.__version__)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
import tensorflow as tf
model_path = 'converted_keras//keras_model.h5'
labels_path = 'converted_keras//labels.txt'
classifier = Classifier("model_path", "labels_path")
offset = 20
imgSize = 300
counter = 0
labels = ["Hello","ThankYou","Yes","No","Want","Dont want","Car","Bathroom","Fine","Finish","Help","Go","Like","More","Need","Right","Wrong","Forget"]


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)

       
        cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  

        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)



"""
import cv2
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
# Load TensorFlow
import tensorflow as tf
print(tf.__version__)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Use the model for predictions, evaluation, etc.

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Paths to the model and labels
model_path = 'converted_keras//keras_model.h5'
labels_path = 'converted_keras//labels.txt'
import h5py
from tensorflow import keras
with h5py.File('keras_model.h5', 'r') as file:
    model = keras.models.load_model('keras_model.h5')
# Initialize the classifier
classifier = Classifier(model_path, labels_path)

# Offset and image size
offset = 20
imgSize = 300

# List of labels
labels = ["Hello", "ThankYou", "Yes", "No", "Want", "Dont want", "Car", "Bathroom", "Fine", "Finish", "Help", "Go", "Like", "More", "Need", "Right", "Wrong", "Forget"]

while True:
    # Read frame from the camera
    success, img = cap.read()
    if not success:
        print("Failed to read frame.")
        break

    imgOutput = img.copy()

    # Find hands in the frame
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize and fill the white background
        imgWhite[:imgCrop.shape[0], :imgCrop.shape[1]] = imgCrop

        # Get prediction from the classifier
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Draw bounding box and label on the output image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

    # Display the output images
    cv2.imshow('ImageCrop', imgCrop)
    cv2.imshow('ImageWhite', imgWhite)
    cv2.imshow('Image', imgOutput)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
"""
"""  
import cv2
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Paths to the model and labels
model_path = 'converted_keras//keras_model.h5'
labels_path = 'converted_keras//labels.txt'

# Load the Keras model
model = keras.models.load_model(model_path)

# Initialize the classifier
classifier = Classifier(model_path, labels_path)

# Offset and image size
offset = 20
imgSize = 300

# List of labels
labels = ["Hello", "ThankYou", "Yes", "No", "Want", "Dont want", "Car", "Bathroom", "Fine", "Finish", "Help", "Go", "Like", "More", "Need", "Right", "Wrong", "Forget"]

while True:
    # Read frame from the camera
    success, img = cap.read()
    if not success:
        print("Failed to read frame.")
        break

    imgOutput = img.copy()

    # Find hands in the frame
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize and fill the white background
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
        imgWhite[:imgResize.shape[0], :imgResize.shape[1]] = imgResize

        # Get prediction from the classifier
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Draw bounding box and label on the output image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

    # Display the output images
    cv2.imshow('ImageCrop', imgCrop)
    cv2.imshow('ImageWhite', imgWhite)
    cv2.imshow('Image', imgOutput)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
"""