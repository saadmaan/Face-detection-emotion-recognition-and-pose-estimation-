# 1. Library imports
import uvicorn
import streamlit
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import cv2
from deepface import DeepFace
# 2. Create the app object
app = FastAPI()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        result = DeepFace.analyze(frame, actions = ['emotion'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.1,4)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, result['dominant_emotion'], (50,50), font, 3, (0,0,255), 2, cv2.LINE_4)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
@app.get('/')
async def index():
    return {'message': 'Hello, World'}

@app.get('/predict')
async def predict():
    return faces()
    
    
   