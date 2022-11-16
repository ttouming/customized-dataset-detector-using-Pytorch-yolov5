# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:57:44 2019

@author: seraj
"""
import time
import cv2 
import torch
from flask import Flask, render_template, Response

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor
# input pre-trained weight
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best_v20n.pt', force_reload=True) 
 

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

# video stteraming generator
def gen():
    """Video streaming generator function."""

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcamccc")

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            # resize frame
            img = cv2.resize(img, (800, 600), fx=0.5, fy=0.5)
            # change color BGR2RGB(default colour of openCV is BGR, yolov5 is RGB) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
            # inferencing
            results = model(img)
            # extract data
            s = results.pandas().xyxy[0]
            if(s.empty==False):
                print("-8984841848-")
                print("x: ",s.loc[0]["xmin"])
                print("y: ",int(s.loc[0]["ymax"]))
                x, y, w, h = int(s.loc[0]["xmin"]), int(s.loc[0]["ymin"]), int(s.loc[0]["xmax"]-s.loc[0]["xmin"]), int(s.loc[0]["ymax"]-s.loc[0]["ymin"])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # org
                org = (int(s.loc[0]["xmin"]),int(s.loc[0]["ymin"]))  
                # fontScale
                fontScale = 1
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 2
                # Using cv2.putText() method
                cv2.putText(img, str(s.loc[0]["name"])+str(round(s.loc[0]["confidence"], 2)), org, font, fontScale, color, thickness, cv2.LINE_AA)
            # change color back 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # time.sleep(0.1)
            time.sleep(0.001)
        else: 
            break

    
 
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    print("here is sasad")
    app.run(debug=True)


    
