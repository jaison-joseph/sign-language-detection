'''
the file to add logic and run the GUI
'''

from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os

# imports for feature drawing & extraction
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# the trained SVM modeL
from libsvm.svmutil import *
from libsvm.svm import *

# data handling
import numpy as np

# features of each frame
global features
features = np.zeros(63)

global store, storeIdx, frameIdx
store = np.zeros((10, 100, 63))
storeIdx = -1
frameIdx = 0

global m
m = svm_load_model('a2z_model.model')

global capture,rec_frame, toggle_prediction, switch, neg, face, rec, out 
capture = False
toggle_prediction = 0
neg = 0
face = 0
switch = 1
rec = 0

global test_var
test_var = 'mic check'

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame) 

'''
the generator used to yield successive frames from the camera,
    then encoded as a JPG, and yielded

this generator is used to stream the video from the python application to the webpage

see more in the video_feed route
'''
def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame, test_var, m, store, storeIdx, frameIdx
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (0, 0, 0)
    thickness = 2
    j = 0
    hands = mp_hands.Hands(
        model_complexity = 0,
        min_detection_confidence = 0.5,
        max_num_hands = 1,
        min_tracking_confidence = 0.5
    )
    while True:
        success, frame = camera.read() 
        if not success: 
            break
        
        if(rec):
            rec_frame=frame
            frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
            frame=cv2.flip(frame,1)

        # feature extraction & updating frame
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            # for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            if toggle_prediction or capture:
                for i, j in enumerate([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]):
                    features[j] = hand_landmarks.landmark[i].x
                    features[j+1] = hand_landmarks.landmark[i].y
                    features[j+2] = hand_landmarks.landmark[i].z
                if capture:
                    if frameIdx < 100:
                        store[storeIdx, frameIdx] = features
                        frameIdx += 1
                    else:
                        capture = False
                        print('done capturing')
                # Convert a Python-format instance to svm_nodearray, a ctypes structure
                else:
                    converted, __ = gen_svm_nodearray(features)
                    label = libsvm.svm_predict(m, converted)
                    label = chr(int(label) + ord('A'))
                    cv2.putText(frame, label, org, font, font_scale, color, thickness, cv2.LINE_AA)
            

        _, buffer = cv2.imencode('.bmp', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', test_var = test_var)
    
    
@app.route('/video_feed')
def video_feed():
    '''
    the "multipart/x-mixed-replace; boundary=frame" tag is used to display media, 
        this tag also tells the browser to constantly update the media that it's displaying
    the media that is displayed is provided using the first argument to the response object, the generator gen_frames()
    each call to the function yields a new frame obtained from the camera
    '''
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/actions', methods = ['GET', 'POST'])
def actions():
    global switch, camera, capture, toggle_prediction, storeIdx, frameIdx 
    '''
    the buttons in the GUI are part of an HTML form (each one triggering a submit) that makes a POST request to this route
    '''
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            if storeIdx == 9:
                print('store full')
            else:
                storeIdx += 1
                capture = True
                frameIdx = 0
        elif request.form.get('toggle_prediction') == 'Toggle_prediction':
            toggle_prediction = not toggle_prediction 
        
        elif request.form.get('stop') == 'Stop/Start':
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1

    return render_template('index.html', test_var = test_var)


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()  