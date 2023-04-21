'''
the file to add logic and run the GUI
'''

from flask import Flask, render_template, Response, request
import cv2
import time
from datetime import datetime
import os

# imports for feature drawing & extraction
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
mp_hands = mp.solutions.hands # type: ignore

# the trained SVM modeL
from libsvm.svmutil import *
from libsvm.svm import *

# data handling
import numpy as np
from string import ascii_uppercase, ascii_lowercase

# timer for capture
from threading import Thread


'''
Countdown class
    - Has 2 attributes, duration and a label
    - the duration is the time in seconds that the countdown lasts for
    - It has a method, work(), that counts down 'duration' seconds by updating the 'label' attribute 
    - the work() methods uses time.sleep(), so it should be spawned on another thread
    - This class is used to show the countdown on the camera output before the camera starts capturing
'''
class Countdown:
    def __init__(self, duration: int) -> None:
        if duration <= 0:
            raise ValueError
        self.duration = duration
        self.label = str(duration)
    
    def work(self):
        counter = self.duration
        while counter > 0:
            time.sleep(1)
            counter -= 1
            self.label = str(counter)

global countdown_thread, countdown
countdown = Countdown(3)
countdown_thread = Thread(target=countdown.work)

# features of each frame
features = np.zeros(63)

class Store:
    '''
    Constructor
    recordCount: int: the # of records to store
    recordDims: list[int]: the dimensions of each record
    '''
    def __init__(self, recordCount, recordDims) -> None:
        self.store_ = np.zeros([recordCount]+recordDims)
        self.storeIdx_ = 0
        self.recordShape_ = self.store_.shape[1:]
        self.storeSize_ = recordCount
        self.labels_ = ['' for _ in range(recordCount)]

    # x is a record to store, a numpy array of dimension self.store_.shape[1:]
    def storeRecord(self, x, y):
        if self.storeIdx_ == self.storeSize_:
            raise MemoryError
        if x.shape != self.recordShape_:
            raise ValueError()
        self.store_[self.storeIdx_] = x
        self.labels_[self.storeIdx_] = y
        self.storeIdx_ += 1

    def saveStore(self):
        if self.storeIdx_ == 0:
            print("Nothing to save; store is empty")
            return
        for alphabet in set(self.labels_):
            idxs = [i for i, j in enumerate(self.labels_) if j == alphabet]
            folderPath = './train_data_6/'+alphabet
            if not os.path.exists(folderPath):
                os.mkdir(folderPath)
            fileName = alphabet + datetime.now().strftime("%m_%d_%y %H-%M-%S") + ".txt"
            np.savetxt(
                folderPath + '/' + fileName, 
                np.concatenate([self.store_[i] for i in idxs])
            )

    def canStore(self):
        return self.storeIdx_ < self.storeSize_
    
    def isEmpty(self):
        return self.storeIdx_ == 0

    def flushStore(self):
        self.storeIdx_ = 0

    def deleteEntry(self):
        pass    

# global store, frameIdx, current_label
# store = np.zeros((20, 1000, 63))
store = Store(20, [1000, 63])
# uppercase alphabets representing the labels of the sets of training samples collected
one_recording = np.zeros((1000, 63))
current_label = ''
frameIdx = 0

def saveStore():
    global store, storeIdx, labels
    # each unique alphabet is a separate file to save
    for alphabet in set(labels):
        idxs = [i for i, j in enumerate(labels) if j == alphabet]
        folderPath = './train_data/'+alphabet
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        fileName = alphabet + datetime.now().strftime("%m_%d_%y %H-%M-%S") + ".txt"
        np.savetxt(
            folderPath + '/' + fileName, 
            np.concatenate([store[i] for i in idxs])
        )

global m
m = svm_load_model('a2z_v9_model.model')

global capture_features, toggle_prediction, switch, rec
capture_features = False
toggle_prediction = 0
switch = 1
rec = 0

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

global camera
camera = cv2.VideoCapture(0)

'''
the generator used to yield successive frames from the camera,
    then encoded as a JPG, and yielded

this generator is used to stream the video from the python application to the webpage

see more in the video_feed route
'''
def gen_frames():  # generate frame by frame from camera
    global out, capture_features,rec_frame, m, store, storeIdx, frameIdx, labels, countdown, countdown_thread
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (0, 0, 0)
    thickness = 2
    idxs = {i: (3*i, 3*i+1, 3*i+2) for i in range(21)}
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
        
        # feature extraction & updating frame
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not results.multi_hand_landmarks:
            _, buffer = cv2.imencode('.bmp', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')
            continue
        
        # if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        if capture_features:
            for i, j in idxs.items():
                features[j[0]] = hand_landmarks.landmark[i].x
                features[j[1]] = hand_landmarks.landmark[i].y
                features[j[2]] = hand_landmarks.landmark[i].z
            if countdown_thread.is_alive():
                cv2.putText(frame, countdown.label, org, font, font_scale, color, thickness, cv2.LINE_AA)
            else:
                if frameIdx < 1000:
                    converted, __ = gen_svm_nodearray(features)

                    label = libsvm.svm_predict(m, converted)
                    label = chr(int(label) + 65) + ' (Recording)'
                    cv2.putText(frame, label, org, font, font_scale, color, thickness, cv2.LINE_AA)
                    
                    # cv2.putText(frame, 'Recording...', org, font, font_scale, color, thickness, cv2.LINE_AA)
                    
                    one_recording[frameIdx] = features
                    frameIdx += 1
                else:
                    capture_features = False
                    store.storeRecord(one_recording, current_label)
                    frameIdx = 0
                    print('done capturing')
        # Convert a Python-format instance to svm_nodearray, a ctypes structure
        elif toggle_prediction:
            for i, j in enumerate([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]):
                features[j] = hand_landmarks.landmark[i].x
                features[j+1] = hand_landmarks.landmark[i].y
                features[j+2] = hand_landmarks.landmark[i].z

            converted, __ = gen_svm_nodearray(features)
            label = libsvm.svm_predict(m, converted)
            label = chr(int(label) + 65)
            cv2.putText(frame, label, org, font, font_scale, color, thickness, cv2.LINE_AA)
            label = ', '.join([str(features[0]*100)[:5], str(features[1]*100)[:5], str(features[2]*100)[:5]])
            cv2.putText(frame, label, (org[0], org[1]+50), font, 0.5, color, 1, cv2.LINE_AA)
            

        _, buffer = cv2.imencode('.bmp', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', parts = {"video": True, "alphabet": False}, msg = '')

@app.route('/test/alert')
def test_alert():
    return render_template('index.html', parts = {"video": False, "alphabet": False}, msg = 'This is a test alert for the alert parameter')

    
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
    global switch, camera, capture_features, toggle_prediction, frameIdx, current_label, store
    '''
    the buttons in the GUI are part of an HTML form (each one triggering a submit) that makes a POST request to this route
    '''
    if request.method == 'GET':
        return render_template('index.html', parts={"video": True, "alphabet": False}, msg = '')
    
    choice = request.form.get('click')

    # logic reaches here if request is a POST
    if choice == 'Capture_alphabet_samples':
        if store.canStore():
            # capture_features = True
            # frameIdx = 0
            # this response will cause the webpage to show a form to enter the label (aphabet) for the new recording
            return render_template('index.html', parts={"video": False, "alphabet": True}, msg = '')
        else:
            return render_template('index.html', parts={"video": False, "alphabet": False}, msg = 'Store full, please save samples before adding new ones')
            
    
    elif choice == 'Toggle_alphabet_prediction':
        toggle_prediction = not toggle_prediction 
        return render_template('index.html', parts={"video": True, "alphabet": False}, msg = '')

    elif choice == 'Save_training_samples':
        if store.isEmpty():
            return render_template('index.html', parts={"video": False, "alphabet": False}, msg = 'No recording avaiable to save.')
        store.saveStore()
        store.flushStore()
        return render_template('index.html', parts={"video": False, "alphabet": False}, msg = 'Save complete.')
    
    elif choice == 'Stop/Start_camera':
        if camera.isOpened():
            camera.release()
            cv2.destroyAllWindows()
            return render_template('index.html', parts={"video": False, "alphabet": False}, msg = '')
        else:
            camera = cv2.VideoCapture(0)
            return render_template('index.html', parts={"video": True, "alphabet": False}, msg = '')

    return render_template('404.html')

@app.route('/alphabet', methods = ['POST'])
def alphabet():
    global current_label, capture_features, countdown_thread
    action = request.form.get('action')
    if action == 'Submit':
        char = request.form.get('alphabet')
        if char is None:
            return render_template('index.html', parts={"video": False, "alphabet": True}, msg = 'Please enter a valid alphabet')
        char = char.upper()
        if char not in ascii_uppercase or char == '':
            return render_template('index.html', parts={"video": False, "alphabet": True}, msg = 'Please enter a valid alphabet')
        current_label = char
        countdown_thread = Thread(target=countdown.work)
        countdown_thread.start()
        capture_features = True
    return render_template('index.html', parts={"video": True, "alphabet": False}, msg = '')

if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()  