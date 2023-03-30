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

global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

hands = mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    max_num_hands = 1,
    min_tracking_confidence = 0.5
)

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
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if not success: 
            break
        if(capture):
            capture=0
            now = datetime.datetime.now()
            p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
            cv2.imwrite(p, frame)
        
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
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    '''
    the "multipart/x-mixed-replace; boundary=frame" tag is used to display media, 
        this tag also tells the browser to constantly update the media that it's displaying
    the media that is displayed is provided using the first argument to the response object, the generator gen_frames()
    each call to the function yields a new frame obtained from the camera
    '''
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/actions')
def actions():
    global switch, camera, capture, grey, neg, face
    '''
    the buttons in the GUI are part of an HTML form (each one triggering a submit) that makes a POST request to this route
    '''
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            capture = 1
        elif request.form.get('grey') == 'Grey':
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            face = not face 
            if face:
                time.sleep(4)   
        elif request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()  