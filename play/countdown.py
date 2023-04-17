'''
the file to add logic and run the GUI
'''

from flask import Flask, render_template, Response, request
import cv2
import datetime
from time import sleep

# timer for capture
from threading import Thread, Lock

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

# start camera
camera = cv2.VideoCapture(0)

class Countdown:
    def __init__(self, duration: int) -> None:
        if duration <= 0:
            raise ValueError
        self.duration = duration
        self.label = str(duration)
        self.lock = Lock()
        self.wait = False
        self.doWork = False

    def work_manager(self):
        while True:
            self.lock.acquire()
            
    
    def work(self):
        counter = self.duration
        self.start_lock.acquire()
        while self.doWork:
            self.lock.acquire()
            if not self.doWork:
                return
            while counter > 0:
                sleep(1)
                counter -= 1
                self.label = str(counter)
            self.lock.release()
            sleep(0.5)
        

global start_record, start_record_countdown, t
start_record = False
countdown = Countdown(3)
countdown_thread = Thread(target=countdown.work)

'''
the generator used to yield successive frames from the camera,
    then encoded as a JPG, and yielded

this generator is used to stream the video from the python application to the webpage

see more in the video_feed route
'''
def gen_frames():  # generate frame by frame from camera
    global start_record, start_record_capture, countdown, countdown_thread
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (0, 0, 0)
    thickness = 2
    while True:
        success, frame = camera.read() 
        if not success: 
            break

        if start_record:
            if countdown_thread.is_alive():
                cv2.putText(frame, countdown.label, org, font, font_scale, color, thickness, cv2.LINE_AA)
        
        # feature extraction & updating frame
        frame.flags.writeable = False

        _, buffer = cv2.imencode('.bmp', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', parts = {"video": True, "alphabet": False}, msg = '')
    
    
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
    global start_record, countdown_thread
    choice = request.form.get('click')
    print('choice: ', choice)
    if choice == 'Capture':
        countdown_thread.start()
        start_record = True
    return render_template('index.html', parts={"video": True, "alphabet": False}, msg = '')

@app.route('/alphabet', methods = ['POST'])
def alphabet():
    return render_template('index.html', parts={"video": True, "alphabet": False}, msg = '')

if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()  