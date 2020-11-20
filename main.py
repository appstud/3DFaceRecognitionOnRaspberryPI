import cv2
import os
import numpy as np
import pdb
import scipy.io as sio
import imutils
import frontalize
import camera_calibration as calib
import face_recognition_HAAR
import pickle
from imutils.video import VideoStream
from picamera.array import PiRGBArray
from PIL import Image
import time
import os
from flask import Response
from flask import Flask
from flask import render_template
import threading
from imutils.video.pivideostream import PiVideoStream

from argparse import ArgumentParser

lock=threading.Lock()
outputFrame=None
registeredFace=None
app=Flask(__name__)
vs=PiVideoStream(resolution=(320,240),framerate=32).start()
time.sleep(2)
    
def process():
    global vs, outputFrame, lock, registeredFace,lock

    imgArray,labels=face_recognition_HAAR.readImageOnly("./Database_aligned_3D")
    print("Database contains ",len(imgArray)," images consider adding/removing images to balance accuracy and speed \n")
    names=list(set(labels))
    names.sort()
    
    print("Available persons in database:"+ str(names))
    labelNumPersonDict = { name : i for i,name in  enumerate(names) }
    labelNumPerson=[labelNumPersonDict.get(n, n) for n in labels]
    
    model3D = frontalize.ThreeD_Model( "./frontalization_models/model3Ddlib.mat", 'model_dlib')
    model3D.ref_U=imutils.resize(model3D.ref_U[52:250,91:225,:],height=60) 
    
    eyeMask = np.asarray(sio.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    eyeMask=imutils.resize(eyeMask[52:250,91:225],height=60) 
    
    model3D.out_A=np.asmatrix(np.array([[291.0,0,158.0],[0,291.0,117.0],[0,0,1]]))
    model3D.distCoeff=np.array([0.03885,-0.2265,0.005354,-0.0016,0.1353])
    #model3D.out_A=np.asmatrix(np.array([[0.5*506.696672,0,0.5*324.202],[0, 0.5*506.3752, 0.5*245.7785096],[0,0,1]]), dtype='float32') 
    #model3D.distCoeff=None
   
   
    print("Training lbph recognizer")
    recognizer=cv2.face.LBPHFaceRecognizer_create(1,8,8,8)
    recognizer.train(imgArray,np.array(labelNumPerson))
    print("Training finished, begin streaming...") 
    while(True):

        frame = vs.read()
        image,front= face_recognition_HAAR.performFaceRecognitionWithFrontalisationV2(frame,recognizer, model3D, eyeMask,labelNumPersonDict)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        with lock:
            outputFrame=np.copy(image)
            registeredFace=np.copy(front)


    
       
   
def generate():
    global outputFrame,lock
    while( True):
        with lock:
            if(outputFrame is None):
                continue

            (flag,encodedImage)=cv2.imencode(".jpg",outputFrame)
            if not flag:
                continue
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' +bytearray(encodedImage) +b'\r\n')


def generateFront():
    global registeredFace,lock
    while( True):
        with lock:
            if(registeredFace is None):
                continue
            (flag,encodedImage)=cv2.imencode(".jpg",registeredFace)
            if not flag:
                continue
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' +bytearray(encodedImage) +b'\r\n')



@app.route("/video_feed")
def video_feed():
    return Response(generate(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_front")
def video_front():
    return Response(generateFront(),mimetype="multipart/x-mixed-replace; boundary=frame")





@app.route("/")
def index():
    return render_template("indexV2.html")


if(__name__=='__main__'):

    # construct the argument parser and parse command line arguments
    ap = ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,help="port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
                         
    t=threading.Thread(target=process)
    t.daemon=True
    t.start()
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

vs.stop()
