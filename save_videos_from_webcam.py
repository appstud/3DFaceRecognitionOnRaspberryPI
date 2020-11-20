import cv2
import os
import numpy as np
import imutils
import pickle
from imutils.video import VideoStream
from picamera.array import PiRGBArray
from PIL import Image
import time
import os
import threading
from imutils.video.pivideostream import PiVideoStream
from pathlib import Path

from argparse import ArgumentParser

   

def saveVideo(destination_folder):
    vs=PiVideoStream(resolution=(320,240),framerate=32).start()
    time.sleep(2)

    videoWriter=cv2.VideoWriter(os.path.join(destination,str(len(os.listdir(destination)))+".mp4"),cv2.VideoWriter_fourcc(*'XVID'),32,(320,240))
    print("Press CTRL C to stop recording...")
    while(True):
        try:
            frame = vs.read()
            videoWriter.write(frame)
        except KeyboardInterrupt:
            print("Saving video...")
            videoWriter.release()
            
            vs.stop()
            break
 
def process():
    global vs, outputFrame, lock, registeredFace,lock

    imgArray,labels=face_recognition_HAAR.readImageOnly("./Database_aligned_3D")
    names=list(set(labels))
    names.sort()
    
    print("Available persons in database:"+ str(names))
    labelNumPersonDict = { name : i for i,name in  enumerate(names) }
    labelNumPerson=[labelNumPersonDict.get(n, n) for n in labels]
    
    model3D = frontalize.ThreeD_Model( "./frontalization_models/model3Ddlib.mat", 'model_dlib')
    model3D.ref_U=imutils.resize(model3D.ref_U[52:250,91:225,:],height=60) 
    
    eyeMask = np.asarray(sio.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    eyeMask=imutils.resize(eyeMask[52:250,91:225],height=60) 
   
    model3D.out_A=np.asmatrix(np.array([[0.5*506.696672,0,0.5*324.202],[0, 0.5*506.3752, 0.5*245.7785096],[0,0,1]]), dtype='float32') 
    model3D.distCoeff=None
   
   
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


    
       
 
if(__name__=='__main__'):

    # construct the argument parser and parse command line arguments
    ap = ArgumentParser()
    ap.add_argument("--path_to_base_folder",  type=str, required=False, help="path to base folder",default="./Database")
    ap.add_argument("--name_of_person", type=str, required=True,help="name of the person",default="me")
    args = vars(ap.parse_args())
    
    path_to_base_folder=args["path_to_base_folder"]
    name_of_person=args["name_of_person"]
    destination=os.path.join(path_to_base_folder,name_of_person)
    Path(destination).mkdir(parents=True,exist_ok=True)

    saveVideo(destination)
