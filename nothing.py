

from imutils.video.pivideostream import PiVideoStream

import cv2
import os
import dlib
import numpy as np
import pdb
import scipy.io as sio
import imutils
from picamera.array import PiRGBArray
from picamera import PiCamera 
import time
from PIL import Image
detector = dlib.get_frontal_face_detector()

def runFaceRecognition():    

    vs = PiVideoStream(resolution=(640,480),framerate=32).start()
    time.sleep(2)
    while(True):
        start=time.time() 
        image=vs.read()
        
                
        


if __name__ == "__main__":
    runFaceRecognition()


