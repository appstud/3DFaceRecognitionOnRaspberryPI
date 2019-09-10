import cv2
import os
import dlib
import numpy as np
import scipy.io as sio
import imutils
import frontalize
import camera_calibration as calib
import time


predictor_path =os.path.join(os.path.dirname(__file__),'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
goodAlignement_cascade = cv2.CascadeClassifier('cascadeV4.xml')
detector = dlib.get_frontal_face_detector()


def createDirectory(path) :
    try:
        os.mkdir(path)     
    except FileExistsError:
        pass
    
def readVideosAndLabels(directory,alignment="2D"):
    namesUnique=os.listdir(directory)
    labels=[]
    faceMatrix=None
    outPath=directory+"_aligned_"+alignment
    createDirectory(outPath)
    
    if(alignment!="2D"):
        model3D = frontalize.ThreeD_Model( "./frontalization_models/model3Ddlib.mat", 'model_dlib')
        eyemask = np.asarray(sio.loadmat('frontalization_models/eyemask.mat')['eyemask'])          
        eyemask=imutils.resize(eyemask[52:250,91:225],height=60)
        model3D.ref_U=imutils.resize(model3D.ref_U[52:250,91:225,:],height=60)
        
        
        model3D.out_A=np.asmatrix(np.array([[0.5*506.696672,0,0.5*324.202],[0,0.5*506.3752,0.5*245.7785096],[0,0,1]]), dtype='float32') #3x3
     

        model3D.distCoeff=None
    
    count=0
    for ind,folder in enumerate(namesUnique):
        createDirectory(os.path.join(outPath,folder))

        for path in os.listdir(os.path.join(directory,folder)):
            
            path=os.path.join(directory,folder,path)
            cap=cv2.VideoCapture(path)
            trackingLost=True
            
            while(True):
                
                ret,imageO=cap.read()
                
                
                if(not ret):
                    break
                landmarks,faceROI,trackingLost,image=trackFaceInANeighborhoodAndDetectLandmarks(np.copy(imageO),faceROI=[0, 0,imageO.shape[0]-1, imageO.shape[1]-1],drawBoundingBoxes=True)
                if(trackingLost):
                        continue
                for k,landmark in enumerate(landmarks):
                    
                    if(alignment=="2D"):
                        registeredFace=performFaceAlignment(imageO,landmark,cols=600,rows=600)
                    else:
                        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmark)
                        frontal_raw, registeredFace = frontalize.frontalize(imageO, proj_matrix, model3D.ref_U, eyemask)
                        xdir=2*np.array([-49.6694, -0.3201, 1.0163])
                        ydir=4*np.array([-0.9852,-3.1128,15.0628])
                        zdir=-np.array([-1.658,747.159,154.29])/5.0
                        origin=np.array([-0.0845, -74.7281, 27.2774])
                        image,_=model3D.drawCoordinateSystems(np.hstack((rmat,tvec)),image,_3Dpoints=np.array([origin,origin+xdir,origin+ydir,origin+zdir]))
                        #image=model3D.drawCandideMesh(np.hstack((rmat,tvec)),image)
                        
                    if(registeredFace is not None):
                        
                      
                        cv2.polylines(image,np.int32(landmark.reshape((-1,1,2))),True,(0,0,255),3)

                        box=goodAlignement_cascade.detectMultiScale(registeredFace, 1.1, 1,minSize=(32,32) )
                        
                        try:

                            """count=count+1
                            cv2.imwrite(os.path.join(outPath,folder,str(count)+'.jpg'), registeredFace)  
                            """
                            if(len(box)>0):
                                
                                cv2.imwrite(os.path.join(outPath,folder,str(count)+'.jpg'),imutils.resize(registeredFace,height=48))  
                                
                                cv2.rectangle(registeredFace,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,0,255),2)             
                                count=count+1
                        
                        except Exception as e:
                            print(e)
           
                   
               
               
            cap.release()
    return faceMatrix,labels 

def readImageOnly(directory):
    listOfImages=[] 
    labels=[]
    for ind,folder in enumerate(os.listdir(directory)):
        for path in os.listdir(os.path.join(directory,folder)):
            path=os.path.join(directory,folder,path)
            image=imutils.resize(cv2.imread(path), height=48)

            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            listOfImages.append(image)
            labels.append(folder)
            #print(folder)
       

    print(".......................................... \n")    
    return listOfImages,labels
    
def trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI,drawBoundingBoxes=True):
       
     N=25
     
     x_top_left=max(faceROI[0]-N,0)
     x_bottom_right=min(image.shape[0],faceROI[0]+faceROI[2]+N)
     y_top_left=max(faceROI[1]-N,0)
     y_bottom_right=min(faceROI[1]+faceROI[3]+N,image.shape[1])
     landmarkList=[]
     boundingBoxList=[]
     if(drawBoundingBoxes):
         cv2.rectangle(image,(y_top_left,x_top_left),(y_bottom_right,x_bottom_right),(0,255,0),1)
     trackingLost=True
     faceBoundingBoxes=[]
            
     try:
         box=face_cascade.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 1.1, 2,minSize=(50,50),maxSize=(55,55))
         for i in range(box.shape[0]):
             
             faceBoundingBoxes.append(tuple((i,dlib.rectangle(int(box[i,0]),int(box[i,1]),int(box[i,0]+box[i,2]), int(box[i,1]+box[i,3]))  )))
             
     except Exception as e:
         faceBoundingBoxes=()
     
        
     for k,d in faceBoundingBoxes:
         landmarks=np.zeros([68,2])
         shape = predictor(cv2.cvtColor(image[x_top_left:x_bottom_right,y_top_left:y_bottom_right,:],cv2.COLOR_BGR2GRAY),d)
         for i in range(0,68):
             landmarks[i,:]=np.array([shape.part(i).x+y_top_left,shape.part(i).y+x_top_left])
          
         faceROI=[x_top_left+d.top(), y_top_left+d.left(),d.height(), d.width()]
         
         trackingLost=False
         landmarkList.append(landmarks)
         boundingBoxList.append(faceROI)
         
         if(drawBoundingBoxes):      
             cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)             
     return landmarkList,boundingBoxList,trackingLost,image


def searchForFaceInTheWholeImage(image):
           
    faceROI=[0, 0,image.shape[0]-1, image.shape[1]-1]

    landmarks,faceROI,_,image=trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI)
    for landmark in landmarks:
        cv2.polylines(image,np.int32(landmark.reshape((-1,1,2))),True,(0,0,255),3)
    
    return image,landmarks,faceROI
    

   


def performFaceRecognitionWithFrontalisationV2(image,recognizer, model3D, eyemask,names):
    start=time.time() 
    imageWithLandmarks,landmarks,faceROIs=searchForFaceInTheWholeImage(np.copy(image))
    result=[]


    registeredFaceColor=255
    raw=255
    for landmark,faceROI in zip(landmarks,faceROIs):

        goodAlignment=False
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmark)
        raw,registeredFaceColor = frontalize.frontalize(image, proj_matrix, model3D.ref_U, eyemask)
        registeredFaceGray=cv2.cvtColor(registeredFaceColor,cv2.COLOR_BGR2GRAY)
        box=goodAlignement_cascade.detectMultiScale(registeredFaceGray, 1.1, 1,minSize=(32,32) )

        ###############
        registeredFace = imutils.resize(registeredFaceGray, height=48)
       

        try:
            cv2.rectangle(registeredFace,(box[0][1],box[0][0]),(box[0][1]+box[0][3],box[0][0]+box[0][2]),(255,0,255),2)
            goodAlignment=True
        except:

            pass
        if(goodAlignment): 
            pred,conf=recognizer.predict(registeredFace)
        
                 
            xdir=2*np.array([-49.6694, -0.3201, 1.0163])
            ydir=4*np.array([-0.9852,-3.1128,15.0628])
            zdir=-np.array([-1.658,747.159,154.29])/5.0
            origin=np.array([-0.0845, -74.7281, 27.2774])
            try:
                image,_=model3D.drawCoordinateSystems(np.hstack((rmat,tvec)),imageWithLandmarks,_3Dpoints=np.array([origin,origin+xdir,origin+ydir,origin+zdir]))
            except:
                pass
            
            identity=list(names.keys())[list(names.values()).index(pred)]
            cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)
            cv2.putText(image,identity ,(faceROI[1],faceROI[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=1)
        
    end=time.time()



    cv2.putText(image,"FPS: "+"{0:.2f}".format(round(1.0/(end-start),2)) ,(15,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),thickness=1)
    
    return image,registeredFaceColor


if __name__ =="__main__":

    print("Creating the database by 3D aligning the faces from the videos")

    readVideosAndLabels(".//Database",alignment="3D")

    print("Finished")

    
        
       
   
    
