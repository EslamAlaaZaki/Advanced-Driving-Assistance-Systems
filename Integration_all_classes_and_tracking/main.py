#_____FPS calculator________
from imutils.video import FPS

#_____detection and tracking___
from yolo import YOLO, Processing
import cv2
import numpy as np

#________depth_______
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn import ensemble

#__________ parameters for depth _____________

presize=np.array([1392,512,1392,512])
mean=[612.767785 ,162.28982962, 661.18900404, 259.03917091]
std=[280.5976799, 16.85319477 ,284.90970296 ,52.1789747 ]
mean_y=[18.56490427]
std_y=[12.45210228]



#load depth model

#load model
depth_call = pickle.load(open('Integration_all_classes_and_tracking/model.sav','rb'))

def predict(box,Size,zoom=1):
    size=np.array([Size[0],Size[1],Size[0],Size[1]])
    box=(box*presize)/(zoom*size) 
    box=(box-mean)/std
    box=np.reshape(box,(1,-1))
    return float(((std_y* depth_call.predict(box))+mean_y))


def Draw(objects,frame,labels):
  """ 
  objects : list bounding boxes and objects' id 
  colors  : colors of bounding boxes
  frame   : frame to draw boxes in 
  """
  #draw bounding boxes 
  classescolors={"person":(179,41,142),"car":(0,0,255),"motorbike":(0,255,0),"bus":(255,255,0),"traffic light":(50,125,255),"stop sign":(125,125,125)}
  
  for i in range(0,len(objects)):
    d = objects[i]
    d=[int(i) for i in d]
    text=""
    
    if labels[i]=="person":
      text = "#"+str("%d" % d[4])+"person"+":"+"{:.1f}".format(predict([d[0],d[1],d[2],d[3]],frame.shape,zoom=1))+" m"
    else:
      text =labels[i]+":"+"{:.1f}".format(predict([d[0],d[1],d[2],d[3]],frame.shape,zoom=1))+" m"
      
    color=classescolors[labels[i]]
    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]),color,                      (frame.shape[0]+frame.shape[1])// 300)
    cv2.putText(frame, text ,(d[0], d[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), thickness=2)


detector=YOLO()

if __name__ == '__main__':
  
  #taking frame by frame from input video
  vid = cv2.VideoCapture("input_videos/test.avi")
  if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
    
  #initialize output video's parameters 
  video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
  video_fps       = vid.get(cv2.CAP_PROP_FPS)
  video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  #Set Output video parameters  
  out = cv2.VideoWriter("integration_all_tracking_output3.mp4", video_FourCC, video_fps, video_size)
  
   #starting fps calculator object 
  fps = FPS().start()
    
  while True:
    return_value, frame = vid.read()
    if not return_value: break
    objects,labels=Processing(detector,frame)
    Draw(objects,frame,labels)
    fps.update()     
    out.write(frame)
               
  fps.stop()
  print("FPS :" , fps.fps())
  out.release()
    
 
