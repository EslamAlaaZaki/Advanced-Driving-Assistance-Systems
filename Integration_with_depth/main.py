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
mean=[511.77739707 ,172.070199 ,  613.31818431, 245.51132065]
std=[275.86396727 , 22.69350105, 272.54653349 , 53.4001591 ]
mean_y=[27.42377063]
std_y=[17.47018457]


#load depth model

#load model
depth_call = pickle.load(open('Integration_with_depth/model.sav','rb'))

def predict(box,Size,zoom=1):
    size=np.array([Size[0],Size[1],Size[0],Size[1]])
    box=(box*presize)/(zoom*size) 
    box=(box-mean)/std
    box=np.reshape(box,(1,-1))
    return float(((std_y* depth_call.predict(box))+mean_y)*0.5)


def Draw(objects,colors,frame):
  """ 
  objects : list bounding boxes and objects' id 
  colors  : colors of bounding boxes
  frame   : frame to draw boxes in 
  """
  #draw bounding boxes 
  for i in range(0,len(objects)):
    d = objects[i].astype(np.int32)
    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]),colors[i],                       (frame.shape[0]+frame.shape[1])// 300)
    text = "#"+str("%d" % d[4])+":"+"{:.1f}".format(predict([d[0],d[1],d[2],d[3]],frame.shape,zoom=1))+" m" #object_id
    cv2.putText(frame, text ,(d[0], d[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,201), thickness=2)




detector=YOLO()
#Lane_detector=Lane_Model("Integration/lane_model_40_epoch.h5")

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
  out = cv2.VideoWriter("test_depth_video.mp4", video_FourCC, video_fps, video_size)
  
   #starting fps calculator object 
  fps = FPS().start()
    
  while True:
    return_value, frame = vid.read()
    if not return_value: break
    objects,colors=Processing(detector,frame)
    #Mask=Lane_detector.predict(frame,1)
    Draw(objects,colors,frame)
    #result = cv2.addWeighted(frame, 1, Mask, 1, 0)
    fps.update()
    #asses each frame to the output video          
    out.write(frame)
               
  fps.stop()
  print("FPS :" , fps.fps())
  out.release()
    
 
