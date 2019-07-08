"""
Class definition of YOLO_v3 style detection model on image and video
"""

#Model_Imports 
import os
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
from timeit import default_timer as timer


from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

# Tracking_Imports
import imutils
import dlib
from imutils.video import FPS


def init_tracker_KCF( frame, bboxes ):
    trackers = []
    for box in bboxes:
        startY,startX,endY,endX= box
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame,(startX,startY,endX-startX,endY-startY))
        trackers.append(tracker)
    return trackers

  
def update_tracker_KCF(frame,trackers,bboxes,scores,labels):
    for i in range (0,len(trackers)):
        _,newbox = trackers[i].update(frame)
        # unpack the position object
        if newbox !=(0.0,0.0,0.0,0.0):
            bboxes[i]=(int(newbox[1]),int(newbox[0]), int(newbox[1] + newbox[3]),int(newbox[0] + newbox[2]))
    return delele_box(frame , trackers,bboxes,scores,labels)



def init_tracker( frame, bboxes ):
    trackers = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for box in bboxes:
        startY,startX,endY,endX= box
        #print(box)
        # construct a dlib rectangle object from the bounding
        # box coordinates and start the correlation tracker
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        t.start_track(rgb, rect)
        trackers.append(t)
    return trackers

def delele_box (frame ,trackers,bboxes,scores,labels):
    flag=False
    z=0
    for i in range(0,len(trackers)):
        Y,X,endY,endX=bboxes[z]
        if (X > frame.shape[1]) or (Y > frame.shape[0]) or(X<0) or(Y<0):
            del trackers[z]
            del bboxes[z]
            del scores[z]
            del labels[z]
          
            flag=True
            z=z-1
        z=z+1
    return flag



def update_tracker(frame , trackers,bboxes,scores,labels):
    # update the tracker and grab the position of the tracked object
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for i in range (0,len(trackers)):
        trackers[i].update(rgb)
        pos = trackers[i].get_position()
        # unpack the position object
        bboxes[i]=(int(pos.top()),int(pos.left()),int(pos.bottom()),int(pos.right()))
    
    return delele_box(frame , trackers,bboxes,scores,labels)



# Object detection class using keras platform with yolo3
class YOLO(object):
    # using tiny version of yolo3
    _defaults = {
        "model_path": 'models/v4_tiny/tiny.h5',
        "anchors_path": 'models/v4_tiny/tiny_anchors.txt',
        "classes_path": 'models/v4_tiny/coconames.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
   
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

       
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    ##################################################################################
      
    def detect_image(self, image):
        """ 
        returns : 
               bboxes : positions of detected objects (top, left, bottom, right)
               scores : probabilty of correctness of detection
               labels : type of detected object
               """
               
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')    
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # initialize returned lists        
        bboxes=[] 
        scores=[]
        labels=[]
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            # working on person detection only !
            if predicted_class != "person" : continue

            box = out_boxes[i]
            scores.append(out_scores[i])
            labels.append(predicted_class)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            bboxes.append((top, left, bottom, right))            

        return bboxes,scores,labels
      
    ##########################################################################
    
    def close_session(self):
        self.sess.close()

    ##########################################################################

def detect_video(yolo, video_path, output_path=""):
  """ 
   inputs : 
          yolo : instance of yolo detector class
          video_path : path to input video
          output_path : path to detected video
          """
  #taking frame by frame from input video
  vid = cv2.VideoCapture(video_path)
  if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
    
  #initialize output video's parameters 
  video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
  video_fps       = vid.get(cv2.CAP_PROP_FPS)
  video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
  # resize input images in case large one to make model faster   
  resize=False
    
  if video_size[0]>640:
    resize=True      
    video_size=(640,int( (video_size[1]/video_size[0])*640 ))
      
  #Set Output video parameters  
  out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    
  counter=0
  maxNumberOfTrackers=6
  detectionAndTrackingCircularSize=6
  trackers=[]
  bboxes=[]
  scores=[]
  labels=[]
  
  fps = FPS().start()
  while True:
    return_value, frame = vid.read()
    if not return_value: break
    if resize :
      frame = cv2.resize(frame, video_size, interpolation = cv2.INTER_AREA)
    image = Image.fromarray(frame)
    if(counter==0) or (len(bboxes)>maxNumberOfTrackers):
      bboxes,scores,labels = yolo.detect_image(image)
      if(len(bboxes)<=maxNumberOfTrackers):
        trackers.clear()
        trackers=init_tracker_KCF(frame,bboxes)
        counter=counter+1
            
    else:
      update_tracker_KCF(frame,trackers,bboxes,scores,labels)
      counter=counter+1
        
    if(counter==detectionAndTrackingCircularSize):
      counter=0
        
    for i in range(0,len(bboxes)):
      (startY,startX,endY,endX) = bboxes[i]
       #Depth = DEP_func([startX,startY,endX,endY],[frame.shape[1],frame.shape[0]])
      cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,255),                     (frame.shape[0]+frame.shape[1])// 300)
      #text = labels[i]+","+str("%.2f" % scores[i])
      #add text to % info
      """cv2.putText(frame, +str(Depth)+" M", (startX, startY - 15),                 cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,201), thickness=2)"""
      #cv2.putText(frame, text ,(startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX,                            0.3,(0,0,201), thickness=2)
        
    fps.update()
    #asses each frame to the output video          
    out.write(frame)
               
  fps.stop()
  print("FPS :" , fps.fps())
  out.release()
  yolo.close_session()
