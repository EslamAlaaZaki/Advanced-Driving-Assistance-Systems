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
from sort import Sort
from imutils.video import FPS

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
    
  #Create instance of the SORT tracker
  tracker =  Sort() 
        
  #Output lists initialization 
  bboxes=[]
  scores=[]
  labels=[]
    
  #starting fps calculator object 
  fps = FPS().start()
    
  while True:
    
    return_value, frame = vid.read()
    if not return_value: break
          
    #resize video by reducing video frames size to decrease detection time if needed 
    if resize :
      frame = cv2.resize(frame, video_size, interpolation = cv2.INTER_AREA)
        
    #converting image type from nparray(cv2) to array(PIL) type
    image = Image.fromarray(frame)
        
    #detect objects in the current frame
    bboxes,scores,labels= yolo.detect_image(image)
        
    #create detections list for updating trackers    
        
    detections=[]                      # [[xmin,ymin,xmax,ymax,score],[],....]
    for i in range(0,len(bboxes)):
      (startY,startX,endY,endX) = bboxes[i]
      temp=[startX,startY,endX,endY,scores[i]]
      detections.append(temp)
      
    detections=np.asarray(detections)
        
    #update trackers 
    trackers,colors = tracker.update(detections,frame
                                    )
        
    #draw bounding boxes 
    for i in range(0,len(trackers)):
      d = trackers[i].astype(np.int32)
      cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]),colors[i],                         (frame.shape[0]+frame.shape[1])// 300)
      text = str("%d" % d[4]) #object_id
      cv2.putText(frame, text ,(d[0], d[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,201), thickness=2)
        
       
    fps.update()
    #asses each frame to the output video          
    out.write(frame)
               
  fps.stop()
  print("FPS :" , fps.fps())
  out.release()
  yolo.close_session()

