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

    def __init__(self):
        #Create instance of the SORT tracker]
        self.personTracker =  Sort("person")  
        self.carTracker =  Sort("car") 
        self.motorbikeTracker =  Sort("motorbike") 
        self.busTracker =  Sort("bus")
        self.trafficLightTracker =  Sort("traffic light") 
        self.stopSignTracker =  Sort("stop sign") 
        
        self.__dict__.update(self._defaults) # set up default values
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
        outClasses=["person","car","motorbike","bus","traffic light","stop sign"]
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            # working on traffic elements!
            if predicted_class not in outClasses : continue

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
    #def __del__(self):
      #self.sess.close()

    ##########################################################################

def Processing(yolo, frame):
  """ 
   inputs : 
          yolo : instance of yolo detector class
          frame: get frame by frame to process on  
  """
 
  #Output lists initialization 
  bboxes=[]
  scores=[]
  labels=[]
       
  #converting image type from nparray(cv2) to array(PIL) type
  image = Image.fromarray(frame)
        
  #detect objects in the current frame
  bboxes,scores,labels= yolo.detect_image(image)
        
  #create detections list for updating trackers    
        
  PersonDetections=[]                      # [[xmin,ymin,xmax,ymax,score],[],....]
  carDetections=[]
  motorbikeDetections=[]
  busDetections=[]
  trafficLightDetections=[]
  stopSignDetections=[]
  for i in range(0,len(bboxes)):
    (startY,startX,endY,endX) = bboxes[i]
    temp=[startX,startY,endX,endY,scores[i]]
    if labels[i] =="person":
      PersonDetections.append(temp)
    elif labels[i] =="car":
      carDetections.append(temp)
    elif labels[i] =="motorbike":
      motorbikeDetections.append(temp)
    elif labels[i] =="bus":
      busDetections.append(temp)
    elif labels[i] =="traffic light":
      trafficLightDetections.append(temp)
    elif labels[i] =="stop sign":
      stopSignDetections.append(temp)
  
  PersonDetections=np.asarray(PersonDetections)
  carDetections=np.asarray(carDetections)
  motorbikeDetections=np.asarray(motorbikeDetections)
  busDetections=np.asarray(busDetections)
  trafficLightDetections=np.asarray(trafficLightDetections)
  stopSignDetections=np.asarray(stopSignDetections)
  
  #update trackers 
  persons=yolo.personTracker.update(PersonDetections)
  cars=yolo.carTracker.update(carDetections)
  motorbikes=yolo.motorbikeTracker.update(motorbikeDetections)
  buses=yolo.busTracker.update(busDetections)
  trafficLights=yolo.trafficLightTracker.update(trafficLightDetections)
  stopSigns=yolo.stopSignTracker.update(stopSignDetections)
  
  classeslabels=(["person"]*len(persons))+(["car"]*len(cars))+(["motorbike"]*len(motorbikes))+(["bus"]*len(buses))+(["traffic light"]*len(trafficLights))+(["stop sign"]*len(stopSigns))
  
  outputBoxes=persons+cars+motorbikes+buses+trafficLights+stopSigns
  
  return outputBoxes,classeslabels
