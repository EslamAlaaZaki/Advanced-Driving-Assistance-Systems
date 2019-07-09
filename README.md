# Advanced-Driving-Assistance-Systems

# **This a graduation project in faculty of engineering Ain Shams university computer and system department**



## **Description:**

In this project, we built an Advanced Driver Assistant (ADAS) system based on an android application.The application is developed to assist drivers, independently of the kind of car they are using, the type of road and the skill of the driver. The application main function is to analyze the road in front of the driver to detect any possible threats and notify him.



**Note :**

this repo belongs to sub team for this project , this team works of pedestrian detection and tracking with depth estimation, we used  YOLO3 tiny version as a detector and we try three different trackers with it , Kalman Filter, Correlation Filter , and Kernelized Correlation Filter (KCF) . After that we used them with SORT Algorithm

**Our results:** [link](https://drive.google.com/drive/folders/1aCMPYg894rILTYFv1fiV96ErSH7MkmP9?usp=sharing)

**Requirements:**

- Python 3.5.2
- Keras 2.1.5
- Tensorflow 1.6.0
- OpenCV
- Dlib
- Numpy
- Filterpy
- CUDA
- nvidia gtx 850 or higher 

if you don't have these requirements you can use colab server but the results will be slower   

**Quick use**

- download kerase model from [here](https://drive.google.com/open?id=10Y61QqqqTFSuaOB6JJZ96pF3IwCtxieS) and put it in model folder
- select which output you want correlation or Kalman or KCF with or without sort then go to the folder and run this command python yolo\_video.py - -input inputVideo.mp4 - - output outputVideo.mp4
- if you face any problem with model paths update it in the code
- if you want to run with depth go to the integration\_with\_depth folder then run python main.py /\* input video and output video have fixed path in main.py you can change it
- if you want to detect person &amp; motorbike &amp; bus &amp; car &amp; traffic light &amp; stop sign  go to Integration\_all\_classes\_and\_tracking folder then run python main.py

**Training**

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---
