# Advanced-Driving-Assistance-Systems

# **This a graduation project in faculty of engineering Ain Shams university computer and system department**



## **Description:**

**In this project, we built an Advanced Driver Assistant (ADAS) system based on an android application.The application is developed to assist drivers, independently of the kind of car they are using, the type of road and the skill of the driver. The application main function is to analyze the road in front of the driver to detect any possible threats and notify him **.



**Note :**

this repo belongs to sub team for this project , this team works of pedestrian detection and tracking with depth estimation, we used  YOLO3 tiny version as a detector and we try three different trackers with it , Kalman Filter, Correlation Filter , and Kernelized Correlation Filter (KCF) . After that we used them with SORT Algorithm



**Requirements:**

- Python 3.5.2
- Keras 2.1.5
- Tensorflow 1.6.0
- OpenCV
- Dlib
- Numpy
- Filterpy
- CUDA

**Quick use**

- download kerase model from here and put it in model folder
- select which output you want correlation or Kalman or KCF with or without sort then go to the folder and run this command python yolo\_video.py - -input inputVideo.mp4 - - output outputVideo.mp4
- if you face any problem with model paths update it in the code
- if you want to run with depth go to the integration\_with\_depth folder then run python main.py /\* input video and output video have fixed path in main.py you can change it
- if you want to detect person &amp; motorbike &amp; bus &amp; car &amp; traffic light &amp; stop sign  go to Integration\_all\_classes\_and\_tracking folder then run python main.py
