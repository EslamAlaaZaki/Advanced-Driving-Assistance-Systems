
from dlib import correlation_tracker, rectangle
from random import randint
class CorrelationTracker:

  count = 0
  def __init__(self,bbox,img):
    self.tracker = correlation_tracker()
    rect =rectangle(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
    self.tracker.start_track(img,rect)
    self.confidence = 0. # measures how confident the tracker is! (a.k.a. correlation score)

    self.time_since_update = 0
    self.id = CorrelationTracker.count
    CorrelationTracker.count += 1
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.color=(randint(0, 255),randint(0, 255),randint(0,255))

  def predict(self,img):
    self.confidence = self.tracker.update(img)

    self.age += 1
    if (self.time_since_update > 0):
      self.hit_streak = 0
    self.time_since_update += 1

    return self.get_state()

  def update(self,bbox,img):
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1

    '''re-start the tracker with detected positions (it detector was active)'''
    if bbox != []:
      self.tracker.start_track(img,rectangle(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])))
    '''
    Note: another approach is to re-start the tracker only when the correlation score fall below some threshold
    i.e.: if bbox !=[] and self.confidence < 10.
    but this will reduce the algo. ability to track objects through longer periods of occlusions.
    '''

  def get_state(self):
    pos = self.tracker.get_position()
    return [pos.left(), pos.top(),pos.right(),pos.bottom()]