
import numpy as np
from correlation_tracker import CorrelationTracker
from data_association import associate_detections_to_trackers


class Sort:

  def __init__(self,max_age=7,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
  

  def update(self,dets,img = None):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
   
    self.frame_count += 1
    trks = np.zeros((len(self.trackers),5))
    
    """
    trks   : array of trackers' position with the score of each tracker
    to_del : array that contain indices of trackers need to be deleted as they are invalid 
    ret    : array of returned trackers [pos,id]
    colors : list of colors of the bounding boxes
    """
    to_del = []
    ret = []
    colors=[]
    
    
    #get predicted locations from existing trackers.
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(img) 
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
      
    #Compare detections to trackers and fil the matched, unmatched_dets, unmatched_trks lists
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0],img) ## for dlib re-intialize the trackers ?!

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      trk = CorrelationTracker(dets[i,:],img)
      self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
      if dets == []:
        trk.update([],img)   
      d = trk.get_state()
      if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
        ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
        colors.append(trk.color)
      i -= 1
      #remove dead tracklet
      if(trk.time_since_update > self.max_age):
        self.trackers.pop(i)
      else:
        ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
        colors.append(trk.color)
          
    if(len(ret)>0):
      return np.concatenate(ret),colors
    return np.empty((0,5)),[]