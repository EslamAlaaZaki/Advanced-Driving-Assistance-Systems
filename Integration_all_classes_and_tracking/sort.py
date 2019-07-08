"""
    SORT: A Simple, Online and Realtime Tracker
"""
from __future__ import print_function
import numpy as np
from kalman_tracker import KalmanBoxTracker
from data_association import associate_detections_to_trackers



class Sort(object):
  def __init__(self,classlabel,max_age = 7 , min_hits = 2):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.classlabel=classlabel

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    if dets==[] and self.trackers==[]:
      return []
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i,:],self.classlabel)
      self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        i -= 1
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits )):
          ret.append(d.tolist()+[trk.id])
        #remove dead tracklet
        elif(trk.time_since_update > self.max_age):
          self.trackers.pop(i) 
        else:
          ret.append(d.tolist()+[trk.id])
          
    if(len(ret)>0):
      return ret
    return []
    
