import random
import datetime
from   copy import deepcopy
import math
import bisect
try:
  from .OPL3 import OPL3
except:
  from OPL3 import OPL3

random.seed(datetime.datetime.now().timestamp())

opl3 = OPL3()

# single frame of a wav2vgm project file
class W2VFrame:
  def __init__(self, orig_spect=None, synth_cfg_bytes=None ):
    global opl3
    self.orig_spect = None
    self.synth_cfg_bytes = None
    if (orig_spect is not None) and (synth_cfg_bytes is not None):
      self.orig_spect = deepcopy(orig_spect)
      self.synth_cfg_bytes = deepcopy(synth_cfg_bytes)
      self.fitness, self.synth_spect = opl3.fitness(deepcopy(self.orig_spect), deepcopy(self.synth_cfg_bytes))
      self.synth_cfg_vect = opl3.rfToV(self.synth_cfg_bytes)
    else:
      self.fitness, self.synth_spect = -1, None
      self.synth_cfg_vect = None
  def __str__(self):
    s=f'fit={self.fitness}'
    return s
  def serialize(self):
    pj = {
      'orig_spect':self.orig_spect,
      'synth_cfg_bytes':self.synth_cfg_bytes,
      'fitness':self.fitness,
      'synth_cfg_vect':self.synth_cfg_vect,
      'synth_spect':self.synth_spect,
    }
    return pj

# project file for a WAV2VGM
class W2VProj:
  def __init__(self,proj_name,num_frames,orig_spects=None,synth_cfgs_bytes=None):
    self.proj_name = proj_name
    self.num_frames = num_frames
    self.frames = []
    for i in range(0,num_frames):
      orig_spect = None
      synth_cfg_bytes = None
      if orig_spects is not None:
        orig_spect = orig_spects[i]
        if synth_cfgs_bytes is not None:
          synth_cfg_bytes = synth_cfgs_bytes[i]
        else:
          synth_cfg_bytes = self.initSynthCfg(orig_spect)
      else:
        orig_spect = None
        synth_cfg_bytes = None
      self.frames.append( W2VFrame(orig_spect, synth_cfg_bytes))
  def initSynthConfig(self, orig_spect):
    '''Make a default synth configuration'''
    global opl3
    rf = opl3.initRegFile()
    return rf
  def __str__(self):
    s=f'w2VProj ({self.num_frames} frames):\n'
    for i,f in enumerate(self.frames):
      s+=f'    {i:4d}: {str(f)}'
    return s
  def serialize(self):
    pj = {
      'proj_name': self.proj_name,
      'num_frames': self.num_frames,
      'frames': []
    }
    for f in self.frames:
      pj['frames'].append(f.serialize())
    return pj



