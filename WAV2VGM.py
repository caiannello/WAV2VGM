# -----------------------------------------------------------------------------
# WAV2VGM
#
# Analyzes an input sound and decomposes it into a sum of sine waves.
#
# This is then resynthesized for Yamaha YMF262 (OPL3) music synth chip and
# output as VGM format file, playable on OPL3-capable sound cards or via 
# emulation. (e.g. via Winamp or VGMPlay(
#
# Craig Iannello 2024/11/04
#
# -----------------------------------------------------------------------------
import os
import sys
import time
import math
import struct
import gzip
import random
import datetime
from pprint import pprint

import pygame
from pygame.locals import *

import numpy as np
from scipy import signal
import torch
import torch.nn as nn  

from src import spect as sp
from src.OPL3 import OPL3
from src.w2vproj import W2VProj
from src import gene                # Import 
from src.model_definitions import OPL3Model  # AI model defs

from copy import deepcopy
# randomize the PRNG
random.seed(datetime.datetime.now().timestamp())
# -----------------------------------------------------------------------------
screen_width=1920 
screen_height=1080
#SLICE_FREQ = 86.1328125  # frequency at which the synth parameters are altered
                         # (exactly two specra per synth frame)
SLICE_FREQ = 91.552734375   # old setting for arduino
MAX_SYNTH_CHANS = 18   # polyphony of synth (num independent oscillators)
OPL3_MAX_FREQ = 6208.431  # highest freq we can assign to a voice (fnum, block)
SPECT_VERT_SCALE = 3  # set max vert spect axis to 7350 Hz max rather than the orig 22050 Hz  
GENE_MAX_GENERATIONS = 500
SPECS_PER_FRAME = 3  

# -----------------------------------------------------------------------------
NO_QUIT   = 0  # neither
SOFT_QUIT = 1  # esc-pressed
HARD_QUIT = 2  # window close

best_vects = []
# opl emulator, register data, and helper functions for converting
# between binary and float32[] config vector for AI training and inference

opl3 = OPL3()

# these two objects are redundant and should be removed,
# (the whole deterministic wav2vgm code can be cleaned
# up a whole lot! )

regofs = [  # one not accounting for bank
0x00,0x01,0x02,0x03,
0x04,0x05,0x0a,0x0b,
0x0c,0x0d,0x10,0x11,
0x12,0x13,0x14,0x15,
0x00,0x01,0x02,0x03,
0x04,0x05,0x0a,0x0b,
0x0c,0x0d,0x10,0x11,
0x12,0x13,0x14,0x15,
]

# these are the operator indexes per each of the 18 output channels
# in 2-op mode.
opidxs_per_chan = [
  (0,3),(1,4),(2,5),(6,9),(7,10),(8,11),(12,15),(13,16),(14,17),
  (18,21),(19,22),(20,23),(24,27),(25,28),(26,29),(30,33),(31,34),(32,35)]

# todo: make folders we need which don't exist
tmpfolder = 'temp/'
infolder = 'input/'
outfolder = 'output/'
modelsfolder = "models/"
trainfolder = "training_sets/"

origspect=None

pygame.init()
try:
    pygame.mixer.init(44100, -16, 1)
except pygame.error as e:
    print(str(e))
    exit()
print('One moment, one moment.')
if len(sys.argv)==2: 
  wavname = sys.argv[1]
else:
  # default input file during dev, if no file specified on the commandline.
  wavname = infolder
  wavname += 'HAL 9000 - Human Error.wav'
  #wavname += 'Grand Piano.wav'  
  #wavname += 'JFK Inaguration.wav'
  #wavname += 'Ghouls and Ghosts - The Village Of Decay.wav'
  #wavname += 'Portal-Still Alive.wav'
  #wavname += 'Amiga-Leander-Intro.wav'

output_vgm_name = outfolder+os.path.basename(wavname[0:-3])+"vgz"
temp_regfile_name = tmpfolder+os.path.basename(wavname[0:-3])+'bin'
origspect = sp.spect(wav_filename=wavname,nperseg=4096,quiet=False,clip=False)
clock = pygame.time.Clock()
screen=pygame.display.set_mode([screen_width,screen_height])#,flags=pygame.FULLSCREEN)
pygame.display.set_caption(f'Mimic - Spectrogram - {wavname}')
pygame_icon = pygame.image.load('src/spect_icon.png')
pygame.display.set_icon(pygame_icon)
pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
font = pygame.font.SysFont(None, 24)
smallfont = pygame.font.SysFont(None, 16)

roi=-1
cutoff_freq=-1
amp_cutoff=-1
amp_cutoff_y=-1
CLICKED_SPECTRUM = 0
CLICKED_SPECTROGRAM = 1
last_clicked = -1
T_AXIS_HEIGHT = 21
# -----------------------------------------------------------------------------
# draws the source spectrogram, scaled to the current display size and 
# SPECT_VERT_SCALE factor.
# -----------------------------------------------------------------------------
def drawSpect(spect = None, xx=0,yy=0,ww=screen_width,hh=screen_height):
  global screen, font, smallfont, roi
  xx=int(xx)
  yy=int(yy)
  ww=int(ww)
  hh=int(hh)
  # draw spectrum
  surf = pygame.transform.scale(spect.surf,(ww,hh*SPECT_VERT_SCALE-T_AXIS_HEIGHT))
  screen.blit(surf, (xx, yy-(SPECT_VERT_SCALE-1)*hh))
  # draw horizontal axis (time)  labels and tickmarks
  # hardcoded to 50 ticks per second
  secs_per_t_tick = 1/50
  pygame.draw.rect(screen, (40,40,40), (0,hh-T_AXIS_HEIGHT,screen_width,T_AXIS_HEIGHT))
  tstart = 0                    # todo zooming
  onscreen_dur = spect.dur_secs # todo  
  t=tstart
  while t<(tstart+onscreen_dur):    
    scr_x = int((t-tstart)*screen_width/(tstart+onscreen_dur))      
    ft = t-int(t)
    if ((ft<0.005) and (t>0.1)) or (abs(ft-0.5)<0.005):
      pygame.draw.line(screen, (255,255,255), (scr_x,hh-T_AXIS_HEIGHT+1),(scr_x,hh-T_AXIS_HEIGHT+6))
      l='{:0.2f}'.format(t)
      img = smallfont.render(l, True, (255,255,255))
      w=img.get_width()
      h=img.get_height()
      screen.blit(img, (int(scr_x-w/2), hh-1-h))    
    else:
      pygame.draw.line(screen, (192,192,192), (scr_x,hh-T_AXIS_HEIGHT+1),(scr_x,hh-T_AXIS_HEIGHT+3))
    t+=secs_per_t_tick
  # highlight selected spectrum on spectrogram (dotted)
  if roi>=0:
    scr_x = int(roi*screen_width/spect.maxrowidx)      
    scr_x1 = int((roi+1)*screen_width/spect.maxrowidx)-1
    scr_y = int(screen_height/4)
    scr_y1 = screen_height
    for y in range(scr_y,scr_y1):
      for x in range(scr_x,scr_x1):
        if ((x&1)+(y&1))&1:
          pygame.draw.line(screen, (255,255,255), (x,y),(x,y))
  # show freq cutoff horiz line on spectrogram (dotted)
  if cutoff_freq>=0:  
    hsr = spect.sample_rate/2
    y = screen_height-1 -T_AXIS_HEIGHT - (cutoff_freq * (screen_height-T_AXIS_HEIGHT) / hsr)
    for x in range(0,screen_width-1):
      if (x&1) == 0:
        pygame.draw.line(screen, (0,255,255), (x,y),(x,y))
  pygame.display.update()
# -----------------------------------------------------------------------------
# draw string, in white, to screen at (x,y)
# -----------------------------------------------------------------------------
def plotText(s,x,y):
  global screen, origspect
  img = font.render(s, True, (255,255,255))
  w=img.get_width()
  h=img.get_height()
  pygame.draw.rect(screen,(0,0,0),(x,y,w,h))
  screen.blit(img, (x, y))

def plotColorText(s,c,x,y):
  global screen, origspect
  img = font.render(s, True, c)
  w=img.get_width()
  h=img.get_height()
  pygame.draw.rect(screen,(0,0,0),(x,y,w,h))
  screen.blit(img, (x, y))

# -----------------------------------------------------------------------------
# given a single spectrum (a vertical slice of spectrogram), identify 
# all prominent peaks. Returns their freqs in Hz and heights in dBFS.
# -----------------------------------------------------------------------------
def getRankedPeaks(tsp, minv, maxv, do_limit_freqs=True,dist=5,prom=5):
  hh = int(screen_height/4)
  ll = len(tsp)
  hsr = origspect.sample_rate/2
  hnps = origspect.nperseg/2 
  # maybe todo: center-of-mass refinement of peak freqs?
  mypeaks=[]
  peaks = signal.find_peaks(tsp, height = -100, distance=dist, prominence=prom)
  for peakidx,binnum in enumerate(peaks[0]):
    peak_freq = hsr-binnum*hsr/hnps
    if do_limit_freqs and (peak_freq>OPL3_MAX_FREQ):
      continue
    peak_height = peaks[1]['peak_heights'][peakidx]
    peak_x = binnum*screen_width/ll
    peak_y = hh-1-((peak_height-minv)*hh/(maxv-minv))
    peak_prominence = peaks[1]['prominences'][peakidx]
    mypeaks.append((peak_height,peak_freq,peak_x,peak_y,peak_prominence))
  mypeaks.sort(key=lambda tup: -tup[0])
  return mypeaks
# -----------------------------------------------------------------------------
# draw a single spectrum along the top of screen with peaks marked and
# optional cursors 
# -----------------------------------------------------------------------------
def plotTestSpect(tsp,minv,maxv,gcolor=(255,255,255),yofs=0):
  global screen
  ll = len(tsp)
  ww = int(screen_width)
  hh = int(screen_height/4)
  hsr = origspect.sample_rate/2

  if minv==maxv: # dont wanna /0
    return

  # draw freq cutoff line 
  if cutoff_freq>=0:
    x = screen_width-1 - (cutoff_freq * screen_width / hsr)
    pygame.draw.line(screen, (0,255,255), (x,0+yofs),(x,hh+yofs))

  # draw amp cutoff line 
  if amp_cutoff_y>=0:
    pygame.draw.line(screen, (255,255,0), (0,amp_cutoff_y+yofs),(ww-1,amp_cutoff_y+yofs))
    plotText("cutoff={}".format(amp_cutoff),0,amp_cutoff_y)

  # draw spectrum peaks ith kinda-gradient colors
  mypeaks=getRankedPeaks(tsp, minv, maxv, False)
  for i,(peak_height,peak_freq,peak_x,peak_y,peak_prominence) in enumerate(mypeaks):
    r = 255*(peak_height-minv)/(maxv-minv)
    if i==0:
      pygame.draw.line(screen, (255,255,0), (peak_x,(peak_y-10)+yofs),(peak_x,(peak_y+10)+yofs))
    else:
      pygame.draw.line(screen, (r,r/8,255-r), (peak_x,(peak_y-10)+yofs),(peak_x,(peak_y+10)+yofs))

  # draw spectrum 
  for i in range(0,ll-1):
    x0=int(i*screen_width/ll)
    x1=int((i+1)*screen_width/ll)    
    v0=int((tsp[i]-minv)*hh/(maxv-minv))
    v1=int((tsp[i+1]-minv)*hh/(maxv-minv))
    if tsp[i]<minv or tsp[i+1]<minv:      
      continue
    y0=hh-1-v0
    y1=hh-1-v1
    if x0>=ww and x1>ww:
      break
    try:
      pygame.draw.line(screen, gcolor, (x0,y0+yofs),(x1,y1+yofs))
    except Exception as e:
      print(f'{e} {(x0,y0)=}-{(x1,y1)=} {i=} {screen_width=} {ll=}')

  return mypeaks
# -----------------------------------------------------------------------------
# draw a single spectrum along the top of screen with peaks marked and
# optional cursors 
# -----------------------------------------------------------------------------
def plotHalfSpect(tsp,minv,maxv,gcolor=(255,255,255),yofs=0):
  global screen
  ll = len(tsp)
  ww = int(screen_width)
  hh = int(screen_height/2)
  hsr = origspect.sample_rate/2

  if minv==maxv: # dont wanna /0
    return

  # opl freq cutoff line
  x = screen_width-1 - (OPL3_MAX_FREQ * screen_width / hsr)
  pygame.draw.line(screen, (0,255,255), (x,0+yofs),(x,hh+yofs))

  # opl amp cutoff line
  v0=int(((-48)-minv)*hh/(maxv-minv))
  y0=hh-1-v0
  pygame.draw.line(screen, (255,255,0), (0,y0+yofs),(ww-1,y0+yofs))

  # draw spectrum 
  for i in range(0,ll-1):
    x0=int(i*screen_width/ll)
    x1=int((i+1)*screen_width/ll)    
    v0=int((tsp[i]-minv)*hh/(maxv-minv))
    v1=int((tsp[i+1]-minv)*hh/(maxv-minv))
    if tsp[i]<minv or tsp[i+1]<minv:      
      continue
    y0=hh-1-v0
    y1=hh-1-v1
    if x0>=ww and x1>ww:
      break
    try:
      pygame.draw.line(screen, gcolor, (x0,y0+yofs),(x1,y1+yofs))
    except Exception as e:
      print(f'plotHalfSpect(): Exception: {e} {(x0,y0)=}-{(x1,y1)=} {i=} {screen_width=} {ll=}')
# -----------------------------------------------------------------------------
# draw a single spectrum along the top of screen without peaks marked
# -----------------------------------------------------------------------------
def plotTestSpectSimple(tsp,minv, maxv,gcolor=(255,255,255)):
  global screen
  ll = len(tsp)
  ww = int(screen_width)
  hh = int(screen_height/4)
  hsr = origspect.sample_rate/2
  if minv==maxv:     # dont wanna /0
    return
  for i in range(0,ll-1):
    x0=int(i*screen_width/ll)
    x1=int((i+1)*screen_width/ll)    
    v0=int((tsp[i]-minv)*hh/(maxv-minv))
    v1=int((tsp[i+1]-minv)*hh/(maxv-minv))
    if tsp[i]<minv or tsp[i+1]<minv:      
      continue
    y0=hh-1-v0
    y1=hh-1-v1
    if x0>=ww and x1>ww:
      break
    pygame.draw.line(screen, gcolor, (x0,y0),(x1,y1))
# -----------------------------------------------------------------------------
# dBFS value per gradient keycolor      
p0 = -115.0   # black
p1 = -75.0    # blue
p2 = -50.0    # red
p3 = -25.0    # yellow
p4 =  0.0     # white
# converts dBFS values to gradient colors
def fred(x):  
  if x<p1:
    return 0
  elif x<p2:
    return (x-p1)*255/(p2-p1)          
  else:
    return 255
def fgrn(x):
  if x<p2:
    return 0
  elif x<p3:
    return (x-p2)*255/(p3-p2)
  else:
    return 255
def fblu(x):
  if x<p0:
    return 0
  elif x<p1:
    return (x-p0)*255/(p1-p0)
  elif x<p2:
    return 255-((x-p2)*255/(p3-p2))
  elif x<p3:
    return 0
  else:
    return (x-p3)*255/(p4-p3)
def gradColor(x):
  r=int(fred(x))
  if r>255:
    r=255
  g=int(fgrn(x))
  if g>255:
    g=255
  b=int(fblu(x))
  if b>255:
    b=255
  c=(r,g,b)
  return c
# -----------------------------------------------------------------------------
# draw single spectrum's peaks on its spot in the displayed spectrogram
# -----------------------------------------------------------------------------
def overlayPeaksOnSpectrum(roi,slen, tsp):
  global screen
  ll = len(tsp)
  ww = int(screen_width)
  hh = int(screen_height) - T_AXIS_HEIGHT
  hsr =  (origspect.sample_rate/2)/SPECT_VERT_SCALE
  minv = origspect.minval
  maxv = origspect.maxval
  rx = roi*ww/slen
  rw = (((roi+SPECS_PER_FRAME)*ww/slen)-rx)+1

  # todo: draw a transparent black rect over the 
  # original spectrum so our detected peaks pop.
  # curently, this bar is opaque, fully hiding the spectrum
  pygame.draw.rect(screen, (0,0,0), (rx,0,rw,hh))

  mypeaks=getRankedPeaks(tsp,minv, maxv,True)
  s0 = 0
  s1 = len(mypeaks)
  points = []
  for i,(peak_height,peak_freq,peak_x,peak_y,peak_prominence) in enumerate(mypeaks[s0:s1]):
    adj_height = peak_height-maxv # normalize volume
    c=gradColor(adj_height)
    ry = hh-peak_freq*hh/hsr
    pygame.draw.rect(screen, c, (rx,ry,rw,2))
    if adj_height >= -48.0:  # loud enough for OPL channel volume setting
      points.append((float(peak_freq), float(adj_height)))
  return points
# -----------------------------------------------------------------------------
# do peak detect on whole displayed spectrogram, find runs of peaks, 
# and draw them
# -----------------------------------------------------------------------------  
def analyzePeakRuns():
  global origspect, roi
  global screen
  global screen_width
  global screen_height  

  slen = len(origspect.spectrogram)
  maxbin = origspect.maxval
  # keeps track of runs of spectral peaks
  freq_runs = {}
  prior_peaks = []
  for roi in range(0,slen,SPECS_PER_FRAME):  # for each spectrum in spectrogram
    # Check if user wants to close program
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  
        return HARD_QUIT  
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return SOFT_QUIT        
    # draws a vertical slice of spectral peaks over top of 
    # spectrum[roi] of the spectrogram, and returns a list of peak 
    # (freq,height) values sorted by descending height
    peaks = overlayPeaksOnSpectrum(roi,slen,origspect.spectrogram[roi][0:-1])

    # see which peaks are new, and which are continuations of prior peaks.
    # New ones set a new named run-record, and others get appended onto the
    # existing one.

    #print(f"{t:6.3f} ")
    new_prior_peaks = []
    for i,(f,h) in enumerate(peaks):
      mindif = 50000
      closest = None
      for j,(pf,ph,rkey) in enumerate(prior_peaks):
        fdif = abs(f-pf)
        if fdif<mindif:
          mindif = fdif
          closest = (j,pf,ph,rkey) 
      continuation = False
      if closest is not None:
        if mindif<30:          
          #print(f'{f:0.1f}->{closest[1]:0.1f} ', end='')
          continuation=True
          rkey = closest[3]
          freq_runs[rkey].append((roi,f,h))
      if not continuation:
        #print(f'{f:0.1f} ', end='')
        rkey = (roi,f)
        freq_runs[rkey] = [(roi,f,h)] # new run
      new_prior_peaks.append((f,h,rkey))
    prior_peaks = new_prior_peaks
    #print()
    pygame.display.update()

  draw_runs = True  
  if draw_runs:     # draw the freq runs in dif colors
    num_frames = slen // SPECS_PER_FRAME
    frames = [ None ] * num_frames
    ll = len(origspect.spectrogram[roi][0:-1])
    ww = int(screen_width)
    hh = int(screen_height) - T_AXIS_HEIGHT
    hsr =  (origspect.sample_rate/2)/SPECT_VERT_SCALE
    minv = origspect.minval
    maxv = origspect.maxval
    pygame.draw.rect(screen, (0,0,0), (0,0,ww,hh))
    rkeys = []
    for kidx, rkey in enumerate(freq_runs):
      run = freq_runs[rkey]
      if len(run)>=2:
        #print(f'{rkey}:')
        c0 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        r,f,h = run[0]
        #print(f'    {(t,f,h)}')    
        r0 = r
        if h>=-48.0 and f<=OPL3_MAX_FREQ:
          frame = r // SPECS_PER_FRAME
          if frames[frame] is None:
            frames[frame] = [(kidx,f,h)]
          else:
            frames[frame].append((kidx,f,h))
        x0 = int(r0*ww/slen)
        y0 = hh-int(f*hh/hsr)
        h0 = h

        for r,f,h in run[1:]:
          if h>=-48.0 and f<=OPL3_MAX_FREQ:
            frame = r // SPECS_PER_FRAME
            if frame>=len(frames):
              break
            if frames[frame] is None:
              frames[frame] = [(kidx,f,h)]
            else:
              frames[frame].append((kidx,f,h))
          r1 = r
          x1 = int(r1*ww/slen)
          y1 = hh-int(f*hh/hsr)
          h1 = h
          #print(f'    {(t,f,h)}')  
          m = 1.0-(h/(-48))
          rr = int(c0[0]*m)
          gg = int(c0[1]*m)
          bb = int(c0[2]*m)
          c1 = (rr,gg,bb)        
          pygame.draw.line(screen, c1, (x0,y0),(x1,y1),2)
          r0=r1
          x0=x1
          y0=y1
          h0=h1
        pygame.display.update()
        #print()

    # show cutoff freq as horiz line on peak runs spectrum
    hsr = origspect.sample_rate/2
    y = screen_height-1 -T_AXIS_HEIGHT - ( OPL3_MAX_FREQ * (screen_height-T_AXIS_HEIGHT) / hsr)
    for x in range(0,screen_width-1):
      if (x&1) == 0:
        pygame.draw.line(screen, (0,255,255), (x,y),(x,y))

    
    for roi in range(0,slen,SPECS_PER_FRAME): 
      x0 = int(roi*ww/slen)
      pygame.draw.line(screen, (32,64,0), (x0,0),(x0,screen_height-1))

    outframes = []
    for frame in frames:
      frame.sort(key=lambda x: -x[2])
    chans = [ None ] * 18
    for fi,frame in enumerate(frames):
      #print(f'Frame {fi:3d}: ',end='')
      unassigned = []
      unupdated = [ i for i in range(0,18)]
      for pi,f in enumerate(frame):
        kidx,freq,h = f
        if fi == 0:
          if pi<=17:
            chans[pi] = f
            del unupdated[unupdated.index(pi)]
        else:
          chanat = None
          for ch in range(0,18):
            if chans[ch][0] == kidx:
              chanat = ch
              break
          if chanat is not None:
            chans[chanat] = f
            del unupdated[unupdated.index(chanat)]
          else:
            unassigned.append(f)
      for x,u in enumerate(unupdated):        
        if x<len(unassigned):
          chans[u] = unassigned[x]
        else:
          chans[u] = (0,0,-99.0)
      #print(f'{chans=} {unassigned=}')
      outframes.append(deepcopy(chans))
      #print(f'({kidx:d},{freq:0.1f},{int(h):d}), ',end='')
      #print()
    #exit()
    #print('\nSynth output plan:\n')
    yeahframes = []
    for fi,o in enumerate(outframes):
      #print(f'Frame #{fi:3d}: ',end='')
      frame = []
      for ci,(ki,fr,h) in enumerate(o):
        ct = (ci, fr, h)
        frame.append(ct)
      yeahframes.append(frame)
      #print()

    #print('\n\n')
    #for frame in yeahframes:
    #  print(frame)
    #exit()
    pygame.display.update()

    return NO_QUIT, yeahframes
# -----------------------------------------------------------------------------
# WIP UI stuff
# -----------------------------------------------------------------------------
def handleMouseWheel(event):
  global screen
  global screen_width
  global screen_height
  mx, my = pygame.mouse.get_pos() 
  hsr = origspect.sample_rate/2
  freq = hsr-(my*hsr/screen_height)
  t = mx*origspect.dur_secs/screen_width
  s='mx,my,x,y = {:4d},{:4d},{:2d},{:2d} freq={:5.2f}, t={:4.4f}    '.format(mx,my,event.x, event.y, freq,t)
  plotText(s,4,4)
  pygame.display.update()
  #print(event.flipped)
  #print(event.which)
# -----------------------------------------------------------------------------
def newROI():
  global screen, origspect
  global screen_width
  global screen_height
  global roi
  drawSpect(origspect,0,0,screen_width,screen_height)
  if roi>=0 and roi<len(origspect.spectrogram[0:-1]):
    plotTestSpect(origspect.spectrogram[roi][0:-1],origspect.minval,origspect.maxval)
  pygame.display.update()
# -----------------------------------------------------------------------------  
# WIP UI stuff
# -----------------------------------------------------------------------------  
def handleMouseDown(event):
  global screen, origspect
  global screen_width
  global screen_height
  global roi,cutoff_freq,amp_cutoff, amp_cutoff_y
  global last_clicked

  mx, my = event.pos #pygame.mouse.get_pos()
  #print(event)
  hsr = origspect.sample_rate/2
  
  if my<int(screen_height/4) and (roi>=0) and (roi<len(origspect.spectrogram)):     # clicked on spectum up top
    last_clicked = CLICKED_SPECTRUM
    freq = hsr-(mx*hsr/screen_width)
    if event.button == 1: # left click selects cutoff frequency on spectrogram
      cutoff_freq = freq
    elif event.button == 3: # right click selects cutoff amplitude on spectrum
      amp_cutoff=origspect.maxval-(my*(origspect.maxval-origspect.minval)/(screen_height/4))
      amp_cutoff_y=my

  elif my>=int(screen_height/4):   # clicked on spectrogram
    last_clicked = CLICKED_SPECTROGRAM
    freq = hsr-(my*hsr/(screen_height-T_AXIS_HEIGHT))  
    if event.button == 1: # left click selects spectrum in spectrogram
      t = mx*origspect.dur_secs/screen_width
      sx=int(mx*origspect.maxrowidx/screen_width)
      sy=int(my*origspect.maxcolidx/(screen_height-T_AXIS_HEIGHT))
      v=origspect.spectrogram[sx][sy]
      roi = sx
      print(f'ROI {sx}')
      #s='mx,my = {:4d},{:4d} freq={:5.2f}, t={:4.4f}, v={:4.1f}    '.format(mx,my,freq,t,v)
      #plotText(s,4,4)
    elif event.button == 3: # right click selects cutoff frequency on spectrogram
      cutoff_freq = freq

  newROI()
# -----------------------------------------------------------------------------
# Plays the sound that the spectrogram is made from
# -----------------------------------------------------------------------------  
def playWave():
  global origspect
  swave = []
  for s in origspect.samples:
    i = int(s)
    swave.append([i,i])
  print('Playing original wave...')  
  s=pygame.sndarray.make_sound(np.array(swave, dtype="int16"))
  pygame.mixer.Sound.play(s)
  while pygame.mixer.get_busy(): 
    pygame.time.Clock().tick(10)         
# -----------------------------------------------------------------------------
def opToOfs(opidx):
  global regofs
  return regofs[opidx]
# -----------------------------------------------------------------------------
# resets the opl configuration to just the fixed-value parts
# -----------------------------------------------------------------------------
def initRegs():
  global regofs
  opl_regs = { (0,0x01): 0x20, (1,0x05): 0x01, (0,0x08): 0x00, (0,0xBD): 0, (1,0x04):0 }
  for b in range(0,2):
    for j in regofs:
      opl_regs[(b,j+0x20)] = 0x20
      opl_regs[(b,j+0x40)] = 0x20
      opl_regs[(b,j+0x60)] = 0xff
      opl_regs[(b,j+0x80)] = 0x0f
      opl_regs[(b,j+0xe0)] = 0x00
    for j in range(0,0x9):
      opl_regs[(b,j+0xA0)] = 0x00
      opl_regs[(b,j+0xB0)] = 0x00
      opl_regs[(b,j+0xC0)] = 0xf0   
  return opl_regs
# -----------------------------------------------------------------------------
# genetic algo calls this to impose a mutation on a genome (float32[] cfg vect)
# -----------------------------------------------------------------------------
def mutatefcn(genome, desperation, permutables):
  global opl3
  ncdist = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,5,6,7]
  numchanges = random.choice(ncdist) + desperation
  coup = 0
  if desperation>10:
    coup = 0.1*(desperation-10)
    if coup > 0.8:
      coup = 0.8
  for c in range(0,numchanges):
    x = random.choice(permutables)
    if random.random()<=(0.1+coup):              # normally a 10% chance to fully re-randomize the element,
      genome[x] = random.random()
    else:                                 # 90% chance for an incremental bump.
      wb = opl3.vec_elem_bits[x]
      mag = (1<<wb)-1
      i = opl3.vecFltToInt(genome[x],wb)
      if random.randint(0,1):
        i+=1
      else:
        i-=1
      if i<0:
        i=0
      elif i>mag:
        i=mag
      genome[x] = opl3.vecIntToFlt(i,wb)
  return genome  
# -----------------------------------------------------------------------------
# Given an OPL3 cfg vector and an ideal spectrum, goes through every element 
# of the vector, tweaking each element up/down by one count, retaining any
# changes that improved the fit. 
# Expensive, so only done by the genetic algorithm for generations where no 
# improved offspring were seen.
# -----------------------------------------------------------------------------
def autoTweak(ospect, initfit, v, permutables):
  global opl3
  newv = deepcopy(v)
  vl = len(newv)
  tweaks = 0
  lastfit = initfit
  for j in range(0,len(permutables)):
    x = permutables[j]
    bw = opl3.vec_elem_bits[x]
    mag = (1<<bw)-1
    f  = newv[x]
    i  = opl3.vecFltToInt(f, bw)
    if i>0:
      fa = opl3.vecIntToFlt(i-1,bw)
      newv[x] = fa
      fita, specta = opl3.fitness(ospect, newv)
      if fita<lastfit:
        lastfit = fita
        newv[x] = fa
        tweaks+=1
        continue
    if i<mag:
      fb = opl3.vecIntToFlt(i+1,bw)
      newv[x] = fb
      fitb, spectb = opl3.fitness(ospect, newv)
      if fitb<lastfit:
        lastfit = fitb
        newv[x] = fb
        tweaks+=1
        continue
    newv[x]=f  # no improvement, revert to orig val 
  return lastfit, newv, tweaks
# -----------------------------------------------------------------------------
# repeatedly calls calls the genetic algorithm's generate() method and shows
# the current best result.
#
# Returns the best OPL3 configuration achieved after either max iterations 
# reached, or we stopped seeing improvements for a long time,
# -----------------------------------------------------------------------------
ga = None
def improveMatch(frame, tot_frames, ospect, permutables):
  global screen,screen_width,screen_height
  global GENE_MAX_GENERATIONS
  global desperation
  global ga
  # width and height of test spectrum displayed during genetic loop
  ww = screen_width
  hh = screen_height//4
  # y ofs to draw position of the display
  yofs = screen_height//2
  # blank whole bottom half so we can show convengence plot in bottom quarter
  pygame.draw.rect(screen,(0,0,0),(0,yofs,ww,hh*2))  
  desperation = 0 # increased when even auto tweak fails. 
  lx = ly = px = py = 0
  tstart = time.time()
  initfit = None
  best_spect = None
  best_vec = None
  best_fit = None
  for gen in range(0,GENE_MAX_GENERATIONS):    
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  
        return HARD_QUIT
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return SOFT_QUIT
        elif event.key == pygame.K_RETURN:
          ga.verifyAll()          
          return NO_QUIT
    hfit = float(ga.p[0]['fit'])
    hspect = deepcopy(ga.p[0]['spect'])
    hgenome = deepcopy(ga.p[0]['genome'])
    improved = False
    quiet = True
    if hspect is not None:
      if initfit is None:
        initfit = hfit
      if best_fit is None:
        best_fit = hfit + 1
      tnow = time.time()
      tdelt = tnow-tstart
      if tdelt>=10.0:
        quiet = False
        tstart = tnow  
        print(f'Frame:{frame+1}/{tot_frames}, Gen.:{gen:3d}, fit:{hfit:14.9f}, tdelt:{tdelt:0.2f}, desperation:{int(desperation):2d}, ',end='')
        sys.stdout.flush()
      if hfit < best_fit:
        improved = True
        best_fit = hfit
        desperation = 0
      # show current best spectrum
      best_spect = hspect
      best_vec = hgenome
      pygame.draw.rect(screen,(0,0,0),(0,yofs,ww,hh))  
      plotTestSpect(ospect,-115.0,0,gcolor=(255,255,255),yofs=yofs)
      plotTestSpect(hspect,-115.0,0,gcolor=(255,255,0),yofs=yofs)
      plotText("Test Spectrum",8,yofs+8)
      plotText("Fitness",8,(screen_height-1)-24)    
      # show fitness plot in green      
      px = gen*ww//GENE_MAX_GENERATIONS
      py = (screen_height-1)-int(hh*hfit/initfit)
      if gen>0:
        pygame.draw.line(screen,(0,255,0),(lx,ly),(px,py))
      lx=px
      ly=py
      pygame.display.update()
    tweaks = -1
    if (not improved) and (best_spect is not None):
      ogenome = deepcopy(hgenome)
      fit, newv, tweaks = autoTweak(ospect, hfit, ogenome, permutables)
      if tweaks == 0:
        desperation += 1
        if desperation >= 50:
          print(' -- Moving on.\n')
          fitness, tspect = opl3.fitness(ospect, ga.p[0]['genome'])
          print(f'improveMatch(): Best fit was {ga.p[0]['fit']}.  verification {fitness}')
          ga.verifyAll()
          return NO_QUIT  
      else:      
        ga.add(newv)  
    ga.generate(desperation, quiet)
    if not quiet:
      if tweaks != -1:
        print(f', tweaks: {tweaks}')
      else:
        print()
  fitness, tspect = opl3.fitness(ospect, ga.p[0]['genome'])
  print(f'improveMatch(): Best fit was {ga.p[0]['fit']}.  verification {fitness}')
  ga.verifyAll()
  return NO_QUIT  
# -----------------------------------------------------------------------------
# brute force - start with sum of sines, then do GA
# -----------------------------------------------------------------------------
def do_brute(frame, tot_frames, ospect,permutables):
  global opl3
  global ga

  ovects = []
  try:
    with open(temp_regfile_name,'rb') as f:
      while True:
        r = f.read(512)
        if len(r)!=512:
          break
        v = opl3.rfToV(r)
        ovects.append(v)
  except Exception as e:
    print(e)
  peaks = getRankedPeaks(ospect, -115.0, 0, True, 5, 5)
  # make original estimate by setting the OPL to 
  # generate a sine wave per each spectral peak
  l = ''
  vvals = []
  for chan,p in enumerate(peaks):
    pheight, pfreq, _, _, _ = p
    if chan>=12:
      break
    if pheight<-48.0:
      break 
    vfreq = pfreq/OPL3_MAX_FREQ
    vamp = pheight/-48  # -48dBFS...0 dBFS : float 1.0...0.0
    vvals.append([float(vfreq),float(vamp)])  
  
  ch = 0
  rf = opl3.initRegFile()
  v = opl3.rfToV(rf)
  for vfreq, vamp in vvals:  # for the loudest peaks, assign a sine
    opl3.setNamedVecElemFloat(v,f'Freq.c{ch}',vfreq)  # Peak freq    
    opl3.setNamedVecElemFloat(v,f'KeyOn.c{ch}',1.0)   # Key ON
    opl3.setNamedVecElemFloat(v,f'SnTyp.c{ch}',1.0)   # 2-op AM
    oidxs = opidxs_per_chan[ch]
    for q,o in enumerate(oidxs):
      opl3.setNamedVecElemFloat(v,f'AttnLv.o{o}',vamp)  # amplitude
      opl3.setNamedVecElemInt(v,f'FMul.o{o}',1)   # op phase multiple
      opl3.setNamedVecElemInt(v,f'KSAtnLv.o{o}',2) # some attenuation at higher freqs
    ch+=1
    if ch>=18:
      break


  del ga
  ga = gene.gene(500, ospect, [i for i in range(0,222)], mutatefcn)
  if len(ovects):
    for i in range(len(ovects)):
      ga.add(ovects[i])
  for i in range(50):
    ga.add(v)
  if len(ovects):
    for i in range(len(ovects)):
      ga.add(ovects[i])
  for i in range(50):
    ga.add(v)

  do_quit = improveMatch(frame, tot_frames, ospect, permutables)
  regfile = None
  if not do_quit:
    regfile = opl3.vToRf(ga.p[0]['genome'])
  return regfile, do_quit
# -----------------------------------------------------------------------------
# WIP EXPERIMENT- 
# Tries to brute-force a solution using either a (slow) genetic algorithm or a 
# convolutional neural-network.
# -----------------------------------------------------------------------------
def bruteForce(brute = False, genetic = False, ai = False):
  global origspect
  global screen
  global screen_width
  global screen_height  
  global opl3
  global ga
  global SPECS_PER_FRAME
  ww = int(screen_width)
  hh = int(screen_height//4)
  slen = len(origspect.spectrogram)

    # The spectragram has 172.265625 spectra/sec,
  # so if we do every third spectrum in this brute-force
  # loop, that'll be a frame rate of 57.421875 Hz.
  try:
    if ai:
      model = OPL3Model()
      model.load_state_dict(torch.load(modelsfolder+'torch_model.pth'))
      model.eval()  # Set the model to evaluation mode
  except:
    print('''
#########################################################
SORRY!
------
I couldn't find the AI model at 'models/torch_model.pth'.
Github doesn't want me to push my copy, as it is 600MB.
You can train one yourself with these provided utils:

'src/generate_training_set.py' - Makes training data
'src/pytorch_train_NN.py'      - Trains the AI model

My results are currently pretty poor though, even with
a 1 GB training set and 600MB model.

If I start having success with the AI stuff, I will
share the model on my personal website and update 
this message and the READMEs.  Stay Tuned!

#########################################################
''')

    return False
  print('\n##############################################################################\n')
  print(f'Starting Brute Force: {wavname}')

  # init intermediate output file of opl3 reg settings for
  # later conversion to a VGM. (TODO!)
  # See if we have a work-in-progress file.
  tmpfile = temp_regfile_name
  try:
    tsize = os.path.getsize(tmpfile)
  except:
    tsize = 0
  # if file full, or empty, or len is not multiple of 512, start over.
  if (tsize==0) or (tsize%512) or (tsize>=(512*(slen//SPECS_PER_FRAME))):
    with open(tmpfile,'wb') as f:      
      start_roi = 0
      start_frame = 0
      print('Starting from the beginning.\n')
      regfile = None
      prior_best = None
  else:
    with open(tmpfile,'rb') as f:
      while True:
        rf = f.read(512)
        if len(rf)<512:
          break
        prior_best = opl3.rfToV(rf)
        regfile = rf
    with open(tmpfile,'ab') as f:
      start_roi = (tsize//512)*SPECS_PER_FRAME
      start_frame = start_roi // SPECS_PER_FRAME
      print('Found working file. Resuming.\n')

  tot_frames = slen//SPECS_PER_FRAME

  # brute-force loop: -----
  tstart = time.time()
  tstepdur = 0
  for roi in range(start_roi,slen,SPECS_PER_FRAME): 

    # check if user is a quitter
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return HARD_QUIT
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return SOFT_QUIT

    # show progess
    tnow = time.time()
    tmins = (tnow - tstart)/60.0
    frame = roi // SPECS_PER_FRAME
    pct = frame * 100.0 / tot_frames
    s = f'Brute Force: frame {frame+1}/{tot_frames} - Progress: {pct:6.2f}% - Duration: {tmins:5.1f} mins '
    elap_frames=(roi-start_roi) // SPECS_PER_FRAME
    if elap_frames>0:
      t_per_frame = (tnow-tstart)/elap_frames
      trem = ((tot_frames-frame)*t_per_frame)/3600.0
      s+= f'- Est. Rem. {trem:5.2f} hrs '
    s += '-'*(132-len(s))
    print(s)

    # this is the spectrum we want recreate.
    ospect = deepcopy(origspect.spectrogram[roi][0:-1])

    if brute:
      # Init an approximate sum-of-sines solution,
      # then sets the genetic algorithm, to work on
      # it, noting the spectrum we hope to achieve.
      print('@@@ BRUTE')
      regfile = None
      tstepstart = time.time()
      regfile, do_quit = do_brute(frame, tot_frames, ospect, [i for i in range(0,222)])
      if do_quit:
        return do_quit
      tstepdur = (time.time() - tstepstart)/60
      print(f'Brute improvement duration: {tstepdur:8.1f} mins')

    # NEURAL NETWORK FUN ------------------
    if ai:          
      print('@@@ AI')
      # reformat the original spectrum to 
      # an input vector for the inferencer
      inp = []
      for i,b in enumerate(ospect):
        o = abs(int(b))
        if o > 255:
          o = 255
        o = 255 - o
        inp.append(float(o)/255.0)  
      # Shape becomes (1, 2048)
      sample_input = torch.tensor(inp, dtype=torch.float32).reshape(1, 2048)
      # Disable gradient calculation for inference
      with torch.no_grad():
          predicted_output = model(sample_input)
      # make prediction
      predicted_output = predicted_output.numpy().flatten()
      # convert output cfg vector to a byte[512] opl3 register file   
      q = opl3.rfToV(opl3.initRegFile())
      opl3.showVector(predicted_output) 
      regfile = opl3.vToRf(predicted_output)
    # -------------------------------------
    # GENETIC ALGORITHM FUN ---------------
    if genetic:
      # Init a fully random population for the genetic 
      # algorithm, noting the spectrum we hope to achieve.
      print('@@@ GENE')
      ga = gene.gene(500, ospect, [i for i in range(0,222)], mutatefcn)
      v = opl3.rfToV(opl3.initRegFile())
      for i in range(0,500):
        genome = opl3.rfToV(opl3.initRegFile())
        for x in range(len(v)):
          genome[x] = random.random()
        ga.add(genome)
      print('Starting permutations.')
      # Do a (slow) genetic annealing process to
      # try to improve the result. 
      do_quit = improveMatch(frame, tot_frames, ospect, [i for i in range(0,222)])
      if do_quit:
        print('Brute force - quitting!')
        if do_quit == SOFT_QUIT:
          drawSpect(origspect,0,0,screen_width,screen_height)
        return do_quit
      regfile = opl3.vToRf(ga.p[0]['genome'])
    # -------------------------------------
    # give register cfg to opl3 emulator and render a spectrum
    wave, tspect = opl3.renderOPLFrame(regfile)
    if tspect is not None: 
      pygame.draw.rect(screen,(0,0,0),(0,0,ww,hh))
      plotTestSpect(ospect,-115,0,(255,255,255))   # plot original spect      
      plotTestSpect(tspect,-115,0,(255,255,0))   # plot spect from predicted config
      pygame.display.update()
    # Output our best register file result to intermediate file, 
    # for later conversion to VGM. (TODO!!)
    if regfile is not None:
      v = opl3.rfToV(regfile)
      fitness, spect = opl3.fitness(ospect, v)   
      print(f'output fitness: {fitness=:8.2f}')
      with open(tmpfile,'ab') as f:
        f.write(regfile)
  return NO_QUIT
# -----------------------------------------------------------------------------
# loading and processing the output of bruteforce
# -----------------------------------------------------------------------------
_2op_chans = {
   0:[ 0, 3], 1 :[ 1, 4],  2:[ 2, 5],  3:[ 6, 9],
   4:[ 7,10], 5 :[ 8,11],  6:[12,15],  7:[13,16],
   8:[14,17], 9 :[18,21], 10:[19,22], 11:[20,23],
  12:[24,27], 13:[25,28], 14:[26,29], 15:[30,33],
  16:[31,34], 17:[32,35] }

def loadRegfile():
  global origspect
  global screen
  global screen_width
  global screen_height
  global opl3 
  global SPECS_PER_FRAME, _2op_chans
  ww = int(screen_width)
  hh = int(screen_height//4)

  # time delta per spectrogram (secs): 0.00580498866213152
  # the regfile does every third spectrum, so it has 
  # a framerate of 57.421875 synth configs per second

  print('\n##############################################################################\n')
  # init intermediate output file of opl3 reg settings for
  # later conversion to a VGM. (TODO!)
  # See if we have a work-in-progress file.
  tmpfile = temp_regfile_name
  try:
    tsize = os.path.getsize(tmpfile)
  except:
    tsize = 0
  # if file full, or empty, or len is not multiple of 512, start over.
  if (tsize==0) or (tsize%512):
    print('loadRegfile(): ABORT: No suitable regfile found.\n')
    return NO_QUIT

  print('loadRegfile(): Starting...\n')
  infile = open(tmpfile,'rb')
  tot_frames = tsize // 512
  
  prevv = None

  frame_vects = []  # gather opl3 register changes per frame
  fits = []
  # data input loop
  for frame in range(tot_frames):  # for each frame (opl config) in file

    # check if user is a quitter
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return HARD_QUIT
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return SOFT_QUIT
    
    # get next synth configuration 
    regfile = infile.read(512)
    v = opl3.rfToV(regfile)

    if prevv is not None:
      print(f'#{frame:3d}:',end='')
      newchans = [i for i in range(0,18)]
      for prevch in range(0,18):
        prevkeyon = opl3.vecGetNamedFloat(prevv, f'KeyOn.c{prevch}')
        prevfreq = opl3.vecGetNamedFloat(prevv, f'Freq.c{prevch}')*OPL3_MAX_FREQ
        oidx = _2op_chans[prevch][0]
        prevvol = opl3.vecGetNamedFloat(prevv, f'AttnLv.o{oidx}')
        mindif = 99999
        minat = -1
        if prevkeyon>=0.5:
          print(f'({prevch:2d},{prevfreq:6.1f},{prevvol:5.3f}) ',end='')
      print(f'')

      # rearrange this vector to align its channels to
      # similar-sounding channels of the predecessor.
      # This may sound smoother (less clicks) and reduce
      # the number of register writes needed in the VGM file.
      pass

    ospect = origspect.spectrogram[frame*SPECS_PER_FRAME][0:-1]
    fitness,spect = opl3.fitness(ospect, v)
    fits.append({'frame':frame, 'fit':fitness, 'spect':spect, 'ospect':deepcopy(ospect)})
    #print(f'frame: {frame+1}/{tot_frames}: {fitness=:8.2f}')

    # render the latest synth configuration and draw spectrum
    wave, tspect = opl3.renderOPLFrame(v)
    if tspect is not None: 
      pygame.draw.rect(screen,(0,0,0),(0,0,ww,hh))
      plotTestSpect(ospect,-115,0,(255,255,255))   # plot spect from predicted config
      if fitness < 150:
        color = (0,255,0)
      elif fitness < 300:
        color = (255,255,0)
      else:  
        color = (255,0,0)

      plotTestSpect(tspect,-115,0,color)   # plot spect from predicted config
      pygame.display.update()
      #time.sleep(0.5)
    # get ready for next frame
    frame_vects.append( v )
    prevv = v
  # end data input loop
  infile.close()

  #fits.sort(key=lambda m:-m['fit'] )
  #for f in fits:
  #  print(f['frame'],f['fit'])

  # todo: render output, make VGM.

  # todo: play output

  return NO_QUIT

# -----------------------------------------------------------------------------
# Make sequence to init the OPL3 chip.
#
# Currently, we set it up to do 18 plain sine waves.
# -----------------------------------------------------------------------------
prev_sets = {}
op_reg_ofs = [ 
  0x000,  0x001,  0x002,  0x003,  0x004,  0x005,  0x008,  0x009,  0x00A,  # bank 0
  0x00B,  0x00C,  0x00D,  0x010,  0x011,  0x012,  0x013,  0x014,  0x015,
  0x100,  0x101,  0x102,  0x103,  0x104,  0x105,  0x108,  0x109,  0x10A,  # bank 1 (OPL3 only)
  0x10B,  0x10C,  0x10D,  0x110,  0x111,  0x112,  0x113,  0x114,  0x115,
  ]

def opl3init2():
  '''
      rf = self.RF(rf,0x105,0x01)
    for i in range(0,36):
      o=self.op_reg_ofs[i]
      rf = self.RF(rf,0x20|o,0b00100000)   # enable sustain
      rf = self.RF(rf,0x60|o,0xff)   # fast envelopes
      rf = self.RF(rf,0x80|o,0x0f)   # sustain level to loudeest
  '''
  global prev_sets, opl3, op_reg_ofs
  prev_sets = {}

  opl3._do_init()
  res = b''
  # enable opl3 mode
  r = 0x05
  v = 0x01
  opl3._writeReg(1,r,v);
  res+=struct.pack('BBB',0x5f,r,v)

  r = 0x04
  v = 0x00
  opl3._writeReg(1,r,v);
  res+=struct.pack('BBB',0x5f,r,v)

  for i in range(0,36):
    o = op_reg_ofs[i]

    r = 0x20|o
    v = 0b00100001
    if r>0xff:
      opl3._writeReg(1,r&0xff,v);
      res+=struct.pack('BBB',0x5f,r&0xff,v)
    else:
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)

    r = 0x60|o
    v = 0xff
    if r>0xff:
      opl3._writeReg(1,r&0xff,v);
      res+=struct.pack('BBB',0x5f,r&0xff,v)
    else:
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)

    r = 0x80|o
    v = 0x0f
    if r>0xff:
      opl3._writeReg(1,r&0xff,v);
      res+=struct.pack('BBB',0x5f,r&0xff,v)
    else:
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)

    r = 0xe0|o
    v = 0x00
    if r>0xff:
      opl3._writeReg(1,r&0xff,v);
      res+=struct.pack('BBB',0x5f,r&0xff,v)
    else:
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)

  for i in range(0,9):
    opl3._writeReg(1,0xc0+i,0x31);
    res+=struct.pack('BBB',0x5f,0xc0+i,0x31)
    opl3._writeReg(0,0xc0+i,0x31);
    res+=struct.pack('BBB',0x5e,0xc0+i,0x31)
    

  return res



def opl3init():
  global prev_sets, opl3
  prev_sets = {}

  opl3._do_init()
  res = b''
  # enable opl3 mode
  r = 0x05
  v = 0x01
  opl3._writeReg(1,r,v);

  res+=struct.pack('BBB',0x5f,r,v)
  # enable waveform select
  #r = 0x01
  #v = 0x20
  #opl3._writeReg(1,r,v);
  #res+=struct.pack('BBB',0x5f,r,v)

  #rf = self.RF(rf,0x20|o,0b00100000)   # enable sustain
  #rf = self.RF(rf,0x60|o,0xff)   # fast envelopes
  #rf = self.RF(rf,0x80|o,0x0f) 

  for chan in range(0,18):
    opidxs = opidxs_per_chan[chan]
    if chan<9:      
      # sustain, vibrato,opfreqmult
      r = 0x20 + opToOfs(opidxs[0])
      v = 0x21
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x20 + opToOfs(opidxs[1])
      v = 0x21
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # keyscalelevel, output level
      # setting keyscale level to 6.0 dB of attenuation per rise in octave
      # (we really want more than this and need to implement something ourselves)
      r = 0x40 + opToOfs(opidxs[0])
      v = 0x30  # 30
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x40 + opToOfs(opidxs[1])
      v = 0x30
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # attack rate, decay rate
      r = 0x60 + opToOfs(opidxs[0])
      v = 0xff
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x60 + opToOfs(opidxs[1])
      v = 0xff
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # sust level, release rate
      r = 0x80 + opToOfs(opidxs[0])
      v = 0x0f
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x80 + opToOfs(opidxs[1])
      v = 0x0f
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # output channels L&R and additive synth
      r = 0xc0 + chan
      v = 0x31
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # waveform select sine
      r = 0xe0 + opToOfs(opidxs[0])
      v = 0x00
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0xe0 + opToOfs(opidxs[1])
      v = 0x00
      opl3._writeReg(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      a = opToOfs(opidxs[0]-18)
      b = opToOfs(opidxs[1]-18)
      # sustain, vibrato,opfreqmult
      r = 0x20 + a
      v = 0x21
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x20 + b
      v = 0x21
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # keyscalelevel, output level
      r = 0x40 + a
      v = 0x30
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x40 + b
      v = 0x30    #  30
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # attack rate, decay rate
      r = 0x60 + a
      v = 0xff
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x60 + b
      v = 0xff
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # sust level, release rate
      r = 0x80 + a
      v = 0x0f
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x80 + b
      v = 0x0f
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # output channels L&R and additive synth
      r = 0xc0 + chan
      v = 0x31
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # waveform select sine (todo: use different waveforms when appropriate!)
      r = 0xe0 + a
      v = 0x00
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0xe0 + b
      v = 0x00
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
  return res
# -----------------------------------------------------------------------------
# Returns an OPL3 register set sequence to set a specified channel to 
# the specified frequency, volume, and/or key on/off.
# -----------------------------------------------------------------------------
def opl3params(freq,namp,chan, keyon):
  global prev_sets, opl3
  fnum, block = opl3.freqToFNumBlk(freq)
  opidxs = opidxs_per_chan[chan]
  res = b''
  schan = chan
  if chan in prev_sets:     # note prior settings so we only reconfigure
    pchan = prev_sets[chan] # registers which change.
  else:
    pchan = None

  if not keyon:               # key off
    if pchan is not None and pchan[2]:
      if chan<9:
        r = 0xb0 + chan
        v = ((fnum>>8)&3) | (block<<2)
        opl3._writeReg(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
      else:
        chan-=9
        r = 0xb0 + chan
        v = ((fnum>>8)&3) | (block<<2)
        opl3._writeReg(1,r,v);      
        res+=struct.pack('BBB',0x5f,r,v)
    prev_sets[chan] = (0,0,False)
  else:                         # key on
    aval = int(0x3F*namp)
    if aval>0x3f:
      aval = 0x3f
    if aval<0:
      aval = 0
    if chan<9:
      if (pchan is None) or (pchan[1]!=namp):
        r = 0x40 + opToOfs(opidxs[0])  # set volume
        v = aval
        opl3._writeReg(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
        r = 0x40 + opToOfs(opidxs[1])
        v = aval
        opl3._writeReg(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
      if (pchan is None) or (pchan[0]!=freq):
        r = 0xA0 + chan  # set low bits of frequency
        v = fnum&0xff
        opl3._writeReg(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
        r = 0xb0 + chan  # set key-on and high bits of frequency
        v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
        opl3._writeReg(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      a = opToOfs(opidxs[0] - 18)
      b = opToOfs(opidxs[1] - 18)
      if (pchan is None) or (pchan[1]!=namp):
        r = 0x40 + a # volume
        v = aval
        opl3._writeReg(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
        r = 0x40 + b
        v = aval
        opl3._writeReg(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
      if (pchan is None) or (pchan[0]!=freq):
        r = 0xA0 + chan # low bits of freq
        v = fnum&0xff
        opl3._writeReg(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
        r = 0xb0 + chan    # key-on and high bits of freq
        v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
        opl3._writeReg(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
    prev_sets[schan] = (freq,namp,keyon)
  return res
# -----------------------------------------------------------------------------
# Returns an OPL3 register set sequence to set a specified channel to 
# the specified frequency, volume, and/or key on/off.
# -----------------------------------------------------------------------------
chan_reg_ofs = [
  0x000, 0x001, 0x002, 0x003, 0x004, 0x005, 0x006, 0x007, 0x008, 
  0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108
]


def opl3params2(freq,namp,chan, keyon):
  global prev_sets, opl3, op_reg_ofs, chan_reg_ofs

  fnum, block = opl3.freqToFNumBlk(freq)
  opidxs = opidxs_per_chan[chan]

  res = b''
  if not keyon:               # key off
    if chan<9:
      r = 0xb0 + chan
      v = ((fnum>>8)&3) | (block<<2)
      opl3._writeReg(0,r,v);              
      res+=struct.pack('BBB',0x5e,r,v)      
    else:
      chan-=9
      r = 0xb0 + chan
      v = ((fnum>>8)&3) | (block<<2)
      opl3._writeReg(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
  else:                         # key on
    aval = int(0x3F*namp)
    if aval>0x3f:
      aval = 0x3f
    if aval<0:
      aval = 0
    if chan<9:
      r = 0x40 + opToOfs(opidxs[0])  # set volume
      v = 0b01000000|aval
      opl3._writeReg(0,r,v);              
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x40 + opToOfs(opidxs[1])
      v = 0b01000000|aval
      opl3._writeReg(0,r,v);              
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0xA0 + chan  # set low bits of frequency
      v = fnum&0xff
      opl3._writeReg(0,r,v);              
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0xb0 + chan  # set key-on and high bits of frequency
      v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
      opl3._writeReg(0,r,v);              
      res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      a = opToOfs(opidxs[0] - 18)
      b = opToOfs(opidxs[1] - 18)
      r = 0x40 + a # volume
      v = 0b01000000|aval
      opl3._writeReg(1,r,v);              
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x40 + b
      v = 0b01000000|aval
      opl3._writeReg(1,r,v);
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0xA0 + chan # low bits of freq
      v = fnum&0xff
      opl3._writeReg(1,r,v);              
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0xb0 + chan    # key-on and high bits of freq
      v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
      opl3._writeReg(1,r,v);              
      res+=struct.pack('BBB',0x5f,r,v)
  return res  
# -----------------------------------------------------------------------------
# fill out the information section (GD3) of the output VGM file
# -----------------------------------------------------------------------------
def utf_to_utf16le(text):
    return text.encode('utf-16le')+b'\0\0'  
def makeGD3():
  utxt = utf_to_utf16le(f"{wavname[0:-4]}")  # trackname
  utxt += utf_to_utf16le(f"")  # jap trackname
  utxt += utf_to_utf16le(f"PUGPUTER 6309")  # gamename
  utxt += utf_to_utf16le(f"")  # jap gamename
  utxt += utf_to_utf16le(f"PUGPUTER 6309")  # system name
  utxt += utf_to_utf16le(f"")  # jap systemname
  utxt += utf_to_utf16le(f"WAV TO VGM")  # orig author
  utxt += utf_to_utf16le(f"")
  utxt += utf_to_utf16le(f"2024/11/05")  # date
  utxt += utf_to_utf16le(f"Craig Iannello")
  utxt += utf_to_utf16le(f"Converted from waveform file using spectral analysis")  
  gd3 = b'Gd3 \x00\x01\x00\x00'+struct.pack('<I',len(utxt))+utxt
  return gd3
# -----------------------------------------------------------------------------
# peak-detect the whole spectrogram and convert it to a VGM file, which is 
# a sequence of OPL3 register settings, interspersed with short delays.
#
# Writes the output as as a .VGZ file (zipped VGM) to the output dir.
# -----------------------------------------------------------------------------
def fastAnalyze():
  global origspect
  global screen
  global screen_width
  global screen_height  
  global opl3
  minv = origspect.minval
  maxv = origspect.maxval

  print('Doing spectral peak-detection.')
  all_peaks = []
  slices=0
  ll = len(origspect.spectrogram)
  tt = origspect.maxtime - origspect.mintime
  for r in range(0,ll):
    all_peaks.append(getRankedPeaks(origspect.spectrogram[r][0:-1],minv,maxv,True))
    slices+=1
  max_amp = 0
  min_amp = 9999999
  max_height = -999999
  min_height = 9999999

  opl3 = OPL3()
  opl3._sample_overflow = 0
  opl3._output = bytes() 
  rows = []
  lt = 0
  for pi, peaks in enumerate(all_peaks):
    t = pi * tt / ll   # get slice timestamp
    do_slice = False
    if (t-lt) >= (1.0/SLICE_FREQ):
      lt = t
      do_slice = True
    if do_slice:
      row = []
      for i,(peak_height,peak_freq,peak_x,peak_y,peak_prominence) in enumerate(peaks):
        wave_amp = 37369.0 * math.exp(0.1151 * peak_height)
        if i>=MAX_SYNTH_CHANS:
          break
        if wave_amp < min_amp:
          min_amp = wave_amp
        if wave_amp > max_amp:
          max_amp = wave_amp
        if peak_height > max_height:
          max_height = peak_height
        if peak_height < min_height:
          min_height = peak_height
        row.append([peak_freq, wave_amp, peak_height])
      k = len(row)
      while k<MAX_SYNTH_CHANS:
        row.append([0,0,-96.0])
        k+=1
      rows.append(row)
  

  # make and write the VGZ output file
  print('Making output file (VGZ)...')
  # these bytes are the VGM header which specifies that we want OPL3 at 14.318 MHz!
  # TODO: Any of the dozen other synth types supported by the VGM file format!
  outvgm = b'Vgm \xd3\xcb\x00\x00Q\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xcd\xca\x00\x00\x8a\x12}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00dz\xda\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
  outvgm+=opl3init()
  for row in rows:
    chan = 0
    for freq,amp,peakheight in row:
      if freq==0:
        continue
      nph = (peakheight-max_height)
      keyon = False
      if nph>=-48:
        namp = nph/-48
        keyon=True
      else:
        namp = 1
      outvgm += opl3params(freq,namp,chan,keyon)
      chan += 1
    opl3._render_ms(1000/SLICE_FREQ)
    outvgm+=struct.pack("<BH",0x61,int(44100/SLICE_FREQ))

  outvgm+=b'\x66'
  gd3_ofs = len(outvgm)-0x14
  gd3_dat = makeGD3()
  outvgm+=gd3_dat
  vgm_eof_ofs = len(outvgm)-0x04
  outvgm=outvgm[0:4]+struct.pack('<I',vgm_eof_ofs)+outvgm[8:0x14]+struct.pack('<I',gd3_ofs)+outvgm[0x18:]
  with gzip.open(output_vgm_name, 'wb') as f:
      f.write(outvgm)    

  print(f'{slices} slices, {len(rows)} frames')
  print(f'{min_height=} {max_height=}')
  stereo_wave, mono_wave = opl3.stereoBytesToNumpy(opl3._output)
  print('\nMaking spectrogram of result...')
  outspect = sp.spect(wav_filename=None,samples=mono_wave,nperseg=4096,quiet=False,clip=True)
  drawSpect(outspect,0,0,screen_width,screen_height)
  stereo_wave=pygame.sndarray.make_sound(stereo_wave)
  print('\nPlaying result...',end='')
  sys.stdout.flush()
  pygame.mixer.Sound.play(stereo_wave)
  while pygame.mixer.get_busy(): 
    pygame.time.Clock().tick(10)
  print('Done!')
# -----------------------------------------------------------------------------
def makeVGM(yayframes):
  global origspect
  global screen
  global screen_width
  global screen_height  
  global opl3
  minv = origspect.minval
  maxv = origspect.maxval

  opl3 = OPL3()
  opl3._sample_overflow = 0
  opl3._output = bytes() 

  # make and write the VGZ output file
  print('Making output file (VGZ)...')
  # these bytes are the VGM header which specifies that we want OPL3 at 14.318 MHz!
  # TODO: Any of the dozen other synth types supported by the VGM file format!
  outvgm = b'Vgm \xd3\xcb\x00\x00Q\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xcd\xca\x00\x00\x8a\x12}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00dz\xda\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
  outvgm+=opl3init2()
  for fidx,frame in enumerate(yayframes):
    print(f'{fidx=}: ',end='')
    for ct in frame:
      chan, freq, peakheight = ct
      peakheight -= maxv
      keyon = True
      if peakheight < -48:
        keyon = False
        namp = 1.0
      else:
        namp = peakheight/-48.0
        if namp>1.0:
          namp =  1.0
        elif namp<0.0:
          namp =  0.0
      print(f'({chan:2d},{freq:6.1f},{int(peakheight):3d},{namp:4.2f}),', end='')
      outvgm += opl3params2(freq,namp,chan,keyon)
    print()
    opl3._render_ms(1000/57.4149659864)
    outvgm+=struct.pack("<BH",0x61,int(44100/57.4149659864))

  outvgm+=b'\x66'
  gd3_ofs = len(outvgm)-0x14
  gd3_dat = makeGD3()
  outvgm+=gd3_dat
  vgm_eof_ofs = len(outvgm)-0x04
  outvgm=outvgm[0:4]+struct.pack('<I',vgm_eof_ofs)+outvgm[8:0x14]+struct.pack('<I',gd3_ofs)+outvgm[0x18:]
  with gzip.open(output_vgm_name, 'wb') as f:
      f.write(outvgm)    

  stereo_wave, mono_wave = opl3.stereoBytesToNumpy(opl3._output)
  print('\nMaking spectrogram of result...')
  outspect = sp.spect(wav_filename=None,samples=mono_wave,nperseg=4096,quiet=False,clip=True)
  drawSpect(outspect,0,0,screen_width,screen_height)
  stereo_wave=pygame.sndarray.make_sound(stereo_wave)
  print('\nPlaying result...',end='')
  sys.stdout.flush()
  pygame.mixer.Sound.play(stereo_wave)
  while pygame.mixer.get_busy(): 
    pygame.time.Clock().tick(10)
  print('Done!')  
# -----------------------------------------------------------------------------
def writeVgmFile(vgmdata):
  # these bytes are the VGM header which specifies that we want OPL3 at 14.318 MHz!
  # TODO: Any of the dozen other synth types supported by the VGM file format!
  outvgm = b'Vgm \xd3\xcb\x00\x00Q\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xcd\xca\x00\x00\x8a\x12}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00dz\xda\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
  outvgm+=vgmdata
  gd3_ofs = len(outvgm)-0x14
  gd3_dat = makeGD3()
  outvgm+=gd3_dat
  vgm_eof_ofs = len(outvgm)-0x04
  outvgm=outvgm[0:4]+struct.pack('<I',vgm_eof_ofs)+outvgm[8:0x14]+struct.pack('<I',gd3_ofs)+outvgm[0x18:]
  with gzip.open(output_vgm_name, 'wb') as f:
      f.write(outvgm)    
  print('Wrote VGM file!')   
# -----------------------------------------------------------------------------
# iterates through the test set (spectrum + cfg) and verifies that the each 
# spectrun is the result of that synth config.
# The spect bins were converted to bytes like this:  uint8=255-(-dBFS)&0xFF
# And each config is a float32[222] synth config vector.
# -----------------------------------------------------------------------------
def testTrainingSet():
  global screen_width
  global screen_height  
  global opl3
  ww = int(screen_width)
  hh = int(screen_height//4)
  REDRAW_INTERVAL = 500
  worstfit = None
  print('testTrainingSet(): Starting.')
  with open(trainfolder+'opl3_training_spects.bin','rb') as sfile:
    with open(trainfolder+'opl3_training_regs.bin','rb') as rfile:
      row = 0
      while True:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
            return HARD_QUIT
          elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
              return SOFT_QUIT
        bs = sfile.read(2048) # bin spect
        br = rfile.read(222*4) # bin regs
        # check for end of file
        if not isinstance(bs,bytes) or len(bs)!=2048:
          print('EOF')
          break
        # conv spect to dBFS[]
        spect = []
        for i in range(0,2048):          
          spect.append(-(255-bs[i]))
        # and vect to float32[222]
        v = []
        for i in range(0,222*4,4):
          f = struct.unpack('<f',br[i:i+4])[0]
          v.append(f)
        # compare output spect ro provides spect
        fit,tspect = opl3.fitness(spect,v)
        if worstfit is None or fit>worstfit:
          worstfit = fit
        row+=1
        if not row % REDRAW_INTERVAL:
          pygame.draw.rect(screen,(0,0,0),(0,0,ww,hh))
          plotTestSpect(spect,-115,0,(255,255,255))   # plot original spect
          if tspect is not None: 
            plotTestSpect(tspect,-115,0,(255,255,0))   # plot spect from predicted config
          pygame.display.update()
          print(f'testTrainingSet(): Row:{row:9}, Fit:{round(fit)}, Worst:{round(worstfit)}')

  print('testTrainingSet(): Ending.')
  return NO_QUIT
# -----------------------------------------------------------------------------  
# UI for stepping through all the frames of the loaded spectrogram and 
# allowing manual/automatic tuning of all 222 synth params per each frame.
#
# all tentative frame configs are kept in a tempfile named after orig filename
# and can be resumed after restart.
# -----------------------------------------------------------------------------  
def manualControls():
  global origspect
  global SPECS_PER_FRAME
  global opl3
  global screen_width, screen_height
  hh = screen_height // 2
  ww = screen_width
  print('\n\n\nStarting manual control:\n\n')
  chromosome_lens = [25,25,25,12,12,12,25,25,25,12,12,12]  
  manualEvtLoop = True

  slen = len(origspect.spectrogram)
  num_frames = slen//SPECS_PER_FRAME

  mindbfs = origspect.minval
  maxdbfs = origspect.maxval

  # todo: load tmpfile, if any.
  # init any undefined opl3 frame configs with 
  # some kind of quick defaults.
  # update tmpfile format to include each of the following 
  # for each frame: cfg, ospect, tspect, fitness.
  # Also add some global metadata: which frame was last 
  # being worked on, src wavfile name, total num frames.
  tmpfile = temp_regfile_name
  loaded_cfgs = []
  try:
    tsize = os.path.getsize(tmpfile)
    infile = open(tmpfile,'rb')
    num_loaded_frames = tsize // 512
    while True:
      cfg = infile.read(512)
      if len(cfg)!=512:
        break
      loaded_cfgs.append(cfg)
    infile.close()
  except:
    tsize = 0
  print("Processing audio...")
  spects = []
  configs = []
  vects = [ ]
  pmax = -9999
  prev_vvals = None
  for frame in range(0,num_frames):
    ospect = origspect.spectrogram[frame*SPECS_PER_FRAME][0:-1]
    spects.append(deepcopy(ospect))
    if False: #frame<len(loaded_cfgs):
      # get working estimate from file
      rf = loaded_cfgs[frame]
      v = opl3.rfToV(rf)
      configs.append(deepcopy(rf))
      vects.append(deepcopy(v))
    else:      
      peaks = getRankedPeaks(ospect, -115.0, 0, True, 5, 5)
      # no estimate on file, make a simple additive-sine one.
      l = ''
      vvals = []
      for chan,p in enumerate(peaks):
        pheight, pfreq, _, _, _ = p
        pheight -= maxdbfs
        #print(f'{int(pheight):4d} ',end='')
        if pheight>pmax:
          pmax=pheight

        if chan>=18:
          break
        if pheight<-48.0:
          break 
        vfreq = pfreq/OPL3_MAX_FREQ
        vamp = (pheight+10)/-48  # -48dBFS...0 dBFS : float 1.0...0.0
        if vamp>1.0:
          vamp=1.0
        elif vamp<0.0:
          vamp=0.0
        vvals.append([float(vfreq),float(vamp)])      
      # sort vvals to align channel positions
      # to similar ones in the preceeding frame
      if prev_vvals is not None:
        used_js=[]
        new_vvals = []
        for i,(fi,vi) in enumerate(prev_vvals):
          bestdif = 3.0
          bestj = -1
          for j,(fj,vj) in enumerate(vvals):
            if j not in used_js:
              dif = abs(fj-fi)
              if dif<bestdif:
                bestdif = dif
                bestj = j
          if bestdif<=0.05:
            used_js.append(bestj)
            fj, vj = vvals[bestj]
            new_vvals.append([fj,vj])
        if len(used_js) < len(vvals):
          for j,(fj,vj) in enumerate(vvals):
            if j not in used_js:
              new_vvals.append([fj,vj])
        '''
        for i,(f,v) in enumerate(prev_vvals):
          print(f'({f:6.4f},{v:6.4f}), ',end='')
        print()
        for i,(f,v) in enumerate(vvals):
          print(f'({f:6.4f},{v:6.4f}), ',end='')
        print()
        for i,(f,v) in enumerate(new_vvals):
          print(f'({f:6.4f},{v:6.4f}), ',end='')
        print()
        exit()
        '''
        vvals = deepcopy(new_vvals)
      prev_vvals = deepcopy(vvals)
      #print()
      ch = 0
      rf = opl3.initRegFile()
      v = opl3.rfToV(rf)
      for vfreq, vamp in vvals:  # for the loudest peaks, assign a sine
        opl3.setNamedVecElemFloat(v,f'Freq.c{ch}',vfreq)  # Peak freq    
        opl3.setNamedVecElemFloat(v,f'KeyOn.c{ch}',1.0)   # Key ON
        opl3.setNamedVecElemFloat(v,f'SnTyp.c{ch}',1.0)   # 2-op AM
        oidxs = opidxs_per_chan[ch]
        for q,o in enumerate(oidxs):
          opl3.setNamedVecElemFloat(v,f'AttnLv.o{o}',vamp)  # amplitude
          opl3.setNamedVecElemInt(v,f'FMul.o{o}',1)   # op phase multiple
          opl3.setNamedVecElemInt(v,f'KSAtnLv.o{o}',1)#2) # some attenuation at higher freqs
        ch+=1
        if ch>=18:
          break
      if ch<18:
        for cq in range(ch,18):
          #opl3.setNamedVecElemFloat(v,f'KeyOn.c{cq}',0.0)   # Key ON
          oidxs = opidxs_per_chan[cq]
          for q,o in enumerate(oidxs):
            opl3.setNamedVecElemFloat(v,f'AttnLv.o{o}',1.0)  # amplitude
            opl3.setNamedVecElemInt(v,f'FMul.o{o}',1)   # op phase multiple
            opl3.setNamedVecElemInt(v,f'KSAtnLv.o{o}',1)#2) # some attenuation at higher freqs
      configs.append(opl3.vToRf(v))
      vects.append(deepcopy(v))
  #print(f'{pmax=}')
  print("saving regfile...")
  # write regfile
  with open(temp_regfile_name,'wb') as fout:
    for fidx in range(0,num_frames):
      fout.write(configs[fidx])
  print("saved. making project...")

  proj_name = os.path.basename(tmpfile)
  pfile = W2VProj(proj_name, num_frames, spects, configs)
  #pprint(pfile.serialize(),indent=2)
  #exit()
  print('ready.')

  frame = 0
  while manualEvtLoop:
    ospect = origspect.spectrogram[frame*SPECS_PER_FRAME][0:-1]
    pygame.draw.rect(screen, (0,0,0),(0,0,ww,hh))
    tspect = None
    #v = vects[frame]
    #fit, tspect = opl3.fitness(ospect, v)
    f = pfile.frames[frame]
    v = f.synth_cfg_vect
    tspect = f.synth_spect
    fit = f.fitness
    if tspect is not None:
      c = (255,0,0)
      if fit<=150:
        c=(0,255,0)
      elif fit<=300:
        c=(255,255,0)
      plotHalfSpect(tspect,-115,0,c,0)
      plotColorText(f'{fit:10.5f}',c,8,8)
    plotHalfSpect(ospect,-115,0,(255,255,255),0)
    pygame.display.update()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
        return HARD_QUIT
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return SOFT_QUIT
        elif event.key == pygame.K_LEFT:
          frame -= 1
          if frame < 0:
            frame = num_frames-1
        elif event.key == pygame.K_RIGHT:
          frame += 1
          if frame >= num_frames:
            frame = 0
        elif event.key == pygame.K_p:
          print('rendering...')
          opl3 = OPL3()
          opl3._sample_overflow = 0
          opl3._output = bytes()
          origstate = b'\0'*512
          framedifs = []
          for fidx in range(0,num_frames):
            regs = pfile.frames[fidx].synth_cfg_bytes
            difs = []
            for i in range(0,512):
              a = origstate[i]
              b = regs[i]
              if a!=b:
                difs.append([i,b])
            framedifs.append(difs)
            opl3._writeRegFile(regs)
            origstate = deepcopy(regs)
            opl3._render_ms(1000/57.4149659864)
            #opl3._render_samples(768)  
          vgmdata = b''
          for di,dframe in enumerate(framedifs):
            #print(f'{di:3d} ')
            for r,v in dframe:
              #print(f'(0x{r:03X},0x{v:02x}),')
              if r>=0x100:
                vgmdata+=struct.pack('BBB',0x5F,r&0xff,v)
              else:
                vgmdata+=struct.pack('BBB',0x5E,r,v)
            vgmdata+=struct.pack("<BH",0x61,int(44100/57.4149659864))
            #print()
          vgmdata+=struct.pack('B',0x66)
          writeVgmFile(vgmdata)
          print('playing...')
          stereo_wave, wave = opl3.stereoBytesToNumpy(opl3._output)
          stereo_wave=pygame.sndarray.make_sound(stereo_wave)
          pygame.mixer.Sound.play(stereo_wave)
          while pygame.mixer.get_busy(): 
            pygame.time.Clock().tick(10)  

# -----------------------------------------------------------------------------
# arrows move cursor, if any, which could be either on spectrum or spectrogram
# spectrum: 
#     l/r select spectrum (vertical cursor on spectrogram)
#     u/d select frequency cutoff (horizontal cursor on spectrogram, vert on spectrum)
# spectrogram: 
#     l/r adjust frequency cutoff (horizontal cursor on spectrogram, vert on spectrum)
#     u/d select amplitude cutoff (horizontal cursor on spectrum)
# -----------------------------------------------------------------------------
def loop():
  global roi
  global roi,cutoff_freq,amp_cutoff, amp_cutoff_y
  global last_clicked

  hsr = origspect.sample_rate/2
  for event in pygame.event.get():
    if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
      return True# raise SystemExit
    elif event.type == pygame.MOUSEWHEEL:
      handleMouseWheel(event)
    elif event.type == pygame.MOUSEBUTTONDOWN:
      handleMouseDown(event)      
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_ESCAPE:
        return True
      elif event.key == pygame.K_a:
        doquit, yayframes = analyzePeakRuns()
        if doquit == HARD_QUIT:
          return True
        elif doquit == NO_QUIT:
          makeVGM(yayframes)
        #drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_f:
        fastAnalyze()
        drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_p:
        playWave()
      elif event.key == pygame.K_t:
        do_quit = testTrainingSet()
        if do_quit == HARD_QUIT:
          return True
        drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_l:
        do_quit = loadRegfile()
        if do_quit == HARD_QUIT:
          return True
        drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_m:
        do_quit = manualControls()
        if do_quit == HARD_QUIT:
          return True
        drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_n:
        do_quit = bruteForce(brute=False, genetic=False, ai=True)
        if do_quit == HARD_QUIT:
          return True
        drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_g:
        do_quit = bruteForce(brute=False, genetic=True, ai=False)
        if do_quit == HARD_QUIT:
          return True
        drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_b:
        do_quit = bruteForce(brute=True, genetic=False, ai=False)
        if do_quit == HARD_QUIT:
          return True
        drawSpect(origspect,0,0,screen_width,screen_height)
      elif event.key == pygame.K_d:
        allthumbs = {}
        for roi in range(44,46):
          allthumbs[roi] = spectThumbnails(origspect.spectrogram[roi][0:-1])
        pprint.pprint(allthumbs, indent=2)
      elif event.key == pygame.K_UP:
        if last_clicked == CLICKED_SPECTROGRAM:
          if cutoff_freq<(hsr/2-4):
            cutoff_freq+=4
        elif last_clicked == CLICKED_SPECTRUM:
          pass
        newROI()
      elif event.key == pygame.K_DOWN:
        if last_clicked == CLICKED_SPECTROGRAM:
          if cutoff_freq>-4:
            cutoff_freq-=4
        elif last_clicked == CLICKED_SPECTRUM:
          pass
        newROI()
      elif event.key == pygame.K_LEFT:
        if last_clicked == CLICKED_SPECTROGRAM:
          if roi>-1:
            roi-=1
        elif last_clicked == CLICKED_SPECTRUM:
          pass
        newROI()
      elif event.key == pygame.K_RIGHT:
        if last_clicked == CLICKED_SPECTROGRAM:
          if roi<len(origspect.spectrogram):
            roi+=1
        elif last_clicked == CLICKED_SPECTRUM:
          pass
        newROI()
# -----------------------------------------------------------------------------  
def main():
  global origspect
  if origspect.spectrogram is not None:
    drawSpect(origspect,0,0,screen_width,screen_height)
    done=False
    while not done:
      done=loop()
      clock.tick(60)    

###############################################################################
# ENTRYPOINT
###############################################################################
if __name__ == '__main__':
  main()
###############################################################################
# EOF
###############################################################################
