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
import time
import pygame
from pygame.locals import *
import numpy as np
import math
from scipy import signal
import numpy as np
from src import spect as sp
from src import opl_emu as opl
from src import gene

import numpy as np
import torch
import torch.nn as nn  


import struct
import gzip
import sys
import pyopl
import random
import datetime
import pprint
import os
from copy import deepcopy

# randomize the PRNG
random.seed(datetime.datetime.now().timestamp())
# -----------------------------------------------------------------------------
screen_width=1920 
screen_height=1080
origspect=None
#SLICE_FREQ = 86.1328125  # frequency at which the synth parameters are altered
                         # (exactly two specra per synth frame)
SLICE_FREQ = 91.552734375   # old setting for arduino
MAX_SYNTH_CHANS = 18   # polyphony of synth (num independent oscillators)
OPL3_MAX_FREQ = 6208.431  # highest freq we can assign to a voice (fnum, block)
SPECT_VERT_SCALE = 3  # set max vert spect axis to 7350 Hz max rather than the orig 22050 Hz  

GENE_MAX_GENERATIONS = 500

# -----------------------------------------------------------------------------
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
  wavname = 'HAL 9000 - Human Error.wav'
  #wavname = 'JFK Inaguration.wav'
  #wavname = 'Ghouls and Ghosts - The Village Of Decay.wav'
  #wavname = 'Portal-Still Alive.wav'
infolder = 'input\\'
outfolder = 'output\\'

vpath = outfolder+wavname[0:-3]+"vgz"
origspect = sp.spect(wav_filename='input\\'+wavname,nperseg=4096,quiet=False,clip=False)
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
  rw = (((roi+1)*ww/slen)-rx)+1

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
  # keeps track of runs of spectral peaks
  freq_runs = {}
  prior_peaks = []
  for roi in range(0,slen):  # for each spectrum in spectrogram
    # Check if user wants to close program
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  
        return True# raise SystemExit    
    # start time of this spectrum within the sound
    t = roi*origspect.dur_secs/slen        

    # draws a vertical slice of spectral peaks over top of 
    # spectrum[roi] of the spectrogram, and returns a list of peak 
    # (freq,height) values sorted by descending height
    peaks = overlayPeaksOnSpectrum(roi,slen,origspect.spectrogram[roi])

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
        if mindif<35:          
          #print(f'{f:0.1f}->{closest[1]:0.1f} ', end='')
          continuation=True
          rkey = closest[3]
          freq_runs[rkey].append((t,f,h))
      if not continuation:
        #print(f'{f:0.1f} ', end='')
        rkey = (t,f)
        freq_runs[rkey] = [(t,f,h)] # new run
      new_prior_peaks.append((f,h,rkey))
    prior_peaks = new_prior_peaks
    #print()
    pygame.display.update()

  draw_runs = False
  if draw_runs:     # draw the freq runs in dif colors
    ll = len(origspect.spectrogram[roi])
    ww = int(screen_width)
    hh = int(screen_height) - T_AXIS_HEIGHT
    hsr =  (origspect.sample_rate/2)/SPECT_VERT_SCALE
    minv = origspect.minval
    maxv = origspect.maxval
    pygame.draw.rect(screen, (0,0,0), (0,0,ww,hh))
    for rkey in freq_runs:
      run = freq_runs[rkey]
      if len(run)>=3:
        #print(f'{rkey}:')
        c0 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        t,f,h = run[0]
        #print(f'    {(t,f,h)}')    
        r0 = t*slen/origspect.dur_secs
        x0 = int(r0*ww/slen)
        y0 = hh-int(f*hh/hsr)
        h0 = h
        for t,f,h in run[1:]:
          r1 = t*slen/origspect.dur_secs
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
    return False
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
  if roi>=0 and roi<len(origspect.spectrogram):
    plotTestSpect(origspect.spectrogram[roi],origspect.minval,origspect.maxval)
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
# Start of OPL3 stuff
# -----------------------------------------------------------------------------
max_freq_per_block = [48.503,97.006,194.013,388.026,776.053,1552.107,3104.215,6208.431]
fsamp = 14318181.0/288.0
# these are the operator indexes per each of the 18 output channels
# in 2-op mode.
opidxs_per_chan = [
  (0,3),(1,4),(2,5),(6,9),(7,10),(8,11),(12,15),(13,16),(14,17),
  (18,21),(19,22),(20,23),(24,27),(25,28),(26,29),(30,33),(31,34),(32,35)]
# -----------------------------------------------------------------------------
# given a freq in hz, returns the OPL3 F-Num and Blocknum needed to achieve it.
# -----------------------------------------------------------------------------
def getFnumBlock(freq):
  for block,maxfreq in enumerate(max_freq_per_block):
    if maxfreq>=freq:
      break
  fnum = round(freq*pow(2,19)/fsamp/pow(2,block-1))
  return fnum,block
# -----------------------------------------------------------------------------
# inverse of above function
# -----------------------------------------------------------------------------
def getFreq(fnum, block):
  return fnum/(pow(2,19)/fsamp/pow(2,block-1))
# -----------------------------------------------------------------------------
def showRegDict(opl_reg_dict):
    keys = list(opl_reg_dict.keys())
    keys.sort()
    for (b,r) in keys:
      v = opl_reg_dict[(b,r)]
      print(f'    {b},${r:02X} : ${v:02X}')  
# -----------------------------------------------------------------------------
# Setup the OPL emulator with the specified register values, generate 4096 
# audio samples, and return resultant frequency spectrum.
# -----------------------------------------------------------------------------
def renderOPLFrame(opl_reg_dict):
  o = opl.opl_emu()
  o.do_init()
  for (b,r) in opl_reg_dict:
    v = opl_reg_dict[(b,r)]
    o.write(b,r,v)
  o._output = bytes()
  o._render_samples(4096)  
  w = []
  for i in range(0,len(o._output),4):
    l=struct.unpack('<h',o._output[i:i+2])[0]
    r=struct.unpack('<h',o._output[i+2:i+4])[0]
    w.append((l+r)//2)
  w = np.array(w, dtype="int16")
  if w.sum():
    newspect = sp.spect(wav_filename = None, sample_rate=44100,samples=w,nperseg=4096, quiet=True)
  else:
    newspect  = None
    #showRegDict(opl_reg_dict)
  return newspect
# -----------------------------------------------------------------------------
# $001        %00100000
# $105        %00000001
# $008        %00000000
# $0BD        %00000000 - percussion currently unused
# $X60...$X75 %11111111 - ADSR stuff, currently fixed to be always-on.
# $X80...$X95 %00001111 - 
#
# $104        %00CCCCCC - 2op / 4op selection bits
#
# per-operator settings (22*2 operators) ------
#
# $X20...$X35:%0010MMMM 
# $X40...$X55 %KKTTTTTT - ksl atten, total level
# $XE0...$XF5 %00000WWW - waveform sel
#
# per-channel settings (9*2 channels) -----
# 
# $XA0...$XA8 %FFFFFFFF - fnum low
# $XB0...$XB8 %00KBBBFF - keyon, block, fnum hi
# $XC0...$XC8 %1111FFFC - feedback, connection set

# -----------------------------------------------------------------------------
# regfile indexes [0...512) known to have a function in OPL3
# -----------------------------------------------------------------------------
permutable_regidxs = [0x104]

permutable_splits = { 0x104:[('_','00'),('4o0',1),('4o1',1),('4o2',1),('4o3',1),('4o4',1),('4o5',1)]}

for i in range(0,0x16):
  permutable_regidxs.append(0x020+i)
  permutable_splits[0x020+i] = [('_',"0010"),('OMulA',4)]
  permutable_regidxs.append(0x120+i)
  permutable_splits[0x120+i] = [('_',"0010"),('OMulB',4)]
  permutable_regidxs.append(0x040+i)
  permutable_splits[0x040+i] = [('KSLA',2),('TlvA',6)]
  permutable_regidxs.append(0x140+i)
  permutable_splits[0x140+i] = [('KSLB',2),('TlvB',6)]
  permutable_regidxs.append(0x0E0+i)
  permutable_splits[0x0E0+i] = [('_',"00000"),('WavA',3)]
  permutable_regidxs.append(0x1E0+i)
  permutable_splits[0x1E0+i] = [('_',"00000"),('WavB',3)]

for i in range(0,0x09):
  permutable_regidxs.append(0x0A0+i)
  permutable_splits[0x0A0+i] = [('FnLA',8)]

  permutable_regidxs.append(0x1A0+i)
  permutable_splits[0x1A0+i] = [('FnLB',8)]

  permutable_regidxs.append(0x0B0+i)
  permutable_splits[0x0B0+i] = [('_',"00"),('KonA',1),('BlkA',3),('FnHA',2)]

  permutable_regidxs.append(0x1B0+i)
  permutable_splits[0x1B0+i] = [('_',"00"),('KonB',1),('BlkB',3),('FnHB',2)]

  permutable_regidxs.append(0x0C0+i)
  permutable_splits[0x0C0+i] = [('_',"1111"),('FbA',3),('CcB',1)]

  permutable_regidxs.append(0x1C0+i)
  permutable_splits[0x1C0+i] = [('_',"1111"),('FbB',3),('CcB',1)]

permutable_regidxs.sort()

permute_counts = {}
for i,ridx in enumerate(permutable_regidxs):  
  b = (ridx>>8)&1
  r = ridx&0xff
  for si,(ll,bb) in enumerate(permutable_splits[ridx]):
    if ll!='_':
      permute_counts[(b,r,si)] = 0

# -----------------------------------------------------------------------------
# resets the opl configuration to just the fixed-value parts
# -----------------------------------------------------------------------------
def initRegs():
  opl_regs = { (0,0x01): 0x20, (1,0x05): 0x01, (0,0x08): 0x00, (0,0xBD): 0, (1,0x04):0 }
  for b in range(0,2):
    for j in range(0,0x16):
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

# $001        %00100000
# $105        %00000001
# $008        %00000000
# $0BD        %00000000 - percussion currently unused
# $X60...$X75 %11111111 - ADSR stuff, currently fixed to be always-on.
# $X80...$X95 %00001111 - 
#
# $104        %00CCCCCC - 2op / 4op selection bits
#
# per-operator settings (22*2 operators) ------
#
# $X20...$X35:%0010MMMM 
# $X40...$X55 %KKTTTTTT - ksl atten, total level
# $XE0...$XF5 %00000WWW - waveform sel
#
# per-channel settings (9*2 channels) -----
# 
# $XA0...$XA8 %FFFFFFFF - fnum low
# $XB0...$XB8 %00KBBBFF - keyon, block, fnum hi
# $XC0...$XC8 %1111FFFC - feedback, connection set  
# -----------------------------------------------------------------------------
# given a byte value, start bit idx [0,7] and end bit idx [0,7]
# extracts the bit string from byte b and converts it to a float mag [0.0,1.0]
# -----------------------------------------------------------------------------
def getBitField(v,lbl,sb,eb):
  slen = sb-eb
  mag = (1<<slen)-1
  v = (v<<(8-sb)) & 0xff
  v = (v>>(8-slen))
  r = float(v)/float(mag)
  #print(f'{lbl=}: {v=} {r=:5.2f}')
  return r

# -----------------------------------------------------------------------------
# convert an opl congiguration dict into a vector representation like what was
# used to training the AI. (Each field gets a float [0.0,1.0] in the vector)
# -----------------------------------------------------------------------------
def regDictToVect(opl_regs):
  global permutable_regidxs, permutable_splits
  permarray = []
  for i,ridx in enumerate(permutable_regidxs):
    b = (ridx>>8) & 1
    r = ridx & 0xff
    v = 0
    if (b,r) in opl_regs:
      v = opl_regs[(b,r)]
    sbit = 8
    for (lbl,bts) in permutable_splits[ridx]:
      if lbl == '_':
        sbit-=len(bts)
      else:
        ebit = sbit-bts
        permarray.append(getBitField(v,lbl,sbit,ebit))
        sbit = ebit
  return permarray
# -----------------------------------------------------------------------------
# convert a float configuration vector into a configuration dictionary
# -----------------------------------------------------------------------------
def intToBits(v,nbit):
  bs = bin(v)[2:]
  pbs = '0'*(nbit-len(bs))+bs
  #print(f'      intToBits({v=},{nbit=}): {bs=} {pbs=}')
  return pbs

def bitsToInt(bs):
  return int(bs,2)

def insBitsFloat(v,lbl,sbit,nbit,f):
  #if 'Kon' in lbl:
  #  print(f'      insBitsFloat({v=},{lbl=},{sbit=},{nbit=},{f=:5.2f}): ',end='')
  orig = intToBits(v,8)
  if nbit>1:  
    mag = (1<<nbit)-1
    newv = intToBits(round(f*mag),nbit)
  else:
    mag = 1
    newv="0"
    if f >= 0.45:
      newv = "1"
  a = 8-sbit
  b = a+nbit
  bs = orig[0:a] + newv[-nbit:] + orig[b:]
  i = bitsToInt(bs)
  #if 'Kon' in lbl:
  #  print(f'{mag=} origv=%{orig} newv=%{newv} {a=} {b=} {bs=} {i=}')
  return i

def insBitsString(v,sbit,nbit,s):
  mag = (1<<nbit)-1
  f = int(s,2)/mag
  #print(f'      insBitsString({v=},{sbit=},{nbit=},{s=}): {mag=} {f=}')
  return insBitsFloat(v,'_',sbit,nbit,f)

def vectToRegDict(vec):
  global permutable_regidxs, permutable_splits
  regs = initRegs()
  j=0
  for i,ridx in enumerate(permutable_regidxs):
    b = (ridx>>8) & 1
    r = ridx & 0xff
    #print(f'vectToRegDict(): ({b},${r:01X}) ',end='')
    if not (b,r) in regs:
      regs[(b,r)]=0x00
      print('(created)')
    #else:
    #  print()

    v = 0x00
    sbit = 8
    for (lbl,bts) in permutable_splits[ridx]:
      #print(f'    ({lbl=}:{bts=}):')
      if lbl == '_':
        v=insBitsString(v,sbit,len(bts),bts)
        regs[(b,r)]=v
        sbit-=len(bts)
      else:
        ebit = sbit-bts
        f = vec[j]
        j+=1
        v=insBitsFloat(v,lbl,sbit,sbit-ebit,f)
        regs[(b,r)]=v
        sbit = ebit

  return regs

# -----------------------------------------------------------------------------
def regFileToDict(regfile):
  regdict = {}
  for i,v in enumerate(regfile):
    b = (i>>8)&1
    r = i&0xff
    regdict[(b,r)]=v
  return regdict
# -----------------------------------------------------------------------------
def regDictToFile(regdict):
  sbin = b'\0' * 512
  for (b,r) in regdict:
    v = regdict[(b,r)]
    idx = b*256 + r
    try:
      sbin = sbin[0:idx]+struct.pack('B',v)+sbin[idx+1:]
    except:
      print(f'{b=} {r=:02X} {v=}')
      exit()
  return sbin
# -----------------------------------------------------------------------------
# genetic algo calls this to impose a mutation on a genome (opl register file)
# -----------------------------------------------------------------------------
def mutatefcn(regfile):
  global permutable_regidxs
  regdict = regFileToDict(regfile)

  ncdist = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,5,6,7]
  numchanges = random.choice(ncdist)

  for c in range(0,numchanges):
    j = random.choice(permutable_regidxs)
    b = (j>>8) & 1      # bank
    r = j & 0xff        # reg
    if random.randint(0,5) == 0:  # 20% chance to completely randomize the reg
      regdict[(b,r)] = random.randint(0,255)
    else:                         # else do some small shift
      v = regdict[(b,r)]
      d = random.randint(0,1)
      if d==0:
        d=-1
      if r==0x04:  # 0x104: six connection sel bits
        a = random.randint(0,5)
        v ^= (1<<a)
      elif r>=0x20 and r<=0x35:
        m = v & 0b1111
        m = m + d 
        if m<0:
          m=0
        elif m>15:
          m=15
        v = 0b00100000|m
      elif r>=0x40 and r<=0x55:
        c = random.randint(0,1)  
        if c==0:  # ksl - freq atten
          k = v>>6
          k=k+d
          if k<0:
            k=0
          elif k>3:
            k=3
          v=(v&0b111111)|(k<<6)
        else:     # tl - total level
          t = v & 0b111111
          t+=d
          if t<0:
            t=0
          elif t>0b111111:
            t=0b111111
          v = (v&0b11000000) | t
      elif r>=0xe0 and r<=0xf5:  # waveform
        v+=d
        if v>7:  
          v=7
        elif v<0:
          v=0
      elif (r>=0xa0 and r<=0xa8) or (r>=0xb0 and r<=0xb8):  # f-num: low byte or %00KBBBFF
        if random.randint(0,3) == 0:  # toggle keyon
          if r<0xb0:
            r+=0x10
          v=v^0b0010000
        else:    # bump frequency
          if r<0xb0:
            ofs = r-0xa0
          else:
            ofs = r-0xb0
          aa = regdict[(b,0xa0+ofs)]
          bb = regdict[(b,0xb0+ofs)]
          fnum = aa|((bb&3)<<8)
          block = (bb>>2)&7

          freq = getFreq(fnum, block)
          freq +=d
          if freq<0:
            freq=0
          elif freq>OPL3_MAX_FREQ:
            freq=OPL3_MAX_FREQ
          fnum,block = getFnumBlock(freq)

          # set both freq regs and keyon and continue
          regdict[(b,0xa0+ofs)] = fnum&0xff
          regdict[(b,0xb0+ofs)] = 0b00100000 | ((fnum>>8)&3) | (block<<2)
          continue
      elif r>=0xc0 and r<=0xc8:     # 1111FFFC  : feedback, connection sel
        if random.randint(0,1) == 0:  # bump feedback
          f=(v>>1)&0b111
          f+=d
          if f<0:
            f=0
          elif f>7:
            f=7
          v=(v&0b11110001) | (f<<1)
        else:     
          v=v^1                       # toggle connection sel
      regdict[(b,r)] = v
  rf = regDictToFile(regdict)
  return rf
# -----------------------------------------------------------------------------
#one-off setting
#
# $104        %00CCCCCC - 2op / 4op selection bits
#
# per operator settings
#
# $X20...$X35:%0010MMMM - op mult
# $X40...$X55 %KKTTTTTT - ksl atten, total level
# $XE0...$XF5 %00000WWW - waveform sel
#
# per-channel settings (9*2 channels) -----
# 
# $XA0...$XA8 %FFFFFFFF - fnum low
# $XB0...$XB8 %00KBBBFF - keyon, block, fnum hi
# $XC0...$XC8 %1111FFFC - feedback, connection set
# -----------------------------------------------------------------------------
# repeatedly calls calls the genetic algorithm's generate() method and shows
# the current best result.
#
# Returns the best register file after max iterations.
# -----------------------------------------------------------------------------
def improveMatch(roi, ospect, g):
  global screen,screen_width,screen_height
  global GENE_MAX_GENERATIONS
  # width and height of test spectrs drawn here
  ww = screen_width
  hh = screen_height//4
  # y ofs to draw position
  yofs = screen_height//2

  # blank whole bottom half so we can show convengence plor in bottom quarter
  pygame.draw.rect(screen,(0,0,0),(0,yofs,ww,hh*2))  
  initfit = None
  lastfit = 0
  lx = ly = px = py = 0
  tstart = time.time()
  for iter in range(0,GENE_MAX_GENERATIONS):    
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  
        return g.p[0].genome    
    tspect = g.p[0].spect
    if tspect is not None:
      fit = g.p[0].fit
      if initfit is None:
        initfit = fit
      tnow = time.time()
      tdelt = tnow-tstart
      tstart = tnow
      print(f'Generation: {iter:3d}, fit:{fit:0.9f}, tdelt:{tdelt:0.2f} ',end='')
      sys.stdout.flush()
      if fit!=lastfit:
        lastfit=fit
        pygame.draw.rect(screen,(0,0,0),(0,yofs,ww,hh))  
        plotTestSpect(tspect,-115.0,0,gcolor=(255,255,0),yofs=yofs)
        plotTestSpect(ospect,-115.0,0,gcolor=(255,255,255),yofs=yofs)
      px = iter*ww//GENE_MAX_GENERATIONS
      py = (screen_height-1)-(hh*fit//initfit)
      if iter>0:
        pygame.draw.line(screen,(0,255,0),(lx,ly),(px,py))
      lx=px
      ly=py
      pygame.display.update()
    g.generate(mutatefcn)

  regfile = g.p[0].genome
  return regfile
# -----------------------------------------------------------------------------
# fitness function for genetic algorithm.
#
# Renders the opl waveform for the given genome (opl register set), then 
# returns the difference between its frequency spectrum and ospect, the ideal.
# -----------------------------------------------------------------------------
def fitfunc(ospect,regfile):
  rd = regFileToDict(regfile)
  tspect = renderOPLFrame(rd)
  if tspect is None:
    return 999999999999, None
  tspect = tspect.spectrogram[0]
  dif = 0
  for i in range(2048):
    a = ospect[i]-tspect[i]
    dif += a*a
  return math.sqrt(dif), tspect
# -----------------------------------------------------------------------------
# WIP EXPERIMENT- Try to use a neural network to find the best OPL3 settings
# for each frame of the spectrogram.
# -----------------------------------------------------------------------------

# TODO: this is duplicated in the training code. We should
# make a shared model-definition file!

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )        
        # Fully connected layers
        self.model = nn.Sequential(
            nn.Linear(32 * 2048, 2048),  # Adjusted input size for flattened Conv1d output
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 290)
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, 2048]        
        x = self.conv_layers(x)        
        x = x.view(x.size(0), -1)  # Shape becomes [batch_size, 32 * 2048]        
        x = self.model(x)        
        return x
# -----------------------------------------------------------------------------
# Code to brute-force a solution using either a (slow) genetic algorithm or a 
# convolutional neural-network.
# -----------------------------------------------------------------------------
def bruteForce(genetic = False, ai = True):
  global origspect
  global screen
  global screen_width
  global screen_height  
  ww = int(screen_width)
  hh = int(screen_height//4)
  slen = len(origspect.spectrogram)

  try:
    if ai:
      model = Model()
      model.load_state_dict(torch.load('models/torch_model.pth'))
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
a 1 GB training set and 600MB model. Stay Tuned!

#########################################################
''')

    return
  # init intermediate output file of opl3 reg settings for
  # later conversion to a VGM. (TODO!)
  with open(outfolder+'reg_files.bin','wb') as f:
    pass

  # brute-force loop: -----
  # for every-other spectrum in original spectrogram:
  for roi in range(0,slen,2): 

    # check if user is a quitter
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return

    # this is the spectrum we want recreate.
    ospect = origspect.spectrogram[roi]

    # NEURAL NETWORK FUN ------------------
    if ai:    
      # reformat the original spectrum to 
      # an input vector for the inferencer
      inp = []
      for i,b in enumerate(ospect[0:-1]):
        o = abs(int(b))
        if o>255:
          o=255
        o = 255-o
        inp.append(float(o)/255.0)  
      # Shape becomes (1, 2048)
      sample_input = torch.tensor(inp, dtype=torch.float32).reshape(1, 2048)
      # Disable gradient calculation for inference
      with torch.no_grad():
          predicted_output = model(sample_input)
      # make prediction
      predicted_output = predicted_output.numpy().flatten()
      # convert output cfg vector to an opl3 register dictionary    
      regdict = vectToRegDict(predicted_output)
    # -------------------------------------

    # GENETIC ALGORITHM FUN ---------------
    if genetic:
      # Init an empty population for the genetic algorithm,
      # noting the spectrum we hope to achieve.
      print('Making initial population')
      g = gene.gene(1000, ospect, fitfunc)
      for i in range(0,1000):
        opl_regs = initRegs()
        for i in permutable_regidxs:
          b = (i>>8)&1
          r = i&0xff
          opl_regs[(b,r)] = random.randint(0,255)
        genome = regDictToFile(opl_regs)
        fit, spect = fitfunc(ospect, genome)    
        g.add(i, fit, spect, genome)
      print('ok')
      # Do a (slow) genetic annealing process to
      # try to improve the result. 
      regfile = improveMatch(roi, ospect, g)
    # -------------------------------------

    # give register cfg to opl3 emulator and render a spectrum
    tspect = renderOPLFrame(regdict)

    pygame.draw.rect(screen,(0,0,0),(0,0,ww,hh))
    plotTestSpect(ospect,-115,0,(255,255,255))   # plot original spect

    if tspect is not None: 
      tspect = tspect.spectrogram[0]
      plotTestSpect(tspect,-115,0,(255,255,0))   # plot spect from predicted config
    
    pygame.display.update()

    # Output our best register file result to intermediate file, 
    # for later conversion to VGM. (TODO!!)
    with open(outfolder+'reg_files.bin','ab') as f:
      regfile = regDictToFile(regdict)
      f.write(regfile)
# -----------------------------------------------------------------------------
# Make sequence to init the OPL3 chip.
#
# Currently, we set it up to do 18 plain sine waves.
# -----------------------------------------------------------------------------
prev_sets = {}
oplemu = None

def opl3init():
  global prev_sets, oplemu
  prev_sets = {}

  oplemu = opl.opl_emu()
  res = b''
  # enable opl3 mode
  r = 0x05
  v = 0x01
  oplemu.write(1,r,v);

  res+=struct.pack('BBB',0x5f,r,v)
  # enable waveform select
  r = 0x01
  v = 0x20
  oplemu.write(1,r,v);
  res+=struct.pack('BBB',0x5f,r,v)

  for chan in range(0,18):
    opidxs = opidxs_per_chan[chan]
    if chan<9:      
      # sustain, vibrato,opfreqmult
      r = 0x20 + opidxs[0]
      v = 0x21
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x20 + opidxs[1]
      v = 0x21
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # keyscalelevel, output level
      # setting keyscale level to 6.0 dB of attenuation per rise in octave
      # (we really want more than this and need to implement something ourselves)
      r = 0x40 + opidxs[0]
      v = 0x30  # 30
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x40 + opidxs[1]
      v = 0x30
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # attack rate, decay rate
      r = 0x60 + opidxs[0]
      v = 0xff
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x60 + opidxs[1]
      v = 0xff
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # sust level, release rate
      r = 0x80 + opidxs[0]
      v = 0x0f
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x80 + opidxs[1]
      v = 0x0f
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # output channels L&R and additive synth
      r = 0xc0 + chan
      v = 0x31
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      # waveform select sine
      r = 0xe0 + opidxs[0]
      v = 0x00
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0xe0 + opidxs[1]
      v = 0x00
      oplemu.write(0,r,v);
      res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      a = opidxs[0]-18
      b = opidxs[1]-18
      # sustain, vibrato,opfreqmult
      r = 0x20 + a
      v = 0x21
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x20 + b
      v = 0x21
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # keyscalelevel, output level
      r = 0x40 + a
      v = 0x30
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x40 + b
      v = 0x30    #  30
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # attack rate, decay rate
      r = 0x60 + a
      v = 0xff
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x60 + b
      v = 0xff
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # sust level, release rate
      r = 0x80 + a
      v = 0x0f
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x80 + b
      v = 0x0f
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # output channels L&R and additive synth
      r = 0xc0 + chan
      v = 0x31
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      # waveform select sine (todo: use different waveforms when appropriate!)
      r = 0xe0 + a
      v = 0x00
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0xe0 + b
      v = 0x00
      oplemu.write(1,r,v);      
      res+=struct.pack('BBB',0x5f,r,v)
  return res
# -----------------------------------------------------------------------------
# Returns an OPL3 register set sequence to set a specified channel to 
# the specified frequency, volume, and/or key on/off.
# -----------------------------------------------------------------------------
def opl3params(freq,namp,chan, keyon):
  global prev_sets, oplemu
  fnum, block = getFnumBlock(freq)
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
        oplemu.write(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
      else:
        chan-=9
        r = 0xb0 + chan - 9
        v = ((fnum>>8)&3) | (block<<2)
        oplemu.write(1,r,v);      
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
        r = 0x40 + opidxs[0]  # set volume
        v = aval
        oplemu.write(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
        r = 0x40 + opidxs[1]
        v = aval
        oplemu.write(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
      if (pchan is None) or (pchan[0]!=freq):
        r = 0xA0 + chan  # set low bits of frequency
        v = fnum&0xff
        oplemu.write(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
        r = 0xb0 + chan  # set key-on and high bits of frequency
        v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
        oplemu.write(0,r,v);              
        res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      a = opidxs[0] - 18
      b = opidxs[1] - 18
      if (pchan is None) or (pchan[1]!=namp):
        r = 0x40 + a # volume
        v = aval
        oplemu.write(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
        r = 0x40 + b
        v = aval
        oplemu.write(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
      if (pchan is None) or (pchan[0]!=freq):
        r = 0xA0 + chan # low bits of freq
        v = fnum&0xff
        oplemu.write(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
        r = 0xb0 + chan - 9    # key-on and high bits of freq
        v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
        oplemu.write(1,r,v);              
        res+=struct.pack('BBB',0x5f,r,v)
    prev_sets[schan] = (freq,namp,keyon)


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
  global oplemu
  minv = origspect.minval
  maxv = origspect.maxval

  print('Doing spectral peak-detection.')
  all_peaks = []
  slices=0
  ll = len(origspect.spectrogram)
  tt = origspect.maxtime - origspect.mintime
  for r in range(0,ll):
    all_peaks.append(getRankedPeaks(origspect.spectrogram[r],minv,maxv,True))
    slices+=1
  max_amp = 0
  min_amp = 9999999
  max_height = -999999
  min_height = 9999999

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
    oplemu.render_ms(1000/SLICE_FREQ)
    outvgm+=struct.pack("<BH",0x61,int(44100/SLICE_FREQ))

  outvgm+=b'\x66'
  gd3_ofs = len(outvgm)-0x14
  gd3_dat = makeGD3()
  outvgm+=gd3_dat
  vgm_eof_ofs = len(outvgm)-0x04
  outvgm=outvgm[0:4]+struct.pack('<I',vgm_eof_ofs)+outvgm[8:0x14]+struct.pack('<I',gd3_ofs)+outvgm[0x18:]
  with gzip.open(vpath, 'wb') as f:
      f.write(outvgm)    

  print(f'{slices} slices, {len(rows)} frames')
  print(f'{min_height=} {max_height=}')
  print('\nPlaying synthesized result...',end='')
  sys.stdout.flush()
  s=[]
  gs=[]
  for i in range(0,len(oplemu._output),4):
    l=struct.unpack('<h',oplemu._output[i:i+2])[0]
    r=struct.unpack('<h',oplemu._output[i+2:i+4])[0]
    s.append(np.array([l,r],dtype='int16'))
    gs.append( (l+r)//2)

  gs = np.array(gs,dtype='int16')
  outspect = sp.spect(wav_filename=None,samples=gs,nperseg=4096,quiet=False,clip=True)
  drawSpect(outspect,0,0,screen_width,screen_height)

  s=pygame.sndarray.make_sound(np.array(s))
  pygame.mixer.Sound.play(s)
  while pygame.mixer.get_busy(): 
    pygame.time.Clock().tick(10)

  print('Done')
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
        doquit = analyzePeakRuns()
        if doquit:
          return True
      elif event.key == pygame.K_f:
        fastAnalyze()
      elif event.key == pygame.K_p:
        playWave()
      elif event.key == pygame.K_b:
        bruteForce(genetic=False, ai=True)
      elif event.key == pygame.K_g:
        bruteForce(genetic=True, ai=False)
      elif event.key == pygame.K_d:
        allthumbs = {}
        for roi in range(44,46):
          allthumbs[roi] = spectThumbnails(origspect.spectrogram[roi])
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
