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
import struct
import gzip
import sys
# -----------------------------------------------------------------------------
# TODO:  Make a bunch of this stuff user-configurable!

screen_width=1920 
screen_height=1080
origspect=None
SLICE_FREQ = 91.552734375   # frequency at which the synth parameters are altered
MAX_SYNTH_CHANS = 18   # polyphony of synth (num independent oscillators)
OPL3_MAX_FREQ = 6208.431  # highest freq we can assign to an operator
SPECT_VERT_SCALE = 3  # set vert axis to 7350 Hz max rather than 22050 Hz                      
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
  #wavname = 'HAL 9000 - Human Error.wav'
  #wavname = 'JFK Inaguration.wav'
  wavname = 'Ghouls and Ghosts - The Village Of Decay.wav'

tempfolder = 'temp\\'
outfolder = 'output\\'
vpath = outfolder+wavname[0:-3]+"vgz"
origspect = sp.spect(wav_filename='input\\'+wavname,nperseg=4096)
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
# draws the pygame surface member of the spectrogram class, 
# scaled to the current display size and SPECT_VERT_SCALE factor.
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
  '''
  # draw vertical gridlines
  tstart = 0
  t=tstart
  specdrawheight = screen_height-T_AXIS_HEIGHT
  while t<(tstart+onscreen_dur):    
    scr_x = int((t-tstart)*screen_width/(tstart+onscreen_dur)) 
    for y in range(0,specdrawheight): 
      if (y%3) == 0:
        #c=screen.get_at((scr_x, y))
        #ic = (255-c[0],255-c[1],255-c[2])
        pygame.draw.line(screen, (0,0,0), (scr_x,y),(scr_x,y))
    t+=secs_per_t_tick
  '''
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
# all prominent peak. Returns their freqs in Hz and heights in dBFS.
# -----------------------------------------------------------------------------
def getRankedPeaks(tsp):
  hh = int(screen_height/4)
  ll = len(tsp)
  hsr = origspect.sample_rate/2
  hnps = origspect.nperseg/2 
  minv = origspect.minval
  maxv = origspect.maxval   
  # get initial peak estimates
  # todo: center of mass refinement of peak freqs
  mypeaks=[]
  peaks = signal.find_peaks(tsp, height = -100, distance=5, prominence=10) # was d 5 p 10
  for peakidx,binnum in enumerate(peaks[0]):
    peak_freq = hsr-binnum*hsr/hnps
    if peak_freq>OPL3_MAX_FREQ:
      continue
    peak_height = peaks[1]['peak_heights'][peakidx]
    peak_x = binnum*screen_width/ll
    peak_y = hh-1-((peak_height-minv)*hh/(maxv-minv))
    peak_prominence = peaks[1]['prominences'][peakidx]
    mypeaks.append((peak_height,peak_freq,peak_x,peak_y,peak_prominence))
  mypeaks.sort(key=lambda tup: -tup[0])
  return mypeaks
# -----------------------------------------------------------------------------
# draw a single spectrum along the top of screen and mark peaks
# -----------------------------------------------------------------------------
def plotTestSpect(tsp):
  global screen
  ll = len(tsp)
  ww = int(screen_width)
  hh = int(screen_height/4)
  hsr = origspect.sample_rate/2
  minv = origspect.minval
  maxv = origspect.maxval

  # draw freq cutoff line 
  if cutoff_freq>=0:
    x = screen_width-1 - (cutoff_freq * screen_width / hsr)
    pygame.draw.line(screen, (0,255,255), (x,0),(x,hh))

  # draw amp cutoff line 
  if amp_cutoff_y>=0:
    pygame.draw.line(screen, (255,255,0), (0,amp_cutoff_y),(ww-1,amp_cutoff_y))
    plotText("cutoff={}".format(amp_cutoff),0,amp_cutoff_y)

  # draw single spectrum's peaks

  mypeaks=getRankedPeaks(tsp)

  for i,(peak_height,peak_freq,peak_x,peak_y,peak_prominence) in enumerate(mypeaks):
    r = 255*(peak_height-minv)/(maxv-minv)
    if i==0:
      pygame.draw.line(screen, (255,255,0), (peak_x,peak_y-10),(peak_x,peak_y+10))
    else:
      pygame.draw.line(screen, (r,r/8,255-r), (peak_x,peak_y-10),(peak_x,peak_y+10))

  # draw single spectrum 
  for i in range(0,ll-1):
    x0=i*screen_width/ll
    x1=(i+1)*screen_width/ll
    v0=(tsp[i]-minv)*hh/(maxv-minv)
    v1=(tsp[i+1]-minv)*hh/(maxv-minv)
    y0=hh-1-v0
    y1=hh-1-v1
    pygame.draw.line(screen, (255,255,255), (x0,y0),(x1,y1))

  return mypeaks
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
def plotPeaks(roi,slen, dur_secs, tsp):
  global screen
  ll = len(tsp)
  ww = int(screen_width)
  hh = int(screen_height) - T_AXIS_HEIGHT
  hsr =  (origspect.sample_rate/2)/SPECT_VERT_SCALE
  minv = origspect.minval
  maxv = origspect.maxval
  rx = roi*ww/slen
  rw = (((roi+1)*ww/slen)-rx)+1
  pygame.draw.rect(screen, (0,0,0), (rx,0,rw,hh))
  mypeaks=getRankedPeaks(tsp)
  spect_t = roi*dur_secs/slen
  if len(mypeaks)<=(MAX_SYNTH_CHANS*2):
    s0 = 0
    s1 = len(mypeaks)
  else:
    # if theres a lot of peaks, eliminate some poorly ranked ones from the plot.
    s0 = 0
    s1 = (MAX_SYNTH_CHANS*2)
  print(f'{spect_t:9.4f}: ',end='')
  points = []
  loudest = None
  for i,(peak_height,peak_freq,peak_x,peak_y,peak_prominence) in enumerate(mypeaks[s0:s1]):
    if peak_height < -96:
      continue
    r = 255*(peak_height-minv)/(maxv-minv)
    if i<MAX_SYNTH_CHANS and  peak_height >= -48:
      print(f'({peak_freq:6.1f},{peak_height:4.0f}) ',end='')
    if i<MAX_SYNTH_CHANS:
      # top peaks get normal colors
      c=gradColor(peak_height)
    else:
      # less prominent ones get drab monochrome colors
      c=(r,r,r)
    ry = hh-peak_freq*hh/hsr
    pygame.draw.rect(screen, c, (rx,ry,rw,2))
    points.append((spect_t, float(peak_freq), float(peak_height)))
    if loudest is None or peak_height>loudest:
      loudest=peak_height
  if loudest is None:
    print(f' <None>')
  else:
    print(f' <{loudest:6.2f}>')
  return points
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
    plotTestSpect(origspect.spectrogram[roi])
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
      #s='mx,my = {:4d},{:4d} freq={:5.2f}, t={:4.4f}, v={:4.1f}    '.format(mx,my,freq,t,v)
      #plotText(s,4,4)
    elif event.button == 3: # right click selects cutoff frequency on spectrogram
      cutoff_freq = freq

  newROI()
# -----------------------------------------------------------------------------
# do peak detect on whole displayed spectrogram and draw those peaks
# -----------------------------------------------------------------------------  
def analyze():
  global origspect, roi
  global screen
  global screen_width
  global screen_height  
  point_cloud = []
  for roi in range(0,len(origspect.spectrogram)):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
        return True# raise SystemExit    
    #pygame.draw.rect(screen, (0,0,0), (0,0,screen_width,int(screen_height/4)))
    #mypeaks = plotTestSpect(origspect.spectrogram[roi])    
    point_cloud += plotPeaks(roi,len(origspect.spectrogram),origspect.dur_secs, origspect.spectrogram[roi])
    pygame.display.update()
  point_cloud.sort(key=lambda tup: -tup[2])
  print(point_cloud)
  return False
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
# Given a series of spectral peaks, resynthesize the sound by summing 
# sine waves of various appropriate freqs and amplitudes
# -----------------------------------------------------------------------------  
def playrows(rows):
  print('Synthesizing sound...')
  swave = []
  wav = []
  phases = [0]*MAX_SYNTH_CHANS
  for row in rows:
    freqs = []
    amps = []    
    for freq, amp, pheight in row:
      freqs.append(freq)
      amps.append(amp)
    for j in range(0,int(44100/SLICE_FREQ)):
      ssum = 0
      for ch in range(0,MAX_SYNTH_CHANS):
        ssum += amps[ch]*math.sin(phases[ch])
        phases[ch] += freqs[ch]*2.0*3.14159 / 44100.0
      isum = int(ssum/MAX_SYNTH_CHANS)
      wav.append(isum)
      swave.append([isum,isum])
  print('Drawing spectrum of synthesized sound...')  
  wav = np.array(wav)
  origspect = sp.spect(samples=wav)    
  sh2=screen_height
  drawSpect(origspect,0,0,screen_width,sh2)
  print('Playing sound...')  
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
# Sequence to init the OPL3 chip to allow 18 channels of plain sine waves.
# -----------------------------------------------------------------------------
def opl3init():
  res = b''
  # enable opl3 mode
  r = 0x05
  v = 0x01
  res+=struct.pack('BBB',0x5f,r,v)
  # enable waveform select
  r = 0x01
  v = 0x20
  res+=struct.pack('BBB',0x5f,r,v)

  for chan in range(0,18):
    opidxs = opidxs_per_chan[chan]
    if chan<9:      
      # sustain, vibrato,opfreqmult
      r = 0x20 + opidxs[0]
      v = 0x21
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x20 + opidxs[1]
      v = 0x21
      res+=struct.pack('BBB',0x5e,r,v)
      # keyscalelevel, output level
      # setting keyscale level to 6.0 dB of attenuation per rise in octave
      # (we really want more than this and need to implement something ourselves)
      r = 0x40 + opidxs[0]
      v = 0x30  # 30
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x40 + opidxs[1]
      v = 0x30
      res+=struct.pack('BBB',0x5e,r,v)
      # attack rate, decay rate
      r = 0x60 + opidxs[0]
      v = 0xff
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x60 + opidxs[1]
      v = 0xff
      res+=struct.pack('BBB',0x5e,r,v)
      # sust level, release rate
      r = 0x80 + opidxs[0]
      v = 0x0f
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0x80 + opidxs[1]
      v = 0x0f
      res+=struct.pack('BBB',0x5e,r,v)
      # output channels L&R and additive synth
      r = 0xc0 + chan
      v = 0x31
      res+=struct.pack('BBB',0x5e,r,v)
      # waveform select sine
      r = 0xe0 + opidxs[0]
      v = 0x00
      res+=struct.pack('BBB',0x5e,r,v)
      r = 0xe0 + opidxs[1]
      v = 0x00
      res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      a = opidxs[0]-18
      b = opidxs[1]-18
      # sustain, vibrato,opfreqmult
      r = 0x20 + a
      v = 0x21
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x20 + b
      v = 0x21
      res+=struct.pack('BBB',0x5f,r,v)
      # keyscalelevel, output level
      r = 0x40 + a
      v = 0x30
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x40 + b
      v = 0x30    #  30
      res+=struct.pack('BBB',0x5f,r,v)
      # attack rate, decay rate
      r = 0x60 + a
      v = 0xff
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x60 + b
      v = 0xff
      res+=struct.pack('BBB',0x5f,r,v)
      # sust level, release rate
      r = 0x80 + a
      v = 0x0f
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x80 + b
      v = 0x0f
      res+=struct.pack('BBB',0x5f,r,v)
      # output channels L&R and additive synth
      r = 0xc0 + chan
      v = 0x31
      res+=struct.pack('BBB',0x5f,r,v)
      # waveform select sine (todo: use different waveforms when appropriate!)
      r = 0xe0 + a
      v = 0x00
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0xe0 + b
      v = 0x00
      res+=struct.pack('BBB',0x5f,r,v)
  return res
# -----------------------------------------------------------------------------
# Returns an OPL3 register set sequence to set a specified channel to 
# the specified frequency, volume, and/or key on/off.
#
# TODO: we really shouldn't need to set everything on every channel/frame!
# We should instead pigenhole peaks to specific channels and keep them there
# for as many frames as needed. That way, we only need to set things that have
# changed since the previous frame. This should reduce the output file size
# and may even sound better!
# 
# Also TODO: OPL3 can do way more than just play sine waves. It would be 
# nice if the analysis could identify use cases for  2-op and 4-op 
# instruments, percussion, noise, vibratoo, tremolo, and whatever else we can
# take advantage of! (Not easy. Might need an AI, eventually, or at least 
# some better pattern-matching than the simple spectral peak-detect we're 
# currently doing.)
# -----------------------------------------------------------------------------
def opl3params(freq,namp,chan, keyon):
  fnum, block = getFnumBlock(freq)
  opidxs = opidxs_per_chan[chan]
  res = b''

  if not keyon:
    if chan<9:
      r = 0xb0 + chan
      v = ((fnum>>8)&3) | (block<<2)
      res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      r = 0xb0 + chan - 9
      v = ((fnum>>8)&3) | (block<<2)
      res+=struct.pack('BBB',0x5f,r,v)
  else:
    aval = int(0x3F*namp)
    if aval>0x3f:
      aval = 0x3f
    if aval<0:
      aval = 0

    if chan<9:
      r = 0x40 + opidxs[0]  # set volume
      v = aval
      res+=struct.pack('BBB',0x5e,r,v)

      r = 0x40 + opidxs[1]
      v = aval
      res+=struct.pack('BBB',0x5e,r,v)

      r = 0xA0 + chan  # set low bits of frequency
      v = fnum&0xff
      res+=struct.pack('BBB',0x5e,r,v)

      r = 0xb0 + chan  # set key-on and high bits of frequency
      v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
      res+=struct.pack('BBB',0x5e,r,v)
    else:
      chan-=9
      a = opidxs[0] - 18
      b = opidxs[1] - 18

      r = 0x40 + a # volume
      v = aval
      res+=struct.pack('BBB',0x5f,r,v)
      r = 0x40 + b
      v = aval
      res+=struct.pack('BBB',0x5f,r,v)

      r = 0xA0 + chan # low bits of freq
      v = fnum&0xff
      res+=struct.pack('BBB',0x5f,r,v)

      r = 0xb0 + chan - 9    # key-on and high bits of freq
      v = 0b00100000 | ((fnum>>8)&3) | (block<<2)
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

  print('Doing spectral peak-detection.')
  all_peaks = []
  slices=0
  ll = len(origspect.spectrogram)
  tt = origspect.maxtime - origspect.mintime
  for r in range(0,ll):
    all_peaks.append(getRankedPeaks(origspect.spectrogram[r]))
    slices+=1
  max_amp = 0
  min_amp = 9999999
  max_height = -999999
  min_height = 9999999

  '''
  # show all spect peaks, not just the timeslices
  print('ALL PEAKS')
  for pi, peaks in enumerate(all_peaks):
    t = pi * tt / ll   # get slice timestamp
    print('Slice {:6d} ({:3.3f}s):'.format(pi, t),end='')
    row = []
    for i,(peak_height,peak_freq,peak_x,peak_y,peak_prominence) in enumerate(peaks):
      if i>=MAX_SYNTH_CHANS:
        break
      print('({:8.2f},{:5d}), '.format(peak_freq,int(peak_height)),end='')
    print()
  print('\n\n\n')
  '''

  #print('SLICE PEAKS')
  rows = []
  lt = 0
  for pi, peaks in enumerate(all_peaks):
    t = pi * tt / ll   # get slice timestamp
    do_slice = False
    if (t-lt) >= (1.0/SLICE_FREQ):
      lt = t
      do_slice = True
    if do_slice:
      #print('Slice {:6d}: t {:3.2f}:'.format(pi, t),end='')
      row = []
      for i,(peak_height,peak_freq,peak_x,peak_y,peak_prominence) in enumerate(peaks):
        #print('({:8.2f},{:4d}), '.format(peak_freq,int(peak_height)),end='')
        wave_amp = 37369.0 * math.exp(0.1151 * peak_height)
        #print("wa vs ph",wave_amp, peak_height)
        if i>=MAX_SYNTH_CHANS:
          break
        #print('({:8.2f},{:4d},{:4d}), '.format(peak_freq,int(peak_height),int(wave_amp)),end='')
        if wave_amp < min_amp:
          min_amp = wave_amp
        if wave_amp > max_amp:
          max_amp = wave_amp
        if peak_height > max_height:
          max_height = peak_height
        if peak_height < min_height:
          min_height = peak_height
        #print('({:8.2f},{:5d}), '.format(peak_freq,int(peak_height)),end='')
        row.append([peak_freq, wave_amp, peak_height])
      k = len(row)
      while k<MAX_SYNTH_CHANS:
        row.append([0,0,-96.0])
        k+=1
      rows.append(row)
      #print()
  #print('\n\n\n')


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

  playrows(rows)
# -----------------------------------------------------------------------------
# experimental stuff regarding additive synthesis with sinewaves
# -----------------------------------------------------------------------------
def calibrate():
  global origspect

  print("Correlating wave amplitude to dbFS")
  res = ''
  ns=int(44100/8)
  for v0 in [8192, 3096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:    
    ct=0
    av=0
    for f0 in range(1000,20000,1000):
      wav = []
      for i in range(0,ns):
        f = (i*2.0*3.1415926536/44100.0)
        a0 = v0*math.sin(f * (f0))
        a1 = v0*math.sin(f * (f0+250))
        a2 = v0*math.sin(f * (f0+500))
        a3 = v0*math.sin(f * (f0+750))
        wav.append(a0+a1+a2+a3)
      wav = np.array(wav)
      otherspect = sp.spect(samples=wav)
      mv=otherspect.maxval
      av+=mv
      ct+=1
      # otherspect = sp.spect(wav_filename='explainer.wav')
      sh2=screen_height
      drawSpect(otherspect,0,0,screen_width,sh2)
      for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
          raise SystemExit      
      clock.tick(30)      
    s='{:5d}\t{:f}'.format(v0,av/ct)
    res+=s+'\n'
  print('cal results\nwave_amp\tdbFS')
  print(res)

  # dBFS = 8.6859 * ln(wav_amplitude) - 91.45
  # wave_amplitude = 37369 * e ^ (0.1151 * dBFS)
# -----------------------------------------------------------------------------
# arpeggio synthesis experiment
# -----------------------------------------------------------------------------
def synth():
  global origspect
 
  ns=int(44100*4)
  wav = []

  a0=0
  a1=0
  a2=0
  a3=0

  v0=0
  v1=0
  v2=0
  v3=0

  dv0=0
  dv1=0.03
  dv2=0.03
  dv3=0.03
  
  j=0
  k=0
  for i in range(0,ns):
    j+=1
    if j==2000:
      j=0
      k+=1
      if k==3:
        k=0

    f = (i*2.0*3.1415926536/44100.0)
    a0 = v0*math.sin(f * 440.0*3)
    a1 = v1*math.sin(f * 523.25)
    a2 = v2*math.sin(f * 415.30)
    a3 = v3*math.sin(f * 349.23)
    v0+=dv0
    v1+=dv1
    v2+=dv2
    v3+=dv3
    #wav.append(a1+a2+a3)
    if k==0:
      wav.append(a1)
    elif k==1:
      wav.append(a2)
    elif k==2:
      wav.append(a3)
  wav = np.array(wav)
  origspect = sp.spect(samples=wav)    
  sh2=screen_height
  drawSpect(origspect,0,0,screen_width,sh2)

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
      elif event.key == pygame.K_s:
        synth()
      elif event.key == pygame.K_c:
        calibrate()
      elif event.key == pygame.K_a:
        doquit = analyze()
        if doquit:
          return True
      elif event.key == pygame.K_f:
        fastAnalyze()
      elif event.key == pygame.K_p:
        playWave()
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
        #print("Player moved left!")
        if last_clicked == CLICKED_SPECTROGRAM:
          if roi>-1:
            roi-=1
        elif last_clicked == CLICKED_SPECTRUM:
          pass
        newROI()
      elif event.key == pygame.K_RIGHT:
        #print("Player moved right!")  
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
