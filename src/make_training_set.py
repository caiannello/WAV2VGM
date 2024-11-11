# -----------------------------------------------------------------------------
# WAV2VGM - MAKE TRAINING SET
#
# This proggie does random permutations of OPL3 register settings, renders
# a short waveform using the OPL3 emulator, and then get the frequerncy 
# spectrum of it. These are both output to a big data file.
#
# The plan is to use this file to train a deep-learning algorithm to
# output OPL configuration data in response to arbitrary input spectra.
#
# Hopefully, the permutations chosen have good enough coverage of that the
# OPL3 chip can do. We're not using volume envelopes, vibrato, or tremolo 
# though, since we currently only care about one output spectrum per 
# permutation.
#
# Craig Iannello 2024/11/04
#
# -----------------------------------------------------------------------------
import numpy as np
import math
from   scipy import signal
import spect as sp
import opl_emu as opl
import struct
import pyopl
import random
import datetime
import time
import pygame
from pygame.locals import *
import struct
import gzip
import sys
from copy import deepcopy
# -----------------------------------------------------------------------------
# display window width and height
ww=1920 
hh=1080
# init pygame for graphics
pygame.init()
screen=pygame.display.set_mode([ww,hh])#,flags=pygame.FULLSCREEN)
pygame.display.set_caption(f'Making OPL3 Training Set')

# randomize the PRNG
random.seed(datetime.datetime.now().timestamp())

reinit_freq = 100  # bigger means less chance of resetting the opl to defaults
OPL3_MAX_FREQ = 6208.431  # highest freq we can assign to an operator

# min/max when plotting spectrum bins onscreen
MAX_DB = 0
MIN_DB = -150.0

# Statistics covering all iterations ----------------------

# bin heights of raw spectrograms out of scipy.signal

rawbin_high = -9999999
rawbin_low  =  9999999

# bin bounds of spect after converting it to dbFS. (decibels of full scale)

dbin_high = -9999999
dbin_low  =  9999999

# waveform bounds out of OPL emulator. 
# After a ron of iterations, for some reason, I'm seeing (-17888,29400) 
# rather than the expected (-32768,32767).

wave_high = -9999999
wave_low  =  9999999

# -----------------------------------------------------------------------------
# draw a single spectrum
# -----------------------------------------------------------------------------
def plotSpectrum(spec):
  global screen, dbin_high, dbin_low
  global MIN_DB,MAX_DB
  global ww,hh  

  ll = len(spec)
  if MIN_DB==MAX_DB: # dont wanna /0
    return
  for i in range(0,ll-1):
    x0=int(i*ww/ll)
    x1=int((i+1)*ww/ll)    
    v0=int((spec[i]-MIN_DB)*hh/(MAX_DB-MIN_DB))
    v1=int((spec[i+1]-MIN_DB)*hh/(MAX_DB-MIN_DB))
    if spec[i]>dbin_high:
      dbin_high = int(spec[i])
    if spec[i+1]>dbin_high:
      dbin_high = int(spec[i+1])
    if spec[i]<dbin_low:
      dbin_low = int(spec[i])
    if spec[i+1]<dbin_low:
      dbin_low = int(spec[i+1])    
    if spec[i]<MIN_DB or spec[i+1]<MIN_DB:      
      continue
    y0=hh-1-v0
    y1=hh-1-v1
    pygame.draw.line(screen, (255,255,255), (x0,y0),(x1,y1))
  # show high-water mark of spectral bin dBFS
  ymax = hh-1-int((dbin_high-MIN_DB)*hh/(MAX_DB-MIN_DB))
  pygame.draw.line(screen, (255,0,0), (0,ymax),(ww-1,ymax))
  # and quiet bin cutoff at -115 dBFS
  ymax = hh-1-int((-115-MIN_DB)*hh/(MAX_DB-MIN_DB))
  pygame.draw.line(screen, (0,0,255), (0,ymax),(ww-1,ymax))
# -----------------------------------------------------------------------------
# draw a mono 16-bit sound waveform in yellow
# -----------------------------------------------------------------------------
def plotWaveform(wave):
  global screen,ww,hh
  ll = len(wave)
  for i in range(0,ll-1):
    s0 = int(wave[i])
    s1 = int(wave[i+1])
    x0=int(i*ww/ll)
    x1=int((i+1)*ww/ll)    
    y0=int((s0+32768)*hh/65536)
    y1=int((s1+32768)*hh/65536)
    pygame.draw.line(screen, (255,255,0), (x0,y0),(x1,y1))
# -----------------------------------------------------------------------------
# show min/max lines for every 32 bins of spectrum
# -----------------------------------------------------------------------------
def plotThumb(t):
  global screen
  global MIN_DB,MAX_DB
  global ww,hh  
  l = len(t)
  hsr = 22050
  if MIN_DB==MAX_DB:  # dont wanna /0
    return
  # draw single spectrum 
  for i in range(0,l,2):
    x0=int(i*ww/l)
    x1=int((i+2)*ww/l)
    a,b = struct.unpack('BB',t[i:i+2])
    a = -a
    b = -b
    v0=int((a-MIN_DB)*hh/(MAX_DB-MIN_DB))
    y0=hh-1-v0
    if y0>=0 and y0<hh:
      pygame.draw.line(screen, (128,255,128), (x0,y0),(x1,y0))

    v1=int((b-MIN_DB)*hh/(MAX_DB-MIN_DB))
    y1=hh-1-v1
    if y1>=0 and y1<hh:
      pygame.draw.line(screen, (255,128,128), (x0,y1),(x1,y1))    
# -----------------------------------------------------------------------------
# Setup the OPL emulator with the specified register values, generate 4096 
# audio samples, and return resultant frequency spectrum.
# -----------------------------------------------------------------------------
def renderOPLFrame(opl_regs):
  global rawbin_low, rawbin_high, wave_low, wave_high
  # init opl emulator
  o = opl.opl_emu()
  o.do_init()
  for (b,r) in opl_regs:
    v = opl_regs[(b,r)]
    o.write(b,r,v)
  o._output = bytes()
  # render 4096 samples
  o._render_samples(4096)  
  # convert to mono, and note min/max sample for statistics
  ll = len(o._output)
  wave = []
  for i in range(0,ll,4):
    l=struct.unpack('<h',o._output[i:i+2])[0]
    r=struct.unpack('<h',o._output[i+2:i+4])[0]
    if l<wave_low:
      wave_low = l
    if r<wave_low:
      wave_low = r
    if l>wave_high:
      wave_high = l
    if r>wave_high:
      wave_high = r
    wave.append((l+r)//2)  
  wave = np.array(wave, dtype="int16")
  # if not flat-line, generate spectrogram
  if wave.sum():
    spec = sp.spect(wav_filename = None, sample_rate=44100,samples=wave,nperseg=4096, quiet=True, clip = False)    
    rbl = int(spec.rawbin_low)
    if rbl<rawbin_low:      
      rawbin_low = rbl
    rbh = int(spec.rawbin_high)
    if rbh>rawbin_high:      
      rawbin_high = rbh
    # we want only the first spectrum of spectogram
    spec = spec.spectrogram[0]
  else:
    spec  = None
  # return waveform and spectrogram, if any
  return wave, spec
# -----------------------------------------------------------------------------
# takes a float[2048] spectrum and squishes it down to 64 bins, each containing
# the min/max of the 32 bins it represents.
# -----------------------------------------------------------------------------
def spectThumbnail(spect):
  thumb = b''
  for bidx in range(0,2048,2048//64):
    i = abs(int(min(spect[bidx:bidx+2048//64])))
    x = abs(int(max(spect[bidx:bidx+2048//64])))
    if i>255:
      i=255
    if x>255:
      x=255
    thumb+=struct.pack('BB',i,x)
  return thumb
# -----------------------------------------------------------------------------  
# Some OPL3 registers we're interested in:  X means can be 0 or 1/
#
# $001        %xxWxxxxx - W: waveform select enable
# $008        %xNxxxxxx - N: note select (keyboard split mode)
# $X60...$X75 %AAAADDDD - A: Attack Rate: D: Decay rate
# $105        %xxxxxxxN - N: OPL3 enable
# $0BD        %AVRBSTCH - A: trem dep, V:vib depth, R:perc mode enable, B:BD, S:SD, T:TOM, C: TC, H:HIHAT
# $X80...$X95 %SSSSRRRR - S: Sustain level, R: release rate
# $X20...$X35:%AVEKMMMM - A: amplitude mod, V: Vibrato, E: EGS 1:sustained/0:not, 
#                         K: Key:scale env rate, M: mult: 0:0.5, 1:1, 2:2...
# $X40...$X55 %KKTTTTTT - K: KSL atten, TL: total level
# $XA0...$XA8 %FFFFFFFF - F: f-num (low)
# $XB0...$XB8 %xxKBBBFF - K: keyon, B: block, F: f-num (high)
# $XC0...$XC8 %DCBAFFFC - D: chand, C:chanc: B:Right, A:Left, F: FBAK, C:CNT
# $XE0...$XF5 %xxxxxWWW - W: op waveform sel
# $104        %xxCCCCCC - C: Operator connection select
# -------------------------------------------------------------------------------
# Permutation map: (0/1: fixed values, alphabetics: permuted values)
#
# Constant settings ------
#
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
#
# -----------------------------------------------------------------------------
# resets the opl configuration to just the fixed-value parts
# -----------------------------------------------------------------------------
def initRegs():
  opl_regs = { (0,0x01): 0x20, (1,0x05): 0x01, (0,0x08): 0x00, (0,0xBD): 0 }
  for b in range(0,2):
    for j in range(0,0x16):
      opl_regs[(b,j+0x60)] = 0xff
      opl_regs[(b,j+0x80)] = 0x0f
  #for k in opl_regs:
  #  v = opl_regs[k]
  #  print(f'${k[0]:02X}, ${k[1]:02X} = ${v:02X}')
  return opl_regs
# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------
def main():
  global rawbin_high, wave_low, wave_high, reinit_freq
  global ww,hh  
  opl_regs = initRegs()
  j=0
  sfile = open('..\\repertoire\\opl3_training_set.bin', 'wb')         # list of (reg_config, spectrum[2046)
  tfile = open('..\\repertoire\\opl3_training_set_thumbs.bin', 'wb')  # list of corresponding 128-byte squished-spectra
  fsz = 0
  ic = reinit_freq  # reinitialization countdown
  lastszmb = -1
  iters=0
  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
        tfile.close()
        sfile.close()
        return 

    m = random.randint(0,1511)
    didrv = False
    if (m>=0x20 and m<=0x35) or (m>=0x120 and m<=0x135): # op multiplier
      b = (m>>8)&1
      r = m&0xff
      v = 0b00100000 | random.randint(0,15)
      didrv = True
    elif (m>=0x40 and m<=0x55) or (m>=0x140 and m<=0x155): # ksl atten 0..3, total level: 0..63
      b = (m>>8)&1
      r = m&0xff
      v = random.randint(0,255)
      didrv = True
    elif (m>=0xa0 and m<=0xa8) or (m>=0x1a0 and m<=0x1a8): # fnum low
      b = (m>>8)&1
      r = m&0xff
      v = random.randint(0,255)
      didrv = True
    elif (m>=0xb0 and m<=0xb8) or (m>=0x1b0 and m<=0x1b8): # Kon 0/1, block 0..7, fnumhi 0..4
      b = (m>>8)&1
      r = m&0xff
      v = random.randint(0,0b111111)
      didrv = True
    elif (m>=0xc0 and m<=0xc8) or (m>=0x1c0 and m<=0x1c8): # feedback 0..7, connection set: 0/1
      b = (m>>8)&1
      r = m&0xff
      v = 0xf0 + random.randint(0,15)
      didrv = True
    elif (m>=0xe0 and m<=0xf5) or (m>=0x1e0 and m<=0x1f5): # waveform sel 0...7
      b = (m>>8)&1
      r = m&0xff
      v = random.randint(0,7)
      didrv = True
    elif m==511:
      ic -= 1
      if ic<=0:
        ic = reinit_freq
        opl_regs = initRegs()
        print('reinit opl regs')
    elif m==0x104:
      b=1
      r=0x04
      v = random.randint(0,0b111111) # reconfigure connection sel
      didrv = True    
    elif m>511 and m<1400:  # change freqs a lot
      a = random.randint(0,1)
      b = random.randint(0,1)
      if a==0:
        r = random.randint(0xa0,0xa8)
        v = random.randint(0,255)
      else:
        r = random.randint(0xb0,0xb8)
        v = random.randint(0,0b111111)
      didrv = True
    elif m>1400:  # change total levels a lot
      b = random.randint(0,1)
      r = random.randint(0x40,0x55)
      v = random.randint(0,255)
      didrv = True
        
    if didrv:   # if any opl setting permuted

      # make settings change 
      opl_regs[(b,r)]=v

      # render a 4096-point waveform and its spectrum
      waveform, tspect = renderOPLFrame(opl_regs)

      # if successful, 
      if tspect is not None:

        # make binary of opl register configuration
        sbin = b'\0' * 512
        for (b,r) in opl_regs:
          v = opl_regs[(b,r)]
          idx = b*256 + r
          sbin = sbin[0:idx]+struct.pack('B',v)+sbin[idx+1:]

        # make binary of spectrum (1 byte-per-bin)
        for i in range(0,2048):
          b = abs(int(tspect[i]))
          if b>255:
            b=255
          sbin+=struct.pack('B',b)    #  0: 0.0 dBFS ... 255: -255.0 dBFS

        # write 512 byte register config + 2048-byte spectrum to training set file
        sfile.write(sbin)

        # note how big the training set is now
        fsz += len(sbin)
        # and how many spectrums are in it
        iters+=1

        # make a 128-byte thumbnail of this spectrum
        thumb=spectThumbnail(tspect)
        # and output to the thumbnails reference file
        tfile.write(thumb)

        j+=1
        if j==10:  # show every 10th set on screen
          j=0        
          try:
            pygame.draw.rect(screen,(0,0,0),(0,0,ww,hh))
            # waveform
            plotWaveform(waveform)
            plotSpectrum(tspect)
            plotThumb(thumb)
            pygame.display.update()

            # get output file size in MB
            fszmb = int(fsz/1024.0/1024.0)
            if fszmb!=lastszmb: # every 1MB out, show a status update.
              lastszmb = fszmb
              sfsz = f'({fszmb:d} MB),'
              print(f'iter: {iters:12d} {sfsz:<16} rawbin max:{rawbin_high:5d}, dBFS min/max:({dbin_low:4d},{dbin_high:4d}), wav min/max:({wave_low:6d},{wave_high:6d})')
              if fszmb >= 20000:  # if we hit 20 GB, stop!
                sfile.close()
                exit()
          except:
            # sometimes we get a divide by zero
            pass

###############################################################################
# ENTRYPOINT
###############################################################################
if __name__ == '__main__':
  main()
###############################################################################
# EOF
###############################################################################
