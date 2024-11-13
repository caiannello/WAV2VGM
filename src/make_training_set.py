###############################################################################
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
# As a possible TODO, since these models are not training so great yet,
# it might help to have a mode here where it will output some simpler
# one-voice and two-voice permutations, along with the current method of
# throwing the kitchen sink at it the whole time. Maybe that'll help? (I'm 
# new to this AI stuff.)
#
# Craig Iannello 2024/11/04
###############################################################################
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
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

sfilename = dir_path+'/../training_sets/opl3_training_spects.bin'
rfilename = dir_path+'/../training_sets/opl3_training_regs.bin'

print('''
-------------------------------------------------------------------------------
Notes:

The purpose of this utility is to generate a training set for the AI used in 
the WAV2VGM project. This version is geared towards OPL3 synthesis- It makes 
tons of random OPL3 register settings, and for each one, the PyOPL emulator 
is used to render a short 4096-point waveform output. If the wave is all 
zeroes, it is thrown out and we start over with another config.

If the wave has sound in it, a frequency spectrum is calculated, and a set
of data is added to the training set output, which consists of two files:

                File Name | Record Type | Rec. Description
                ----------+-------------+-----------------
   opl3_training_regs.bin | float[290]  | Synth configuration vector
 opl3_training_spects.bin | byte[2048]  | Currenponding frequency spectrum

The spectra are used as the inputs for training, and the configs are used
as the ground-truth outputs we would like the AI to achieve when given
such a spectrum.

*** Note on re-running this utility: (Will I lose my precious data???!)

If closed and restarted, the utility will check for pre-existing training
data. If some exists, and if both files are sized correctly, (in multiples
of the record size) the files will be kept and expanded. 

If the files aren't the right sizes, though, they'll get overwritten!

-------------------------------------------------------------------------------

Now Starting...
''')
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
def plotSpectrum(spec, gcolor=(255,255,255)):
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
    pygame.draw.line(screen, gcolor, (x0,y0),(x1,y1))
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
def showRegs(opl_regs):  
  keys = list(opl_regs.keys())
  keys.sort()
  for k in keys:
    (b,r)=k
    v=opl_regs[k]
    print(f'({b},${r:02X}): ${v:02X}')
# -----------------------------------------------------------------------------
# Setup the OPL emulator with the specified register values, generate 4096 
# audio samples, and return resultant frequency spectrum.
# -----------------------------------------------------------------------------
def renderOPLFrame(opl_regs):
  global rawbin_low, rawbin_high, wave_low, wave_high
  # init opl emulator
  o = opl.opl_emu()
  o.do_init()

  keys = list(opl_regs.keys())
  keys.sort()

  for (b,r) in keys:
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
    # we want only the first spectrum of spectogram
    spec = spec.spectrogram[0]
  else:
    spec  = None
    #showregs(opl_regs)
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
# convert an opl congiguration dict into a vector representation for
# training the AI.  (each field gets a float [0.0,1.0] in the vector)
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

def insBitsFloat(v,sbit,nbit,f):
  #print(f'      insBitsFloat({v=},{sbit=},{nbit=},{f=:5.2f}): ',end='')
  mag = (1<<nbit)-1
  orig = intToBits(v,8)
  newv = intToBits(round(f*mag),nbit)
  a = 8-sbit
  b = a+nbit
  bs = orig[0:a] + newv[-nbit:] + orig[b:]
  i = bitsToInt(bs)
  #print(f'{mag=} {orig=} {newv=} {a=} {b=} {bs=} {i=}')
  return i

def insBitsString(v,sbit,nbit,s):
  mag = (1<<nbit)-1
  f = int(s,2)/mag
  #print(f'      insBitsString({v=},{sbit=},{nbit=},{s=}): {mag=} {f=}')
  return insBitsFloat(v,sbit,nbit,f)

def vectToRegDict(vec):
  global permutable_regidxs, permutable_splits
  regs = initRegs()

  j=0
  for i,ridx in enumerate(permutable_regidxs):
    b = (ridx>>8) & 1
    r = ridx & 0xff
    if not (b,r) in regs:
      regs[(b,r)]=0x00
    v = 0x00

    sbit = 8
    for (lbl,bts) in permutable_splits[ridx]:
      if lbl == '_':
        v=insBitsString(v,sbit,len(bts),bts)
        regs[(b,r)]=v
        sbit-=len(bts)
      else:
        ebit = sbit-bts
        f = vec[j]
        j+=1
        v=insBitsFloat(v,sbit,sbit-ebit,f)
        regs[(b,r)]=v
        sbit = ebit

  return regs
# -----------------------------------------------------------------------------
def showPermutationCounts():
  global permute_counts
  keys = list(permute_counts.keys())
  keys.sort()

  for k in keys:
    v = permute_counts[k]
    (b,r,fi) = k
    print(f'  ({b},{r:02X},{fi}): {v}')
# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------
def main():
  global rawbin_high, wave_low, wave_high, reinit_freq
  global ww,hh,permutable_splits,permutable_regidxs, permute_counts  
  opl_regs = initRegs()
  j=0

  try:
    ssize = os.path.getsize(sfilename)
    rsize = os.path.getsize(rfilename)
    smod = ssize % 2048
    rmod = rsize % (290*115)
    print(f'{ssize=}, {smod=}, {rsize=}, {rmod=}')
  except Exception as e:
    ssize = 0
    rsize = 0
    smod = -1
    rmd  = -1

  if (not ssize) or (ssize%2048) or (not rsize) or (rsize%(290*4)):
    print("A training file doesn't exist or is of unexpected size and will be rewritten.")
    sfile = open(sfilename, 'wb')         # list of (reg_config, spectrum[2046)
    rfile = open(rfilename, 'wb')  # list of corresponding 128-byte squished-spectra
  else:
    print('Found existing training files. They will be expanded.')
    sfile = open(sfilename, 'ab')         # list of (reg_config, spectrum[2046)
    rfile = open(rfilename, 'ab')  # list of corresponding 128-byte squished-spectra

    # todo: still, we should have a way to ensure the two files are in-sync
    # all the way to the end bore commencing the hourslong AI training
  
  fsz = ssize
  ic = reinit_freq  # reinitialization countdown
  lastszmb = -1
  iters=0
  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
        rfile.close()
        sfile.close()
        print('\npermute counts:\n')
        showPermutationCounts()
        print('\n final regvals\n')
        showRegs(opl_regs)
        return 

    while True:
      ridx = random.choice(permutable_regidxs)
      b = (ridx>>8)&1
      r = ridx&0xff
      ll,bb = random.choice(permutable_splits[ridx])
      if ll!='_':
        break
    
    # get startine and ending bitpos to fuzz
    splits = permutable_splits[ridx]
    sidx = 0
    sbit=8
    while True:
      al,ab = splits[sidx]
      if al==ll:
        break
      if al=='_':
        sbit-=len(ab)
      else:
        sbit-=ab
      sidx+=1
      
    v = opl_regs[(b,r)]
    f = random.random()
    v = insBitsFloat(v,sbit,bb, f)
    opl_regs[(b,r)]=v

    try:
      permute_counts[(b,r,sidx)]+=1
    except:
      print(f'PNF! ({b},{r:02X},{sidx})')
      exit()

    # render a 4096-point waveform and its spectrum
    waveform, tspect = renderOPLFrame(opl_regs)

    # make a vector representation of the opl config
    # and render it for comparison to the orig cfg dict
    vec = regDictToVect(opl_regs)
    #vregs = vectToRegDict(vec)
    #waveform1, tspect1 = renderOPLFrame(vregs)
    '''
    # also ensure it is reversible
    good = True
    for k in opl_regs:
      if not k in vregs:
        print(f'{k} NOT FOUND')
        good=False
      else:
        v0 = opl_regs[k]
        v1 = vregs[k]
        if v0!=v1:
          print(f'({k[0]},{k[1]:02X}) MISMATCH {v0=:02X} {v1=:02X}')
          good=False
    if not good:
      print('NOT GOOD!')
      exit()
    '''

    # if successful, 
    if tspect is not None:

      # write the opl reg configuration vector float[290]
      # to the cfg training set file
      sbin = b''
      for f in vec:
        sbin+=struct.pack('<f',f)
      rfile.write(sbin)
      
      # write binary of corresponding spectrum (1 byte-per-bin)
      # to the spect training set file
      sbin = b''
      for i in range(0,2048):
        b = abs(int(tspect[i]))
        if b>255:
          b=255
        b = 255-b
        sbin+=struct.pack('B',b)    #  0: 0.0 dBFS ... 255: -255.0 dBFS
      # write 2048-byte spectrum to spectrum file
      sfile.write(sbin)

      # note how big the spectrum file is now
      fsz += len(sbin)
      # and how many spectrums are in it
      iters+=1


      j+=1
      if j==10:  # show every 10th set on screen
        j=0        
        try:
          thumb=spectThumbnail(tspect)
          pygame.draw.rect(screen,(0,0,0),(0,0,ww,hh))
          # waveform
          plotWaveform(waveform)
          plotSpectrum(tspect)
          #if not tspect1 is None:
          #  plotSpectrum(tspect1,(255,128,128))
          plotThumb(thumb)
          pygame.display.update()

          # get output file size in MB
          fszmb = int(fsz/1024.0/1024.0)
          if fszmb!=lastszmb: # every 1MB out, show a status update.
            lastszmb = fszmb
            sfsz = f','
            print(f'iteration: {iters:12d} ({fszmb:d} MB)')
            if fszmb >= 20000:  # if we hit 20 GB, stop!
              sfile.close()
              rfile.close()
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
