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
# Rather than throwing the kitchen sink at it the whole time, maybe there
# should be some modes where we do something simpler, like one-voice and 
# two-voice permutations, and some of the automatic fits like WAV2VGM does 
# now, with the 1-op sine waves?.
#
# Hopefully, some simpler training cases might help the algorithm figure 
# things out?
#
# (I'm a total noob to this AI stuff.)
#
# Craig Iannello 2024/11/04
###############################################################################
import numpy as np
import math
from   scipy import signal
import spect as sp
from   OPL3 import OPL3
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
from pprint import pprint

# randomize the PRNG
random.seed(datetime.datetime.now().timestamp())
dir_path = os.path.dirname(os.path.realpath(__file__))

sfilename = dir_path+'/../training_sets/opl3_training_spects.bin'
rfilename = dir_path+'/../training_sets/opl3_training_regs.bin'

print('''
-------------------------------------------------------------------------------
Notes:

The purpose of this utility is to generate a training set for the AI used in 
the WAV2VGM project. This version is geared towards OPL3 synthesis- It makes 
tons of random OPL3 register settings, and for each one, the PyOPL emulator 
is used to render a short, 4096-point waveform. If the wave is all zeroes, 
it is thrown out, and another config is chosen. If the wave has sound in it,
a frequency spectrum is calculated, and a set of data is added to the training
set output, which consists of two files:

                File Name | Record Type | Rec. Description
   -----------------------+-------------+-----------------
   opl3_training_regs.bin | float[290]  | Synth configuration vector
 opl3_training_spects.bin | byte[2048]  | 2048-bin frequency spectrum

The spectra are used as the inputs for training, and the configs are used
as the ground-truth outputs that we'd like the AI to produce when given
such a spectrum.

*** Note on re-running this utility:

If closed and restarted, this tool will check for pre-existing training
data. If some exists, and if both files are sized correctly, (in multiples
of the record size) the files will be kept and expanded upon. 

If the files aren't the right sizes though, they'll get overwritten!

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

# each operator (indexes [0..35]) has a specific byte-offset to
# locate it within various sections of the opl3 register bank.
#
# Notice how these are NOT CONTINUOUS!  (Caused me some headaches!)  

op_reg_ofs = [ 
  0x000,  0x001,  0x002,  0x003,  0x004,  0x005,  0x008,  0x009,  0x00A,  # bank 0
  0x00B,  0x00C,  0x00D,  0x010,  0x011,  0x012,  0x013,  0x014,  0x015,
  0x100,  0x101,  0x102,  0x103,  0x104,  0x105,  0x108,  0x109,  0x10A,  # bank 1 (OPL3 only)
  0x10B,  0x10C,  0x10D,  0x110,  0x111,  0x112,  0x113,  0x114,  0x115,
  ]

chan_reg_ofs = [
  0x000, 0x001, 0x002, 0x003, 0x004, 0x005, 0x006, 0x007, 0x008, 
  0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108
]

# given a float vector element f, with range [0.0,1.0], and a 
# desired width in bits, rescales the float to an integer of 
# that size.

def vecFltToInt(f, bwid):
  mag = (1<<bwid)-1
  return round(f*mag)

# reverse of the above operation

def vecIntToFlt(i, bwid):
  mag = (1<<bwid)-1
  return float(i/mag)

# Convert regions of interest from an OPL3 register file
# into an synth configuration vector for the AI. 

vector_elem_labels = [
  '11f14','10f13','9f12','2f5','1f4','0f3',
]

keyfreq_vec_idxs = None

def rfToV(rf):
  global op_reg_ofs, chan_reg_ofs, keyfreq_vec_idxs
  note_freqs = False
  if keyfreq_vec_idxs is None:
    note_freqs = True
    keyfreq_vec_idxs = []
  v=[]
  # one chip-wide cfg reg becomes 6 vector elements
  # 0x104: [(0,2),('11f12',1),('10f13',1),('9f12',1),('2f5',1),('1f4',1),('0f4',1)], # sets to 4-op the indicated channel pair
  b = rf[0x104]
  mask = 0b00100000
  while mask:
    if b & mask:
      v.append(1.0)
    else:
      v.append(0.0)
    mask>>=1
  # chan-related things
  # 0xA0: [('FnLow',8)],
  # 0xB0: [(0,2),('KeyOn',1),('Block',3),('FnHi',2)],
  # 0xC0: [('OutD',1),('OutC',1),('OutR',1),('OutL',1),('FbFct',3),('SnTyp',1)]  
  for i in range(0,18):
    o = chan_reg_ofs[i]
    flow = rf[0xA0|o]
    b = rf[0xB0|o]
    c = rf[0xC0|o]
    fhi = b&3
    block = (b>>2)&7
    keyon = (b>>5)&1
    sntype = c&1
    fbcnt = (c>>1)&3
    freq = flow|(fhi<<8)|(block<<10)
    v.append(float(keyon))
    vector_elem_labels.append('KeyOn.c'+str(i))
    if note_freqs:
      keyfreq_vec_idxs.append(len(v))
    v.append(vecIntToFlt(freq,13))
    vector_elem_labels.append('Freq.c'+str(i))
    v.append(vecIntToFlt(fbcnt,3))
    vector_elem_labels.append('FbCnt.c'+str(i))
    v.append(float(sntype))
    vector_elem_labels.append('SnTyp.c'+str(i))
  # op-related_things:
  # 0x20: [('Trem',1),('Vibr',1),('Sust',1),('KSEnvRt',1),('FMul',4)],
  # 0x40: [('KSAtnLv',2),('OutLv',6)],
  # 0x60: [('AttRt',4),('DcyRt',4)],
  # 0x80: [('SusLv',4),('RelRt',4)],  
  for i in range(0,36):
    o=op_reg_ofs[i]
    fmul = rf[0x20|o]&15
    f = rf[0x40|o]
    ws = rf[0xE0|o]&7
    outlv = f&63
    ksatnlv = (f>>6)&3
    v.append(vecIntToFlt(fmul,4))
    vector_elem_labels.append('FMul.o'+str(i))
    v.append(vecIntToFlt(ksatnlv,2))
    vector_elem_labels.append('KSAtnLv.o'+str(i))    
    v.append(vecIntToFlt(outlv,6))
    vector_elem_labels.append('OutLv.o'+str(i))
    v.append(vecIntToFlt(ws,3))
    vector_elem_labels.append('WavSel.o'+str(i))
  '''
  z = zip(vector_elem_labels, v)
  for i,zi in enumerate(z):
    a,b = zi
    print(f'{a:>12}: {b:5.2f}')
  exit()
  '''
  return v

# opposite of the above operation, also sets some 
# defaults that don't concern the AI.
def RF(rf, idx, v):
  return rf[0:idx] + struct.pack('B',v) + rf[idx+1:]


def initRegFile():
  rf = b'\0'*512
  rf = RF(rf,0x105,0x01)
  for i in range(0,36):
    o=op_reg_ofs[i]
    rf = RF(rf,0x20|o,0b00100000)        
    rf = RF(rf,0x60|o,0xff)    
    rf = RF(rf,0x80|o,0x0f)    
  return rf

def vToRf(v):
  global op_reg_ofs, chan_reg_ofs
  rf = initRegFile()
  i = 0
  for j in range(0,6):
    i<<=1
    if v[0+j]>=0.5:
      i|=1
  rf = RF(rf, 0x104, i)
  j=6
  # chan-related things
  # 0xA0: [('FnLow',8)],
  # 0xB0: [(0,2),('KeyOn',1),('Block',3),('FnHi',2)],
  # 0xC0: [('OutD',1),('OutC',1),('OutR',1),('OutL',1),('FbFct',3),('SnTyp',1)]  
  for i in range(0,18):
    o = chan_reg_ofs[i]
    keyon = 1 if v[j+0] >= 0.5 else 0
    freq = vecFltToInt(v[j+1],13)
    fbcnt = vecFltToInt(v[j+2],3)
    sntyp = 1 if v[j+3] >= 0.5 else 0
    j+=4
    flow = freq&0xff
    freq>>=8
    fhi = freq&3
    freq>>=2
    blk = freq&7
    blk>>=3
    sntyp = freq&1
    rf = RF(rf,0xA0|o,flow)
    rf = RF(rf,0xB0|o,(keyon<<5)|(blk<<2)|fhi)
    rf = RF(rf,0xC0|o,0b00110000 | (fbcnt<<1) | sntyp)

  # op-related_things:
  # 0x20: [('Trem',1),('Vibr',1),('Sust',1),('KSEnvRt',1),('FMul',4)],
  # 0x40: [('KSAtnLv',2),('OutLv',6)],
  # 0x60: [('AttRt',4),('DcyRt',4)],
  # 0x80: [('SusLv',4),('RelRt',4)],  
  for i in range(0,36):
    o=op_reg_ofs[i]
    fmul = vecFltToInt(v[j+0],4)
    ksatnlv = vecFltToInt(v[j+1],2)
    outlv = vecFltToInt(v[j+2],6)
    wavsel = vecFltToInt(v[j+3],3)
    j+=4
    rf = RF(rf,0x20|o,0b00100000 | fmul)    
    rf = RF(rf,0x40|o,outlv | (ksatnlv<<6))    
    rf = RF(rf,0x60|o,0xff)    
    rf = RF(rf,0x80|o,0x0f)    
    rf = RF(rf,0xe0|o,wavsel)    
  return rf


_2op_chans = {
   0:[ 0, 3], 1 :[ 1, 4],  2:[ 2, 5],  3:[ 6, 9],
   4:[ 7,10], 5 :[ 8,11],  6:[12,15],  7:[13,16],
   8:[14,17], 9 :[18,21], 10:[19,22], 11:[20,23],
  12:[24,27], 13:[25,28], 14:[26,29], 15:[30,33],
  16:[31,34], 17:[32,35] }

_4op_chan_combos = [
  (0,3),
  (1,4),
  (2,5),
  (9,12),
  (10,13),
  (11,14),
]

def combineChans(dc, c0, c1):
  a = dc[c0]
  b = dc[c1]
  c = a+b
  dc[c0]=c
  del dc[c1]
  return dc

# call after changing any of v[0]...v[5]
def vecGetPermutableIndxs(v):
  global _2op_chans, _4op_chan_combos
  chans = deepcopy(_2op_chans)
  # based on v[0]...v[5] determine available channels
  # and which operators are associated with each.
  for i in range(0,6):
    if v[i] >= 0.5:
      c0,c1 = _4op_chan_combos[5-i]
      chans = combineChans(chans,c0,c1)

  print(v[0:6],chans)

  idxs = {}
  keyons = {}
  lvls = {}
  freqs = {}  
  for c in chans:
    idxs[c]=[]    
    keyons[c]=[]
    lvls[c]=[]
    freqs[c]=[]
    # include indexes of fields relevant to this channel
    s = f'.c{c}'
    for vi,lbl in enumerate(vector_elem_labels):
      if lbl.endswith(s):
        if 'KeyOn' in lbl:
          keyons[c].append(vi)
        else:
          idxs[c].append(vi)
        if 'Freq' in lbl:
          freqs[c].append(vi)
    # and each of the operators
    opidxs = chans[c]
    for oi in opidxs:
      s = f'.o{oi}'
      for vi,lbl in enumerate(vector_elem_labels):
        if lbl.endswith(s):
          idxs[c].append(vi)
          if 'OutLv' in lbl:
            lvls[c].append(vi)

  return idxs,keyons,lvls,freqs

'''
for i in [0,1]:
  rf = initRegFile()
  rf = RF(rf,0x104,i)
  v = rfToV(rf)
  print(i,v[0:6])
  r = vecGetPermutableIndxs(v)
  for c in r:
    vv = r[c]
    print(c,vv)

exit()
'''

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
minwave =  99999999
maxwave = -99999999

def plotWaveform(wave):
  global screen,ww,hh,minwave,maxwave
  ll = len(wave)
  for i in range(0,ll-1):
    s0 = int(wave[i])    
    s1 = int(wave[i+1])
    if s0<minwave:
      minwave = s0
    if s1<minwave:
      minwave = s1
    if s0>maxwave:
      maxwave = s0
    if s1>maxwave:
      maxwave = s1

    x0=int(i*ww/ll)
    x1=int((i+1)*ww/ll)    
    y0=int((s0+32768)*hh/65536)
    y1=int((s1+32768)*hh/65536)
    pygame.draw.line(screen, (255,255,0), (x0,y0),(x1,y1))
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
def renderOPLFrame(regfile):
  global rawbin_low, rawbin_high, wave_low, wave_high
  # init opl emulator
  o = OPL3()
  o.do_init()

  o.writeregfile(regfile)

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
# main loop
# -----------------------------------------------------------------------------
def main():
  global rawbin_high, wave_low, wave_high, reinit_freq
  global ww,hh

  j=0
  opl_vec = rfToV(initRegFile())
  vl = len(opl_vec)
  print(f'vector length: {vl}')

  try:
    ssize = os.path.getsize(sfilename)
    rsize = os.path.getsize(rfilename)
    smod = ssize % 2048
    rmod = rsize % (vl*115)
    print(f'{ssize=}, {smod=}, {rsize=}, {rmod=}')
  except Exception as e:
    ssize = 0
    rsize = 0
    smod = -1
    rmd  = -1

  if (not ssize) or (ssize%2048) or (not rsize) or (rsize%(vl*4)):
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
  perms_this_mode = 0
  REINIT_PERIOD = 10000000

  fuzzchan = 0 #random.randint(0,17)
  permidxs,keyons,lvls,freqs = vecGetPermutableIndxs(opl_vec)
  permidxs = permidxs[fuzzchan]
  keyons = keyons[fuzzchan]
  lvls = lvls[fuzzchan]
  freqs = freqs[fuzzchan]
  print(f'Initial fuzz channel: {fuzzchan}')
  print('permutables',permidxs)
  print('keyons',keyons)
  print('lvls',lvls)      
  print('freqs',freqs)

  for ko in keyons:
    opl_vec[ko] = 1.0
  #for lo in lvls:
  #  opl_vec[lo] = 0 #random.random()*0.2

  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
        rfile.close()
        sfile.close()
        return 

    if random.random()<0.01:
      x = random.choice(permidxs)
    else:
      x = random.choice(freqs)

    if (random.random() < 0.05):
      #if x in lvls:
      #  opl_vec[x] = random.random()*0.2
      #else:
      opl_vec[x] = random.random()
    else:
      o = (random.random()*0.05) - 0.025
      o = opl_vec[x] + o
      if o<0.0:
        o=0.0
      elif o>1.0:
        o=1.0
      #if x in lvls:
      #  if o<0.05:
      #    o=0.05
      #  elif o > 0.5:
      #    o=0.5
      opl_vec[x] = o
    # todo: sort channels by keyon, frequency, amplitude
    # to try to standardize the training inputs?
    # I need to try to figure out how to indicate to the 
    # model that each spectrum can be made by several equivalent 
    # synth configs with channels swapped.

    rf = vToRf(opl_vec)
    # render a 4096-point waveform and its spectrum
    waveform, tspect = renderOPLFrame(rf)


    # if successful, 
    if tspect is not None:
      # write the opl reg configuration vector float[vl]
      # to the cfg training set file
      sbin = b''
      for f in opl_vec:
        sbin+=struct.pack('<f',f)
      rfile.write(sbin)
      
      # write binary of corresponding spectrum (1 byte-per-bin)
      # to the spect training set file
      sbin = b''
      for i in range(0,2048):
        try:
          b = abs(int(tspect[i]))
        except:
          b=0
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
          pygame.draw.rect(screen,(0,0,0),(0,0,ww,hh))
          # waveform
          plotWaveform(waveform)
          plotSpectrum(tspect)
          #if not tspect1 is None:
          #  plotSpectrum(tspect1,(255,128,128))
          pygame.display.update()
          # get output file size in MB
          fszmb = int(fsz/1024.0/1024.0)
          if fszmb!=lastszmb: # every 1MB out, show a status update.
            lastszmb = fszmb
            sfsz = f','
            print(f'iteration: {iters:12d} ({fszmb:d} MB) {minwave=} {maxwave=}')
            if fszmb >= 20000:  # if we hit 20 GB, stop!
              sfile.close()
              rfile.close()
              exit()
        except:
          # sometimes we get a divide by zero
          pass
      # check to see if we need to switch complexity modes
      # or reinit the opl3 registers
      perms_this_mode += 1      
      if  perms_this_mode % REINIT_PERIOD == 0:
        perms_this_mode = 0
        REINIT_PERIOD = random.randint(10000,500000)
        opl_vec = rfToV(initRegFile())
        fuzzchan = 0 #random.randint(0,17)
        print(f'Cleanslate. Switch fuzzed channel to {fuzzchan}')
        didit = False
        while not didit:
          permidxs,keyons,lvls = vecGetPermutableIndxs(opl_vec)
          permidxs = permidxs[fuzzchan]
          keyons = keyons[fuzzchan]
          lvls = lvls[fuzzchan]
          freqs = freqs[fuzzchan]
          print('permutables',permidxs)
          print('keyons',keyons)
          print('lvls',lvls)      
          print('freqs',freqs)
          try:    
            for ko in keyons:            
              opl_vec[ko] = 1.0
            didit=True
          except:
            print('Keyonswitch failed')
            pass
        #for lo in lvls:
        #  opl_vec[lo] = random.random()*0.2

###############################################################################
# ENTRYPOINT
###############################################################################
if __name__ == '__main__':
  main()
###############################################################################
# EOF
###############################################################################
