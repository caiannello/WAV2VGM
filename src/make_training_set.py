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


DISPLAY_INTERVAL = 10     # permutations per display refresh
REINIT_PERIOD = 100000    # permutations per resetting the OPL to defaults

# display window width and height
ww=1920 
hh=1080


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

# First six names of the OPL3 configuration vector. 
# (The other 216 names are the first time we
# convert a an OPL3 register array into a 
# float32[222] synth configuration vector using the
# rfToV() function.

vector_elem_labels = [
  '11f14','10f13','9f12','2f5','1f4','0f3',
]

# like the above, but the bit-width of each field 
# as it is in the opl registers
vector_elem_bits = [
  1,1,1,1,1,1
]

# which two operators are assigned to each channel
# in 2-op mode.

_2op_chans = {
   0:[ 0, 3], 1 :[ 1, 4],  2:[ 2, 5],  3:[ 6, 9],
   4:[ 7,10], 5 :[ 8,11],  6:[12,15],  7:[13,16],
   8:[14,17], 9 :[18,21], 10:[19,22], 11:[20,23],
  12:[24,27], 13:[25,28], 14:[26,29], 15:[30,33],
  16:[31,34], 17:[32,35] }

# which channels can be paired into a single 
# 4-op channel

_4op_chan_combos = [
  (0,3),    # Makes 4-op channel 0, channel 3 goes away
  (1,4),
  (2,5),
  (9,12),
  (10,13),
  (11,14),
]

# things used to convert between frequency in Hz and 
# the OPL3 equivalent: (uint10 fnum, uint3 block)
max_freq_per_block = [48.503,97.006,194.013,388.026,776.053,1552.107,3104.215,6208.431]
fsamp = 14318181.0/288.0


make_labels = True  # True until we have named all vector elements

reinit_freq = 100  # bigger means less chance of resetting the opl to defaults
OPL3_MAX_FREQ = 6208.431  # highest freq we can assign to an operator

# min/max when plotting spectrum bins onscreen
MAX_DB = 0
MIN_DB = -150.0

# Statistics covering all iterations ----------------------

# bin bounds of spect after converting it to dbFS. (decibels of full scale)

dbin_high = -9999999
dbin_low  =  9999999

# waveform min/max seen coming out of the OPL3 emulator. 

wave_high = -9999999
wave_low  =  9999999

# used by the code to combine two 2-op channels
# in dc dict, c0 and c1, into a single 4-op 
# channel at c0.  Channel c1 is goes away.

def combineChans(dc, c0, c1):
  a = dc[c0]
  b = dc[c1]
  c = a+b
  dc[c0]=c
  del dc[c1]
  return dc

# given a frequency in Hz, converts it to the OPL3
# equivalent:  (uint10 fnum, uint3 block)
def freqToFNumBlk(freq):
  global max_freq_per_block, fsamp
  for block,maxfreq in enumerate(max_freq_per_block):
    if maxfreq>=freq:
      break
  fnum = round(freq*pow(2,19)/fsamp/pow(2,block-1))
  return fnum,block

# inverse of the above
def fNumBlkToFreq(fnum, block):
  freq = fnum/(pow(2,19)/fsamp/pow(2,block-1))
  return freq

# given a float vector element f, with range [0.0,1.0], and a 
# desired width in bits, rescales the float to an integer of 
# that size.

def vecFltToInt(f, bwid):
  mag = (1<<bwid)-1
  i = round(f*mag)
  if i<0:
    i=0
  elif i>mag:
    i=mag
  return i

# reverse of the above operation

def vecIntToFlt(i, bwid):
  mag = (1<<bwid)-1
  f = float(i/mag)
  if f<0.0:
    f=0.0
  elif f>1.0:
    f=1.0
  return f

# the first time through rfToV(), this fcn
# is used to note vector element names.

def nameVecElem(s,bwid):
  global make_labels, vector_elem_labels, vector_elem_bits
  if make_labels:
    vector_elem_labels.append(s)
    vector_elem_bits.append(bwid)

# Convert regions of interest from a 512-byte 
# OPL3 register file into float32[222] synth configuration
# vector for use during AI training and infrerencing. 
#
# The first time though, we also build an array of what 
# each vector element is named: 
# (e.g. fnum+block for channel 0 gets called "Freq.c0" 
# operator 3 output level gets called "OutLv.o3")

def rfToV(rf):
  global op_reg_ofs, chan_reg_ofs, make_labels, OPL3_MAX_FREQ
  #
  # Chip wide things start off our vector:
  #
  # one cfg reg (at 0x104) has six bits in it which become 
  # the first six elements of the vector:
  #
  # 0x104: bit 5 : '11f14', bit 4 : '10f13', bit 3 :'9f12',
  #        bit 2 : '2f5',   bit 1 : '1f4',   bit 0 :'0f3'
  #
  # Setting a bit will pair the two, 2-op channels indicated by
  # the name into a single 4-op channel.
  #
  # e.g. bit zero becomes vector elemment 5, and if set, that
  # element will have a value of 1.0, and synth channels 0 & 3
  # are to be combined into a single 4-operator channel 0.
  #

  v=[]
  b = rf[0x104]
  mask = 0b00100000
  while mask:
    if b & mask:
      v.append(1.0)  # vectorizing the high bit first
    else:
      v.append(0.0)
    mask>>=1
  #
  # Channel-related things come next:
  #
  # 0xA0: [('FnLow',8)],  
  # 0xB0: [(0,2),('KeyOn',1),('Block',3),('FnHi',2)],
  #
  # three of the above (fnum hi/low and block) get 
  # concatenated to make one vector element called 
  # "frequency" which controls the channel frequency.
  #
  # 0xC0: [('OutD',1),('OutC',1),('OutR',1),('OutL',1),('FbFct',3),('SnTyp',1)]  
  #
  # we are hardcoding the fist four fields of this (controls
  # output speaker) and so they arent included in the vector.
  # This is done to simplify things since we're doing only 
  # mono sounds.
  #
  for i in range(0,18):
    o = chan_reg_ofs[i]
    flow = rf[0xA0|o]
    b = rf[0xB0|o]
    c = rf[0xC0|o]
    fhi = b&3
    block = (b>>2)&7
    keyon = (b>>5)&1
    sntype = c&1
    fbcnt = (c>>1)&7
    fnum = flow|(fhi<<8)
    freq = fNumBlkToFreq(fnum, block)
    v.append(float(keyon))
    nameVecElem('KeyOn.c'+str(i),1)
    v.append(freq / OPL3_MAX_FREQ)
    nameVecElem('Freq.c'+str(i),13)
    v.append(vecIntToFlt(fbcnt,3))
    nameVecElem('FbCnt.c'+str(i),3)
    v.append(float(sntype))
    nameVecElem('SnTyp.c'+str(i),1)
  #
  # Operator related things come last:
  #
  # 0x20: [('Trem',1),('Vibr',1),('Sust',1),('KSEnvRt',1),('FMul',4)],
  # 0x40: [('KSAtnLv',2),('OutLv',6)],
  # 0xE0: [('_',5),'WavSel':3]
  #
  # # 0x60: [('AttRt',4),('DcyRt',4)],
  # # 0x80: [('SusLv',4),('RelRt',4)],
  #
  # Envelope related (0x60 and 0x80) are not vectorized 
  # and are instead hard-coded in our app.
  for i in range(0,36):
    o=op_reg_ofs[i]
    fmul = rf[0x20|o]&15
    f = rf[0x40|o]
    ws = rf[0xE0|o]&7
    outlv = f&63
    ksatnlv = (f>>6)&3
    v.append(vecIntToFlt(fmul,4))
    nameVecElem('FMul.o'+str(i),4)    # operator phase multiple
    v.append(vecIntToFlt(ksatnlv,2))
    nameVecElem('KSAtnLv.o'+str(i),2) # attenuation of higher freqs
    v.append(vecIntToFlt(outlv,6))
    nameVecElem('OutLv.o'+str(i),6)   # overall attenuation
    v.append(vecIntToFlt(ws,3))
    nameVecElem('WavSel.o'+str(i),3)  # waveform selection 0..7

  make_labels = False   # all vector elements were named.  
  return v

# Show label:value of each element of the
# specified float32[222] vector.

def showVector(v):
  global vector_elem_labels
  z = zip(vector_elem_labels, v)
  j = 0
  l=''
  print('------------------------------- [')
  for i,zi in enumerate(z):
    a,b = zi
    l+=f'{a:>12}: {b:5.2f}, '
    j+=1
    if j>5:
      j=0
      print(l)
      l=''
  if len(l):
    print(l)
  print('] -------------------------------')



# opposite of the above operation, also sets some 
# defaults that don't concern the AI.
def RF(rf, idx, v):
  return rf[0:idx] + struct.pack('B',v) + rf[idx+1:]

# Returns initial synth settings as a 512-byte OPL3
# register value file.
#
# We hard-code certain things for our application:
# all envelopes rates are set to fastest rate, 
# sustain level set to loudest, and sustain and 
# OPL3 mode are enabled.

def initRegFile():
  rf = b'\0'*512
  rf = RF(rf,0x105,0x01)
  for i in range(0,36):
    o=op_reg_ofs[i]
    rf = RF(rf,0x20|o,0b00100000)   # enable sustain
    rf = RF(rf,0x60|o,0xff)   # fast envelopes
    rf = RF(rf,0x80|o,0x0f)   # sustain level to loudeest
  return rf

# Converts a float[222] synth configuration vector
# into a 512-byte OPL3 register array
def vToRf(v):
  global op_reg_ofs, chan_reg_ofs, OPL3_MAX_FREQ
  rf = initRegFile()

  # chipwide things (0..5)
  i = 0
  for j in range(0,6):
    i<<=1
    if v[0+j]>=0.5:
      i|=1
  rf = RF(rf, 0x104, i)
  j=6

  # channel-related things
  for i in range(0,18):
    o = chan_reg_ofs[i]
    keyon = 1 if v[j+0] >= 0.5 else 0
    freq = v[j+1]
    fbcnt = vecFltToInt(v[j+2],3)
    sntyp = 1 if v[j+3] >= 0.5 else 0
    j+=4

    
    fnum, blk = freqToFNumBlk( freq * OPL3_MAX_FREQ )
    flow = fnum&0xff
    fhi = (fnum>>8)&3

    rf = RF(rf,0xA0|o,flow)
    rf = RF(rf,0xB0|o,(keyon<<5)|(blk<<2)|fhi)
    rf = RF(rf,0xC0|o,0b00110000 | (fbcnt<<1) | sntyp)

  # operator-related_things:
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

# call after changing any of v[0]...v[5]
# to get what synth cfg vector element indices will 
# have any effect on the output sound

def vecGetPermutableIndxs(v):
  global _2op_chans, _4op_chan_combos
  chans = deepcopy(_2op_chans)
  # based on v[0]...v[5] determine available channels
  # and which operators are associated with each.
  for i in range(0,6):
    if v[i] >= 0.5:
      c0,c1 = _4op_chan_combos[5-i]
      chans = combineChans(chans,c0,c1)
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
  # return waveform and spectrogram, if any
  return wave, spec

# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------
def main():
  global rawbin_high, wave_low, wave_high, dbin_low, dbin_high
  global reinit_freq, ww,hh, vector_elem_bits
  global REINIT_PERIOD, DISPLAY_INTERVAL

  # init config vector
  j=0
  opl_vec = rfToV(initRegFile())
  vl = len(opl_vec)

  # pick one channel to fuzz, for now.

  # see what elements are permutable per each channel
  permidxs,keyons,lvls,freqs = vecGetPermutableIndxs(opl_vec)

  fuzzchan = 0 #random.randint(0,17)
  print(f'Channnel to fuzz: {fuzzchan}')
  
  permidxs = permidxs[fuzzchan]   # permutable vector elems for selected chan(s)
  lvls = lvls[fuzzchan]           # operator attenuation levels
  freqs = freqs[fuzzchan]         # channel frequency setting(s)
  keyons = keyons[fuzzchan]       # channel key-on setting(s)
  for ko in keyons:     # ensure keyon is set for selected channel(s)
    opl_vec[ko] = 1.0

  print('all permutables: ',permidxs)
  print('  op atten. lvs: ',lvls)      
  print('   chan. keyons: ',keyons)
  print('    chan. freqs: ',freqs)

  print(f'\nInitial float32[{len(opl_vec)}] Config. Vector:')
  showVector(opl_vec)

  # check for training file
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
  lastszmb = -1
  iters=ssize//2048
  perms_this_mode = 0

  fbumpdir = 1

  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  # Usually wise to be able to close your program.
        rfile.close()
        sfile.close()
        return 

    if random.random()<0.98:
      x = random.choice(freqs)
      o = opl_vec[x]
      o += fbumpdir * 0.0005
      if o>1.0:
        o=1.0
        fbumpdir = -1
      elif o<0.0:
        o=0.0
        fbumpdir = 1
      opl_vec[x] = o
    else:  
      x = random.choice(permidxs)
      opl_vec[x]=random.random()
      if x in lvls:
        opl_vec[x]/=32

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
      if j==DISPLAY_INTERVAL:  # show every 10th set on screen
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
            l = f'iteration: {iters:12d} ({fszmb:d} MB), samp min/max:({wave_low},{wave_high}), bin min/max:({dbin_low},{dbin_high}), cur_lvls:['
            for x in lvls:
              l += f'{opl_vec[x]:4.2f},'            
            print(l[0:-1]+']')
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
        permidxs,keyons,lvls,freqs = vecGetPermutableIndxs(opl_vec)
        fuzzchan = 0 #random.randint(0,17)
        print(f'Channnel to fuzz: {fuzzchan}')    
        permidxs = permidxs[fuzzchan]   # permutable vector elems for selected chan(s)
        lvls = lvls[fuzzchan]           # operator attenuation levels
        freqs = freqs[fuzzchan]         # channel frequency setting(s)
        keyons = keyons[fuzzchan]       # channel key-on setting(s)
        for ko in keyons:     # ensure keyon is set for selected channel(s)
          opl_vec[ko] = 1.0

###############################################################################
# ENTRYPOINT
###############################################################################
if __name__ == '__main__':
  main()
###############################################################################
# EOF
###############################################################################
