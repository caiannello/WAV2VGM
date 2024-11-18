
# eventually want to have these different OPL3-related
# features in here (as one or more classes:)

# 1: The OPL3 sound emulator (already present) 
#
# 2: a. OPL3 convenience class that adds a per-channel interface abstraction,
#    b. input and output of AI-friendly synth configuration vectors and metadata
#    c. Functions to permute specific settings/channels for training set generator.
#
# 3: VGM data writer
#
# 4: The pytorch OPL3 Model and Dataset getters
#
#import array
#from enum import Enum
#import struct
#import typing
import pyopl
import numpy as np
import math
from   scipy import signal
try:
  import spect as sp
except:
  from . import spect as sp
import struct
import random
from copy import deepcopy
# OPL3 Chip class: using Audio emulation from DOSBox ---
# Adapted from opl_emu-Master: regression test
# adapted from DRO Trimmer by Laurence Dougal Myers.
# Extended for WAV2VGM by C. Iannello
# The methods from the emu example have a preceeding 
# underscore and are snake case.  Additions don't, and
# are they're camelCase.
class OPL3:
  reg_settings = {}  # a dict of (b,r)=v  for all settings

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

  # all permutable opl3 fields, named and sorted into related objects
  # to help genetic crossover and maybe the AI.
  '''
  def makeVecElemNames(self):
    shop_list = [
      '0f3',   'c0',    'o0',    'o3',    'c3',    'o6',    'o9',
      '1f4',   'c1',    'o1',    'o4',    'c4',    'o7',    'o10',
      '2f5',   'c2',    'o2',    'o5',    'c5',    'o8',    'o11',
      'c6',    'o12',   'o15',
      'c7',    'o13',   'o16',
      'c8',    'o14',   'o17',
      '9f12',  'c9',    'o18',   'o21',   'c12',   'o24',   'o27',
      '10f13', 'c10',   'o19',   'o22',   'c13',   'o25',   'o28',
      '11f14', 'c11',   'o20',   'o23',   'c14',   'o26',   'o29',
      'c15',   'o30',   'o33',
      'c16',   'o31',   'o34',
      'c17',   'o32',   'o35',
    ]
  '''
  # for speeding up rfToV and vToRf
  vec_reloc = {

  }
  got_reloc = False

  vec_elem_names = [
    '0f3', 'KeyOn.c0', 'Freq.c0', 'FbCnt.c0', 'SnTyp.c0', 'FMul.o0', 'KSAtnLv.o0', 'AttnLv.o0', 
    'WavSel.o0', 'FMul.o3', 'KSAtnLv.o3', 'AttnLv.o3', 'WavSel.o3', 'KeyOn.c3', 'Freq.c3', 
    'FbCnt.c3', 'SnTyp.c3', 'FMul.o6', 'KSAtnLv.o6', 'AttnLv.o6', 'WavSel.o6', 'FMul.o9', 
    'KSAtnLv.o9', 'AttnLv.o9', 'WavSel.o9', '1f4', 'KeyOn.c1', 'Freq.c1', 'FbCnt.c1', 'SnTyp.c1', 
    'FMul.o1', 'KSAtnLv.o1', 'AttnLv.o1', 'WavSel.o1', 'FMul.o4', 'KSAtnLv.o4', 'AttnLv.o4', 
    'WavSel.o4', 'KeyOn.c4', 'Freq.c4', 'FbCnt.c4', 'SnTyp.c4', 'FMul.o7', 'KSAtnLv.o7', 
    'AttnLv.o7', 'WavSel.o7', 'FMul.o10', 'KSAtnLv.o10', 'AttnLv.o10', 'WavSel.o10', '2f5', 
    'KeyOn.c2', 'Freq.c2', 'FbCnt.c2', 'SnTyp.c2', 'FMul.o2', 'KSAtnLv.o2', 'AttnLv.o2', 
    'WavSel.o2', 'FMul.o5', 'KSAtnLv.o5', 'AttnLv.o5', 'WavSel.o5', 'KeyOn.c5', 'Freq.c5', 
    'FbCnt.c5', 'SnTyp.c5', 'FMul.o8', 'KSAtnLv.o8', 'AttnLv.o8', 'WavSel.o8', 'FMul.o11', 
    'KSAtnLv.o11', 'AttnLv.o11', 'WavSel.o11', 'KeyOn.c6', 'Freq.c6', 'FbCnt.c6', 'SnTyp.c6', 
    'FMul.o12', 'KSAtnLv.o12', 'AttnLv.o12', 'WavSel.o12', 'FMul.o15', 'KSAtnLv.o15', 'AttnLv.o15', 
    'WavSel.o15', 'KeyOn.c7', 'Freq.c7', 'FbCnt.c7', 'SnTyp.c7', 'FMul.o13', 'KSAtnLv.o13', 
    'AttnLv.o13', 'WavSel.o13', 'FMul.o16', 'KSAtnLv.o16', 'AttnLv.o16', 'WavSel.o16', 'KeyOn.c8', 
    'Freq.c8', 'FbCnt.c8', 'SnTyp.c8', 'FMul.o14', 'KSAtnLv.o14', 'AttnLv.o14', 'WavSel.o14', 
    'FMul.o17', 'KSAtnLv.o17', 'AttnLv.o17', 'WavSel.o17', '9f12', 'KeyOn.c9', 'Freq.c9', 'FbCnt.c9', 
    'SnTyp.c9', 'FMul.o18', 'KSAtnLv.o18', 'AttnLv.o18', 'WavSel.o18', 'FMul.o21', 'KSAtnLv.o21', 
    'AttnLv.o21', 'WavSel.o21', 'KeyOn.c12', 'Freq.c12', 'FbCnt.c12', 'SnTyp.c12', 'FMul.o24', 
    'KSAtnLv.o24', 'AttnLv.o24', 'WavSel.o24', 'FMul.o27', 'KSAtnLv.o27', 'AttnLv.o27', 'WavSel.o27', 
    '10f13', 'KeyOn.c10', 'Freq.c10', 'FbCnt.c10', 'SnTyp.c10', 'FMul.o19', 'KSAtnLv.o19', 
    'AttnLv.o19', 'WavSel.o19', 'FMul.o22', 'KSAtnLv.o22', 'AttnLv.o22', 'WavSel.o22', 'KeyOn.c13', 
    'Freq.c13', 'FbCnt.c13', 'SnTyp.c13', 'FMul.o25', 'KSAtnLv.o25', 'AttnLv.o25', 'WavSel.o25', 
    'FMul.o28', 'KSAtnLv.o28', 'AttnLv.o28', 'WavSel.o28', '11f14', 'KeyOn.c11', 'Freq.c11', 
    'FbCnt.c11', 'SnTyp.c11', 'FMul.o20', 'KSAtnLv.o20', 'AttnLv.o20', 'WavSel.o20', 'FMul.o23', 
    'KSAtnLv.o23', 'AttnLv.o23', 'WavSel.o23', 'KeyOn.c14', 'Freq.c14', 'FbCnt.c14', 'SnTyp.c14', 
    'FMul.o26', 'KSAtnLv.o26', 'AttnLv.o26', 'WavSel.o26', 'FMul.o29', 'KSAtnLv.o29', 'AttnLv.o29', 
    'WavSel.o29', 'KeyOn.c15', 'Freq.c15', 'FbCnt.c15', 'SnTyp.c15', 'FMul.o30', 'KSAtnLv.o30', 
    'AttnLv.o30', 'WavSel.o30', 'FMul.o33', 'KSAtnLv.o33', 'AttnLv.o33', 'WavSel.o33', 'KeyOn.c16', 
    'Freq.c16', 'FbCnt.c16', 'SnTyp.c16', 'FMul.o31', 'KSAtnLv.o31', 'AttnLv.o31', 'WavSel.o31', 
    'FMul.o34', 'KSAtnLv.o34', 'AttnLv.o34', 'WavSel.o34', 'KeyOn.c17', 'Freq.c17', 'FbCnt.c17', 
    'SnTyp.c17', 'FMul.o32', 'KSAtnLv.o32', 'AttnLv.o32', 'WavSel.o32', 'FMul.o35', 'KSAtnLv.o35', 
    'AttnLv.o35', 'WavSel.o35']
  vec_elem_bits = [
    1, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 1, 13, 3, 1, 
    4, 2, 6, 3, 4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 1, 13, 3, 1, 4, 2, 6, 3, 4, 
    2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 13, 3, 
    1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 1, 13, 3, 1, 4, 2, 6, 3, 
    4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 
    13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 
    6, 3, 4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3, 1, 
    13, 3, 1, 4, 2, 6, 3, 4, 2, 6, 3]
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

  OPL3_MAX_FREQ = 6208.431  # highest freq we can assign to an operator

  # waveform sample min/max seen by this instance

  wave_high = -9999999
  wave_low  =  9999999

  # spectral bin min/max seen by this instance

  bin_high  = -9999999
  bin_low   =  9999999

  def __init__(self) -> None:
    self._bit_depth = 16
    self._buffer_size = 512
    self._channels = 2
    self._buffer = self._create_bytearray(self._buffer_size)
    self._frequency = 44100 #49716
    self._opl: pyopl.opl = pyopl.opl(
      self._frequency,
      sampleSize=(self._bit_depth // 8),
      channels=self._channels,
    )
    self._sample_overflow = 0
    self._output = bytes()

  def _create_bytearray(self, size: int) -> bytearray:
    return bytearray(size * (self._bit_depth // 8) * self._channels)

  def _render_ms(self, length_ms: int) -> None:
    samples_to_render = length_ms * self._frequency / 1000.0
    self._render_samples(samples_to_render)

  def _render_samples(self, samples_to_render: float) -> None:
    samples_to_render += self._sample_overflow
    self.sample_overflow = samples_to_render % 1
    if samples_to_render < 2:
      # Limitation: needs a minimum of two samples.
      return
    samples_to_render = int(samples_to_render // 1)
    while samples_to_render > 1:
      if samples_to_render < self._buffer_size:
        tmp_buffer = self._create_bytearray(
          (samples_to_render % self._buffer_size)
        )
        samples_to_render = 0
      else:
        tmp_buffer = self._buffer
        samples_to_render -= self._buffer_size
      self._opl.getSamples(tmp_buffer)
      self._output += tmp_buffer

  def _do_init(self):
    # Reset (may not be required)
    for bank in range(2):
      for reg in range(0x100):
        self._writeReg(bank, reg, 0x00)

  def _writeReg(self, bank: int, register: int, value: int) -> None:
    self.reg_settings[(bank,register)] = value
    #self._opl.writeReg(register | (bank << 2), value)
    self._opl.writeReg(bank<<8|register, value)

  def _writeRegFile(self, rf) -> None:
    for i,v in enumerate(rf):
      self._opl.writeReg( i, v)


  # the "AttnLvl.oXX" field for an operator,
  # an int6 : [0...63],  or vec : [0.0...1.0],
  # ranges from 0.0 dB to 48dB, where 0 is loudest.
  # Picking random amplitudes normally will
  # tend towards quiet signals, so we need this 
  # alternative picking from log-scale things:

  def randomAtten(self):
      # Generate a random number between 0 and 1
      u = random.random()
      # Apply exponential decay to bias toward lower values
      return math.exp(-u * 10.0)

  '''
  for j in range(1000):
    print(f'{randomAtten():6.4f}')
  exit()
  '''

  # used by the code to combine two 2-op channels
  # in dc dict, c0 and c1, into a single 4-op 
  # channel at c0.  Channel c1 is goes away.

  def combineChans(self, dc, c0, c1):
    a = dc[c0]
    b = dc[c1]
    c = a+b
    dc[c0]=c
    del dc[c1]
    return dc

  # given a frequency in Hz, converts it to the OPL3
  # equivalent:  (uint10 fnum, uint3 block)
  def freqToFNumBlk(self, freq):
    for block,maxfreq in enumerate(self.max_freq_per_block):
      if maxfreq>=freq:
        break
    fnum = round(freq*pow(2,19)/self.fsamp/pow(2,block-1))
    return fnum, block

  # inverse of the above
  def fNumBlkToFreq(self, fnum, block):
    freq = fnum/(pow(2,19)/self.fsamp/pow(2,block-1))
    return freq

  # given a float vector element f, with range [0.0,1.0], and a 
  # desired width in bits, rescales the float to an integer of 
  # that size.

  def vecFltToInt(self, f, bwid):
    mag = (1<<bwid)-1
    i = round(f*mag)
    if i<0:
      i=0
    elif i>mag:
      i=mag
    return i

  # reverse of the above operation

  def vecIntToFlt(self, i, bwid):
    mag = (1<<bwid)-1
    f = float(i/mag)
    if f<0.0:
      f=0.0
    elif f>1.0:
      f=1.0
    return f

  def addNamedVecElem(self, v, i, name, val):
    if self.got_reloc:
      x = self.vec_reloc[i]
    else:
      x = self.vec_elem_names.index(name)
      self.vec_reloc[i] = x
    v[x] = val
    return v

  def getNamedVecElemFloat(self, v, i, name):
    if self.got_reloc:
      x = self.vec_reloc[i]
    else:
      x = self.vec_elem_names.index(name)
      self.vec_reloc[i] = x
    f = v[x]
    return f

  def getNamedVecElemInt(self, v, i, name):
    if self.got_reloc:
      x = self.vec_reloc[i]
    else:
      x = self.vec_elem_names.index(name)
      self.vec_reloc[i] = x

    f = v[x]
    bw = self.vec_elem_bits[x]
    return self.vecFltToInt(f,bw)

  def setNamedVecElemFloat(self, v, name,f):
    x = self.vec_elem_names.index(name)    
    v[x] = f
    return v

  def setNamedVecElemInt(self, v, name,i):
    x = self.vec_elem_names.index(name)
    bw = self.vec_elem_bits[x]
    v[x] = self.vecIntToFlt(i,bw)
    return v

  # Convert regions of interest from a 512-byte 
  # OPL3 register file into float32[222] synth configuration
  # vector for use during AI training and infrerencing. 
  #
  # The first time though, we also build an array of what 
  # each vector element is named: 
  # (e.g. fnum+block for channel 0 gets called "Freq.c0" 
  # operator 3 output attenuation level gets called "AttnLv.o3")

  def rfToV(self, rf):
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

    v=[0.0] * len(self.vec_elem_names)
    b = rf[0x104]
    mask = 0b00100000
    j = 0
    names = ['11f14','10f13','9f12','2f5','1f4','0f3']
    while mask:
      if b & mask:
        val = 1.0
      else:
        val = 0.0
      name = names[j]
      v = self.addNamedVecElem(v,j,name,val)
      mask>>=1
      j+=1
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
      o = self.chan_reg_ofs[i]
      flow = rf[0xA0|o]
      b = rf[0xB0|o]
      c = rf[0xC0|o]
      fhi = b&3
      block = (b>>2)&7
      keyon = (b>>5)&1
      sntype = c&1
      fbcnt = (c>>1)&7
      fnum = flow|(fhi<<8)
      freq = self.fNumBlkToFreq(fnum, block)
      v = self.addNamedVecElem(v,i*4+6,'KeyOn.c'+str(i) , float(keyon))
      v = self.addNamedVecElem(v,i*4+7,'Freq.c'+str(i)  , freq / self.OPL3_MAX_FREQ)
      v = self.addNamedVecElem(v,i*4+8,'FbCnt.c'+str(i) , self.vecIntToFlt(fbcnt,3))
      v = self.addNamedVecElem(v,i*4+9,'SnTyp.c'+str(i) , float(sntype))
    #
    # Operator related things come last:
    #
    # 0x20: [('Trem',1),('Vibr',1),('Sust',1),('KSEnvRt',1),('FMul',4)],
    # 0x40: [('KSAtnLv',2),('AttnLv',6)],
    # 0xE0: [('_',5),'WavSel':3]
    #
    # # 0x60: [('AttRt',4),('DcyRt',4)],
    # # 0x80: [('SusLv',4),('RelRt',4)],
    #
    # Envelope related (0x60 and 0x80) are not vectorized 
    # and are instead hard-coded in our app.
    for i in range(0,36):
      o=self.op_reg_ofs[i]
      fmul = rf[0x20|o]&15
      f = rf[0x40|o]
      ws = rf[0xE0|o]&7
      attnlv = f&63
      ksatnlv = (f>>6)&3
      v = self.addNamedVecElem(v,i*4+(4*18)+10,'FMul.o'+str(i),self.vecIntToFlt(fmul,4))
      v = self.addNamedVecElem(v,i*4+(4*18)+11,'KSAtnLv.o'+str(i),self.vecIntToFlt(ksatnlv,2))
      v = self.addNamedVecElem(v,i*4+(4*18)+12,'AttnLv.o'+str(i),self.vecIntToFlt(attnlv,6))
      v = self.addNamedVecElem(v,i*4+(4*18)+13,'WavSel.o'+str(i),self.vecIntToFlt(ws,3))

    self.got_reloc = True
    #print(self.vec_reloc)
    #exit()

    return v

  # Show label:value of each element of the
  # specified float32[222] vector.

  def showVector(self, v):
    z = zip(self.vec_elem_names, v)
    j = 0
    l=''
    print('------------------------------- [')
    for i,zi in enumerate(z):
      a,b = zi
      l+=f'{i:>3d} {a:>12}: {b:5.2f}, '
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
  def RF(self, rf, idx, v):
    try:
      rf = rf[0:idx] + struct.pack('B',v) + rf[idx+1:]
    except Exception as e:
      print(e, v)
      raise Exception(e)
    return rf

  # Returns initial synth settings as a 512-byte OPL3
  # register value file.
  #
  # We hard-code certain things for our application:
  # all envelopes rates are set to fastest rate, 
  # sustain level set to loudest, and sustain and 
  # OPL3 mode are enabled.

  def initRegFile(self):
    rf = b'\0'*512
    rf = self.RF(rf,0x105,0x01)
    for i in range(0,36):
      o=self.op_reg_ofs[i]
      rf = self.RF(rf,0x20|o,0b00100000)   # enable sustain
      rf = self.RF(rf,0x60|o,0xff)   # fast envelopes
      rf = self.RF(rf,0x80|o,0x0f)   # sustain level to loudeest
    return rf

  # Converts a float[222] synth configuration vector
  # into a 512-byte OPL3 register array
  def vToRf(self, v):
    rf = self.initRegFile()

    # chipwide things (0..5)
    names = ['11f14','10f13','9f12','2f5','1f4','0f3']
    i=0
    for j in range(0,6):
      i<<=1
      name = names[j]
      if self.getNamedVecElemFloat(v,j,name)>=0.5:
        i|=1
    rf = self.RF(rf, 0x104, i)

    # channel-related things
    for i in range(0,18):
      o = self.chan_reg_ofs[i]
      keyon = self.getNamedVecElemInt(v,i*4+6,'KeyOn.c'+str(i))
      freq = self.getNamedVecElemFloat(v,i*4+7,'Freq.c'+str(i))
      fbcnt = self.getNamedVecElemInt(v,i*4+8,'FbCnt.c'+str(i))
      sntyp = self.getNamedVecElemInt(v,i*4+9,'SnTyp.c'+str(i))

      fnum, blk = self.freqToFNumBlk( freq * self.OPL3_MAX_FREQ )
      flow = fnum&0xff
      fhi = (fnum>>8)&3

      rf = self.RF(rf,0xA0|o,flow)
      rf = self.RF(rf,0xB0|o,(keyon<<5)|(blk<<2)|fhi)
      rf = self.RF(rf,0xC0|o,0b00110000 | (fbcnt<<1) | sntyp)

    # operator-related_things:
    for i in range(0,36):
      o=self.op_reg_ofs[i]
      fmul = self.getNamedVecElemInt(v,i*4+(4*18)+10,'FMul.o'+str(i))
      ksatnlv = self.getNamedVecElemInt(v,i*4+(4*18)+11,'KSAtnLv.o'+str(i))
      attnlv = self.getNamedVecElemInt(v,i*4+(4*18)+12,'AttnLv.o'+str(i))
      wavsel = self.getNamedVecElemInt(v,i*4+(4*18)+13,'WavSel.o'+str(i))
      j+=4
      rf = self.RF(rf,0x20|o,0b00100000 | fmul)    
      rf = self.RF(rf,0x40|o,attnlv | (ksatnlv<<6))    
      rf = self.RF(rf,0x60|o,0xff)    
      rf = self.RF(rf,0x80|o,0x0f)    
      rf = self.RF(rf,0xe0|o,wavsel)    
    return rf

  # call after changing any of v[0]...v[5]
  # to get what synth cfg vector element indices will 
  # have any effect on the output sound

  def vecGetPermutableIndxs(self, v, inc_keyons=False):
    chans = deepcopy(self._2op_chans)
    # based on v[0]...v[5] determine available channels
    # and which operators are associated with each.
    for i in range(0,6):
      if v[i] >= 0.5:
        c0,c1 = self._4op_chan_combos[i]
        chans = self.combineChans(chans,c0,c1)

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
      for vi,lbl in enumerate(self.vec_elem_names):
        if lbl.endswith(s):
          if 'KeyOn' in lbl:
            keyons[c].append(vi)
            if inc_keyons:
              idxs[c].append(vi)
          else:
            idxs[c].append(vi)
          if 'Freq' in lbl:
            freqs[c].append(vi)
      # and each of the operators
      opidxs = chans[c]
      for oi in opidxs:
        s = f'.o{oi}'
        for vi,lbl in enumerate(self.vec_elem_names):
          if lbl.endswith(s):
            idxs[c].append(vi)
            if 'AttnLv' in lbl:
              lvls[c].append(vi)

    return idxs,keyons,lvls,freqs

  # -----------------------------------------------------------------------------
  def stereoBytesToMonoNumpy(self, bytes_stream):
    stereo_audio = np.frombuffer(bytes_stream, dtype=np.int16)
    stereo_audio = stereo_audio.reshape(-1, 2)  # Each row is [Left, Right]
    mono_audio = stereo_audio.mean(axis=1, dtype=np.int16)
    return mono_audio

  # -----------------------------------------------------------------------------
  # Setup the OPL emulator with the specified register values, generate 4096 
  # audio samples, and return resultant frequency spectrum.
  # -----------------------------------------------------------------------------
  def renderOPLFrame(self, cfg):
    # init opl emulator
    self._do_init()
    if cfg is None:
      return None, None
    if isinstance(cfg, dict):  # if config is an old-style opl reg dict
      rf = self.initRegFile()
      keys = list(cfg.keys())
      keys.sort()
      for key in keys:
        b,r = key
        v = cfg[key]
        self._writeReg(b,r,v)
    elif isinstance(cfg,list):  # float32[] cfg vector
      try:
        rf = self.vToRf(cfg)
      except Exception as e:
        print(e,cfg)
        exit()
      self._writeRegFile(rf)   
    else:
      self._writeRegFile(cfg)   # bytes[512] opl3 register file
    self._output = bytes()
    # render 4096 samples
    self._render_samples(4096)  
    '''
    # convert to mono, and note min/max sample for statistics
    ll = len(self._output)
    wave = []
    for i in range(0,ll,4):
      l=struct.unpack('<h',self._output[i:i+2])[0]
      r=struct.unpack('<h',self._output[i+2:i+4])[0]
      if l<self.wave_low:
        self.wave_low = l
      if r<self.wave_low:
        self.wave_low = r
      if l>self.wave_high:
        self.wave_high = l
      if r>self.wave_high:
        self.wave_high = r
      wave.append((l+r)//2)  
    wave = np.array(wave, dtype="int16")
    '''
    wave = self.stereoBytesToMonoNumpy(self._output)
    # if not flat-line, generate spectrogram
    if wave.sum():
      spec = sp.spect(wav_filename = None, sample_rate=44100,samples=wave,nperseg=4096, quiet=True, clip = False)    
      # we want only the first spectrum of spectogram
      spec = spec.spectrogram[0]
      for b in spec:
        if b < self.bin_low:
          self.bin_low = b
        if b > self.bin_high:
          self.bin_high = b
    else:
      spec  = None
    # return waveform and spectrogram, if any
    return wave, spec


###############################################################################
# ENTRYPOINT
###############################################################################
#if __name__ == '__main__':
#  main()
###############################################################################
# EOF
###############################################################################
