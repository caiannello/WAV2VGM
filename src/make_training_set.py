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
from   OPL3 import OPL3
import struct
import pyopl
import random
import datetime
import time
import pygame
from pygame.locals import *
import gzip
import sys
import os
# -----------------------------------------------------------------------------
DISPLAY_INTERVAL = 50     # permutations per display refresh
REINIT_PERIOD = 10000     # permutations per resetting the OPL to defaults
MAX_DB = 0                # min/max when plotting spectrum bins onscreen
MIN_DB = -150.0
WW=1920                   # display window width and height
HH=1080
MYPATH = os.path.dirname(os.path.realpath(__file__))
SPECT_FILEPATH = MYPATH+'/../training_sets/opl3_training_spects.bin'
REGS_FILEPATH = MYPATH+'/../training_sets/opl3_training_regs.bin'
# -----------------------------------------------------------------------------
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
# randomize the PRNG
random.seed(datetime.datetime.now().timestamp())
# -----------------------------------------------------------------------------
# init pygame for graphics
pygame.init()
screen=pygame.display.set_mode([WW,HH])#,flags=pygame.FULLSCREEN)
pygame.display.set_caption(f'Making OPL3 Training Set')


# Statistics covering all iterations ----------------------

# bin bounds of spect after converting it to dbFS. (decibels of full scale)

dbin_high = -9999999
dbin_low  =  9999999

# opl emulator, register data, and helper functions for converting
# between binary and float32[] config vector for AI training and inference

opl3 = OPL3()

# -----------------------------------------------------------------------------
# draw a single spectrum
# -----------------------------------------------------------------------------
def plotSpectrum(spec, gcolor=(255,255,255)):
  global screen, dbin_high, dbin_low
  global MIN_DB,MAX_DB
  global WW,HH  

  ll = len(spec)
  for i in range(0,ll-1):
    x0=int(i*WW/ll)
    x1=int((i+1)*WW/ll)    
    v0=int((spec[i]-MIN_DB)*HH/(MAX_DB-MIN_DB))
    v1=int((spec[i+1]-MIN_DB)*HH/(MAX_DB-MIN_DB))
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
    y0=HH-1-v0
    y1=HH-1-v1
    pygame.draw.line(screen, gcolor, (x0,y0),(x1,y1))
  # show high-water mark of spectral bin dBFS
  ymax = HH-1-int((dbin_high-MIN_DB)*HH/(MAX_DB-MIN_DB))
  pygame.draw.line(screen, (255,0,0), (0,ymax),(WW-1,ymax))
  # and quiet bin cutoff at -115 dBFS
  ymax = HH-1-int((-115-MIN_DB)*HH/(MAX_DB-MIN_DB))
  pygame.draw.line(screen, (0,0,255), (0,ymax),(WW-1,ymax))
# -----------------------------------------------------------------------------
# draw a mono 16-bit sound waveform in yellow
# -----------------------------------------------------------------------------
def plotWaveform(wave):
  global screen,WW,HH
  ll = len(wave)
  for i in range(0,ll-1):
    s0 = int(wave[i])    
    s1 = int(wave[i+1])
    x0=int(i*WW/ll)
    x1=int((i+1)*WW/ll)    
    y0=int((s0+32768)*HH/65536)
    y1=int((s1+32768)*HH/65536)
    pygame.draw.line(screen, (255,255,0), (x0,y0),(x1,y1))  

# select 2-op/4-op. channel hard coded to zero for now

def setComplexity(v, do_4op):
  if do_4op:
    v[0] = 1.0   # chan 0: 4-op
  else:
    v[0] = 0.0  # channel 0: 2-op

  # see what elements are permutable per each channel
  permidxs,keyons,lvls,freqs = opl3.vecGetPermutableIndxs(v)
  permidxs = permidxs[0]   # permutable vector elems for selected chan(s)
  lvls = lvls[0]           # operator attenuation levels
  freqs = freqs[0]         # channel frequency setting(s)
  keyons = keyons[0]       # channel key-on setting(s)
  for ko in keyons:               # ensure key-on is set for selected channel(s)
    v[ko] = 1.0

  return v, permidxs, lvls, freqs
# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------
def main():
  global opl3,WW,HH,REINIT_PERIOD, DISPLAY_INTERVAL
  # init config vector
  j=0
  opl_vec = opl3.rfToV(opl3.initRegFile())
  vl = len(opl_vec)

  do_4op = False
  opl_vec, permidxs, lvls, freqs = setComplexity(opl_vec, do_4op)
  print('  permutables: ',permidxs)
  print('op atten. lvs: ',lvls)      
  print('  chan. freqs: ',freqs)

  print(f'\nInitial float32[{len(opl_vec)}] Config. Vector:')
  opl3.showVector(opl_vec)

  # check for training file
  try:
    ssize = os.path.getsize(SPECT_FILEPATH)
    rsize = os.path.getsize(REGS_FILEPATH)
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
    sfile = open(SPECT_FILEPATH, 'wb')         # list of (reg_config, spectrum[2046)
    rfile = open(REGS_FILEPATH, 'wb')  # list of corresponding 128-byte squished-spectra
  else:
    print('Found existing training files. They will be expanded.')
    sfile = open(SPECT_FILEPATH, 'ab')         # list of (reg_config, spectrum[2046)
    rfile = open(REGS_FILEPATH, 'ab')  # list of corresponding 128-byte squished-spectra

  # todo: we should somehow ensure the two preexisting files 
  # are in-sync all the way til the end before keeping them 
  # before expanding them further or committing to hours-long 
  # AI training.
  
  fsz                   = ssize       # keep a running tally of the spectrum file, the larger of the two sets.
  lastszmb              = -1          # for filesize status updates
  iters                 = ssize//2048 # start the iteration count higher if some are on disk
  freq_sweep_direction  = 1           # we sweep freq back and forth while permuting things
  perms_this_mode       = 0           # counts iterations before OPL3 clean slate (theres no permutation modes yet)

  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:  # Check for pygame app close
        rfile.close()
        sfile.close()
        return 
    '''
    if random.random()<0.98:
      x = random.choice(freqs)
      o = opl_vec[x]
      o += freq_sweep_direction * 0.0005
      if o>1.0:
        o=1.0
        freq_sweep_direction = -1
      elif o<0.0:
        o=0.0
        freq_sweep_direction = 1
      opl_vec[x] = o
    else:  
    '''
    x = random.choice(permidxs)
    if x in lvls:
      opl_vec[x]= opl3.randomAtten()
    else:
      opl_vec[x]=random.random()

    rf = opl3.vToRf(opl_vec)

    # render a 4096-point waveform and its spectrum
    waveform, tspect = opl3.renderOPLFrame(rf)
    
    if tspect is not None:    # If noise was audiable, 
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
          pygame.draw.rect(screen,(0,0,0),(0,0,WW,HH))
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
            l = f'batches: {iters//32:12d} ({fszmb:d} MB), samp min/max:({opl3.wave_low},{opl3.wave_high}), bin min/max:({dbin_low},{dbin_high}), cur_lvls:['
            for x in lvls:
              l += f'{opl_vec[x]:6.4f},'
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
        opl_vec = opl3.rfToV(opl3.initRegFile())
        perms_this_mode = 0
        do_4op = not do_4op
        if do_4op:
          REINIT_PERIOD = random.randint(10000,50000)
        else:
          REINIT_PERIOD = random.randint(10000,50000)
        opl_vec, permidxs, lvls, freqs = setComplexity(opl_vec, do_4op)
        print('  permutables: ',permidxs)
        print('op atten. lvs: ',lvls)      
        print('  chan. freqs: ',freqs)



###############################################################################
# ENTRYPOINT
###############################################################################
if __name__ == '__main__':
  main()
###############################################################################
# EOF
###############################################################################
