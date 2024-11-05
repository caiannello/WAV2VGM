# -----------------------------------------------------------------------------
#
# RoboMimic
#
# Analyzes an input sound and tries to recreate it using very a very basic 
# additive synthesizer. 
#
# The intent is to use it to help build a minimal speech synthesizer for
# attiny85, but musical applications might be interesting too.
#
# Craig Iannello 20201225
#
# -----------------------------------------------------------------------------
import numpy as np
import math
import time
from scipy import signal
from scipy.io import wavfile
from copy import deepcopy
import pygame
# -----------------------------------------------------------------------------
tstart = time.time()
def lg(s):
  print('{:8.3f} {:s}'.format(time.time()-tstart,s))
# -----------------------------------------------------------------------------
# represents a mono sound with members for waveform and spectrogram
# -----------------------------------------------------------------------------
class spect:  

  wav_filename = None
  sample_rate = None
  num_chans = None
  samples = None
  num_samps = None

  frequencies = None    # y axis list 
  times = None          # x axis list
  spectrogram = None


  lvl_map = None
  red_map = None
  grn_map = None
  blu_map = None
  rgb_map = None

  spectrum_colors = [
    #   dBFM      Red   Grn   Blu 
    [ -115.0,  (  0,    0,    0   )],   # black
    [ -75.0,   (  0,    0,    255 )],   # blue
    [ -50.0,   (  255,  0,    0   )],   # red
    [ -25.0,   (  255,  255,  0   )],   # yellow
    [ -0.0,    (  255,  255,  255 )]    # white
  ]  
  # ---
  def __init__(self,wav_filename = None,sample_rate=44100,samples=None,nperseg=4096):
    self.nperseg = nperseg
    if not wav_filename is None:
      lg('loading wav {}.'.format(wav_filename))
      self.wav_filename = wav_filename
      self.sample_rate, self.samples = wavfile.read(wav_filename)
      if not len(self.samples):
        lg('  ABORT: No waveform in input.')        
        return
      st = type(self.samples[0])
      if (st is np.ndarray) or (self.sample_rate!=44100):
        self.num_channels = len(self.samples[0])
        lg('ABORT: Currently, we only accept Mono, 16-bit WAV files with samplerate 44.1 kHz.')
        exit()
      else:
        self.num_channels = 1
      self.num_samps = len(self.samples)      
      self.minsamp = self.samples.min()
      self.maxsamp = self.samples.max()
      lg(f'{self.sample_rate=} {self.num_channels=} {self.num_samps=} {self.minsamp=} {self.maxsamp=} ')
    else:
      self.sample_rate = sample_rate
      self.samples = samples
      if len(self.samples) and type(self.samples[0]) is list:
        self.num_channels = len(self.samples[0])
      else:
        self.num_channels = 1
      self.num_samps = len(self.samples)      
      self.minsamp = self.samples.min()
      self.maxsamp = self.samples.max()
    self.genSpect()
  # ---
  def genSpect(self):
    if not self.samples is None:
      lg('[+] gen spectrum ({} samps):'.format(self.num_samps))
      # make spectrogram from provided wav
      self.frequencies, self.times, self.spectrogram = signal.spectrogram(
          self.samples, self.sample_rate,
          window=signal.windows.blackmanharris(self.nperseg),
          nperseg=self.nperseg,
          noverlap=self.nperseg-int(self.nperseg/16),  # window step rate
          scaling='spectrum',
          mode='magnitude'
          )
      rawmin=self.spectrogram.min()
      rawmax=self.spectrogram.max()
      lg('   samples min/max {}/{}'.format(self.minsamp,self.maxsamp))      
      lg('raw  spect min/max {:1.2f}/{:.2f}'.format(rawmin,rawmax))

      # highest bin when full-scale 8 kHz sine wave
      absmax=18137
      if rawmax>absmax:
        absmax=rawmax

      # scale spectrogram to dBfS
      lg('  Converting spect bins to dBFS')
      fscale = lambda x: 20.0*np.log10(x/absmax)
      new_spect = fscale(self.spectrogram)
      self.spectrogram = np.rot90(new_spect,k=3)

      # calc some helpful statistics
      self.dur_secs = len(self.samples)/self.sample_rate
      self.minfreq=self.frequencies.min()
      self.maxfreq=self.frequencies.max()
      self.mintime=self.times.min()
      self.maxtime=self.times.max()
      self.minval=self.spectrogram.min()
      self.maxval=self.spectrogram.max()
      self.mincolidx=0
      self.maxcolidx=len(self.spectrogram[0])-1
      self.minrowidx=0
      self.maxrowidx=len(self.spectrogram/len(self.spectrogram[0]))-1      

      lg('dbfs spect min/max {:1.2f}/{:.2f}'.format(self.minval,self.maxval))
      lg(f'frequencies: {self.frequencies}')

      lg('  RGB Colorizing')

      # p4  white   255,  255,  255
      # p3  yellow  255,  255,  0
      # p2  red     255,  0,    0
      # p1  blue    0,    0,    255
      # p0  black   0,    0,    0

      # dBFS value per gradient keycolor      
      p0 = -115.0   # black
      p1 = -75.0    # blue
      p2 = -50.0    # red
      p3 = -25.0    # yellow
      p4 =  0.0     # white

      # clip spectrum bins below p0 to p0
      def fclip(x):
        if x<p0:
          return p0
        return x
      fclip_vec = np.vectorize(fclip)
      self.spectrogram = fclip_vec(self.spectrogram)
      self.minval=self.spectrogram.min()

      # convert dBFS values to gradient colors

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

      fred_vec = np.vectorize(fred)
      fgrn_vec = np.vectorize(fgrn)
      fblu_vec = np.vectorize(fblu)

      self.red_map = fred_vec(self.spectrogram)
      self.grn_map = fgrn_vec(self.spectrogram)
      self.blu_map = fblu_vec(self.spectrogram)
      self.rgb_map = np.dstack((self.red_map,self.grn_map,self.blu_map)).astype('int32') 
      self.surf = pygame.pixelcopy.make_surface(self.rgb_map)

      lg('spectrum recalc done. dur={}s mincol={} maxcol={} minrow={} maxrow={} minbin={} maxbin={} (dif={})'.format(
        self.dur_secs,
        self.mincolidx,
        self.maxcolidx,
        self.minrowidx,
        self.maxrowidx,
        self.minval,self.maxval,
        self.maxval-self.minval))
  # ---
  # ---


