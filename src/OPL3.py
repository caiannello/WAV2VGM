
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

# OPL3 Audio emulation from DOSBox ---
# Adapted from opl_emu-Master: regression test
# adapted from DRO Trimmer by Laurence Dougal Myers.
class OPL3:
  reg_settings = {}
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

  def render_ms(self, length_ms: int) -> None:
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

  def do_init(self):
    # Reset (may not be required)
    for bank in range(2):
      for reg in range(0x100):
        self.write(bank, reg, 0x00)

  def write(self, bank: int, register: int, value: int) -> None:
    self.reg_settings[(bank,register)] = value
    #self._opl.writeReg(register | (bank << 2), value)
    self._opl.writeReg(bank<<8|register, value)

  def writeregfile(self, rf) -> None:
    for i,v in enumerate(rf):
      self._opl.writeReg( i, v)

###############################################################################
# ENTRYPOINT
###############################################################################
#if __name__ == '__main__':
#  main()
###############################################################################
# EOF
###############################################################################
