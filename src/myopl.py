# Adapted from MyOPL-Master: regression test
# adapted from DRO Trimmer by Laurence Dougal Myers.
import array
from enum import Enum
import struct
import typing
import pyopl


class myopl:
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

	def render_dro(self):
		# Reset (may not be required)
		for bank in range(2):
			for reg in range(0x100):
				self.write(bank, reg, 0x00)

		# Render all instructions
		for entry in self._dro:
			if entry[0] == DROInstructionType.DELAY_MS:
				_, delay_ms = entry
				self.render_ms(delay_ms)
				#print(delay_ms)
			else:
				_, bank, reg, val = entry
				#print(bank, reg, val)
				self.write(bank, reg, val)

		return self._output

	def write(self, bank: int, register: int, value: int) -> None:
		self._opl.writeReg(register | (bank << 2), value)
