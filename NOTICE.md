# Third-party components

## dbopl (OPL3 / YMF262 emulator)

`src/third_party/dbopl/` (`dbopl.h`, `dbopl.cpp`) is the OPL3 emulator
core from the [DOSBox](https://www.dosbox.com/) project, copyright (C)
2002-2020 The DOSBox Team, standalone-extracted by rofl0r at
<https://github.com/rofl0r/dbopl>. It is used, unmodified, to power the
FM voices of the OPL3, OPL2 and OPL4 targets in the "Chip" analysis mode.

**License: GNU General Public License version 2 or later (GPL2+)**, the
full text of which is in [`LICENSE-GPL2.txt`](LICENSE-GPL2.txt) at the
repo root. The copyright/license header is preserved as-is at the top of
both vendored files.

Because dbopl is GPL2+, and it is statically linked into the `WAV2VGM`
executable (not run out-of-process or dynamically loaded as a separable
component), **any build of WAV2VGM that includes the OPL3 analysis mode
is a combined work under GPL2+ as a whole.** If you distribute such a
build, you must comply with GPL2+ for the entire distributed binary, not
just the dbopl portion — this includes making corresponding source
available and preserving license notices. This does not affect purely
local/non-distributed use.

## YM2151 phase increment table

`src/chips/OpmPhaseIncRom.h` holds the YM2151 (OPM) phase increment ROM
table (768 values measured from the real chip). It was copied from MAME's
`ym2151.c` by Jarek Burczynski, by way of
[libvgm](https://github.com/ValleyBell/libvgm) (`emu/cores/ym2151.c`).
MAME's sound cores are available under BSD-3-Clause or GPL2+, which fits
the GPL2+ combined work described above.

The other chip targets (`src/chips/`) contain no third-party code. Their
register layouts, pitch formulas and VGM conventions were written using
libvgm's emulator cores and VGM player as a reference.
