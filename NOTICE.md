# Third-party components

## dbopl (OPL3 / YMF262 emulator)

`src/third_party/dbopl/` (`dbopl.h`, `dbopl.cpp`) is the OPL3 emulator
core from the [DOSBox](https://www.dosbox.com/) project, copyright (C)
2002-2020 The DOSBox Team, standalone-extracted by rofl0r at
<https://github.com/rofl0r/dbopl>. It is used, unmodified, to power the
"OPL3" analysis mode.

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
