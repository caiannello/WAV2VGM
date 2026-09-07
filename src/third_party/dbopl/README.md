dbopl OPL3 emulator
===================

this is the so-called dbopl opl3 emulator, ripped from DOSBOX svn rev 4391
(November 2020), with full commit history and some tiny touch-ups to make
it compile standalone. as of the time of this writing, this is the default
OPL emulator used by DOSBOX.

Usage:
======

drop files into your project, include dbopl.h, and use in your code:

    // initialize the chip(s). samplerate is typicallly 44100.
    DBOPL::Handler chip = DBOPL::Handler();

    // write a value into a register.
    chip.WriteReg(reg, val);

    // get a sample from the chip.
    int32_t sample_tmp;
    int16_t sample;
    chip.Generate(&tmp, 1);
    sample = sample_tmp;

then compile it and link dbopl.cpp to your code.
that's pretty much all there is to it.

check https://github.com/digital-sound-antiques/emu2413/blob/master/sample2413.c
for a good example of what and how to feed into the chip, and when to read
samples (and how many) from it.
although the example uses a different chip emulator, the strategy is identical.


License
=======
unfortunately, the code is licensed as GPL2+.

TODO
====
convert to a language that doesn't suck, i.e. C.
also it would be nice to poke the original authors whether they could relicense
the code as MIT.
