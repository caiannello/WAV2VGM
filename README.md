## Introduction
--------------------------------------------------------------------------------
```
WELCOME TO THE WAV2VGM PROJECT !!!

This (messy) utility for Python 3 takes an input WAV file and outputs it as
an VGM file that uses OPL3 synthesis to recreate the original sound.  

Examples are provided demonstrating famous speeches as well as a 
conversion of a classic arcade tune from YM2151 (OPM) to YMF262M! (OPL3)

The method used is to convert the input sound to a spectrogram, then 
detect peak frequencies on the spectrogram and make a set of OPL3 
instructions to play these peaks as a sum of pure sine waves of varying
amplitudes.

Directory Content:
  
  doc -                 Currently just some screenshots
  input -               Some same wave files to use as input
  output -              Those waves, converted to OPL3, as VGM and MP3 file
  repertoire -          WIP crazy experimental stuff
  src -                 Project source code
  WAV2VGM.py -          Main executable

```
## Sample Output

The sound files from the input/ directory have been converted and are provided in the output/
directory as VGM files and MP3 files. These can be played using WINAMP, VGMPLAY, a REAL OPL3 CHIP, etc.:

[JFK Inaguration Clip](https://github.com/caiannello/WAV2VGM/raw/refs/heads/main/output/JFK%20Inaguration.mp3)

[HAL 9000 - Human Error](https://github.com/caiannello/WAV2VGM/raw/refs/heads/main/output/HAL%209000%20-%20Human%20Error.mp3)

[Ghouls and Ghosts - The Village of Decay](https://github.com/caiannello/WAV2VGM/raw/refs/heads/main/output/Ghouls%20and%20Ghosts%20-%20The%20Village%20Of%20Decay.mp3)

[Portal - Still Alive](https://github.com/caiannello/WAV2VGM/raw/refs/heads/main/output/Portal-Still%20Alive.mp3)

## Gallery

Spectrogram view of a song: "Ghouls and Ghosts - The Village of Decay" 
![gg](https://raw.githubusercontent.com/caiannello/WAV2VGM/main/doc/WAV2VGM%20-%20Spectrogram%20-%20Ghouls.png)

Peak Detection view of a speech sample
![gg](https://raw.githubusercontent.com/caiannello/WAV2VGM/main/doc/peak_detect_jfk.png)

## Installation

This utility requires Python 3.8+ to be installed, as well as the package manager, python3-pip.

There are additional dependencies, listed in requirements.txt, which can be be installed 
from the command-line in Windows, Linux, or Mac by doing something similar to the following:

Windows:
```
pip install numpy pygame scipy PyOPL
```

Linux, Mac:
```
sudo pip install numpy pygame scipy PyOPL
```

## Usage

Input files must be 16-bit WAV files, MONO, with a 44.1 kHz samplerate. 

Examples of starting the utility from the command-line:

Windows:
```
cd WAV2VGM
python WAV2VGM.py "input\JFK Inaguration.wav"
```

Linux, Mac:
```
cd WAV2VGM
./WAV2VGM.py "input/JFK Inaguration.wav"
```

## Keyboard Commands

If all is well, after some time, a window should appear which displays a spectrogram of 
the input sound. From here, a few different commands can be issued by pressing the 
following (lowercase) keys:

```
  p - Play original sound (currently, program is unresponsive during playback)
  f - Analyze the sound and outputs a VGM file to the output/ directory.
      The file can be played using WINAMP, VGMPLAY, or even a real YMF262M (OPL3) chip!
      It will also play a rendition of the output. (Again, the program will be unresponsive
      during playback.)
  a - Analyze the spectrum and show frequency peaks.

  Or close the window to quit.

  EXPERIMENTAL STUFF

  b - Brute-forces OPL config per each frame of the spectrugram. THIS TAKES HOURS, and the
      output files aren't yet in VGM format! Also, this experiment requires a "training set"
      to be generated beforehand, using 'src/make_training_set.py', which also takes hours!
      
      The usefulness of the training set for this purpose is dubious as best, since the
      the matches get further refined using a genetic algorithm, which is always fun, so
      maybe can just trart with random-everything and skip the search step before doing
      the genetic stuff.

      TODO: Use training set for its actual intended purpose, which is to train an AI!

```
  Clicking on the displayed spectrogram sets some cursors, but these don't yet 
  do anything useful and might even crash the app, but kinda fun to play with.

## Notes

There is much room for improvement in this project. 

  - The code is quite disorganized and slow, and the user interface is sparse and unintuitive. 

  - The input file format requirements are too strict. Utility should be able to load
    files of different types, with different sample rates and channel counts.

  - There should be some settings to tweak to affect the conversion.

  - The output files are larger than they really need to be. (Redundancy in the instructions
    sent to the OPL3, e.g. the same frequency being set repeatedly, even if unchanged)

  - The quality of the output could be HUGELY BETTER: The resynthesis method currently 
    only contains pure sine waves (1-operator, basically), rather than any of the advanced
    features offered in OPL3 such as 2-op, 4-op, percussion modes, waveforms besides sine,
    volume envelopes, vibrato, tremolo, etc.

  - It would be super easy to add OPL2 support, since it's nearly the same except
    with less channels. (Many of the synthesizers supported by the VGM file format 
    would be a good fit with this project and should be added.)

  - I want to someday make a super slow version that iteratively brute-forces different 
    opl register combos, 2-op, 4-op, etc, to try to better approximate the input at
    each frame. (Something I can brood over for hours.. like genetic annealing, or 
    training a GAN or something... MUHAHA!)

