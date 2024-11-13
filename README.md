## Introduction
--------------------------------------------------------------------------------
```
WELCOME TO THE WAV2VGM PROJECT !!!

This (messy) utility for Python 3 takes an input WAV file and outputs it as
an VGM file that uses OPL3 synthesis to recreate the original sound.  

Examples are provided demonstrating famous speeches as well as a 
conversion of a classic arcade tune from YM2151 (OPM) to YMF262M! (OPL3)

The original method works surprisingly well. It converts the input sound to 
a spectrogram, detects peak frequencies, and makes a set of OPL3 
instructions to play these peaks as a sum of pure sine waves of varying
amplitudes.

Work is underway to leverage neural networks to produce more sophisticated
synthesizer configurations for a given input: Code is provided to produce 
training sets (random OPL3 register settings and resulting spectra), and 
to train the AI model from these data.

Directory Content:
  
  doc -                 Currently just some screenshots
  input -               Some same wave files to use as input
  output -              Those waves, converted to OPL3, as VGM and MP3 file
  training_sets -       Large files for training AI models
  models -              AI Models created using the above data
  src -                 Misc. source code, training set generator, and
                        trainer for the convolutional neural-network.

  WAV2VGM.py -          Main executable

```
## Sample Output

The sound files from the input/ directory have been converted and are provided in the output/ directory as VGM files and MP3 files. These can be played using WINAMP, VGMPLAY, a REAL OPL3 CHIP, etc.:

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

This utility requires Python 3.8+ to be installed, as well as the package manager, python3-pip. There are additional dependencies, listed in requirements.txt, which can be be installed from the command-line in Windows, Linux, or Mac by doing something similar to the following:

### Windows:
```
pip install numpy pygame scipy PyOPL torch
```

### Linux, Mac:
```
sudo pip install numpy pygame scipy PyOPL torch
```

## Advanced Installation

If you want to experiment with training the AI stuff yourself, there are more appropriate versions of torch (pytorch) to install, for example, ones that leverage your system's GPU. Your steps will vary, but in my case, it was from a Windows admin command prompt:

```
pip install numpy pygame scipy matplotlib PyOPL torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


## Usage

Input files must be 16-bit WAV files, MONO, with a 44.1 kHz samplerate. 

Examples of starting the utility from the command-line:

### Windows:
```
cd WAV2VGM
python WAV2VGM.py "input\JFK Inaguration.wav"
```

### Linux, Mac:
```
cd WAV2VGM
./WAV2VGM.py "input/JFK Inaguration.wav"
```

## Keyboard Commands

If all is well, after some time, a window should appear which displays a spectrogram of 
the input sound. From here, a few different commands can be issued by pressing the 
following (lowercase) keys:

```
  p - Play original sound (Currently, program is unresponsive during playback)

  f - Analyzes the sound simply, and outputs a VGM file to the 'output/'
      directory. Also, plays a rendition of the output. (The program will
      be unresponsive during playback.)

      (Output file can be played using WINAMP, VGMPLAY, or a real YMF262M!)

  a - Analyze the spectrum and show frequency peaks.

  Or close the window to quit.

  EXPERIMENTAL STUFF -   (WORK-IN-PROGRESS)

  b - Tries to make a VGM using the convolutional neural-network model at
      'models/torch_model.pth'

  g - Tries to make a VGM using the slow-as-heck genetic algorithm! 

  Clicking on the displayed spectrogram sets some cursors, but these don't yet 
  do anything, and might even crash the app! (Still kinda fun to play with.)

```

## Notes

  - The spectrogram is made by slicing the input wave into 4096 byte samples, 
    with 32-samples of overlap per slice. Each spectrum is 2048-bins spanning a 
    frequency range of 0Hz to 22050 Hz, at about 10.8 Hz per bin. The frequency
    range shown on screen scaled up by three though, from 0Hz (bottom) to
    7350 Hz (top).
    
There is much room for improvement in this project:

  - The code is quite disorganized and slow, and the user interface is sparse
    and unintuitive. 

  - The input file format requirements are too strict. Utility should be able
    to load files of different types, with different sample rates and channel 
    counts.

  - There should be some settings to tweak to affect the conversion.

  - The output files are larger than they really need to be. (Redundancy in
    the instructions sent to the OPL3, e.g. the same frequency being set 
    repeatedly, even if unchanged)

  - It would be super easy to add OPL2 support, since it's nearly the same
    except with less channels. (Many of the synthesizers supported by the 
    VGM file format would be a good fit with this project and should be added.)

  - The quality of the output could be HUGELY BETTER: The re-synthesis method
    currently only contains pure sine waves (1-operator, basically), rather
    than any of the more advanced features offered in OPL3 such as 2-op, 4-op,
    percussion modes, waveforms besides sine, volume envelopes, vibrato, 
    tremolo, etc.  See below.

  - Work is underway to try to leverage more advanced synthesizer capabilities:

      - The first attempt was a super-slow genetic algorithm that generates
        some interesting results. 

      - We are also trying to build a neural network to do it! See the READMEs
        in 'model/' and 'training_sets/' for info about the NN stuff.

