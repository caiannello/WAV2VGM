#pragma once

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

namespace dsp
{

// Percent-only (0-100); dsp stays UI/stage-agnostic, so no stage text here.
using ProgressCallback = std::function<void(int percent)>;

// Runtime-configurable (see AppSettings) rather than compile-time
// constants -- set once at startup from the user's saved settings, before
// any spectrogram/FFT work begins, and treated as read-only from then on.
// consecutive frames overlap by kFftSize - kHopSize samples.
extern int kFftSize;
extern int kHopSize;

struct RGB
{
    unsigned char r, g, b;
};

// Standard 3-term Blackman window of length n.
std::vector<double> MakeBlackmanWindow(int n);

// In-place iterative radix-2 Cooley-Tukey FFT. data.size() must be a power of two.
void FftForwardRadix2(std::vector<std::complex<double>>& data);

// Windows and transforms one frame of `n` real samples (frame must already be
// exactly n samples, zero-padded by the caller if needed), returning per-bin
// dBFS for bins [0, n/2]. A full-scale, bin-centered sinusoid reads ~0 dBFS.
std::vector<double> ComputeFrameDbfs(const float* frame, int n, const std::vector<double>& window);

// Same computation as ComputeFrameDbfs, but writes into caller-supplied
// scratch/output buffers (resized as needed, never reallocated once
// they're the right size) instead of allocating fresh ones on every
// call -- for a hot loop that calls this once per spectrogram column, of
// which a several-minute recording can have hundreds of thousands (see
// AudioProject::GenerateSpectrogram), the repeated heap churn from the
// plain ComputeFrameDbfs is itself a meaningful chunk of the total cost.
// `complexScratch`/`outDbfs` are typically one per worker thread, reused
// across that thread's whole column range.
void ComputeFrameDbfsInto(const float* frame, int n, const std::vector<double>& window,
                           std::vector<std::complex<double>>& complexScratch, std::vector<double>& outDbfs);

// Convenience wrapper around ComputeFrameDbfs: grabs one kFftSize-sample,
// Blackman-windowed frame of `samples` centered on `timeSeconds` (zero-
// padded at the buffer's edges), and returns its per-bin dBFS. The
// single-FFT "spectrum at an instant" building block behind the OPL3
// analysis mode's single-frame fitting and original-vs-rendered
// comparison.
std::vector<double> ComputeSpectrumAtTime(const std::vector<float>& samples, int sampleRate, double timeSeconds);

// Same as ComputeSpectrumAtTime, but into caller-supplied scratch/output
// buffers -- see ComputeFrameDbfsInto's identical reasoning. Meant for a
// hot loop that calls this many times per worker thread (e.g. rebuilding
// the OPL3 mode's whole-recording original-vs-rendered comparison after
// every Start), reusing the same three buffers across that thread's
// whole frame range instead of paying three fresh allocations per call.
void ComputeSpectrumAtTimeInto(const std::vector<float>& samples, int sampleRate, double timeSeconds,
                                std::vector<float>& frameScratch,
                                std::vector<std::complex<double>>& complexScratch, std::vector<double>& outDbfs);

// Maps a dBFS value through the fixed 5-point heatmap gradient:
//   -115 -> black, -75 -> blue, -50 -> red, -25 -> yellow, 0 -> white
// Clamped outside [-115, 0], linearly interpolated between stops otherwise.
RGB DbfsToColor(double dbfs);

// Resamples mono samples from srcRate to targetRate. Identity if rates match.
// Applies an anti-aliasing/anti-imaging lowpass around the resample.
// `progress` (if set) is called ~100 times total, regardless of input size.
std::vector<float> ResampleTo(const std::vector<float>& in, int srcRate, int targetRate,
                               const ProgressCallback& progress = nullptr);

// Scales samples in place so the peak absolute value is 1.0 (no-op if silent).
void NormalizePeak(std::vector<float>& samples);

} // namespace dsp
