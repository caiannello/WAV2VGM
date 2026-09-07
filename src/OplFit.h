#pragma once

#include "Dsp.h"
#include "OplPatch.h"

#include <vector>

// Fits OPL3 channel register settings to a target drawn from an input
// recording, and renders that fit back out through the real emulator for
// verification/display.
//
// Three workflows live here side by side:
//  - FitChannelsInTurn/RenderAllFrames/RenderAllFramesMix: the current
//    workflow driving SpectrogramView's Opl3 mode (Start/VGM export).
//    Peak-tracks the whole recording once (analysis::ComputePeakTrends,
//    the same mutual-nearest-neighbor trend linking Peak Trends mode
//    draws as connecting lines), ranks the resulting trends by energy,
//    and dedicates one channel to each of the top ones for that trend's
//    entire lifetime -- so a given trend is always represented by the
//    same channel, never reassigned frame to frame. The channel's timbre
//    itself is not searched at all -- it's fixed to a pure, additive-mode
//    sine carrier with the modulator fully silenced and disconnected
//    (see PureSineSetup in OplFit.cpp), deterministic and fast, matching
//    what the user's own earlier tool did and avoiding the faint
//    FM-sideband coloration an earlier, searched-timbre version of this
//    code introduced.
//  - FitAllFrames/RenderAllFrames/RenderAllFramesMix: an earlier per-
//    channel workflow that instead fits one channel's frequency/level
//    independently frame by frame (with only a local continuity nudge
//    toward the previous frame, and residual-based targeting against
//    already-added channels) -- no longer used by FitChannelsInTurn since
//    it has no persistent notion of "this channel represents this trend"
//    across frames, but still exercised directly by opl3_selftest and
//    kept working.
//  - FitSingleChannel/FitMultiChannel/RenderChannel/RenderMix: an earlier
//    whole-recording, multi-channel workflow (peak-track the whole file,
//    rank trends by energy, fit one channel per trend) that DOES search
//    FM modulator settings and operator waveforms (see SearchTimbre).
//    Still exercised by opl3_selftest and kept working, but not
//    currently wired into the UI.
namespace oplfit
{

// fidelity in [0,1]: how many of the analytical Tier-1 candidates get
// promoted to real-emulator Tier-2 verification (see OplFit.cpp) --
// higher costs more time for a better-verified timbre choice. Only
// meaningful to the legacy FM-search workflow (FitSingleChannel/
// FitMultiChannel) -- the current pure-sine workflow has nothing left to
// search, so doesn't use this at all.
constexpr double kMinFidelity = 0.0;
constexpr double kMaxFidelity = 1.0;
constexpr double kDefaultFidelity = 0.5;

// Default frame rate for the interactive Opl3 analysis mode -- much lower
// than Peak Trends' own default (see analysis::kDefaultFrameRateHz) since
// this sets both the size of the input segment considered at once (1/rate
// seconds) and how often OPL3 registers actually change in the render;
// range/units are shared with Peak Trends' own frame rate control
// (analysis::kMinFrameRateHz/kMaxFrameRateHz).
constexpr double kDefaultInteractiveFrameRateHz = 100.0;

// One frame's full channel-0 configuration -- both the timbre fields
// (see PureSineSetup -- fixed to the same pure additive-mode sine for
// every frame of a channel, not searched) and the per-frame frequency/
// level, which do vary frame to frame.
struct FrameFit
{
    opl::ChannelStaticSetup setup;
    opl::FramePatch frame;
};

struct AllFramesFit
{
    std::vector<FrameFit> frames; // frames[i] covers [i/frameRateHz, (i+1)/frameRateHz)
    double frameRateHz = 0.0;
};

// Fits one OPL3 channel independently to every frame of the whole
// recording -- each frame's frequency and carrier level are fit
// independently, with the channel's timbre fixed to the same pure
// additive-mode sine for every frame (see PureSineSetup), not searched.
// Frequency selection favors continuity with the previous frame's pitch
// (weighted
// by how loud the current frame is: quiet frames pick purely by
// prominence, since a pitch jump there is inaudible anyway; loud/
// sustained frames favor continuity strongly, to avoid octave-jumping
// between adjacent frames' peak picks) -- this is unrelated to timbre and
// still applies even though timbre itself is no longer searched.
//
// `previousMixRendered`, if given, is the actual rendered audio of every
// channel already added so far, mixed together (see RenderAllFramesMix)
// -- when present, each frame's search target becomes the RESIDUAL left
// after that render's own spectrum is subtracted from the original's,
// per bin, in the linear-magnitude domain (never negative). That's safe
// unlike audio-domain subtraction (which would need previousMixRendered
// to be phase-aligned with the original at every sample -- it isn't):
// magnitude subtraction doesn't care about phase, it's just "how much
// more energy is still needed here". Using the actual combined render of
// every previous channel, rather than summing each one's spectrum in
// isolation, also correctly accounts for any constructive/destructive
// interaction between those channels rather than double-counting their
// individual contributions. Frames the existing channels already explain
// well end up with a near-zero residual and are naturally left silent by
// the same loudness gate that handles genuinely quiet input. Omit (or
// pass empty) for the first channel, which then targets the recording
// directly.
//
// `previousChannels`, if given (normally the same channels
// previousMixRendered was rendered from), supplies each already-added
// channel's own exact frequency per frame -- used on top of the residual
// targeting above to hard-exclude candidates too close to an already-
// covered frequency (residual subtraction alone isn't perfectly clean:
// OPL's fnum/block quantization and 64-step TL granularity mean an
// already-covered peak's own spectral leakage can still look, bin for
// bin, like "the next loudest thing" in the residual) and to mildly
// prefer whichever remaining candidate is more diverse in frequency from
// what's already covered, on an otherwise-close tie.
AllFramesFit FitAllFrames(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                           const dsp::ProgressCallback& progress = nullptr,
                           const std::vector<float>& previousMixRendered = {},
                           const std::vector<AllFramesFit>& previousChannels = {});

// Renders every frame of `fit` back-to-back through one continuous
// OplChip (so oscillator phase and envelope state carry over naturally
// frame to frame, same as real hardware would), mono-downmixed, covering
// the fit's whole duration.
std::vector<float> RenderAllFrames(const AllFramesFit& fit, int sampleRate);

// Renders multiple AllFramesFit results together through one shared
// OplChip (fits[i] on real OPL3 channel i), the way they'd actually
// sound played back together -- the basis for "listen to the
// approximation so far" after each added channel, and for computing the
// next channel's residual target (see FitAllFrames). Fits are typically
// built from repeated FitAllFrames calls against the same recording and
// so normally share a frame rate/count, but each channel is sampled by
// its own frame rate/count independently (looked up by absolute time
// against a shared render tick derived from fits.front()), so this still
// behaves reasonably if they don't exactly match.
std::vector<float> RenderAllFramesMix(const std::vector<AllFramesFit>& fits, int sampleRate);

constexpr int kMinMaxChannels = 1;
constexpr int kMaxMaxChannels = opl::kMaxChannels;
constexpr int kDefaultMaxChannels = kMaxMaxChannels; // use every real oscillator by default

struct MultiChannelSineFit
{
    std::vector<AllFramesFit> channels;
    std::vector<float> mixRendered; // every channel's own solo render, summed -- see FitChannelsInTurn
};

// This is what the Opl3 analysis mode's Start button runs. Peak-tracks
// the whole recording once (analysis::ComputePeakTrends -- the same
// mutual-nearest-neighbor trend linking Peak Trends mode itself draws as
// connecting lines), buckets the resulting trends into log-spaced
// frequency bands spanning whatever range the recording's own content
// actually occupies, and processes them in round-robin order across
// bands (loudest trend within a band first) rather than one flat sort by
// raw energy -- pure loudness ranking systematically favors low/mid
// frequencies (a voice's fundamental and its lower harmonics simply
// carry far more energy than a sibilant or a high formant ever will), so
// spending the channel budget strictly loudest-first made the result
// sound like the input had been run through a low-pass filter. Round-
// robin instead guarantees every band gets a channel before any band
// gets a second one, letting frequency range compete on equal footing
// with amplitude for which regions get represented at all, while
// amplitude still decides ordering within any one region.
//
// Each trend in that order is then greedily allocated like a voice: it's
// placed on the first already-open channel whose own material doesn't
// overlap it in time (a real voice can play many different, time-
// disjoint notes over a recording's life), opening a fresh channel only
// when no existing one has room and maxChannels (clamped to
// [kMinMaxChannels, kMaxMaxChannels]) hasn't been reached yet. A trend
// that finds no room anywhere is simply left out -- there are only
// maxChannels real oscillators to share. Every trend that IS placed gets
// ITS channel's frequency/level for its entire lifetime and never anyone
// else's, so a given trend is always represented by the same channel --
// never reassigned, never abandoned mid-flight for a louder one -- while
// trends that don't overlap in time reuse a channel rather than each
// claiming their own. That reuse matters on its own, independent of the
// frequency banding: a long feature (e.g. a speaker's fundamental pitch)
// is often split into several separate PeakTrend objects by pauses/
// consonants, and packing those time-disjoint fragments onto one shared
// channel frees the rest of the budget for genuinely simultaneous
// content instead of redundant fragments of the same one. Trends under a
// small fraction of their own band's loudest trend are never even
// attempted -- a threshold scoped per band rather than to the single
// loudest trend recording-wide, so a genuinely quiet but real high-
// frequency band isn't excluded just for being quieter than an unrelated
// low-frequency one.
//
// Builds the running mix by rendering each channel SOLO (RenderAllFrames)
// and summing it in, rather than re-simulating every channel together
// from scratch via RenderAllFramesMix -- every channel here uses the same
// additive-mode, fully-disconnected-modulator timbre (PureSineSetup), so
// each one's output is completely independent of every other's: OPL3's
// real combined output genuinely *is* the sample-by-sample sum of each
// active channel's own signal in that case, not merely close to it, so
// this is exact, not an approximation.
MultiChannelSineFit FitChannelsInTurn(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                       int maxChannels = kDefaultMaxChannels,
                                       const dsp::ProgressCallback& progress = nullptr);

// An alternate, much simpler workflow modeled directly on an earlier
// (Python) implementation of this same idea, kept side by side with
// FitChannelsInTurn above for direct A/B comparison rather than as a
// replacement -- on real speech, this one measurably sounds and looks
// denser/richer despite having no persistent-identity mechanism at all.
//
// No trend tracking, no residual computation, no frequency banding: for
// every output frame independently, this takes that frame's own RAW
// spectrum's local maxima (least-distinctness-filtered so a real
// partial's multi-bin mainlobe doesn't register more than once -- see
// FindDistinctPeaks), sorts them loudest first, and assigns channel =
// rank -- channel 0 gets whatever's loudest THIS instant, channel 1 the
// 2nd loudest, and so on, up to maxChannels (clamped to
// [kMinMaxChannels, kMaxMaxChannels]; the result always has exactly this
// many channels, silence-padded). A candidate below the channel's own
// audibility floor (see FitChannelsInTurn's identical reasoning) is
// skipped, same as every candidate quieter than it that frame.
//
// This never drops brief/transient content the way trend-based fitting
// can (a PeakTrend needs >=2 linked detections across frames to exist at
// all, and can lose FitChannelsInTurn's whole-lifetime greedy-packing
// race and be excluded outright) -- every frame gets its own full,
// independent snapshot of local polyphony. The tradeoff is genuine:
// nothing here stops channel 5 from representing a completely different
// partial next frame than it did this one, so there is no guarantee
// against the kind of channel-identity swap FitChannelsInTurn was built
// to prevent. In practice, on real voiced speech, relative loudness
// ranking between simultaneous harmonics doesn't reorder often enough
// for that to be very audible, and the density this buys back apparently
// matters more than the continuity given up for it -- hence keeping both
// for direct comparison rather than picking a winner outright.
//
// Builds the running mix the same way FitChannelsInTurn does (each
// channel rendered solo via RenderAllFrames and summed, exact for this
// shared additive-mode timbre -- see PureSineSetup).
MultiChannelSineFit FitChannelsPerFrame(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                         int maxChannels = kDefaultMaxChannels,
                                         const dsp::ProgressCallback& progress = nullptr);

struct FitResult
{
    opl::ChannelStaticSetup setup;
    std::vector<opl::FramePatch> frames; // one per output frame, contiguous
    double startSeconds = 0.0;           // when `frames[0]` should sound
    double frameRateHz = 0.0;

    // Total (linear-magnitude) energy of the target trend this fit was
    // built from -- bookkeeping for FitMultiChannel's early-stop check,
    // not used by RenderChannel/RenderMix.
    double targetEnergy = 0.0;
};

// Picks the single PeakTrend (via analysis::ComputePeakTrends) with the
// greatest total energy as the target, chooses one static 2-operator
// timbre for it (Tier 1: analytical FM-sideband cost model over a small
// grid of modulator multiplier/level candidates; Tier 2: the top
// `fidelity`-scaled slice of those re-scored by actually rendering them
// through OplChip and comparing real spectra), then maps the target's
// per-frame frequency/amplitude onto OPL3's achievable fnum/block grid and
// a carrier total level calibrated against one real probe render. Returns
// an empty `frames` if no trend was found.
FitResult FitSingleChannel(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                            double fidelity = kDefaultFidelity,
                            const dsp::ProgressCallback& progress = nullptr);

// Renders `fit` (as produced by FitSingleChannel) through a fresh OplChip,
// mono-downmixed, on the same absolute timebase as the source recording
// (silence before fit.startSeconds) so it can be directly re-analyzed or
// compared against the original.
std::vector<float> RenderChannel(const FitResult& fit, int sampleRate);

struct MultiChannelResult
{
    // channels[i] is meant to occupy real OPL3 channel i in a combined
    // render/export (see RenderMix) -- order matters, unlike
    // PeakTrendsResult::trends.
    std::vector<FitResult> channels;
};

// Peak-tracks the whole recording once (analysis::ComputePeakTrends),
// ranks the resulting trends by total energy, and fits one OPL3 channel
// to each of the top ones in turn -- trends are already time/frequency-
// disjoint by construction (that's why the tracker couldn't link them
// into a single trend to begin with), so this naturally spreads channels
// across the recording without needing risky audio-domain residual
// subtraction (an earlier version tried subtracting each rendered
// channel from a working residual buffer; that's only valid if the
// synthesized waveform is phase-aligned with the original at every
// sample, which OPL's arbitrary key-on phase isn't, so it didn't
// actually cancel anything). Stops when a trend's energy drops under a
// small fraction of the first (most significant) channel's, or
// maxChannels (clamped to [kMinMaxChannels, kMaxMaxChannels]) is
// reached, whichever comes first.
MultiChannelResult FitMultiChannel(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                    double fidelity = kDefaultFidelity, int maxChannels = kDefaultMaxChannels,
                                    const dsp::ProgressCallback& progress = nullptr);

// Renders every channel in `channels` together through one shared OplChip
// (channels[i] on real OPL3 channel i), the way they'd actually sound
// played back on hardware or exported to VGM -- each channel is silenced
// once its own frame span ends, so it doesn't hold its last note forever.
std::vector<float> RenderMix(const std::vector<FitResult>& channels, int sampleRate);

} // namespace oplfit
