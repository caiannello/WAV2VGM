#pragma once

#include <cstdint>
#include <functional>

// Thin wrapper around the vendored DBOPL::Handler (src/third_party/dbopl),
// DOSBox's OPL3 (YMF262) emulator core. See src/third_party/dbopl/NOTICE
// (and the repo-root NOTICE.md) for licensing -- dbopl is GPL2+.
class OplChip
{
public:
    explicit OplChip(int sampleRate);
    ~OplChip();
    OplChip(const OplChip&) = delete;
    OplChip& operator=(const OplChip&) = delete;

    // bank 0 = the primary OPL2-compatible register set (0x00-0xF5);
    // bank 1 = OPL3's secondary set (registers 0x100-0x1F5, including
    // 0x105 which enables OPL3/"new" mode -- write that first). This
    // mirrors how VGM files themselves refer to "port 0"/"port 1" writes
    // (commands 0x5E/0x5F), which is why the same 0/1 numbering is used
    // here even though dbopl's own WriteAddr() expects 0/2 internally.
    void WriteReg(int bank, uint8_t reg, uint8_t value);

    // Optional hook, called with the exact (bank, reg, value) of every
    // WriteReg just before it's forwarded to the real emulator -- lets a
    // caller capture the precise register sequence a render used (e.g.
    // for VGM export) without duplicating OplPatch.h's register-
    // generation logic. Pass an empty std::function to clear it.
    void SetWriteObserver(std::function<void(int bank, uint8_t reg, uint8_t value)> observer);

    // Renders `count` stereo sample pairs (interleaved L, R) into
    // outInterleaved (must hold at least count*2 floats), scaled to
    // [-1, 1]. Requires OPL3 mode to already be enabled (bank 1, reg
    // 0x05, value with bit 0 set) for stereo output; otherwise dbopl
    // generates mono and only the left channel of each pair is filled.
    void GenerateStereo(float* outInterleaved, int count);

private:
    // Opaque: avoids pulling dbopl.h's macro-heavy, C-style declarations
    // into every translation unit that just wants to drive the chip.
    struct Impl;
    Impl* m_impl;
    std::function<void(int, uint8_t, uint8_t)> m_writeObserver;
};

// freqHz = fnum * OplBaseRateHz / 2^(20 - block), the standard OPL
// F-Number formula -- OplBaseRateHz (14318180/288 = ~49716.03 Hz) is
// confirmed against dbopl's own OPLRATE constant, not just textbook
// recollection. block is in [0,7], fnum in [0,1023] (10 bits); both
// clamped by these helpers. Higher block roughly halves fnum for the
// same frequency, so HzToFnumBlock picks the smallest block that still
// keeps fnum within range, for the best available frequency resolution.
constexpr double kOplBaseRateHz = 14318180.0 / 288.0;

// The real hardware ceiling: fnum=1023 at block=7 (the largest block,
// giving fnum the widest possible per-step frequency range) is the
// highest frequency any fnum/block pair can represent at all -- computed
// via the exact same formula FnumBlockToHz uses, not a separately-
// maintained magic number. HzToFnumBlock clamps up to this; callers that
// need to know a peak is fundamentally unplayable *before* even trying
// to fit it (see oplfit::FitChannelsInTurn/FitChannelsPerFrame) should
// compare against this directly rather than attempt-then-clamp, since
// clamping alone would collapse every above-ceiling peak onto this exact
// same frequency -- many real, distinct high-frequency partials all
// landing on one oscillator's worth of output, audible as a single loud,
// artificial tone at the ceiling rather than the (unplayable) content
// they were actually approximating.
constexpr double kOplMaxFreqHz = kOplBaseRateHz * 1023.0 / 8192.0; // 2^(20-7) = 8192

struct OplFnumBlock
{
    int fnum;  // 0-1023
    int block; // 0-7
};

OplFnumBlock HzToFnumBlock(double freqHz);
double FnumBlockToHz(int fnum, int block);
