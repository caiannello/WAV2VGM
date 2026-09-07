#include "OplChip.h"

#include "third_party/dbopl/dbopl.h"

#include <algorithm>
#include <cmath>
#include <vector>

struct OplChip::Impl
{
    DBOPL::Handler handler;
};

OplChip::OplChip(int sampleRate)
    : m_impl(new Impl)
{
    m_impl->handler.Init(static_cast<Bitu>(sampleRate));
}

OplChip::~OplChip()
{
    delete m_impl;
}

void OplChip::WriteReg(int bank, uint8_t reg, uint8_t value)
{
    if (m_writeObserver)
        m_writeObserver(bank, reg, value);

    // dbopl's own port numbering follows real AdLib/OPL3 I/O port
    // conventions (address ports 0 and 2, data ports 1 and 3 handled
    // internally by the two-call protocol below) -- confirmed by reading
    // Chip::WriteAddr in dbopl.cpp, which only recognizes port&3 == 0 or
    // 2. That's unrelated to VGM's own "port 0"/"port 1" register-bank
    // terminology (commands 0x5E/0x5F) that this wrapper's `bank`
    // parameter mirrors, so bank 1 maps to dbopl port 2, not port 1.
    const Bit32u dboplPort = (bank == 0) ? 0u : 2u;
    const Bit32u resolvedReg = m_impl->handler.WriteAddr(dboplPort, reg);
    m_impl->handler.WriteReg(resolvedReg, value);
}

void OplChip::SetWriteObserver(std::function<void(int, uint8_t, uint8_t)> observer)
{
    m_writeObserver = std::move(observer);
}

void OplChip::GenerateStereo(float* outInterleaved, int count)
{
    if (count <= 0)
        return;
    std::vector<Bit32s> buf(static_cast<size_t>(count) * 2);
    m_impl->handler.Generate(buf.data(), static_cast<Bitu>(count));
    // dbopl's accumulator is 16-bit-PCM-equivalent per active channel
    // (each operator's table lookup is scaled to roughly a Bit16s
    // range); with several loud channels summed it can exceed that, same
    // as any digital mixer -- clamped here rather than left to wrap.
    for (int i = 0; i < count * 2; ++i)
        outInterleaved[i] = std::clamp(buf[static_cast<size_t>(i)] / 32768.0f, -1.0f, 1.0f);
}

OplFnumBlock HzToFnumBlock(double freqHz)
{
    // freqHz = fnum * kOplBaseRateHz / 2^(20-block)  =>  fnum = freqHz *
    // 2^(20-block) / kOplBaseRateHz. fnum shrinks as block grows, so pick
    // the *smallest* block that still keeps fnum within its 10-bit range
    // -- that's the largest (finest-resolution) fnum available for this
    // frequency; a higher block than necessary just wastes precision by
    // pushing fnum down toward zero for no reason.
    for (int block = 0; block <= 7; ++block)
    {
        const double fnum = freqHz * std::pow(2.0, 20 - block) / kOplBaseRateHz;
        if (fnum <= 1023.0)
        {
            const int fnumInt = std::clamp(static_cast<int>(std::llround(fnum)), 0, 1023);
            return OplFnumBlock{fnumInt, block};
        }
    }
    // Frequency exceeds even block 7's range (~6208 Hz at fnum=1023, the
    // real hardware ceiling) -- clamp to the top of THAT band, not
    // block 0's (an earlier version of this fallback returned block 0,
    // which is actually the lowest achievable frequency for fnum=1023,
    // ~48.5 Hz -- about as wrong an answer to "too high" as possible).
    return OplFnumBlock{1023, 7};
}

double FnumBlockToHz(int fnum, int block)
{
    return static_cast<double>(fnum) * kOplBaseRateHz / std::pow(2.0, 20 - block);
}
