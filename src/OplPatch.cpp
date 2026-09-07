#include "OplPatch.h"

namespace opl
{

namespace
{

struct ChannelOffsets { uint8_t modBase; uint8_t carBase; };

// Standard OPL2/3 per-channel operator base offsets within a bank
// (channel index 0-8): modulator and carrier are 3 apart, with a gap
// baked into the register map itself every 3 channels (0-2, 8-A, 10-12).
constexpr ChannelOffsets kOffsetsInBank[9] = {
    {0x00, 0x03}, {0x01, 0x04}, {0x02, 0x05},
    {0x08, 0x0B}, {0x09, 0x0C}, {0x0A, 0x0D},
    {0x10, 0x13}, {0x11, 0x14}, {0x12, 0x15},
};

void Resolve(int channelIndex, int& bank, uint8_t& modBase, uint8_t& carBase, uint8_t& chanReg)
{
    bank = channelIndex / 9;
    const int chanInBank = channelIndex % 9;
    modBase = kOffsetsInBank[chanInBank].modBase;
    carBase = kOffsetsInBank[chanInBank].carBase;
    chanReg = static_cast<uint8_t>(chanInBank);
}

} // namespace

void SetupChannel(OplChip& chip, int channelIndex, const ChannelStaticSetup& setup)
{
    int bank = 0;
    uint8_t modBase = 0, carBase = 0, chanReg = 0;
    Resolve(channelIndex, bank, modBase, carBase, chanReg);

    // EGT: hold at the sustain level instead of free-running past it to
    // OFF -- see the comment on ChannelStaticSetup.
    constexpr uint8_t kSustainFlag = 0x20;

    chip.WriteReg(bank, static_cast<uint8_t>(0x20 + modBase),
                  static_cast<uint8_t>(kSustainFlag | (setup.modMultipleIndex & 0x0F)));
    chip.WriteReg(bank, static_cast<uint8_t>(0x40 + modBase), static_cast<uint8_t>(setup.modTotalLevel & 0x3F));
    chip.WriteReg(bank, static_cast<uint8_t>(0x60 + modBase), 0xFF); // attack=15 (fastest), decay=15
    chip.WriteReg(bank, static_cast<uint8_t>(0x80 + modBase), 0x0F); // sustain level=0 (max), release=15
    chip.WriteReg(bank, static_cast<uint8_t>(0xE0 + modBase), static_cast<uint8_t>(setup.modWaveform & 0x07));

    chip.WriteReg(bank, static_cast<uint8_t>(0x20 + carBase),
                  static_cast<uint8_t>(kSustainFlag | (setup.carMultipleIndex & 0x0F)));
    chip.WriteReg(bank, static_cast<uint8_t>(0x60 + carBase), 0xF0); // attack=15, decay=0 (never leaves the peak on its own)
    chip.WriteReg(bank, static_cast<uint8_t>(0x80 + carBase), 0x0F);
    chip.WriteReg(bank, static_cast<uint8_t>(0xE0 + carBase), static_cast<uint8_t>(setup.carWaveform & 0x07));

    constexpr uint8_t kPanBothOn = 0x30;
    const uint8_t algorithmBit = setup.algorithmAdditive ? 0x01 : 0x00; // CNT: 0=FM, 1=additive (see ChannelStaticSetup)
    chip.WriteReg(bank, static_cast<uint8_t>(0xC0 + chanReg),
                  static_cast<uint8_t>(kPanBothOn | ((setup.feedback & 0x07) << 1) | algorithmBit));
}

void ApplyFrame(OplChip& chip, int channelIndex, const FramePatch& frame)
{
    int bank = 0;
    uint8_t modBase = 0, carBase = 0, chanReg = 0;
    Resolve(channelIndex, bank, modBase, carBase, chanReg);

    chip.WriteReg(bank, static_cast<uint8_t>(0x40 + carBase), static_cast<uint8_t>(frame.carTotalLevel & 0x3F));
    chip.WriteReg(bank, static_cast<uint8_t>(0xA0 + chanReg), static_cast<uint8_t>(frame.fnum & 0xFF));
    const uint8_t keyOnBit = frame.keyOn ? 0x20 : 0x00;
    chip.WriteReg(bank, static_cast<uint8_t>(0xB0 + chanReg),
                  static_cast<uint8_t>(keyOnBit | ((frame.block & 0x07) << 2) | ((frame.fnum >> 8) & 0x03)));
}

} // namespace opl
