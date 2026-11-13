#include "MidiWriter.h"

#include "chips/MidiTarget.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace midi
{

namespace
{

using chips::MidiTarget;

constexpr double kTicksPerSecond = kTicksPerQuarter * 1e6 / kTempoMicrosPerQuarter;
constexpr int kNoteVelocity = 127; // dynamics come from CC11 instead

void AppendVarLen(std::vector<uint8_t>& out, uint32_t value)
{
    uint8_t buf[5];
    int n = 0;
    buf[n++] = static_cast<uint8_t>(value & 0x7F);
    while (value >>= 7)
        buf[n++] = static_cast<uint8_t>(0x80 | (value & 0x7F));
    while (n > 0)
        out.push_back(buf[--n]);
}

void AppendBE32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int s = 24; s >= 0; s -= 8)
        out.push_back(static_cast<uint8_t>((v >> s) & 0xFF));
}

void AppendBE16(std::vector<uint8_t>& out, uint16_t v)
{
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

// Builds one MTrk body from events at absolute tick times, given in
// non-decreasing order.
class TrackBuilder
{
public:
    void Event(uint32_t tick, std::initializer_list<uint8_t> bytes)
    {
        AppendVarLen(m_data, tick - m_lastTick);
        m_lastTick = tick;
        m_data.insert(m_data.end(), bytes);
    }
    void Meta(uint32_t tick, uint8_t type, const std::vector<uint8_t>& payload)
    {
        AppendVarLen(m_data, tick - m_lastTick);
        m_lastTick = tick;
        m_data.push_back(0xFF);
        m_data.push_back(type);
        AppendVarLen(m_data, static_cast<uint32_t>(payload.size()));
        m_data.insert(m_data.end(), payload.begin(), payload.end());
    }
    void Name(const std::string& name) { Meta(0, 0x03, std::vector<uint8_t>(name.begin(), name.end())); }
    void AppendTo(std::vector<uint8_t>& file, uint32_t endTick)
    {
        Meta(std::max(endTick, m_lastTick), 0x2F, {});
        file.insert(file.end(), {'M', 'T', 'r', 'k'});
        AppendBE32(file, static_cast<uint32_t>(m_data.size()));
        file.insert(file.end(), m_data.begin(), m_data.end());
    }

private:
    std::vector<uint8_t> m_data;
    uint32_t m_lastTick = 0;
};

uint32_t FrameTick(size_t frame, double frameRateHz)
{
    return static_cast<uint32_t>(std::llround(static_cast<double>(frame) * kTicksPerSecond / frameRateHz));
}

int BendValue(double semitonesFromNote, int range)
{
    return std::clamp(static_cast<int>(std::llround(8192.0 + semitonesFromNote * 8192.0 / range)), 0, 16383);
}

void WriteChannelSetup(TrackBuilder& t, uint8_t ch, int bendRange)
{
    const auto cc = [&](uint8_t controller, uint8_t value) {
        t.Event(0, {static_cast<uint8_t>(0xB0 | ch), controller, value});
    };
    cc(121, 0);                              // Reset All Controllers
    cc(0, 0);                                // Bank Select MSB
    cc(32, 0);                               // Bank Select LSB
    t.Event(0, {static_cast<uint8_t>(0xC0 | ch), static_cast<uint8_t>(MidiTarget::kGmOcarinaProgram)});
    cc(124, 0);                              // Omni Off
    cc(126, 1);                              // Mono On, one voice
    cc(7, 127);                              // Channel Volume
    cc(10, 64);                              // Pan center
    cc(11, 127);                             // Expression
    cc(101, 0);                              // RPN MSB  } RPN 0 =
    cc(100, 0);                              // RPN LSB  } pitch-bend sensitivity
    cc(6, static_cast<uint8_t>(bendRange));  // Data Entry MSB: semitones
    cc(38, 0);                               // Data Entry LSB: cents
    cc(101, 127);                            // RPN null, so later
    cc(100, 127);                            //   data entry can't change it
    t.Event(0, {static_cast<uint8_t>(0xE0 | ch), 0x00, 0x40}); // pitch bend center
}

void WriteVoicePerformance(TrackBuilder& t, uint8_t ch, const sinefit::VoiceTrack& track, double frameRateHz,
                           int bendRange, uint32_t endTick)
{
    // Usable bend reach: the top bend value is 8191/8192 of the range.
    const double reach = bendRange * 8191.0 / 8192.0;
    bool sounding = false;
    int note = 0;
    int lastBend = 8192;
    int lastExpression = 127;

    const auto setExpression = [&](uint32_t tick, int expression) {
        if (expression != lastExpression)
        {
            t.Event(tick, {static_cast<uint8_t>(0xB0 | ch), 11, static_cast<uint8_t>(expression)});
            lastExpression = expression;
        }
    };
    const auto setBend = [&](uint32_t tick, int bend) {
        if (bend != lastBend)
        {
            t.Event(tick, {static_cast<uint8_t>(0xE0 | ch), static_cast<uint8_t>(bend & 0x7F),
                           static_cast<uint8_t>((bend >> 7) & 0x7F)});
            lastBend = bend;
        }
    };
    const auto noteOn = [&](uint32_t tick, int n) {
        t.Event(tick, {static_cast<uint8_t>(0x90 | ch), static_cast<uint8_t>(n), static_cast<uint8_t>(kNoteVelocity)});
    };
    const auto noteOff = [&](uint32_t tick, int n) {
        t.Event(tick, {static_cast<uint8_t>(0x80 | ch), static_cast<uint8_t>(n), 0});
    };

    for (size_t f = 0; f < track.frames.size(); ++f)
    {
        const chips::VoiceFrame& frame = track.frames[f];
        const uint32_t tick = FrameTick(f, frameRateHz);
        if (!frame.keyOn)
        {
            if (sounding)
            {
                noteOff(tick, note);
                sounding = false;
            }
            continue;
        }

        const double target = MidiTarget::PitchToNote(frame.pitch);
        setExpression(tick, MidiTarget::ExpressionOf(frame.level));
        if (!sounding || std::abs(target - note) > reach)
        {
            // Strike a new note: bend first so it starts in tune, then
            // the new note-on before the old note-off (legato in mono mode).
            const int newNote = std::clamp(static_cast<int>(std::lround(target)), 0, 127);
            setBend(tick, BendValue(target - newNote, bendRange));
            noteOn(tick, newNote);
            if (sounding)
                noteOff(tick, note);
            note = newNote;
            sounding = true;
        }
        else
        {
            setBend(tick, BendValue(target - note, bendRange));
        }
    }
    if (sounding)
        noteOff(endTick, note);
}

} // namespace

bool BuildMidi(const sinefit::ChipFit& fit, std::vector<uint8_t>& out, std::string& errorMessage)
{
    const auto* target = dynamic_cast<const MidiTarget*>(fit.target.get());
    if (!target)
    {
        errorMessage = "This approximation wasn't made for MIDI output.";
        return false;
    }
    if (fit.tracks.empty() || fit.frameRateHz <= 0.0)
    {
        errorMessage = "No channels to export.";
        return false;
    }

    size_t maxFrames = 0;
    for (const sinefit::VoiceTrack& track : fit.tracks)
        maxFrames = std::max(maxFrames, track.frames.size());
    const uint32_t endTick = FrameTick(maxFrames, fit.frameRateHz);

    // One track per channel that ever sounds, in MIDI channel order.
    std::vector<const sinefit::VoiceTrack*> used;
    for (const sinefit::VoiceTrack& track : fit.tracks)
    {
        if (track.voice < 0 || track.voice >= MidiTarget::kChannels)
        {
            errorMessage = "Channel out of range for MIDI.";
            return false;
        }
        if (std::any_of(track.frames.begin(), track.frames.end(), [](const chips::VoiceFrame& f) { return f.keyOn; }))
            used.push_back(&track);
    }
    std::sort(used.begin(), used.end(), [](const sinefit::VoiceTrack* a, const sinefit::VoiceTrack* b) {
        return MidiTarget::MidiChannelOf(a->voice) < MidiTarget::MidiChannelOf(b->voice);
    });

    out.clear();
    out.insert(out.end(), {'M', 'T', 'h', 'd'});
    AppendBE32(out, 6);
    AppendBE16(out, 1); // format 1
    AppendBE16(out, static_cast<uint16_t>(1 + used.size()));
    AppendBE16(out, static_cast<uint16_t>(kTicksPerQuarter));

    TrackBuilder conductor;
    conductor.Name("WAV2VGM");
    conductor.Meta(0, 0x51, {static_cast<uint8_t>((kTempoMicrosPerQuarter >> 16) & 0xFF),
                             static_cast<uint8_t>((kTempoMicrosPerQuarter >> 8) & 0xFF),
                             static_cast<uint8_t>(kTempoMicrosPerQuarter & 0xFF)});
    conductor.Meta(0, 0x58, {4, 2, 24, 8}); // 4/4
    conductor.AppendTo(out, endTick);

    for (const sinefit::VoiceTrack* track : used)
    {
        const uint8_t ch = static_cast<uint8_t>(MidiTarget::MidiChannelOf(track->voice));
        TrackBuilder t;
        t.Name("Ocarina " + std::to_string(ch + 1));
        WriteChannelSetup(t, ch, target->BendRangeSemitones());
        WriteVoicePerformance(t, ch, *track, fit.frameRateHz, target->BendRangeSemitones(), endTick);
        t.AppendTo(out, endTick);
    }
    return true;
}

bool WriteMidiFile(const std::string& path, const sinefit::ChipFit& fit, std::string& errorMessage)
{
    std::vector<uint8_t> bytes;
    if (!BuildMidi(fit, bytes, errorMessage))
        return false;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
    {
        errorMessage = "Unable to write MIDI file: " + path;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!ofs.good())
    {
        errorMessage = "Unable to write MIDI file: " + path;
        return false;
    }
    return true;
}

} // namespace midi
