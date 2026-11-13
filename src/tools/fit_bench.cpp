// Command-line comparison of the fitting workflows on a real recording --
// the same three choices the Chip tab's "Fitting" radio box offers:
//
//   fit_bench <input.wav> [chip] [voices] [frameRateHz] [outDir]
//
// chip defaults to opl3, voices to the chip's full voice count, frame rate
// to 100 Hz. If outDir is given, each workflow's VGM is written there too.
//
// Each run reports the fit score (see sinefit::FitScore) and how long it
// took, which is the quickest way to judge whether a workflow change is an
// improvement on your own material rather than on a synthetic test.

#include "../AudioProject.h"
#include "../ChipVgmWriter.h"
#include "../SineFit.h"
#include "../chips/ChipTarget.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

struct ChipName
{
    const char* name;
    chips::ChipId id;
};

constexpr ChipName kChips[] = {
    {"opl3", chips::ChipId::Opl3},       {"opl2", chips::ChipId::Opl2},
    {"opl4fm", chips::ChipId::Opl4Fm},   {"opl4pcm", chips::ChipId::Opl4Pcm},
    {"opl4both", chips::ChipId::Opl4Both}, {"opm", chips::ChipId::Opm},
    {"opn", chips::ChipId::Opn},         {"opna", chips::ChipId::Opna},
    {"opn2", chips::ChipId::Opn2},       {"huc6280", chips::ChipId::HuC6280},
    {"vsu", chips::ChipId::Vsu},         {"multipcm", chips::ChipId::MultiPcm},
    {"midi", chips::ChipId::Midi},
};

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <input.wav> [chip] [voices] [frameRateHz] [outDir]\n", argv[0]);
        std::printf("chips:");
        for (const ChipName& c : kChips)
            std::printf(" %s", c.name);
        std::printf("\n");
        return 1;
    }

    chips::ChipId id = chips::ChipId::Opl3;
    if (argc > 2)
    {
        bool found = false;
        for (const ChipName& c : kChips)
            if (std::strcmp(argv[2], c.name) == 0)
            {
                id = c.id;
                found = true;
            }
        if (!found)
        {
            std::printf("unknown chip '%s'\n", argv[2]);
            return 1;
        }
    }
    const auto target = chips::CreateTarget(id);

    AudioProject project;
    std::string err;
    if (!project.ImportFromFile(argv[1], err))
    {
        std::printf("import failed: %s\n", err.c_str());
        return 1;
    }

    const int voices = (argc > 3) ? std::max(1, std::atoi(argv[3])) : target->VoiceCount();
    const double frameRate = (argc > 4) ? std::atof(argv[4]) : 100.0;
    const char* outDir = (argc > 5) ? argv[5] : nullptr;

    std::printf("%s: %s, %d voices, %.1f Hz frames\n", argv[1], target->Name().c_str(), voices, frameRate);

    struct Workflow
    {
        const char* name;
        sinefit::ChipFit (*run)(const std::shared_ptr<const chips::ChipTarget>&, const std::vector<float>&, int,
                                double, int, const dsp::ProgressCallback&);
    };
    const Workflow workflows[] = {
        {"per-frame", &sinefit::FitChannelsPerFrame},
        {"trend", &sinefit::FitChannelsInTurn},
        {"chains", &sinefit::FitChannelsByChains},
    };

    for (const Workflow& workflow : workflows)
    {
        const auto start = std::chrono::steady_clock::now();
        const sinefit::ChipFit fit =
            workflow.run(target, project.Samples(), project.SampleRate(), frameRate, voices, nullptr);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        // Average sounding voices per frame: the density the per-frame
        // workflow guarantees and the others have to earn.
        double voiceSum = 0.0;
        size_t frames = 0;
        for (const sinefit::VoiceTrack& track : fit.tracks)
        {
            frames = std::max(frames, track.frames.size());
            for (const chips::VoiceFrame& frame : track.frames)
                voiceSum += frame.keyOn ? 1.0 : 0.0;
        }
        const double density = frames > 0 ? voiceSum / static_cast<double>(frames) : 0.0;

        std::printf("  %-10s similarity %.3f  err %6.2f dB  key-ons %5d  jumps %5d  voices %5.2f/%3zu  %6lld ms\n",
                    workflow.name, fit.score.similarity, fit.score.meanAbsDbError, fit.score.keyOnCount,
                    fit.score.pitchJumpCount, density, fit.tracks.size(), static_cast<long long>(elapsed));

        if (outDir && !fit.tracks.empty())
        {
            const std::string path = std::string(outDir) + "/" + workflow.name + ".vgm";
            std::string writeErr;
            if (vgm::WriteChipVgmFile(path, fit, writeErr))
                std::printf("             wrote %s\n", path.c_str());
            else
                std::printf("             %s\n", writeErr.c_str());
        }
    }
    return 0;
}
