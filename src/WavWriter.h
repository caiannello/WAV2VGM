#pragma once

#include <string>
#include <vector>

// Minimal standalone mono 16-bit PCM WAV writer, shared by anything that
// needs a playable file on disk from an in-memory sample buffer (OPL3
// analysis mode's channel/mix preview playback) without depending on
// AudioProject's own (private, project-file-specific) writer.
namespace wavwriter
{

bool WriteMonoWavFile(const std::string& path, const std::vector<float>& samples, int sampleRate,
                       std::string& errorMessage);

} // namespace wavwriter
