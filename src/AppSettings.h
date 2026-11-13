#pragma once

#include <wx/string.h>

// User-editable settings, persisted to an INI file next to the executable
// (portable, like the rest of the app -- no registry/AppData use). Two
// remembered dialog directories plus the two spectrogram-shape knobs that
// used to be Dsp.h compile-time constants (dsp::kFftSize/kHopSize), now
// runtime-tunable without a rebuild.
namespace settings
{

struct AppSettings
{
    wxString lastImportDir;
    wxString lastExportDir;
    int fftSize = 4096;
    int hopSize = 32;
    // Last-used output chip in the Chip tab's dropdown (an index into
    // ProjectWindow's chip list) and OPL4 mode (0 = FM, 1 = PCM, 2 = both).
    int targetChip = 0;
    int opl4Mode = 0;
    // MIDI output's pitch-bend range in semitones (RPN 0), 1-24.
    int midiBendRange = 24;
};

// Returns the process-wide settings, loading them from disk (or falling
// back to defaults, e.g. on first run or a missing/invalid INI value) on
// first call. The returned reference is mutable -- update a field in
// place, then call Save() to persist it.
AppSettings& Get();

// Writes the current in-memory settings back to the INI file.
void Save();

} // namespace settings
