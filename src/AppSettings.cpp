#include "AppSettings.h"

#include <wx/fileconf.h>
#include <wx/filename.h>
#include <wx/stdpaths.h>

namespace settings
{

namespace
{

bool IsPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

// Sits next to the executable, matching the rest of the app's "copy the
// build output anywhere and it still works" portability.
wxString IniPath()
{
    wxFileName exe(wxStandardPaths::Get().GetExecutablePath());
    exe.SetFullName("WAV2VGM.ini");
    return exe.GetFullPath();
}

AppSettings LoadFromDisk()
{
    AppSettings s;
    wxFileConfig cfg(wxEmptyString, wxEmptyString, IniPath(), wxEmptyString, wxCONFIG_USE_LOCAL_FILE);

    cfg.Read("LastImportDir", &s.lastImportDir, wxEmptyString);
    cfg.Read("LastExportDir", &s.lastExportDir, wxEmptyString);

    int fftSize = s.fftSize;
    int hopSize = s.hopSize;
    cfg.Read("FftSize", &fftSize, s.fftSize);
    cfg.Read("HopSize", &hopSize, s.hopSize);

    // The FFT requires a power-of-two frame size, and a hop past the frame
    // size would skip audio entirely -- guard against a hand-edited INI
    // producing either rather than letting the spectrogram/FFT code choke
    // on it.
    if (IsPowerOfTwo(fftSize))
        s.fftSize = fftSize;
    if (hopSize >= 1 && hopSize <= s.fftSize)
        s.hopSize = hopSize;

    return s;
}

} // namespace

AppSettings& Get()
{
    static AppSettings s = LoadFromDisk();
    return s;
}

void Save()
{
    const AppSettings& s = Get();
    wxFileConfig cfg(wxEmptyString, wxEmptyString, IniPath(), wxEmptyString, wxCONFIG_USE_LOCAL_FILE);
    cfg.Write("LastImportDir", s.lastImportDir);
    cfg.Write("LastExportDir", s.lastExportDir);
    cfg.Write("FftSize", s.fftSize);
    cfg.Write("HopSize", s.hopSize);
    cfg.Flush();
}

} // namespace settings
