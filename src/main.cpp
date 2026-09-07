#include "MainWindow.h"
#include "AppSettings.h"
#include "Dsp.h"

#include <wx/wx.h>

class WAV2VGMApp : public wxApp
{
public:
    bool OnInit() override
    {
        // Load before any spectrogram/FFT work can happen -- dsp::kFftSize/
        // kHopSize are set once here from the user's saved settings (see
        // AppSettings.h) and treated as read-only for the rest of the run.
        const settings::AppSettings& s = settings::Get();
        dsp::kFftSize = s.fftSize;
        dsp::kHopSize = s.hopSize;

        MainWindow* frame = new MainWindow(nullptr);
        frame->SetSize(1200, 760);
        frame->Show();
        return true;
    }
};

wxIMPLEMENT_APP(WAV2VGMApp);
