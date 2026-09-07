#include "MainWindow.h"
#include "ProjectWindow.h"
#include "AppSettings.h"

#include <wx/filedlg.h>
#include <wx/msgdlg.h>

MainWindow::MainWindow(wxWindow* parent)
    : wxMDIParentFrame(parent, wxID_ANY, "WAV2VGM")
{
    SetSize(1200, 760);
    m_fileMenu = new wxMenu();
    m_fileMenu->Append(ID_Import, "Import WAV...");
    m_fileMenu->AppendSeparator();
    m_fileMenu->Append(wxID_EXIT, "Exit");

    wxMenuBar* menuBar = new wxMenuBar();
    menuBar->Append(m_fileMenu, "File");
    SetMenuBar(menuBar);

    Bind(wxEVT_MENU, &MainWindow::OnImport, this, ID_Import);
    Bind(wxEVT_MENU, &MainWindow::OnExit, this, wxID_EXIT);

    CreateStatusBar();
    SetStatusText("Ready");
}

void MainWindow::OnImport(wxCommandEvent& WXUNUSED(evt))
{
    settings::AppSettings& s = settings::Get();
    wxFileDialog dlg(this, "Open audio file", s.lastImportDir, wxEmptyString,
                     "WAV files (*.wav)|*.wav|All files (*.*)|*.*",
                     wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    if (dlg.ShowModal() != wxID_OK)
        return;

    s.lastImportDir = dlg.GetDirectory();
    settings::Save();

    wxString path = dlg.GetPath();
    ProjectWindow* win = new ProjectWindow(this);
    if (!win->LoadImportedFile(path.ToStdString()))
    {
        win->Destroy();
        wxMessageBox("Import failed", "Error", wxICON_ERROR);
        return;
    }
    win->Show();
    win->Maximize(true);
    SetStatusText("Imported " + path);
}

void MainWindow::OnExit(wxCommandEvent& WXUNUSED(evt))
{
    Close(true);
}
