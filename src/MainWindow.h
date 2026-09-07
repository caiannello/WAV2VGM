#pragma once

#include <wx/wx.h>

class MainWindow : public wxMDIParentFrame
{
public:
    explicit MainWindow(wxWindow* parent = nullptr);

private:
    void OnImport(wxCommandEvent& evt);
    void OnExit(wxCommandEvent& evt);

    enum
    {
        ID_Import = wxID_HIGHEST + 1
    };

    wxMenu* m_fileMenu = nullptr;
};
