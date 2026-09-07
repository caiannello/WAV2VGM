cmake_minimum_required(VERSION 3.21)
project(WAV2VGM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(wxWidgets REQUIRED CONFIG COMPONENTS core base)

add_executable(WAV2VGM WIN32
    src/main.cpp
    src/MainWindow.cpp
    src/ProjectWindow.cpp
    src/AudioProject.cpp
    src/SpectrogramView.cpp
    src/TimeRuler.cpp
    src/VerticalAxis.cpp
    src/Dsp.cpp
    src/PeakAnalysis.cpp
    src/OplChip.cpp
    src/OplPatch.cpp
    src/OplFit.cpp
    src/WavWriter.cpp
    src/VgmWriter.cpp
    src/AppSettings.cpp
    src/third_party/dbopl/dbopl.cpp
)

target_link_libraries(WAV2VGM PRIVATE wxWidgets::wxWidgets)

enable_testing()
add_executable(dsp_selftest src/Dsp.cpp src/tools/dsp_selftest.cpp)
add_test(NAME DspSelfTest COMMAND dsp_selftest)

# dbopl (src/third_party/dbopl) is GPL2+ -- see NOTICE.md. Only the two
# targets that link it (WAV2VGM itself, and this self-test) are affected.
add_executable(opl3_selftest
    src/Dsp.cpp
    src/PeakAnalysis.cpp
    src/OplChip.cpp
    src/OplPatch.cpp
    src/OplFit.cpp
    src/VgmWriter.cpp
    src/third_party/dbopl/dbopl.cpp
    src/tools/opl3_selftest.cpp
)
add_test(NAME Opl3SelfTest COMMAND opl3_selftest)
