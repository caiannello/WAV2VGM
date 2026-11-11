#pragma once

#include "SineFit.h"

#include <cstdint>
#include <string>
#include <vector>

// Writes a sinefit::ChipFit out as a standard VGM file for whichever chip
// it was fit against: the chip's clock goes in its own header field, any
// sample data the target needs (OPL4/MultiPCM sine samples) goes in data
// blocks at the start, and register writes are only emitted when they
// change a value -- same conventions as vgm::WriteVgmFile, which this
// reproduces byte for byte for the OPL3 target.
namespace vgm
{

bool BuildChipVgm(const sinefit::ChipFit& fit, std::vector<uint8_t>& out, std::string& errorMessage);
bool WriteChipVgmFile(const std::string& path, const sinefit::ChipFit& fit, std::string& errorMessage);

} // namespace vgm
