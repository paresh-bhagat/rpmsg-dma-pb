#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <string>
#include <vector>
#include <cstdint>

bool loadAudioFile(const std::string& filename, std::vector<int16_t>& audio_data);
bool saveAudioFile(const std::string& filename, const std::vector<int16_t>& audio_data);

#endif // AUDIO_UTILS_H
