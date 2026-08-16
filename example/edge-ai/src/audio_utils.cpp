#include "audio_utils.h"
#include <iostream>
#include <memory>
#include <sndfile.h>

bool loadAudioFile(const std::string& filename, std::vector<int16_t>& audio_data)
{
    SF_INFO sfinfo{};

    std::unique_ptr<SNDFILE, decltype(&sf_close)> infile{
        sf_open(filename.c_str(), SFM_READ, &sfinfo), &sf_close};
    if (!infile) {
        std::cout << "[App] Error: Failed to open audio file: " << filename << std::endl;
        return false;
    }

    // Validate audio format
    if (sfinfo.channels != 1) {
        std::cout << "[App] Error: Audio must be mono (1 channel), got " << sfinfo.channels << " channels" << std::endl;
        return false;
    }

    if (sfinfo.frames <= 0) {
        std::cout << "[App] Error: Audio file contains no samples" << std::endl;
        return false;
    }

    if (sfinfo.samplerate != 16000) {
        std::cout << "[App] Error: Audio sample rate is " << sfinfo.samplerate
                  << "Hz; this pipeline requires 16kHz" << std::endl;
        return false;
    }

    std::cout << "[App] Audio file info: " << (sfinfo.frames / 160) << " GCRN frames ("
              << sfinfo.frames << " samples), "
              << sfinfo.samplerate << "Hz, " << sfinfo.channels << " channel(s)" << std::endl;

    // Read all audio data
    audio_data.resize(sfinfo.frames);
    const sf_count_t frames_read = sf_readf_short(
        infile.get(), audio_data.data(), sfinfo.frames);

    if (frames_read < 0) {
        std::cout << "[App] Error: Failed while reading audio data" << std::endl;
        return false;
    }
    if (frames_read != sfinfo.frames) {
        std::cout << "[App] Warning: Read " << frames_read << " frames, expected " << sfinfo.frames << std::endl;
        audio_data.resize(frames_read);
    }

    std::cout << "[App] Loaded " << audio_data.size() << " audio samples ("
              << (static_cast<double>(audio_data.size()) / sfinfo.samplerate)
              << " seconds)" << std::endl;

    return true;
}

bool saveAudioFile(const std::string& filename, const std::vector<int16_t>& audio_data)
{
    if (audio_data.empty()) {
        std::cout << "[App] Error: Refusing to write an empty audio file" << std::endl;
        return false;
    }
    SF_INFO sfinfo{};

    // Set output file parameters to match our audio format
    sfinfo.samplerate = 16000;    // 16kHz
    sfinfo.channels = 1;          // mono
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;  // WAV file with 16-bit PCM

    std::unique_ptr<SNDFILE, decltype(&sf_close)> outfile{
        sf_open(filename.c_str(), SFM_WRITE, &sfinfo), &sf_close};
    if (!outfile) {
        std::cout << "[App] Error: Failed to create output audio file: " << filename << std::endl;
        std::cout << "[App] Error details: " << sf_strerror(nullptr) << std::endl;
        return false;
    }

    std::cout << "[App] Saving audio to: " << filename << std::endl;
    std::cout << "[App] Output file info: " << (audio_data.size() / 160) << " GCRN frames ("
              << audio_data.size() << " samples), "
              << sfinfo.samplerate << "Hz, " << sfinfo.channels << " channel(s)" << std::endl;

    // Write audio data to file
    const sf_count_t frames_written = sf_writef_short(
        outfile.get(), audio_data.data(), static_cast<sf_count_t>(audio_data.size()));

    if (frames_written != static_cast<sf_count_t>(audio_data.size())) {
        std::cout << "[App] Warning: Wrote " << frames_written << " frames, expected " << audio_data.size() << std::endl;
        return false;
    }

    std::cout << "[App] Successfully saved " << (frames_written / 160) << " GCRN frames ("
              << frames_written << " samples) to " << filename << std::endl;
    std::cout << "[App] Duration: "
              << (static_cast<double>(frames_written) / sfinfo.samplerate)
              << " seconds" << std::endl;

    return true;
}
