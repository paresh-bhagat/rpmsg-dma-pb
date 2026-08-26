#include "speech_classification_pipeline.h"
#include "pipeline_common.h"
#include "audio_utils.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

size_t require_param(const std::map<std::string, std::string>& params,
                     const char* key, const char* stage)
{
    auto it = params.find(key);
    if (it == params.end())
        throw PipelineError{std::string{"Stage missing required parameter: "} + key +
                            " (stage: " + stage + ")"};
    int v = std::stoi(it->second);
    if (v <= 0)
        throw PipelineError{std::string{"Parameter must be positive: "} + key};
    return static_cast<size_t>(v);
}

} // namespace

PipelineManager::CommandResult run_speech_classification_pipeline(
    PipelineManager::State& state,
    DspTaskClient& dsp_client,
    bool debug)
{
    try {
        if (state.input_type != PipelineManager::InputType::AUDIO_WAV)
            throw PipelineError{"speech_classification pipeline requires a .wav input file"};

        const PipelineManager::PipelineStage* stft_stage_ptr = nullptr;
        for (const auto& stage : state.pipeline_config.stages) {
            if (stage.message_type == "C7X_MSG_STFT_ANALYZE") {
                stft_stage_ptr = &stage;
                break;
            }
        }
        if (!stft_stage_ptr)
            throw PipelineError{"speech_classification pipeline requires a C7X_MSG_STFT_ANALYZE stage"};

        const auto& sp        = stft_stage_ptr->parameters;
        const size_t HOP_SIZE     = require_param(sp, "hop_size",     stft_stage_ptr->stage_id.c_str());
        const size_t MODEL_ELEMS  = require_param(sp, "model_elems",  stft_stage_ptr->stage_id.c_str());
        const size_t TOTAL_FRAMES = require_param(sp, "total_frames", stft_stage_ptr->stage_id.c_str());
        const size_t BATCH_N      = require_param(sp, "batch_n",      stft_stage_ptr->stage_id.c_str());
        const size_t PAD_FRAMES   = TOTAL_FRAMES % BATCH_N;
        const size_t NUM_BATCHES  = (TOTAL_FRAMES + BATCH_N - 1) / BATCH_N;

        const size_t audio_batch_bytes  = BATCH_N      * HOP_SIZE    * sizeof(int16_t);
        const size_t spectral_buf_bytes = TOTAL_FRAMES * MODEL_ELEMS * sizeof(float);

        std::cout << "[App] STFT parameters: HOP=" << HOP_SIZE
                  << " MODEL_ELEMS=" << MODEL_ELEMS
                  << " BATCH_N=" << BATCH_N
                  << " TOTAL_FRAMES=" << TOTAL_FRAMES << std::endl;

        std::vector<int16_t> audio_data;
        if (!loadAudioFile(state.current_input_file, audio_data))
            throw PipelineError{"Failed to load input audio"};

        if (!dsp_client.initialize(state.pipeline_config.dsp_config.proc_id,
                                   state.pipeline_config.dsp_config.endpoint))
            throw PipelineError{"Failed to initialize DSP Task client"};

        DmaBuffer dma_audio{audio_batch_bytes,  "STFT audio input"};
        DmaBuffer dma_mel  {spectral_buf_bytes, "STFT mel output"};

        std::cout << "[App] DMA buffers:" << std::endl;
        std::cout << "[App]   audio in: phys=0x" << std::hex << dma_audio->phys_addr
                  << std::dec << " size=" << dma_audio->size << std::endl;
        std::cout << "[App]   mel out:  phys=0x" << std::hex << dma_mel->phys_addr
                  << std::dec << " size=" << dma_mel->size << std::endl;

        const size_t total_frames = (audio_data.size() + HOP_SIZE - 1) / HOP_SIZE;
        audio_data.resize(total_frames * HOP_SIZE, int16_t{0});

        const size_t num_full_chunks = total_frames / TOTAL_FRAMES;
        const size_t partial_frames  = total_frames % TOTAL_FRAMES;
        const size_t num_chunks      = num_full_chunks + (partial_frames > 0 ? 1 : 0);

        std::cout << "[App] Total frames: " << total_frames
                  << " | Full chunks: " << num_full_chunks
                  << " | Partial: " << partial_frames << " frames" << std::endl;

        // Accumulate all mel floats from every chunk
        std::vector<float> all_mel;
        all_mel.reserve(total_frames * MODEL_ELEMS);

        auto t_total = std::chrono::steady_clock::now();

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const size_t chunk_frame_offset = chunk_idx * TOTAL_FRAMES;
            const size_t real_frames_chunk  = (chunk_idx == num_full_chunks && partial_frames > 0)
                                              ? partial_frames : TOTAL_FRAMES;
            auto t_chunk = std::chrono::steady_clock::now();

            for (size_t batch_idx = 0; batch_idx < NUM_BATCHES; ++batch_idx) {
                const size_t frames_this_batch  = (batch_idx < NUM_BATCHES - 1) ? BATCH_N : PAD_FRAMES;
                const size_t samples_this_batch = frames_this_batch * HOP_SIZE;
                const size_t audio_offset       = (chunk_frame_offset + batch_idx * BATCH_N) * HOP_SIZE;
                const uint64_t spectral_offset  = batch_idx * BATCH_N * MODEL_ELEMS * sizeof(float);

                dma_audio.begin_cpu_access();
                std::fill_n(dma_audio.data<std::byte>(), audio_batch_bytes, std::byte{});
                if (audio_offset < audio_data.size()) {
                    const size_t available = std::min(samples_this_batch,
                                                      audio_data.size() - audio_offset);
                    std::copy_n(audio_data.begin() + static_cast<std::ptrdiff_t>(audio_offset),
                                available, dma_audio.data<int16_t>());
                }
                dma_audio.end_cpu_access();

                auto params = stft_stage_ptr->parameters;
                params["input_buffer"]  = hex_address(dma_audio->phys_addr);
                params["output_buffer"] = hex_address(dma_mel->phys_addr + spectral_offset);
                params["input_frame"]   = std::to_string(frames_this_batch);
                params["output_frame"]  = std::to_string(frames_this_batch);

                if (debug)
                    std::cout << "[App]   batch " << (batch_idx + 1) << "/" << NUM_BATCHES
                              << ": " << frames_this_batch << " frames"
                              << " in=0x" << std::hex << dma_audio->phys_addr
                              << " out=0x" << (dma_mel->phys_addr + spectral_offset)
                              << std::dec << std::endl;

                auto r = dsp_client.process("C7X_MSG_STFT_ANALYZE", params);
                if (!r.success)
                    throw PipelineError{"STFT batch " + std::to_string(batch_idx + 1) +
                                        " failed: " + r.error_message};
            }

            // Collect real mel frames from this chunk
            dma_mel.begin_cpu_access();
            const float* mel_ptr = dma_mel.data<float>();
            all_mel.insert(all_mel.end(), mel_ptr, mel_ptr + real_frames_chunk * MODEL_ELEMS);
            dma_mel.end_cpu_access();

            double t_chunk_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_chunk).count() / 1000.0;

            std::cout << "[App] Chunk " << (chunk_idx + 1) << "/" << num_chunks
                      << " [" << real_frames_chunk << " frames, "
                      << real_frames_chunk * MODEL_ELEMS << " mel floats]"
                      << " " << std::fixed << std::setprecision(1) << t_chunk_ms << "ms"
                      << std::endl;
        }

        double t_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_total).count() / 1000.0;

        std::cout << "[App] Done: " << (all_mel.size() / MODEL_ELEMS) << " mel frames, "
                  << all_mel.size() << " floats"
                  << " | " << std::fixed << std::setprecision(1) << t_total_ms << "ms" << std::endl;

        // Save all mel features to binary file
        const std::string out_path = "mel_features_output.bin";
        std::ofstream out(out_path, std::ios::binary);
        if (!out.is_open())
            throw PipelineError{"Cannot open output file: " + out_path};
        out.write(reinterpret_cast<const char*>(all_mel.data()),
                  static_cast<std::streamsize>(all_mel.size() * sizeof(float)));
        std::cout << "[App] Saved to " << out_path
                  << " (" << (all_mel.size() * sizeof(float)) << " bytes)" << std::endl;

        return PipelineManager::CommandResult::SUCCESS;

    } catch (const std::exception& error) {
        std::cerr << "[App] Pipeline failed: " << error.what() << std::endl;
        return PipelineManager::CommandResult::ERROR;
    }
}
