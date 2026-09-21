#ifndef SPEECH_CHUNK_RANGE_H
#define SPEECH_CHUNK_RANGE_H
#include <algorithm>
#include <cstddef>

struct SpeechChunkRange { size_t begin; size_t end; };

// Adjacent chunks meet at one boundary, including odd overlap frame counts.
// Caller validates overlap_frames < total_frames and matching STFT/ISTFT geometry.
inline SpeechChunkRange speech_chunk_range(size_t index, size_t chunks,
                                           size_t chunk_samples, size_t overlap_frames,
                                           size_t hop_size, size_t original_samples)
{
    const size_t overlap = overlap_frames * hop_size;
    const size_t left = (overlap_frames / 2) * hop_size;
    const size_t right = overlap - left;
    const size_t offset = index * (chunk_samples - overlap);
    const size_t begin = index ? left : 0;
    const size_t end = index + 1 < chunks ? chunk_samples - right : chunk_samples;
    const size_t available = original_samples > offset ? original_samples - offset : 0;
    return {begin, std::min(end, available)};
}
#endif
