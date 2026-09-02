#ifndef PIPELINE_COMMON_H
#define PIPELINE_COMMON_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

extern "C" {
#include "dmabuf.h"
}

class PipelineError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string hex_address(uint64_t address);

size_t require_param(const std::map<std::string, std::string>& params,
                     const char* key, const char* stage);

class DmaBuffer {
public:
    DmaBuffer(size_t bytes, std::string_view purpose);
    ~DmaBuffer();
    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    dma_buf_params* operator->() noexcept { return &params_; }
    const dma_buf_params* operator->() const noexcept { return &params_; }

    template <typename T>
    T* data() noexcept { return reinterpret_cast<T*>(params_.kern_addr); }

    void begin_cpu_access() const;
    void end_cpu_access() const;

private:
    void sync(int operation) const;

    dma_buf_params params_{};
    bool allocated_{false};
};


#endif // PIPELINE_COMMON_H
