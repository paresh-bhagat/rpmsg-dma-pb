#include "pipeline_common.h"
#include <sstream>

std::string hex_address(uint64_t address)
{
    std::ostringstream value;
    value << "0x" << std::hex << address;
    return value.str();
}

DmaBuffer::DmaBuffer(size_t bytes, std::string_view purpose)
{
    if (bytes > std::numeric_limits<uint32_t>::max())
        throw PipelineError{"DMA allocation is larger than the API limit"};
    char heap[] = "linux,cma";
    char remoteproc[] = "/dev/remoteproc0";
    if (dmabuf_heap_init(heap, static_cast<uint32_t>(bytes), remoteproc, &params_) != 0)
        throw PipelineError{"Failed to allocate DMA buffer for " + std::string{purpose}};
    allocated_ = true;
}

DmaBuffer::~DmaBuffer()
{
    if (allocated_) dmabuf_heap_destroy(&params_);
}

void DmaBuffer::begin_cpu_access() const { sync(DMA_BUF_SYNC_START); }
void DmaBuffer::end_cpu_access() const   { sync(DMA_BUF_SYNC_END); }

void DmaBuffer::sync(int operation) const
{
    if (dmabuf_sync(params_.dma_buf_fd, operation) != 0)
        throw PipelineError{"DMA buffer synchronization failed"};
}

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
