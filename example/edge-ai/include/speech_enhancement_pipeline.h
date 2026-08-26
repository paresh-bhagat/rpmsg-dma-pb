#ifndef SPEECH_ENHANCEMENT_PIPELINE_H
#define SPEECH_ENHANCEMENT_PIPELINE_H

#include "pipeline_manager.h"
#include "dsp_task_client.h"
#include "tvm_inference_client.h"

PipelineManager::CommandResult run_speech_enhancement_pipeline(
    PipelineManager::State& state,
    DspTaskClient& dsp_client,
    TvmInferenceClient& tvm_client,
    bool debug);

#endif // SPEECH_ENHANCEMENT_PIPELINE_H
