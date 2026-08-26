#ifndef SPEECH_CLASSIFICATION_PIPELINE_H
#define SPEECH_CLASSIFICATION_PIPELINE_H

#include "pipeline_manager.h"
#include "dsp_task_client.h"

PipelineManager::CommandResult run_speech_classification_pipeline(
    PipelineManager::State& state,
    DspTaskClient& dsp_client,
    bool debug);

#endif // SPEECH_CLASSIFICATION_PIPELINE_H
