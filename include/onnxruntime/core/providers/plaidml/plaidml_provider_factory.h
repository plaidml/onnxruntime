// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_PlaidML, _In_ OrtSessionOptions* options);

#ifdef __cplusplus
}
#endif
