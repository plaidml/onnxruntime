// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

// TODO: Other includes?
#include "core/framework/compute_capability.h"
#include "core/graph/model.h"

#include "plaidml_execution_provider.h"

namespace onnxruntime {

PlaidMLExecutionProvider::PlaidMLExecutionProvider(const PlaidMLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kPlaidMLExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  // TODO: This is a do-nothing ctor; that might be correct, but if not implement!
}

std::vector<std::unique_ptr<ComputeCapability>> PlaidMLExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& /*graph_viewer*/,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // TODO: This is a do-nothing stub. Implement!

  return result;
}

common::Status PlaidMLExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& /*fused_nodes*/,
    std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {

  // TODO: This is a do-nothing stub. Implement!

  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}

}  // namespace onnxruntime
