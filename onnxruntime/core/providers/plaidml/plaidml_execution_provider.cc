// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

// TODO: Other includes?
#include "core/framework/compute_capability.h"
#include "core/graph/model.h"

#include "core/framework/allocatormgr.h"  // TODO: For DeviceAllocatorRegistrationInfo

// TODO: Actually use this
#include "plaidml/edsl/edsl.h"

#include "plaidml_execution_provider.h"

namespace onnxruntime {

PlaidMLExecutionProvider::PlaidMLExecutionProvider(const PlaidMLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kPlaidMLExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  // TODO: I'm reusing the Allocator setup from OpenVINO. Is that right?
  DeviceAllocatorRegistrationInfo device_info(
    {
      OrtMemTypeDefault,
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
          onnxruntime::make_unique<OrtMemoryInfo>(PLAIDML, OrtDeviceAllocator)
        );
      },
      std::numeric_limits<size_t>::max()
    }
  );
  InsertAllocator(CreateAllocator(device_info));
}

std::vector<std::unique_ptr<ComputeCapability>> PlaidMLExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // TODO: This is a basic implementation that does not handle graph partitioning, incompatible
  // operation detection, initializers as inputs (for e.g. weights, reshape, ...), and probably
  // other things. But it should work in the basic case.
  // Loosely based on the nGraph approach
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

  std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

  // If there are no inputs, leave it for constant folding
  if (inputs.empty()) {
    return result;
  }

  // TODO: this is taken fairly directly from nGraph's approach; verify it makes sense
  auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "PlaidML_Fully_Fused_Graph";
  meta_def->domain = kPlaidMLDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  sub_graph->nodes = graph_viewer.GetNodesInTopologicalOrder();
  sub_graph->SetMetaDef(meta_def);
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  // /end TODO: verify

  return result;
}

common::Status PlaidMLExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& /*fused_nodes*/,
    std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {

  // TODO: This is a do-nothing stub. Implement!

  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}

}  // namespace onnxruntime
