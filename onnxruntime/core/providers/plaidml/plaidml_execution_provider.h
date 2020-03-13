// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {

// Information needed to construct PlaidML execution providers.
struct PlaidMLExecutionProviderInfo {
    // TODO: Empty for now. Forever?
};

class PlaidMLExecutionProvider : public IExecutionProvider {
 public:
  explicit PlaidMLExecutionProvider(const PlaidMLExecutionProviderInfo& info);
  ~PlaidMLExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  // TODO: PlaidML-specific state goes here; if info_ is truly empty, don't need it here
  // PlaidMLExecutionProviderInfo info_;
};

}  // namespace onnxruntime