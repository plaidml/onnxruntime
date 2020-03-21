// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/framework/execution_provider.h"

namespace plaidml {
namespace edsl {
class Program;
class Tensor;
}  // namespace edsl
}  // namespace plaidml

namespace onnxruntime {

constexpr const char* PLAIDML = "PlaidML";  // TODO: Borrowed from OpenVINO. Reasonable?

struct PlaidMLProgram {
  // A PlaidML Program bundled with its (ordered) input placeholder tensors
  std::shared_ptr<plaidml::edsl::Program> program;
  std::vector<plaidml::edsl::Tensor> inputs;
};

struct PlaidMLFunctionState {
  std::shared_ptr<PlaidMLProgram> program = nullptr;
};

// Information needed to construct PlaidML execution providers.
struct PlaidMLExecutionProviderInfo {
    // TODO: Empty for now. Forever? -- if so we can scrap this struct altogether
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