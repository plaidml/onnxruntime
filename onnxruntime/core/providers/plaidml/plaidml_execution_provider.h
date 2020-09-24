// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/framework/execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "plaidml_node.h"
#include "plaidml_ops.h"

#include <stdio.h>
#include <string.h>

namespace onnxruntime {

constexpr const char* PLAIDML = "PlaidML";  // TODO (PlaidML): Borrowed from OpenVINO. Reasonable?

struct PlaidMLFunctionState {
  std::shared_ptr<plaidml_ep::PlaidMLProgram> program = nullptr;
};

// Information needed to construct PlaidML execution providers.
struct PlaidMLExecutionProviderInfo {
  // TODO (PlaidML): Empty for now. Forever? -- if so we can scrap this struct altogether
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
  // TODO (PlaidML): -specific state goes here; if info_ is truly empty, don't need it here
  // PlaidMLExecutionProviderInfo info_;
};

}  // namespace onnxruntime