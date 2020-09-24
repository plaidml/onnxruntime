// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "core/providers/plaidml/plaidml_provider_factory.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"

#include "plaidml_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct PlaidMLProviderFactory : IExecutionProviderFactory {
  PlaidMLProviderFactory() = default;
  ~PlaidMLProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    PlaidMLExecutionProviderInfo info{};
    return onnxruntime::make_unique<PlaidMLExecutionProvider>(info);
  }
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_PlaidML() {
  plaidml::edsl::init();
  plaidml::exec::init();
  return std::make_shared<onnxruntime::PlaidMLProviderFactory>();
}

}  // namespace onnxruntime
// TODO (PlaidML): do we need device id here?
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_PlaidML,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_PlaidML());
  return nullptr;
}
