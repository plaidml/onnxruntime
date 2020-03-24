// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <functional>
#include <map>
#include <string>

#include "plaidml/edsl/edsl.h"

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace plaidml_ep {

using OpFunction = std::function<std::vector<plaidml::edsl::Tensor>(const std::vector<plaidml::edsl::Value>& args)>;

std::vector<plaidml::edsl::Tensor> add(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> mul(const std::vector<plaidml::edsl::Value>& args);

std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

}  // namespace plaidml_ep
}  // namespace onnxruntime