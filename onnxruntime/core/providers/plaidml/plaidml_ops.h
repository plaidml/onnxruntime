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

using _OpFunction = std::function<std::vector<plaidml::edsl::Tensor>(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs)>;
std::vector<plaidml::edsl::Tensor> _eye_like(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> logical_and(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> logical_or(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> logical_not(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> logical_xor(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _lp_normalization(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _mod(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> mean(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> erf(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> pow(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sample_op(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sign(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> tile(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> where(const std::vector<plaidml::edsl::Value>& args);


//-------------------------------------------binary ops
std::vector<plaidml::edsl::Tensor> add(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> div(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> mul(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> prelu(const std::vector<plaidml::edsl::Value>& args);//not tested
std::vector<plaidml::edsl::Tensor> sub(const std::vector<plaidml::edsl::Value>& args);


//-----------------------------------------binary compare ops
std::vector<plaidml::edsl::Tensor> equal(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> greater(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> less(const std::vector<plaidml::edsl::Value>& args);


//--------------------------------------------pool ops
std::vector<plaidml::edsl::Tensor> _maxpool(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _average_pool(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
//std::vector<plaidml::edsl::Tensor> global_max_pool(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> global_average_pool(const std::vector<plaidml::edsl::Value>& args);

//---------------------------------------------reduce ops
std::vector<plaidml::edsl::Tensor> _argmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
 std::vector<plaidml::edsl::Tensor> _argmin(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

// std::vector<plaidml::edsl::Tensor> recuce_l1(const std::vector<plaidml::edsl::Value>& args);
// std::vector<plaidml::edsl::Tensor> reduce_l2(const std::vector<plaidml::edsl::Value>& args);
// std::vector<plaidml::edsl::Tensor> reduce_log_sum(const std::vector<plaidml::edsl::Value>& args);
// std::vector<plaidml::edsl::Tensor> reduce_log_sum_exp(const std::vector<plaidml::edsl::Value>& args);
 std::vector<plaidml::edsl::Tensor> _reduce_max(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _reduce_mean(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
 std::vector<plaidml::edsl::Tensor> _reduce_min(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
 std::vector<plaidml::edsl::Tensor> _reduce_prod(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
 std::vector<plaidml::edsl::Tensor> _reduce_sum(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
//  std::vector<plaidml::edsl::Tensor> _reduce_sum_square(
//     const ONNX_NAMESPACE::NodeProto& node,
//     const std::vector<plaidml::edsl::Value>& inputs);


//---------------------------------------------unary ops
std::vector<plaidml::edsl::Tensor> abs(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> affine(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> ceil(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _elu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> exp(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> floor(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _hard_sigmoid(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _leaky_relu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> log(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> neg(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> parametric_softplus(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> reciprocal(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> relu(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> scaled_tanh(const std::vector<plaidml::edsl::Value>& args);//deprcated op
std::vector<plaidml::edsl::Tensor> _selu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> sigmoid(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> softplus(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> softsign(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sqrt(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> tanh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _thresholded_relu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

//------------------------------------------variadic ops
std::vector<plaidml::edsl::Tensor> max(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> min(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sum(const std::vector<plaidml::edsl::Value>& args);


//---------------------------------------other ops
std::vector<plaidml::edsl::Tensor> _cast(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> clip(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _concat(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _conv(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
// std::vector<plaidml::edsl::Tensor> _crop(
//     const ONNX_NAMESPACE::NodeProto& node,
//     const std::vector<plaidml::edsl::Value>& inputs);
//std::vector<plaidml::edsl::Tensor> dropout(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> expand(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _flatten(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
//std::vector<plaidml::edsl::Tensor> gather(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> gather_elements(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> gem(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> identity(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _log_softmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
//std::vector<plaidml::edsl::Tensor> lstm(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> matmul(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> matmul_integer(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> pad(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> reshape(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> shape(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> slice(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _softmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _split(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _squeeze(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
//std::vector<plaidml::edsl::Tensor> transpose(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _transpose(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _unsqueeze(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);




std::vector<plaidml::edsl::Tensor> asin(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> acos(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> atan(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> asinh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cos(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cosh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> _cumsum(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> sin(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sinh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> tan(const std::vector<plaidml::edsl::Value>& args);

std::vector<plaidml::edsl::Tensor> _lrn(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

std::vector<plaidml::edsl::Tensor> _one_hot(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

std::vector<plaidml::edsl::Tensor> _reverse_sequence(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

bool check_op_support(std::string op_name);

}  // namespace plaidml_ep
}  // namespace onnxruntime