// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "plaidml_ops.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"
#include "plaidml_node.h"

// TODO (PlaidML): using value of pi to compute sin from cos (find a better way )
#define _USE_MATH_DEFINES
#include <cmath>

namespace onnxruntime {
namespace plaidml_ep {

std::map<std::string, OpFunction> kSupportedOps = {
    {"Abs", abs},
    {"Add", add},
    {"And", logical_and},
    {"Asin", asin},
    {"Acos", acos},
    {"Atan", atan},
    {"Ceil", ceil},
    {"Cos", cos},
    {"Cosh", cosh},
    {"Div", div},
    {"Equal", equal},
    {"Erf", erf},
    {"Exp", exp},
    {"Floor", floor},
    {"Greater", greater},
    //{"Identity", identity}, // TODO (PlaidML): broken test (1/2 failures) (string type needed)//need to enable for Loop, scan9 etc tests
    {"Less", less},
    {"Log", log},
    {"Max", max},
    {"Mean", mean},
    //{"MatMul", matmul}, // TODO (PlaidML): OP WIP
    {"Min", min},
    {"Mul", mul},
    {"Neg", neg},
    {"Not", logical_not},
    {"Or", logical_or},
    //{"Pow", pow}, // TODO (PlaidML): fix broken tests (double)
    {"PRelu", prelu},
    {"Reciprocal", reciprocal},
    {"Relu", relu},
    //{"Reshape", reshape}, // TODO (PlaidML): OP WIP
    {"SampleOp", sample_op},
    //{"Shape", shape}, // TODO (PlaidML): fix broken tests (11/11 failures)
    //{"Sigmoid", sigmoid}, failing after new build (operation not registered error )
    //{"Sign", sign}, // TODO (PlaidML): fix broken tests (4/5 failures) double uint64 int64 float16 are broken, float works
    {"Sin", sin},
    {"Sinh", sinh},
    {"Sqrt", sqrt},
    {"Sub", sub},
    {"Sum", sum},
    {"Tan", tan},
    {"Tanh", tanh},
    //{"Tile", tile,}, // TODO (PlaidML): OP WIP (11/11 failures)
    {"Where", where},  // TODO (PlaidML): fix broken tests (3/4 failures)
                       //{"Xor", logical_xor,}, // TODO (PlaidML): fix broken tests (2/2 failures)
};

std::map<std::string, _OpFunction> _kSupportedOps =
    {
        //{"ArgMax", _argmax,}, // TODO (PlaidML): fix broken tests (7/7 failures )
        //{"ArgMin", _argmin,},  // TODO (PlaidML): fix (4/5 failures)
        {"AveragePool", _average_pool},  // TODO (PlaidML): fix broken tests (1/4 failures)
        //{"Cast",_cast}, // TODO (PlaidML): OP WIP
        //{"Clip", _clip}, // TODO (PlaidML): fix broken tests (int8) incorrect docs in onnx has min max attributes not inputs
        {"Conv", _conv},                 // TODO (PlaidML): fix broken tests (6/17 failures)
        {"ConvInteger", _conv_integer},  // TODO (PlaidML): need to handle x_zero_point and w_zero_point inputs
        {"Concat", _concat},             // TODO (PlaidML): fix broken tests (3/12 failures) string type, dynamic shapes not handled
        //{"CumSum", _cumsum}, // TODO (PlaidML): fix broken tests
        {"Elu", _elu},
        //{"EyeLike",_eye_like}, // TODO (PlaidML): OP WIP
        {"Flatten", _flatten},
        {"HardSigmoid", _hard_sigmoid},
        {"LeakyRelu", _leaky_relu},
        {"LogSoftmax", _log_softmax},  // TODO (PlaidML): fix broken tests (2/7 failures)
        {"LpNormalization", _lp_normalization},
        //{"LRN",_lrn}, // TODO (PlaidML): fix broken tests (2/2 failures)
        {"MaxPool", _maxpool},  // TODO (PlaidML): fix broken tests (multiple outputs, attribute handling)
        //{"Mod",_mod}, // TODO (PlaidML): fix broken tests (6/15 failures)
        //{"OneHot",_one_hot}, // TODO (PlaidML): OP WIP
        {"ReduceMax", _reduce_max},
        {"ReduceMean", _reduce_mean},
        {"ReduceMin", _reduce_min},
        {"ReduceProd", _reduce_prod},
        {"ReduceSum", _reduce_sum},
        //{"ReverseSequence",_reverse_sequence}, // TODO (PlaidML): OP WIP
        {"Selu", _selu},
        //{"Slice",_slice},
        //{"Softmax",_softmax}, // TODO (PlaidML): fix broken tests (2/8 failures)
        //{"Split",_split}, // TODO (PlaidML): failing split OP WIP
        //{"Squeeze",_squeeze}, // TODO (PlaidML): fix broken tests (5/10 failures)(segfault)
        //{"ThresholdedRelu",_thresholded_relu}, // TODO (PlaidML): fix broken tests (new failure! op not registered )
        {"Transpose", _transpose},  // TODO (PlaidML): fix broken tests (8/17 failures)
        {"Unsqueeze", _unsqueeze},

};

std::vector<plaidml::edsl::Tensor> abs(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {plaidml::op::abs(X)};
}

std::vector<plaidml::edsl::Tensor> add(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A + B};
}

std::vector<plaidml::edsl::Tensor> logical_and(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A && B};
}

std::vector<plaidml::edsl::Tensor> logical_not(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  auto T = plaidml::edsl::Tensor(true);
  auto F = plaidml::edsl::Tensor(false);
  return {plaidml::edsl::select(I, F, T)};
}

std::vector<plaidml::edsl::Tensor> logical_or(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A || B};
}

std::vector<plaidml::edsl::Tensor> logical_xor(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A ^ B};
}

std::vector<plaidml::edsl::Tensor> asin(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::asin(A)};
}

std::vector<plaidml::edsl::Tensor> acos(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::acos(A)};
}

std::vector<plaidml::edsl::Tensor> atan(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::atan(A)};
}

std::vector<plaidml::edsl::Tensor> ceil(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::ceil(A)};
}

std::vector<plaidml::edsl::Tensor> cos(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::cos(A)};
}

std::vector<plaidml::edsl::Tensor> cosh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::cosh(A)};
}

std::vector<plaidml::edsl::Tensor> div(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A / B};
}

std::vector<plaidml::edsl::Tensor> equal(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A == B};
}

std::vector<plaidml::edsl::Tensor> erf(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  return {plaidml::edsl::erf(A)};
}

std::vector<plaidml::edsl::Tensor> exp(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::exp(A)};
}

std::vector<plaidml::edsl::Tensor> floor(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::floor(A)};
}

std::vector<plaidml::edsl::Tensor> greater(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A > B};
}

std::vector<plaidml::edsl::Tensor> identity(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  return {I};
}

std::vector<plaidml::edsl::Tensor> less(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A < B};
}

std::vector<plaidml::edsl::Tensor> log(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::log(A)};
}

std::vector<plaidml::edsl::Tensor> max(const std::vector<plaidml::edsl::Value>& args) {
  plaidml::edsl::Tensor result = args[0].as_tensor();
  if (args.size() == 1) return {result};
  for (size_t i = 1; i < args.size(); i++) {
    const auto& B = args[i].as_tensor();
    result = plaidml::op::maximum(result, B);
  }
  return {result};
}

// TODO (PlaidML): OP WIP
// std::vector<plaidml::edsl::Tensor> matmul(const std::vector<plaidml::edsl::Value>& args) {
//   auto A = args[0].as_tensor();
//   auto B = args[1].as_tensor();
//   //numpy style matmul
//   // if both arguments are 2-D multiply like normal matrices
//   // if dims > 2

//   return {plaidml::op::dot(A,B)};
// }

std::vector<plaidml::edsl::Tensor> mean(const std::vector<plaidml::edsl::Value>& args) {
  int num_tensors = (int)args.size();
  auto result = args[0].as_tensor();
  if (args.size() == 1) return {result};
  for (size_t i = 1; i < args.size(); i++) {
    result = result + args[i].as_tensor();
  }
  return {result / num_tensors};
}

std::vector<plaidml::edsl::Tensor> min(const std::vector<plaidml::edsl::Value>& args) {
  plaidml::edsl::Tensor result = args[0].as_tensor();
  if (args.size() == 1) return {result};
  for (size_t i = 1; i < args.size(); i++) {
    const auto& B = args[i].as_tensor();
    result = plaidml::op::minimum(result, B);
  }
  return {result};
}

std::vector<plaidml::edsl::Tensor> mul(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A * B};
}

std::vector<plaidml::edsl::Tensor> neg(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  return {-A};
}

// TODO (PlaidML): fix broken tests (double)
std::vector<plaidml::edsl::Tensor> pow(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  const auto& p = args[1].as_tensor();
  return {plaidml::edsl::pow(I, p)};
}

std::vector<plaidml::edsl::Tensor> prelu(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  const auto& slope = args[1].as_tensor();
  auto O = select(I < 0.0, slope * I, I);
  return {O};
}

std::vector<plaidml::edsl::Tensor> reciprocal(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {1 / X};
}

std::vector<plaidml::edsl::Tensor> relu(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {plaidml::op::relu(X)};
}

// TODO (PlaidML): OP WIP reshape
std::vector<plaidml::edsl::Tensor> reshape(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  //const auto& shape = args[1];//need to convert this into
  auto shape = plaidml::edsl::shape(I);
  return {plaidml::op::reshape(I, plaidml::edsl::make_tuple(shape))};
}

std::vector<plaidml::edsl::Tensor> sample_op(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {X};
}

// TODO (PlaidML): fix broken tests (11/11 failures)
std::vector<plaidml::edsl::Tensor> shape(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  return {plaidml::edsl::shape(I)};
}

// TODO (PlaidML): fix broken tests (4/5 failures)
std::vector<plaidml::edsl::Tensor> sign(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  auto Z = plaidml::edsl::Tensor(0.0);
  auto O = plaidml::edsl::Tensor(1.0);
  return {plaidml::edsl::select(I > 0, O, Z) + plaidml::edsl::select(I < 0, -1 * O, Z)};
}

std::vector<plaidml::edsl::Tensor> sigmoid(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {plaidml::op::sigmoid(X)};
}

std::vector<plaidml::edsl::Tensor> sin(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  return {plaidml::edsl::cos(M_PI_2 - A)};
}

std::vector<plaidml::edsl::Tensor> sinh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::sinh(A)};
}

std::vector<plaidml::edsl::Tensor> sqrt(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::sqrt(A)};
}

std::vector<plaidml::edsl::Tensor> sub(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A - B};
}

std::vector<plaidml::edsl::Tensor> sum(const std::vector<plaidml::edsl::Value>& args) {
  plaidml::edsl::Tensor result = args[0].as_tensor();
  if (args.size() == 1) return {result};
  for (size_t i = 1; i < args.size(); i++) {
    const auto& B = args[i].as_tensor();
    result = result + B;
  }
  return {result};
}

std::vector<plaidml::edsl::Tensor> tan(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::tan(A)};
}

std::vector<plaidml::edsl::Tensor> tanh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::tanh(A)};
}

std::vector<plaidml::edsl::Tensor> tile(const std::vector<plaidml::edsl::Value>& args) {
  const auto& inputs = args[0].as_tensor();
  //const auto& repeats = args[1].as_tensor();// TODO: need to convert this to vector<int>
  std::vector<int> reps_int;
  return {plaidml::op::tile(inputs, reps_int)};
}

std::vector<plaidml::edsl::Tensor> where(const std::vector<plaidml::edsl::Value>& args) {
  const auto& condition = args[0].as_tensor();
  const auto& X = args[1].as_tensor();
  const auto& Y = args[2].as_tensor();

  auto O = plaidml::edsl::select(condition, X, Y);

  return {O};
}

// TODO (PlaidML): fix broken tests (7/7 failures )
std::vector<plaidml::edsl::Tensor> _argmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", 0);
  auto keep_dims = pnode.get_int_attribute("keep_dims", 1);
  auto select_last_index = pnode.get_int_attribute("select_last_index", 0);
  ;

  if (select_last_index) {
    // TODO (PlaidML): handle select_last_index attribute here
    //temp statement added to bypass compile errors
    select_last_index = !select_last_index;
  } else {
    select_last_index = !select_last_index;
  }
  if (keep_dims == 1) {
    //keep the reduced dimensions
    // TODO (PlaidML): handle keep_dims attribute here
    //temp statement added to bypass compile errors
    keep_dims = 0;
  } else {
    //do not keep the reduced dimensions
    keep_dims = 0;
  }
  return {plaidml::op::argmax(A, plaidml::edsl::make_tuple(axis))};
}

// TODO (PlaidML): fix (4/5 failures)
std::vector<plaidml::edsl::Tensor> _argmin(  // TODO (PlaidML): merge argmax and argmin into one wrapper
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", 0);
  auto keep_dims = pnode.get_int_attribute("keep_dims", 1);
  auto select_last_index = pnode.get_int_attribute("select_last_index", 0);
  ;

  if (select_last_index) {
    // TODO (PlaidML): handle select_last_index attribute here
    //temp statement added to bypass compile errors
    select_last_index = !select_last_index;
  } else {
    select_last_index = !select_last_index;
  }
  if (keep_dims == 1) {
    //keep the reduced dimensions
    // TODO (PlaidML): handle keep_dims attribute here
    //temp statement added to bypass compile errors
    keep_dims = 0;
  } else {
    //do not keep the reduced dimensions
    keep_dims = 0;
  }

  return {plaidml::op::argmax(-A, plaidml::edsl::make_tuple(axis))};
}

// TODO (PlaidML): fix broken tests (1/4 failures)
// fix ceil_mode and count_include_mode handling
std::vector<plaidml::edsl::Tensor> _average_pool(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;
  //NOTSET, SAME_UPPER, SAME_LOWER or VALID
  const auto auto_pad = pnode.get_string_attribute("auto_pad", "NOTSET");
  if (auto_pad == "NOTSET") auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;  //default
  if (auto_pad == "SAME_UPPER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
  if (auto_pad == "SAME_LOWER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_LOWER;
  if (auto_pad == "VALID") auto_pad_mode = plaidml::op::AutoPadMode::VALID;
  if (auto_pad == "") auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;  //default

  auto ceil_mode = pnode.get_int_attribute("ceil_mode", 0);
  bool use_ceil = (ceil_mode == 1);
  // TODO (PlaidML): handle count_include_pad attribue
  //int count_include_pad = pnode.get_attribute("count_include_pad",(int)0);
  std::vector<int> kernel_shape;
  kernel_shape = pnode.get_vector_attribute("kernel_shape", kernel_shape);
  std::vector<int> pads;
  pads = pnode.get_vector_attribute("pads", pads);
  bool has_manual_pads = !pads.empty();
  auto input_order = plaidml::op::TensorLayout::NCX;
  std::vector<int> strides;
  //create default strides
  for (size_t i = 0; i < kernel_shape.size(); i++) {
    strides.push_back(1);
  }
  strides = pnode.get_vector_attribute("strides", strides);
  auto result = plaidml::op::pool(I,
                                  plaidml::op::PoolMode::AVG,
                                  kernel_shape,
                                  strides,
                                  auto_pad_mode,
                                  pads,
                                  input_order,
                                  has_manual_pads,
                                  use_ceil);

  return {result};
}

// TODO (PlaidML): OP WIP
std::vector<plaidml::edsl::Tensor> _cast(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  int to = onnx::TensorProto::UNDEFINED;
  to = pnode.get_int_attribute("to", to);
  auto plaidml_type = plaidml::DType::INVALID;

  switch (to) {
    case onnx::TensorProto::INT8:
      plaidml_type = plaidml::DType::INT8;
      break;
    case onnx::TensorProto::INT16:
      plaidml_type = plaidml::DType::INT16;
      break;
    case onnx::TensorProto::INT32:
      plaidml_type = plaidml::DType::INT32;
      break;
    case onnx::TensorProto::INT64:
      plaidml_type = plaidml::DType::INT64;
      break;
    case onnx::TensorProto::UINT8:
      plaidml_type = plaidml::DType::UINT8;
      break;
    case onnx::TensorProto::UINT16:
      plaidml_type = plaidml::DType::UINT16;
      break;
    case onnx::TensorProto::UINT32:
      plaidml_type = plaidml::DType::UINT32;
      break;
    case onnx::TensorProto::UINT64:
      plaidml_type = plaidml::DType::UINT64;
      break;
    case onnx::TensorProto::BOOL:
      plaidml_type = plaidml::DType::BOOLEAN;
      break;
    case onnx::TensorProto::FLOAT16:
      plaidml_type = plaidml::DType::FLOAT16;
      break;
    case onnx::TensorProto::FLOAT:
      plaidml_type = plaidml::DType::FLOAT32;
      break;
    case onnx::TensorProto::DOUBLE:
      plaidml_type = plaidml::DType::FLOAT64;
      break;
    case onnx::TensorProto::BFLOAT16:
      plaidml_type = plaidml::DType::BFLOAT16;
      break;
    case onnx::TensorProto::STRING:
    case onnx::TensorProto::COMPLEX64:
    case onnx::TensorProto::COMPLEX128:
      throw std::runtime_error("{PlaidML} ERROR: Asked to cast to unsupported data_type");
      break;

    default:
      throw std::runtime_error("{PlaidML} ERROR: Asked to cast to Unrecognized data_type");
  }

  printf("Cast %s to %s\n\n\n", to_string(I.dtype()), to_string(plaidml_type));
  return {plaidml::edsl::cast(I, plaidml_type)};
}

// // TODO (PlaidML): OP WIP clip
std::vector<plaidml::edsl::Tensor> _clip(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  // TODO (PlaidML): need to know type to set min max defaults does not work for int8 uint8 int64 uint64
  auto default_min = std::numeric_limits<float>::lowest();
  auto default_max = std::numeric_limits<float>::max();
  auto min_val = pnode.get_float_attribute("min", default_min);
  auto max_val = pnode.get_float_attribute("max", default_max);

  return {plaidml::op::clip(A, plaidml::edsl::Value(min_val).as_tensor(), plaidml::edsl::Value(max_val).as_tensor())};
}

// TODO (PlaidML): fix broken tests (3/12 failures)
std::vector<plaidml::edsl::Tensor> _concat(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  std::vector<plaidml::edsl::Tensor> tensors;
  for (size_t i = 0; i < inputs.size(); i++) {
    tensors.push_back(inputs[i].as_tensor());
  }
  if (tensors.empty()) {
    throw std::runtime_error("The concatenate op requires at least one input tensor");
  }

  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", 1);

  return {plaidml::op::concatenate(tensors, axis)};
}

// TODO (PlaidML): fix broken tests (6/17 failures)
// TODO (PlaidML): fix failing tests:
// ConvTest.Conv1D_Bias,
// ConvTest.Conv2D_1,
// ConvTest.Conv2D_Bias_1,
// ConvTest.Conv2D_Bias_2,
// ConvTest.Conv2D_group,
// ConvTest.ConvDimWithZero
std::vector<plaidml::edsl::Tensor> _conv(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto I = inputs[0].as_tensor();
  const auto K = inputs[1].as_tensor();
  //auto num_attributes = node.attribute_size();
  auto pnode = plaidml_ep::PlaidMLNode(node);

  //default auto_pad mode
  auto auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;

  const auto auto_pad = pnode.get_string_attribute("auto_pad", "NOTSET");
  if (auto_pad == "SAME_UPPER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
  if (auto_pad == "SAME_LOWER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_LOWER;
  if (auto_pad == "VALID") auto_pad_mode = plaidml::op::AutoPadMode::VALID;
  if (auto_pad == "NOTSET") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
  if (auto_pad == "") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;

  auto auto_group_mode = plaidml::op::AutoGroupMode::UNGROUPED;
  bool has_group_layout = false;
  auto group = pnode.get_int_attribute("group", 1);
  if (group != 1) {
    has_group_layout = true;
    auto_group_mode = plaidml::op::AutoGroupMode::EXPLICIT;
  }
  std::vector<int> dilations;
  dilations = pnode.get_vector_attribute("dilations", dilations);
  bool has_defined_dilations = !dilations.empty();
  std::vector<int> strides;
  strides = pnode.get_vector_attribute("strides", strides);
  bool has_defined_strides = !strides.empty();
  std::vector<int> pads;
  pads = pnode.get_vector_attribute("pads", pads);
  bool has_manual_pads = !pads.empty();
  if (has_manual_pads) auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;
  std::vector<int> kernel_shape;
  kernel_shape = pnode.get_vector_attribute("kernel_shape", kernel_shape);

  auto result = plaidml::op::convolution(I, K)
                    .input_layout(plaidml::op::TensorLayout::NCX)
                    .filter_layout(plaidml::op::TensorLayout::KCX)
                    .autopad_mode(auto_pad_mode)
                    .groups(group)
                    .autogroup_mode(auto_group_mode);

  if (has_defined_strides) result = result.strides(strides);
  if (has_defined_dilations) result = result.dilations(dilations);
  if (has_manual_pads) result = result.manual_padding(pads);
  if (has_group_layout) result = result.group_layout(plaidml::op::GroupLayout::IN_C);
  return {result};
}
// TODO (PlaidML): handle optional inputs:
//        from onnx docs:
// x_zero_point (optional) : T1
// Zero point tensor for input 'x'. It's optional and default value is 0.
// It's a scalar, which means a per-tensor/layer quantization.
// w_zero_point (optional) : T2
// Zero point tensor for input 'w'. It's optional and default value is 0.
// It could be a scalar or a 1-D tensor, which means a per-tensor/layer or
// per output channel quantization. If it's a 1-D tensor, its number of
// elements should be equal to the number of output channels (M)
std::vector<plaidml::edsl::Tensor> _conv_integer(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto I = inputs[0].as_tensor();
  const auto K = inputs[1].as_tensor();
  // plaidml::edsl::Tensor x_zero_point;
  // plaidml::edsl::Tensor w_zero_point;
  // if(inputs.size()>2)
  // {
  //     x_zero_point = inputs[2].as_tensor();
  // }
  // if(inputs.size()>3){
  //     w_zero_point = inputs[3].as_tensor();
  // }
  //auto num_attributes = node.attribute_size();
  auto pnode = plaidml_ep::PlaidMLNode(node);

  //default auto_pad mode
  auto auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;

  const auto auto_pad = pnode.get_string_attribute("auto_pad", "NOTSET");
  if (auto_pad == "SAME_UPPER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
  if (auto_pad == "SAME_LOWER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_LOWER;
  if (auto_pad == "VALID") auto_pad_mode = plaidml::op::AutoPadMode::VALID;
  if (auto_pad == "NOTSET") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
  if (auto_pad == "") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;

  auto auto_group_mode = plaidml::op::AutoGroupMode::UNGROUPED;
  bool has_group_layout = false;
  auto group = pnode.get_int_attribute("group", 1);
  if (group != 1) {
    has_group_layout = true;
    auto_group_mode = plaidml::op::AutoGroupMode::EXPLICIT;
  }
  std::vector<int> dilations;
  dilations = pnode.get_vector_attribute("dilations", dilations);
  bool has_defined_dilations = !dilations.empty();
  std::vector<int> strides;
  strides = pnode.get_vector_attribute("strides", strides);
  bool has_defined_strides = !strides.empty();
  std::vector<int> pads;
  pads = pnode.get_vector_attribute("pads", pads);
  bool has_manual_pads = !pads.empty();
  if (has_manual_pads) auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;
  std::vector<int> kernel_shape;
  kernel_shape = pnode.get_vector_attribute("kernel_shape", kernel_shape);

  auto result = plaidml::op::convolution(I, K)
                    .input_layout(plaidml::op::TensorLayout::NCX)
                    .filter_layout(plaidml::op::TensorLayout::KCX)
                    .autopad_mode(auto_pad_mode)
                    .groups(group)
                    .autogroup_mode(auto_group_mode);

  if (has_defined_strides) result = result.strides(strides);
  if (has_defined_dilations) result = result.dilations(dilations);
  if (has_manual_pads) result = result.manual_padding(pads);
  if (has_group_layout) result = result.group_layout(plaidml::op::GroupLayout::IN_C);
  return {result};
}

// TODO (PlaidML): fix broken tests
std::vector<plaidml::edsl::Tensor> _cumsum(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);

  //const auto& axis = inputs[0].as_tensor();
  // TODO (PlaidML): need to turn tensor into integer
  auto exclusive = pnode.get_int_attribute("exclusive", 0);
  auto reverse = pnode.get_int_attribute("reverse", 0);
  auto axis = inputs[0].as_int();
  //auto int_axis = 0;  ///axis tensor can be int32 or int64

  if (reverse == 1) {
    // TODO (PlaidML): handle reverse
    //perform the sum in reverse direction
    reverse = 0;
  }
  if (exclusive == 1) {
    // TODO (PlaidML): handle exclusive
    //If set to 1 will return exclusive sum in which the top element
    //is not included. In other terms, if set to 1, the j-th output
    //element would be the sum of the first (j-1) elements. Otherwise,
    //it would be the sum of the first j elements
    exclusive = 0;
  }

  return {plaidml::op::cumsum(I, axis)};  //cumsum(Tensor,int)
}

std::vector<plaidml::edsl::Tensor> _elu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto alpha = pnode.get_float_attribute("alpha", 1.0);

  return {plaidml::edsl::select(I >= 0, I, alpha * (plaidml::edsl::exp(I) - 1))};
}

// TODO (PlaidML): OP WIP
std::vector<plaidml::edsl::Tensor> _eye_like(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto I = inputs[0].as_tensor();
  // must be 2D tensor
  if (I.rank() > 2) throw std::runtime_error("{PlaidML} ERROR: EyeLike given tensor of rank !=2 \n");
  auto pnode = plaidml_ep::PlaidMLNode(node);
  // TODO (PlaidML): handle dtype
  //int dtype = pnode.get_attribute("dtype",(int)0); //optional attribue
  bool has_dtype = pnode.has_attribute("dtype");
  auto k = pnode.get_int_attribute("k", 0);
  ;

  if (!has_dtype && k == 0) {
    return {plaidml::edsl::ident(I)};
  }

  //create identity matrix from input tensor
  plaidml::edsl::TensorDim X, Y;
  plaidml::edsl::TensorIndex i, j;
  I.bind_dims(X, Y);
  auto O = plaidml::edsl::TensorOutput(X, Y);

  //auto O = plaidml::edsl::Placeholder(type, shape);
  auto I_zero = plaidml::edsl::Tensor{0};
  auto I_one = plaidml::edsl::Tensor{1};
  O(i, j) = I_zero(i);
  O(i, i + k) = I_one(i);
  //O(x,x+k) = I(x,x+k) + I(x,x+k);
  //O = O - I;
  // O = plaidml::edsl::select(O>0,plaidml::edsl::Tensor{1},O);
  // TODO (PlaidML): maybe select should allow indices
  // O = plaidml::edsl::select(y==x+k,0,1) should get the required identity ?

  // TODO (PlaidML): handle dtye attribute
  //if(has_dtype){cast the tensor into given dtype  }

  return {O};
}

// TODO (PlaidML): fix broken tests (4/6 failures)
std::vector<plaidml::edsl::Tensor> _flatten(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto X = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", 1);
  if (axis < 0) axis = (int)X.rank() + axis;
  if (axis > (int)X.rank() || axis < 0) {
    throw std::runtime_error("{PlaidML ERROR} invalid axis attribute in flatten \n");
  }
  std::vector<plaidml::edsl::TensorDim> X_dims(X.rank());
  X.bind_dims(X_dims);
  if (X_dims.empty()) {
    return {X};
  }
  plaidml::edsl::TensorDim product(1);
  plaidml::edsl::TensorDim first_dim(1);
  if (axis == 0) {
    for (size_t i = 0; i < X.rank(); i++) {
      product = product * X_dims[i];
    }
    return {plaidml::edsl::reshape(X, {first_dim, product})};
  }
  for (size_t i = 0; i < static_cast<size_t>(axis); i++) {
    first_dim = first_dim * X_dims[i];
  }
  for (size_t i = axis; i < X.rank(); i++) {
    product = product * X_dims[i];
  }
  return {plaidml::edsl::reshape(X, {first_dim, product})};
}

std::vector<plaidml::edsl::Tensor> _hard_sigmoid(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto alpha = pnode.get_float_attribute("alpha", 0.2);
  auto beta = pnode.get_float_attribute("beta", 0.5);

  auto O = plaidml::edsl::Tensor(1.0);
  auto Z = plaidml::edsl::Tensor(0.0);

  auto result = alpha * I + beta;
  result = plaidml::edsl::select(result > O, O, result);
  result = plaidml::edsl::select(result < Z, Z, result);

  return {result};
}

// TODO (PlaidML): fix broken tests (2/7 failures)
std::vector<plaidml::edsl::Tensor> _log_softmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);

  auto axis = pnode.get_int_attribute("axis", 1);

  auto O = plaidml::op::softmax(A, axis);
  return {plaidml::edsl::log(O)};
}

std::vector<plaidml::edsl::Tensor> _lp_normalization(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", -1);
  auto p = pnode.get_int_attribute("p", 2);

  std::vector<int64_t> axes;
  size_t last_axis = I.rank() - 1;

  if (axis == -1) {
    axes.push_back(last_axis);
  } else {
    axes.push_back(size_t(axis));
  }

  if (p != 1 && p != 2) {
    throw std::runtime_error("LpNormalize only supports p=1 and p=2");
  }

  if (p == 1) {
    //l1norm
    auto X = plaidml::op::sum(I, plaidml::edsl::make_tuple(axes), 1);
    auto N = plaidml::op::maximum(X, plaidml::edsl::Tensor{0.001});
    return {I / N};
  } else {
    //l2norm
    auto N = plaidml::op::l2norm(I, {axes});
    return {I / N};
  }
}

std::vector<plaidml::edsl::Tensor> _leaky_relu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto alpha = pnode.get_float_attribute("alpha", 0.01);

  return {plaidml::op::relu(I).alpha(plaidml::edsl::Tensor{alpha})};
}

// TODO (PlaidML): fix broken tests (2/2 failures)
std::vector<plaidml::edsl::Tensor> _lrn(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);

  auto alpha = pnode.get_float_attribute("alpha", 0.0001);
  auto beta = pnode.get_float_attribute("beta", 0.75);
  auto bias = pnode.get_float_attribute("bias", 1.0);
  auto size = pnode.get_int_attribute("size", 1.0);
  //if size is set to 1 instead of 5 (as requested by the test) second test (LRNTest.LRN_2) passes

  return {plaidml::op::lrn(A, {static_cast<int64_t>(size)}).alpha(alpha).beta(beta).epsilon(bias).axes({1})};
}

// TODO (PlaidML): fix broken tests
// TODO (PlaidML): Indices  (optional) output not handled
//from onnx docs: Indices (optional, non-differentiable) : I
//      Indices tensor from max pooling across the input tensor.
//      The dimensions of indices are the same as output tensor.
//      The values in indices of are the indices of the selected values during pooling.
//      The indices are computed as flatten 1-D tensor, and the indices do not consider padding.
//      So the values in indices are in [0, N x C x D1 x ... x Dn).
// TODO (PlaidML): attribute "dilations" not handled
std::vector<plaidml::edsl::Tensor> _maxpool(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);

  auto auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;
  const auto auto_pad = pnode.get_string_attribute("auto_pad", "NOTSET");
  if (auto_pad == "NOTSET") auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;  //default
  if (auto_pad == "SAME_UPPER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
  if (auto_pad == "SAME_LOWER") auto_pad_mode = plaidml::op::AutoPadMode::SAME_LOWER;
  if (auto_pad == "VALID") auto_pad_mode = plaidml::op::AutoPadMode::VALID;
  if (auto_pad == "") auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;

  //Whether to use ceil or floor (default) to compute the output shape.
  auto ceil_mode = pnode.get_int_attribute("ceil_mode", 0);
  bool use_ceil = (ceil_mode == 1);
  //std::vector<int> dilations = pnode.get_attribute("dilations");
  //if not present default is 1 along each spacial axis
  //bool has_defined_dilations = !dilations.empty();
  std::vector<int> kernel_shape;
  kernel_shape = pnode.get_vector_attribute("kernel_shape", kernel_shape);
  //If not present, should be inferred from input W
  std::vector<int> pads;
  pads = pnode.get_vector_attribute("pads", pads);
  //If not present, the padding defaults to
  //0 along start and end of each spatial axis.
  bool has_manual_pads = !pads.empty();
  int storage_order = pnode.get_int_attribute("storage_order", 0);
  //The storage order of the tensor. 0 is row major, and 1 is column major.
  auto input_order = plaidml::op::TensorLayout::NCX;
  if (storage_order == 0) input_order = plaidml::op::TensorLayout::NCX;
  if (storage_order == 1) input_order = plaidml::op::TensorLayout::NXC;

  std::vector<int> strides;
  strides = pnode.get_vector_attribute("strides", strides);
  //If not present, the stride defaults is 1 along each spatial axis
  if (strides.empty()) {
    for (size_t i = 0; i < kernel_shape.size(); i++) {
      strides.push_back(1);
    }
  }
  //bool has_defined_strides = !strides.empty();

  auto result = plaidml::op::pool(I,
                                  plaidml::op::PoolMode::MAX,
                                  kernel_shape,
                                  strides,
                                  auto_pad_mode,
                                  pads,
                                  input_order,
                                  has_manual_pads,
                                  use_ceil);

  return {result};
}

// TODO (PlaidML): fix broken tests (6/15 failures)
std::vector<plaidml::edsl::Tensor> _mod(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  const auto& B = inputs[1].as_tensor();

  // TODO (PlaidML): if input is float fmod must be 1
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto fmod = pnode.get_int_attribute("fmod", 0);

  auto result = A % B;  // TODO (PlaidML): need to handle fmod

  if (fmod == 1) {
    //find floating point remainder
    return {plaidml::edsl::select(B > 0, result, -result)};
  } else {
    return {result};
  }
}

// TODO (PlaidML): OP WIP
std::vector<plaidml::edsl::Tensor> _one_hot(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& indices = inputs[0].as_tensor();
  //const auto& depth = inputs[1].as_tensor();// TODO (PlaidML): need to convert this to int
  int fake_depth = 10;
  //const auto& values = inputs[2].as_tensor();//on value off value

  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", -1);

  std::vector<plaidml::edsl::TensorDim> I_dims(indices.rank());
  indices.bind_dims(I_dims);
  std::vector<plaidml::edsl::TensorDim> O_dims(indices.rank() + 1);

  if (axis < 0) axis = indices.rank() + axis;
  size_t j = 0;
  for (size_t i = 0; i < O_dims.size(); i++) {
    if (i == (size_t)axis) {
      O_dims[i] = O_dims[i] * fake_depth;
    } else {
      O_dims[i] = I_dims[j];
      j++;
    }
  }

  plaidml::edsl::TensorIndex v;
  plaidml::edsl::Tensor O = plaidml::edsl::TensorOutput(O_dims);
  std::vector<plaidml::edsl::TensorIndex> O_idxs(indices.rank() + 1);
  std::vector<plaidml::edsl::TensorIndex> I_idxs(indices.rank());
  plaidml::edsl::Tensor count = plaidml::edsl::index(O_dims, axis);
  plaidml::edsl::TensorIndex c;
  O(O_idxs) = indices(I_idxs) == count(c);
  //O = plaidml::edsl::select(O, 0,1);
  return {O};
}

// TODO (PlaidML): fix broken tests (2/9 failures)
std::vector<plaidml::edsl::Tensor> _reduce_max(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> att_axes;
  att_axes = pnode.get_vector_attribute("axes", att_axes);
  std::vector<int> axes;
  axes.assign(att_axes.begin(), att_axes.end());
  if (axes.empty()) {
    for (size_t i = 0; i < I.rank(); i++) {
      axes.push_back(i);
    }
  }
  auto att_keep_dims = pnode.get_int_attribute("keepdims", 1);
  bool keep_dims = true;
  if (att_keep_dims == 0) keep_dims = false;
  return {plaidml::op::max(I, plaidml::edsl::make_tuple(axes), keep_dims)};
}

// TODO (PlaidML): fix broken tests (2/8 failures)
std::vector<plaidml::edsl::Tensor> _reduce_mean(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> att_axes;
  att_axes = pnode.get_vector_attribute("axes", att_axes);
  std::vector<int> axes;
  axes.assign(att_axes.begin(), att_axes.end());
  if (axes.empty()) {
    for (size_t i = 0; i < I.rank(); i++) {
      axes.push_back(i);
    }
  }
  auto att_keep_dims = pnode.get_int_attribute("keepdims", 1);
  bool keep_dims = true;
  if (att_keep_dims == 0) keep_dims = false;
  return {plaidml::op::mean(I, plaidml::edsl::make_tuple(axes), keep_dims)};
}

// TODO (PlaidML): fix broken tests (4/9 failures)
std::vector<plaidml::edsl::Tensor> _reduce_min(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> att_axes;
  att_axes = pnode.get_vector_attribute("axes", att_axes);
  std::vector<int> axes;
  axes.assign(att_axes.begin(), att_axes.end());
  if (axes.empty()) {
    for (size_t i = 0; i < I.rank(); i++) {
      axes.push_back(i);
    }
  }
  auto att_keep_dims = pnode.get_int_attribute("keepdims", 1);
  bool keep_dims = true;
  if (att_keep_dims == 0) keep_dims = false;
  return {plaidml::op::min(I, plaidml::edsl::make_tuple(axes), keep_dims)};
}

// TODO (PlaidML): fix broken tests (2/8 failures)
std::vector<plaidml::edsl::Tensor> _reduce_prod(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> att_axes;
  att_axes = pnode.get_vector_attribute("axes", att_axes);
  std::vector<int> axes;
  axes.assign(att_axes.begin(), att_axes.end());
  if (axes.empty()) {
    for (size_t i = 0; i < I.rank(); i++) {
      axes.push_back(i);
    }
  }
  auto att_keep_dims = pnode.get_int_attribute("keepdims", 1);
  bool keep_dims = true;
  if (att_keep_dims == 0) keep_dims = false;
  return {plaidml::op::prod(I, plaidml::edsl::make_tuple(axes), keep_dims)};
}

// TODO (PlaidML): fix broken tests (2/19 failures)
std::vector<plaidml::edsl::Tensor> _reduce_sum(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> att_axes;
  att_axes = pnode.get_vector_attribute("axes", att_axes);
  std::vector<int> axes;
  axes.assign(att_axes.begin(), att_axes.end());
  if (axes.empty()) {
    for (size_t i = 0; i < I.rank(); i++) {
      axes.push_back(i);
    }
  }
  auto att_keep_dims = pnode.get_int_attribute("keepdims", 1);
  bool keep_dims = true;
  if (att_keep_dims == 0) keep_dims = false;
  return {plaidml::op::sum(I, plaidml::edsl::make_tuple(axes), keep_dims)};
}

// TODO (PlaidML): OP WIP
std::vector<plaidml::edsl::Tensor> _reverse_sequence(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  //const auto& sequence_lens = inputs[1].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto batch_axis = pnode.get_int_attribute("batch_axis", 1);
  auto time_axis = pnode.get_int_attribute("time_axis", 0);

  if (I.rank() < 2) {
    throw std::runtime_error("{PlaidML ERROR} ReverseSequense reqiured an input Tensor of rank >= 2");
  }
  if (time_axis != 0 && time_axis != 1) {
    throw std::runtime_error("{PlaidML ERROR} ReverseSequense was given invalid time axis");
  }
  if (batch_axis != 0 && batch_axis != 1) {
    throw std::runtime_error("{PlaidML ERROR} ReverseSequense was given invalid batch axis");
  }
  auto O = I;

  std::vector<plaidml::edsl::TensorIndex> I_idxs(I.rank());
  std::vector<plaidml::edsl::TensorDim> I_dims;
  std::vector<plaidml::edsl::TensorIndex> I_revidxs(I.rank());
  I.bind_dims(I_dims);
  std::vector<plaidml::edsl::TensorIndex> seq_lens_index(I.rank());
  //plaidml::edsl::TensorIndex seq_index;
  //for each slice on the batch axis
  for (size_t i = 0; i < I.rank(); i++) {
    if (i == static_cast<size_t>(batch_axis))
      I_revidxs[i] = I_idxs[i];
    else
      I_revidxs[i] = I_idxs[i] + seq_lens_index[i];
  }

  //I_revidxs[batch_axis] = I_idxs[batch_axis];

  O(I_idxs) = I(I_revidxs);
  //O.add_constraint(seq_index<I_dims);
  for (size_t i = 0; i < I.rank(); i++) {
    if (i != static_cast<size_t>(batch_axis)) O.add_constraint(seq_lens_index[i] < I_dims[i]);
  }
  return {O};
}

std::vector<plaidml::edsl::Tensor> _selu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto alpha = pnode.get_float_attribute("alpha", 1.67326);
  auto gamma = pnode.get_float_attribute("gamma", 1.0507);
  return {gamma * plaidml::edsl::select(I > 0, I, alpha * (plaidml::edsl::exp(I) - 1))};
}

std::vector<plaidml::edsl::Tensor> _slice(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> starts;
  std::vector<int> ends;
  std::vector<int> axes;

  starts = pnode.get_vector_attribute("starts", starts);
  ends = pnode.get_vector_attribute("ends", ends);
  axes = pnode.get_vector_attribute("axes", axes);
  //const auto starts = args[1].as_tensor();
  //const auto ends = args[2].as_tensor();

  //const axes = args[3].as_tensor();//optional input
  //const steps = args[4].as_tensor();//optional input

  return {plaidml::op::slice(I).add_dims(starts).add_dims(ends)};
}

// TODO (PlaidML): fix broken tests (2/8 failures)
std::vector<plaidml::edsl::Tensor> _softmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", 1);
  return {plaidml::op::softmax(A, axis)};
}

// TODO (PlaidML): failing split OP WIP
std::vector<plaidml::edsl::Tensor> _split(  // TODO (PlaidML): need to handle multiple outputs in makeprogram
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto axis = pnode.get_int_attribute("axis", 0);
  std::vector<int> splits;
  splits = pnode.get_vector_attribute("split", splits);
  std::vector<plaidml::edsl::Tensor> I_split;

  auto ndims = I.rank();
  std::vector<plaidml::edsl::TensorDim> I_dims(ndims);
  I.bind_dims(I_dims);
  std::vector<plaidml::edsl::TensorDim> O_dims(I_dims);
  std::vector<plaidml::edsl::TensorIndex> I_idxs(ndims);
  std::vector<plaidml::edsl::TensorIndex> O_idxs(I_idxs);

  //set up dims for output

  if (splits.size() == 0) {
    return {I};
  }
  int64_t prev_split = 0;
  for (auto split : splits) {
    auto O = plaidml::edsl::TensorOutput(O_dims);
    I_idxs[axis] = I_idxs[axis] + (int64_t)prev_split;
    O(O_idxs) = I(I_idxs);
    O.add_constraint(I_idxs[axis] < (int64_t)split);
    I_split.push_back(O);
    prev_split = (int64_t)split;
  }

  return {I_split};
}

// TODO (PlaidML): fix broken tests (5/10 failures)(segfault)
std::vector<plaidml::edsl::Tensor> _squeeze(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> axes;
  axes = pnode.get_vector_attribute("axes", axes);
  //If axes is not provided, all the single dimensions will be removed from the shape.
  if (axes.empty()) {
    for (size_t i = 0; i < I.rank(); i++) {
      axes.push_back(i);
    }
  }
  // unsorted axis segfaults
  //std::sort(axes.begin(),axes.end(),[](int a, int b) {return a < b; });
  return {plaidml::op::squeeze(I, axes)};
}

// TODO (PlaidML): fix broken tests (new failure! op not registered )
std::vector<plaidml::edsl::Tensor> _thresholded_relu(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& I = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  auto alpha = pnode.get_float_attribute("alpha", 1.0);
  return {plaidml::op::relu(I).threshold(static_cast<double>(alpha))};
}

// TODO (PlaidML): fix broken tests (8/17 failures)
std::vector<plaidml::edsl::Tensor> _transpose(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> att_axes;
  att_axes = pnode.get_vector_attribute("perm", att_axes);
  std::vector<int> axes;
  axes.assign(att_axes.begin(), att_axes.end());
  bool no_perm = axes.empty();

  if (no_perm)
    return {plaidml::op::transpose(A)};
  else
    return {plaidml::op::transpose(A, plaidml::edsl::make_tuple(axes))};
}

std::vector<plaidml::edsl::Tensor> _unsqueeze(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  const auto& A = inputs[0].as_tensor();
  auto pnode = plaidml_ep::PlaidMLNode(node);
  std::vector<int> att_axes;
  att_axes = pnode.get_vector_attribute("axes", att_axes);
  std::vector<int64_t> axes;
  axes.assign(att_axes.begin(), att_axes.end());

  return {plaidml::op::unsqueeze(A, {axes})};
}

std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
  //check if this op is supported
  const std::vector<plaidml::edsl::Value*> args;

  printf("Executing {PlaidML OP}: %s\n\n", node.op_type().c_str());

  auto op_it = plaidml_ep::kSupportedOps.find(node.op_type());
  auto _op_it = plaidml_ep::_kSupportedOps.find(node.op_type());
  if (op_it == plaidml_ep::kSupportedOps.end() && _op_it == plaidml_ep::_kSupportedOps.end()) {
    throw std::runtime_error("{PlaidML ERROR} Asked to executed unsupported op " + node.op_type());
  }
  if (op_it != plaidml_ep::kSupportedOps.end()) {
    return op_it->second(inputs);
  } else {
    return _op_it->second(node, inputs);
  }
}

// TODO (PlaidML): adding this temp solution to bypass test failures
// will be scraped once kernel registry is implemented
bool check_op_support(std::string op_name) {
  auto op_it = plaidml_ep::kSupportedOps.find(op_name);
  auto _op_it = plaidml_ep::_kSupportedOps.find(op_name);
  if (op_it == plaidml_ep::kSupportedOps.end() && _op_it == plaidml_ep::_kSupportedOps.end()) {
    return false;
  } else {
    return true;
  }
}

bool check_attribute_support(const ONNX_NAMESPACE::NodeProto& node) {
  auto pnode = plaidml_ep::PlaidMLNode(node);

  if (node.op_type() == "AveragePool") {
    if (pnode.get_int_attribute("ceil_mode", 0) == 1) return false;
  }
  if (node.op_type() == "MaxPool") {
    //indices output not handled: maxpool only produces one output
    if (node.output_size() > 1) return false;
    //dilations attribute not handled
    if (pnode.has_attribute("dilations")) return false;
  }
  if (node.op_type() == "ConvInteger") {
    // optional inputs (x_zero_point, w_zero_point) not handled
    if (node.input_size() > 2) return false;
  }
  return true;
}

}  // namespace plaidml_ep
}  // namespace onnxruntime
