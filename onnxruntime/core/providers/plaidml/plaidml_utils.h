
#pragma once

#include "core/graph/onnx_protobuf.h"

namespace plaidml {
namespace edsl {
class Program;
class Tensor;
class Value;
}  // namespace edsl
enum class DType;
}  // namespace plaidml

namespace onnxruntime {
namespace plaidml_ep {

// TODO (PlaidML): add error type enumeration

//-----------------------------------------------helper functions-----------------------------
plaidml::DType get_precision(ONNX_NAMESPACE::DataType onnx_type);
//-----------------------------------------------------------------------------------------------

}  // namespace plaidml_ep
}  // namespace onnxruntime
