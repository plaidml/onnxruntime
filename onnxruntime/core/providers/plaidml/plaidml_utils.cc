#include "plaidml_utils.h"
#include "plaidml/edsl/edsl.h"

namespace onnxruntime {
namespace plaidml_ep {

// TODO (PlaidML): add error type enumeration

//-----------------------------------------------helper functions-----------------------------
plaidml::DType get_precision(
    ONNX_NAMESPACE::DataType onnx_type) {
  if (*onnx_type == "double" || *onnx_type == "tensor(double)") {
    return plaidml::DType::FLOAT64;
  } else if (*onnx_type == "float" || *onnx_type == "tensor(float)") {
    return plaidml::DType::FLOAT32;
  } else if (*onnx_type == "float16" || *onnx_type == "tensor(float16)") {
    return plaidml::DType::FLOAT16;
  } else if (*onnx_type == "int32" || *onnx_type == "tensor(int32)") {
    return plaidml::DType::INT32;
  } else if (*onnx_type == "int64" || *onnx_type == "tensor(int64)") {
    return plaidml::DType::INT64;
  } else if (*onnx_type == "int16" || *onnx_type == "tensor(int16)") {
    return plaidml::DType::INT16;
  } else if (*onnx_type == "int8" || *onnx_type == "tensor(int8)") {
    return plaidml::DType::INT8;
  } else if (*onnx_type == "uint64" || *onnx_type == "tensor(uint64)") {
    return plaidml::DType::UINT64;
  } else if (*onnx_type == "uint32" || *onnx_type == "tensor(uint32)") {
    return plaidml::DType::UINT32;
  } else if (*onnx_type == "uint16" || *onnx_type == "tensor(uint16)") {
    return plaidml::DType::UINT16;
  } else if (*onnx_type == "uint8" || *onnx_type == "tensor(uint8)") {
    return plaidml::DType::UINT8;
  } else if (*onnx_type == "bool" || *onnx_type == "tensor(bool)") {
    return plaidml::DType::BOOLEAN;
  } else {
    throw std::runtime_error("{PlaidML ERROR} : invalid data type " + *onnx_type);
    return plaidml::DType::INVALID;
  }
}
//-----------------------------------------------------------------------------------------------

}  // namespace plaidml_ep
}  // namespace onnxruntime
