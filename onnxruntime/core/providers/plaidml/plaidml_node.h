#include "plaidml_utils.h"
#include "core/graph/model.h"
#include "plaidml_ops.h"

namespace onnxruntime {
namespace plaidml_ep {

// TODO (PlaidML): PlaidMLNode should inherit nodeproto ?
class PlaidMLNode {
 public:
  explicit PlaidMLNode(const ONNX_NAMESPACE::NodeProto& node);

  //attribute getters
  bool has_attribute(const std::string& name);
  // TODO (PlaidML): change this to template/overloaded funcs with trailing return type to avoid casting problems
  // this is a temp fix
  int64_t get_int_attribute(const std::string& name, int64_t default_value);
  float get_float_attribute(const std::string& name, float default_value);
  std::string get_string_attribute(const std::string& name, std::string default_value);
  std::vector<int> get_vector_attribute(const std::string& name, std::vector<int> default_value);

 private:
  const ONNX_NAMESPACE::NodeProto& _node;
};

class PlaidMLProgram {
 public:
  // TODO (PlaidML): We might instead implement this on an ONNX ModelProto instead of an ONNX RT Node.
  //      This might have benefits for reuse in a non-RT ONNX context?
  // TODO (PlaidML):In general, inputs are a mix of initializers and input data; this currently assumes they're all the latter
  // TODO (PlaidML):work out if deprecated op is being used and handle it
  // TODO (PlaidML):A node_input's Shape can be nullptr (i.e. if the input isn't a tensor) and we need to handle that case
  // TODO (PlaidML):This doesn't address symbolic shapes
  explicit PlaidMLProgram(const onnxruntime::Node* fused_node);
  std::vector<plaidml::edsl::Tensor> get_program_inputs();
  std::vector<plaidml::edsl::Tensor> get_program_outputs();
  plaidml::edsl::Value get_value_from_dictionary(const std::string& name);
  bool add_value_to_dictionary(const std::string& name, plaidml::edsl::Value val);

  std::shared_ptr<plaidml::edsl::Program> program;

 private:
  std::vector<plaidml::edsl::Tensor> _program_inputs;
  std::vector<plaidml::edsl::Tensor> _program_outputs;
  const onnxruntime::Node* _fused_node;
  /* fused node input dictionary contains the input tensor name and plaidml placeholder*/
  std::map<std::string, plaidml::edsl::Value> _value_dictionary;
  std::map<std::string, plaidml::edsl::Tensor> _output_dictionary;

  std::vector<int64_t> get_input_shape(const onnxruntime::NodeArg* node);
  bool add_fused_node_inputs_to_tensor_dictionary();
  std::vector<plaidml::edsl::Value> get_local_input_values(const onnxruntime::Node* node);
};

}  // namespace plaidml_ep
}  // namespace onnxruntime