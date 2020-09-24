#include "plaidml_node.h"
#include "plaidml/edsl/edsl.h"
namespace onnxruntime {
namespace plaidml_ep {

PlaidMLNode::PlaidMLNode(const ONNX_NAMESPACE::NodeProto& node) : _node(node) {}

bool PlaidMLNode::has_attribute(const std::string& name) {
  auto num_attributes = _node.attribute_size();
  if (num_attributes > 0) {
    auto attributes = _node.attribute();
    for (auto attribute : attributes) {
      if (attribute.name() == name) {
        return true;
      }
    }
  }
  return false;
}

int64_t PlaidMLNode::get_int_attribute(const std::string& name, int64_t default_value) {
  auto num_attributes = _node.attribute_size();
  if (num_attributes > 0) {
    auto attributes = _node.attribute();
    for (auto attribute : attributes) {
      if (attribute.name() == name) {
        return attribute.i();
      }
    }
  }
  return default_value;
}

float PlaidMLNode::get_float_attribute(const std::string& name, float default_value) {
  auto num_attributes = _node.attribute_size();
  if (num_attributes > 0) {
    auto attributes = _node.attribute();
    for (auto attribute : attributes) {
      if (attribute.name() == name) {
        return attribute.f();
      }
    }
  }
  return default_value;
}

std::string PlaidMLNode::get_string_attribute(const std::string& name, std::string default_value) {
  auto num_attributes = _node.attribute_size();
  if (num_attributes > 0) {
    auto attributes = _node.attribute();
    for (auto attribute : attributes) {
      if (attribute.name() == name) {
        return attribute.s();
      }
    }
  }
  return default_value;
}

std::vector<int> PlaidMLNode::get_vector_attribute(const std::string& name, std::vector<int> default_value) {
  auto num_attributes = _node.attribute_size();
  std::vector<int> ret_val;
  if (num_attributes > 0) {
    auto attributes = _node.attribute();
    for (auto attribute : attributes) {
      if (attribute.name() == name) {
        auto att_ints = attribute.ints();
        ret_val.assign(att_ints.begin(), att_ints.end());
        return ret_val;
      }
    }
  }
  if (ret_val.empty()) return default_value;
  return ret_val;
}

PlaidMLProgram::PlaidMLProgram(const onnxruntime::Node* fused_node) : _fused_node(fused_node) {
  /* Add the inputs to tensor dictionary and to plaidml input tensor list */
  // TODO (PlaidML): add error checks
  add_fused_node_inputs_to_tensor_dictionary();

  for (const auto& node : _fused_node->GetFunctionBody()->Body().Nodes()) {
    std::vector<plaidml::edsl::Value> local_input_tensors = get_local_input_values(&node);

    ONNX_NAMESPACE::NodeProto node_proto;
    node.ToProto(node_proto);
    auto local_output_tensors = plaidml_ep::MakePlaidMLOp(node_proto, local_input_tensors);
    // Iterate over output tensors and names in tandem
    if (local_output_tensors.empty()) {
      throw std::runtime_error("{PlaidML ERROR} op produced empty output");
    }
    if (local_output_tensors.size() != node.OutputDefs().size()) {
      throw std::runtime_error("{PlaidML ERROR} op produced inconsistent number of outputs expected " +
                               std::to_string(node.OutputDefs().size()) + ", got " +
                               std::to_string(local_output_tensors.size()));
    }
    auto output_tensor_it = local_output_tensors.begin();
    for (const auto& local_output : node.OutputDefs()) {
      if (add_value_to_dictionary(local_output->Name(), plaidml::edsl::Value(*output_tensor_it)) == false) {
        throw std::runtime_error("Unexpected duplicate name in fused node while adding outputs (possibly intermediate) [TODO better error handling]");
      }
      output_tensor_it++;
    }
  }

  // Lookup outputs from `tensors` dict, use those to call edsl::ProgramBuilder
  for (const auto& node_output : _fused_node->OutputDefs()) {
    auto output = get_value_from_dictionary(node_output->Name());
    _program_outputs.push_back(output.as_tensor());
  }

  program = std::make_shared<plaidml::edsl::Program>(plaidml::edsl::ProgramBuilder(fused_node->Name(), _program_outputs).compile());
}

/* returns the input shapes of a node from the tensor dictionary */
std::vector<plaidml::edsl::Value> PlaidMLProgram::get_local_input_values(const onnxruntime::Node* node) {
  std::vector<plaidml::edsl::Value> local_input_tensors;
  for (const auto& local_input : node->InputDefs()) {
    try {
      if (local_input->Name() != "") {  //TODO (PlaidML): fix this
        auto input = get_value_from_dictionary(local_input->Name());

        local_input_tensors.push_back(plaidml::edsl::Value(input));
      }
    } catch (const std::out_of_range& e) {
      throw std::runtime_error("Could not find expected tensor " + local_input->Name() + " [TODO better error handling]");
    }
  }
  return local_input_tensors;
}

/* For each input, look up shape (or at least rank) and construct a (placeholder) tensor accordingly;
   add this to the `tensors` dict */
bool PlaidMLProgram::add_fused_node_inputs_to_tensor_dictionary() {
  for (const auto& node_input : _fused_node->InputDefs()) {
    // TODO (PlaidML): A node_input's Shape can be nullptr (i.e. if the input isn't a tensor) and we need to handle that case
    // TODO (PlaidML): This doesn't address symbolic shapes
    std::vector<int64_t> shape = get_input_shape(node_input);
    auto type = plaidml_ep::get_precision(node_input->Type());
    auto input_placeholder = plaidml::edsl::Placeholder(type, shape);
    if (!add_value_to_dictionary(node_input->Name(), plaidml::edsl::Value(input_placeholder))) {
      throw std::runtime_error("{PlaidML ERROR} Unexpected duplicate name in fused node while adding inputs");
    }
    _program_inputs.push_back(input_placeholder);
  }
  return true;
}

plaidml::edsl::Value PlaidMLProgram::get_value_from_dictionary(const std::string& name) {
  plaidml::edsl::Value ret_val;
  try {
    ret_val = _value_dictionary.at(name);
  } catch (const std::out_of_range& oor) {
    std::cerr << "{PlaidML ERROR} Out of Range error: " << oor.what() << '\n';
  }
  return ret_val;
}

std::vector<plaidml::edsl::Tensor> PlaidMLProgram::get_program_inputs() {
  return _program_inputs;
}

std::vector<plaidml::edsl::Tensor> PlaidMLProgram::get_program_outputs() {
  return _program_outputs;
}

bool PlaidMLProgram::add_value_to_dictionary(const std::string& name, plaidml::edsl::Value val) {
  auto check = _value_dictionary.insert({name, val});
  return check.second; /*TODO: PlaidML if existed already do we update?*/
}

std::vector<int64_t> PlaidMLProgram::get_input_shape(const onnxruntime::NodeArg* node_input) {
  std::vector<int64_t> shape;
  for (int dim = 0; dim < node_input->Shape()->dim_size(); dim++) {
    shape.push_back(node_input->Shape()->dim(dim).dim_value());
  }
  return shape;
}

}  // namespace plaidml_ep
}  // namespace onnxruntime