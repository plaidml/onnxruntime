// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "plaidml_execution_provider.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

namespace onnxruntime {

PlaidMLExecutionProvider::PlaidMLExecutionProvider(const PlaidMLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kPlaidMLExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  // This Allocator setup is ported fairly directly from the OpenVINO version.
  // TODO (PlaidML): Verify that this is the approach we want to take.
  DeviceAllocatorRegistrationInfo device_info(
      {OrtMemTypeDefault,
       [](int) {
         return std::make_unique<CPUAllocator>(OrtMemoryInfo(PLAIDML, OrtDeviceAllocator));
       },
       std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(device_info));
}

std::vector<std::unique_ptr<ComputeCapability>> PlaidMLExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // TODO (PlaidML): This is a basic implementation that does not handle graph partitioning, incompatible
  // operation detection, initializers as inputs (for e.g. weights, reshape, ...), and probably
  // other things. But it should work in the basic case.
  // Loosely based on the nGraph approach
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

  //add initializers to inputs
  for (auto it : initializers) {
    inputs.push_back(it.first);
  }

  std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

  // If there are no inputs, leave it for constant folding
  if (inputs.empty()) {
    return result;
  }

  auto node_indexes = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : node_indexes) {
    const auto node = graph_viewer.GetNode(index);

    // TODO (PlaidML): does not support STRING types yet
    for (const auto n_input : node->InputDefs()) {
      if (n_input->Type() != nullptr) {
        if (!strcmp(n_input->Type()->c_str(), "string") || !strcmp(n_input->Type()->c_str(), "tensor(string)")) {
          return result;
        }
      }
    }
    for (const auto n_input : node->OutputDefs()) {
      if (n_input->Type() != nullptr) {
        if (!strcmp(n_input->Type()->c_str(), "string") || !strcmp(n_input->Type()->c_str(), "tensor(string)")) {
          return result;
        }
      }
    }
    // TODO (PlaidML): do we need to add a kernel registry instead?
    if (!plaidml_ep::check_op_support(node->OpType())) {
      //throw "Operation is not yet supported by PlaidML Execution Provider";
      return result;
    }
    // TODO (PlaidML): check for unsupported attributes
    ONNX_NAMESPACE::NodeProto node_proto;
    node->ToProto(node_proto);
    if (!plaidml_ep::check_attribute_support(node_proto)) {
      return result;
    }
  }

  // This was modeled off of the metadata that nGraph included
  auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "PlaidML_Fully_Fused_Graph";
  meta_def->domain = kPlaidMLDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  sub_graph->nodes = graph_viewer.GetNodesInTopologicalOrder();
  sub_graph->SetMetaDef(meta_def);
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));

  return result;
}

common::Status PlaidMLExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (auto fused_node : fused_nodes) {
    NodeComputeInfo compute_info;

    compute_info.create_state_func =
        [pml_program = std::make_shared<plaidml_ep::PlaidMLProgram>(plaidml_ep::PlaidMLProgram(fused_node))](ComputeContext* /*context*/, FunctionState* state) {
          auto* pml_state = new PlaidMLFunctionState();
          pml_state->program = pml_program;
          *state = pml_state;
          return 0;
        };
    compute_info.compute_func =
        [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
          // TODO (PlaidML): nGraph code has mutexs all over this stuff, is that something we should be concerned with?

          Ort::CustomOpApi ort{*api};
          // TODO (PlaidML): why is this using custom op api? is it necessary ? is there a neater way?
          auto pml_state = static_cast<PlaidMLFunctionState*>(state);
          pml_state->program->program->compile();
          // Load input data
          auto executable = plaidml::exec::Executable(*pml_state->program->program);

          unsigned input_idx = 0;
          std::vector<plaidml::Buffer> inputs;
          std::vector<plaidml::Buffer> outputs;

          for (auto input_arg : pml_state->program->program->inputs()) {
            inputs.emplace_back(plaidml::Buffer(input_arg));
          }
          for (auto output_arg : pml_state->program->program->outputs()) {
            outputs.emplace_back(plaidml::Buffer(output_arg));
          }
          //auto input_shapes = pml_state->program->program->inputs();
          for (auto input_placeholder : pml_state->program->get_program_inputs()) {
            // program->inputs and ORT inputs are in the same order, so these match
            // TODO (PlaidML): check if a lookup method is required here or it is sufficient to rely
            // on the matching input order
            const OrtValue* input_value = ort.KernelContext_GetInput(context, input_idx++);
            void* input_data = const_cast<void*>(ort.GetTensorData<void>(input_value));
            inputs[input_idx-1].copy_from(input_data);
          }

          plaidml::init();
          plaidml::edsl::init();
          plaidml::op::init();
          plaidml::exec::init();
          executable.run(inputs, outputs);

          // Write output data
          unsigned output_idx = 0;
          for (auto output_arg : pml_state->program->program->outputs()) {
            std::vector<int64_t> ort_shape = output_arg.sizes();
            OrtValue* output_value = ort.KernelContext_GetOutput(context, output_idx++, ort_shape.data(), ort_shape.size());
            void* output_data = ort.GetTensorMutableData<void>(output_value);
            outputs[output_idx-1].copy_into(output_data);
          }
          return Status::OK();
        };

    compute_info.release_state_func =
        [](FunctionState state) {
          if (state) {
            auto* function_state = static_cast<PlaidMLFunctionState*>(state);
            delete function_state;
          }
        };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

}  // namespace onnxruntime
