// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "plaidml_execution_provider.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"
// #include "core/providers/plaidml/plaidml_ops.h"
#include "plaidml_ops.h"

#include <stdio.h>
#include <string.h>
namespace onnxruntime {

  //TODO: move to plaidml_utils
plaidml::DType ConvertPrecisionONNXToPlaidML(
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
  } else if (*onnx_type == "uint16" || *onnx_type == "tensor(uint16)") {
    return plaidml::DType::UINT16;
  } else if (*onnx_type == "uint8" || *onnx_type == "tensor(uint8)") {
    return plaidml::DType::UINT8;
  } else if (*onnx_type == "bool" || *onnx_type == "tensor(bool)") {
    return plaidml::DType::BOOLEAN;
  } 
  else{
    throw std::runtime_error("{PlaidML} ERROR: invalid data type " + *onnx_type);
    return plaidml::DType::INVALID;
  }

}
// TODO: Some of this stuff should probably be separated into new files

PlaidMLProgram MakePlaidMLProgram(const onnxruntime::Node* fused_node) {
  PlaidMLProgram ret;
  std::map<std::string, plaidml::edsl::Value> init_tensors;
  // TODO: We might instead implement this on an ONNX ModelProto instead of an ONNX RT Node.
  //     This might have benefits for reuse in a non-RT ONNX context?

  // TODO: In general, inputs are a mix of initializers and input data; this currently assumes they're all the latter

  // TODO: work out if deprecated op is being used and handle it

  // For each input, look up shape (or at least rank) and construct a (placeholder) tensor accordingly;
  // add this to the `tensors` dict
  
  for (const auto& node_input : fused_node->InputDefs()) {
    // TODO: A node_input's Shape can be nullptr (i.e. if the input isn't a tensor) and we need to handle that case
    // TODO: This doesn't address symbolic shapes
    std::vector<int64_t> shape;
    for (int dim = 0; dim < node_input->Shape()->dim_size(); dim++) {
      shape.push_back(node_input->Shape()->dim(dim).dim_value());
    }
    auto type = ConvertPrecisionONNXToPlaidML(node_input->Type());
    auto input_placeholder = plaidml::edsl::Placeholder(type, shape);
    if (!init_tensors.insert({node_input->Name(), plaidml::edsl::Value(input_placeholder)}).second) {
      throw std::runtime_error("Unexpected duplicate name in fused node while adding inputs [TODO better error handling]");
    }
    ret.inputs.push_back(input_placeholder);
  }

  // For each node in topological order:
  //   * Get its inputs out of the `tensors` dict
  //   * Call `MakePlaidMLOp` and write results into `tensors` dict
  
  for (const auto& node : fused_node->GetFunctionBody()->Body().Nodes()) {
    std::vector<plaidml::edsl::Value> local_input_tensors;
   
    for (const auto& local_input : node.InputDefs()) {
      try {
        if(local_input->Name()!=""){//TODO: PlaidML fix this
          auto input = init_tensors.at(local_input->Name());
          local_input_tensors.push_back(plaidml::edsl::Value(input));
        }
      } catch (const std::out_of_range& e) {
        throw std::runtime_error("Could not find expected tensor " + local_input->Name() + " [TODO better error handling]");
      }
    }
    ONNX_NAMESPACE::NodeProto node_proto;
    node.ToProto(node_proto);
    auto local_output_tensors = plaidml_ep::MakePlaidMLOp(node_proto, local_input_tensors);
    // Iterate over output tensors and names in tandem
    auto output_tensor_it = local_output_tensors.begin();
    for (const auto& local_output : node.OutputDefs()) {
      if (output_tensor_it == local_output_tensors.end()) {
        throw std::runtime_error("Inconsistent number of outputs [TODO better error handling]");
      }
      if (!init_tensors.insert({
          local_output->Name(),
          plaidml::edsl::Value(*output_tensor_it)
      }).second) {
        throw std::runtime_error("Unexpected duplicate name in fused node while adding outputs (possibly intermediate) [TODO better error handling]");
      }
      output_tensor_it++;
    }
    if (output_tensor_it != local_output_tensors.end()) {
      throw std::runtime_error("Inconsistent number of outputs [TODO better error handling]");
    }
  }

  // Lookup outputs from `tensors` dict, use those to call edsl::ProgramBuilder
  std::vector<plaidml::edsl::Tensor> output_tensors;
  for (const auto& node_output : fused_node->OutputDefs()) {
    auto local_output_tensor_it = init_tensors.find(node_output->Name());
    if (local_output_tensor_it == init_tensors.end()) {
      throw std::runtime_error("Expected output tensor " + node_output->Name() + " not found [TODO better error handling]");
    }
    output_tensors.push_back(local_output_tensor_it->second.as_tensor());
  }
  ret.program = std::make_shared<plaidml::edsl::Program>(plaidml::edsl::ProgramBuilder(fused_node->Name(), output_tensors).compile());
  return ret;
}

PlaidMLExecutionProvider::PlaidMLExecutionProvider(const PlaidMLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kPlaidMLExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  // This Allocator setup is ported fairly directly from the OpenVINO version.
  // TODO: Verify that this is the approach we want to take.
  DeviceAllocatorRegistrationInfo device_info(
    {
      OrtMemTypeDefault,
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
          onnxruntime::make_unique<OrtMemoryInfo>(PLAIDML, OrtDeviceAllocator)
        );
      },
      std::numeric_limits<size_t>::max()
    }
  );
  InsertAllocator(CreateAllocator(device_info));
}

std::vector<std::unique_ptr<ComputeCapability>> PlaidMLExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // TODO: This is a basic implementation that does not handle graph partitioning, incompatible
  // operation detection, initializers as inputs (for e.g. weights, reshape, ...), and probably
  // other things. But it should work in the basic case.
  // Loosely based on the nGraph approach
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

  std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

  // If there are no inputs, leave it for constant folding
  if (inputs.empty()) {
    return result;
  }

  auto node_indexes = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : node_indexes) {
    const auto node = graph_viewer.GetNode(index);

    //TODO: PlaidML do we need to add a kernel registry instead?
    if (!plaidml_ep::check_op_support(node->OpType())) {
      {
        //throw "Operation is not yet supported by PlaidML Execution Provider";
        return result;
      }
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
        [pml_program = std::make_shared<PlaidMLProgram>(MakePlaidMLProgram(fused_node))](ComputeContext* /*context*/, FunctionState* state) {
          auto* pml_state = new PlaidMLFunctionState();
          pml_state->program = pml_program;
          *state = pml_state;
          return 0;
        };
    compute_info.compute_func =
        [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
          // TODO: nGraph code has mutexs all over this stuff, is that something we should be concerned with?

          Ort::CustomOpApi ort{*api};
          auto pml_state = static_cast<PlaidMLFunctionState*>(state);
          auto binder = plaidml::exec::Binder(*pml_state->program->program);

          // Load input data
          auto executable = binder.compile();
          unsigned input_idx = 0;
          for (auto input_placeholder : pml_state->program->inputs) {
            // program->inputs and ORT inputs are in the same order, so these match
            const OrtValue* input_value = ort.KernelContext_GetInput(context, input_idx++);
            void* input_data = const_cast<void*>(ort.GetTensorData<void>(input_value));
            binder.input(input_placeholder).copy_from(input_data);
          }
          plaidml::init();
          plaidml::edsl::init();
          plaidml::op::init();
          plaidml::exec::init();
          executable->run();

          // Write output data
          unsigned output_idx = 0;
          for (auto output_arg : pml_state->program->program->outputs()) {
            std::vector<int64_t> ort_shape = output_arg.shape.sizes();
            OrtValue* output_value = ort.KernelContext_GetOutput(context, output_idx++, ort_shape.data(), ort_shape.size());
            void* output_data = ort.GetTensorMutableData<void>(output_value);
            binder.output(output_arg.tensor).copy_into(output_data);
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
