# PlaidML Execution Provider

PlaidML Execution Provider uses a pre-release PlaidML version (plaidml-v1) to generate [MLIR](https://mlir.llvm.org) code for onnxruntime programs. This version is being developed on the plaidml-v1 branch of the PlaidML GitHub repository [here](https://github.com/plaidml/plaidml/tree/plaidml-v1). Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware. Details on [MLIR](https://mlir.llvm.org) are available on their website.


## Build instructions for ONNXRT with PlaidML (plaidml-v1)

PlaidML [version 1](https://github.com/plaidml/plaidml/tree/plaidml-v1) pre-release can be downloaded and build from source [->The PlaidML Github Repository](https://github.com/plaidml/plaidml)

Build the PlaidML library to use with onnxruntime
```
git clone --recursive  --branch plaidml-v1  https://github.com/plaidml/plaidml.git
cd plaidml
./configure
conda activate .cenv/
bazelisk test //...
bazelisk build plaidml:shlib
```

The dylib will be generated inside bazel-bin/plaidml/

download onnxruntime PlaidML branch and run the following commands to build the onnxruntime wheel with PlaidML Execution Provider enabled.  
```
export TODO_TEMP_PLAIDML_DIR=/path/to/plaidml
export TODO_TEMP_PLAIDML_LIB_DIR=~/path/to/plaidml.dylib
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_plaidml
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_plaidml --build_wheel
```

## Tile eDSL (C++ / Python)
PlaidML provides the [Tile eDSL](https://plaidml.readthedocs.io/en/latest/usage/edsl.html#how-to-write-tile-code) (Embedded Domain Specific Language) as a simple way to translate mathematical logic into code. Tile eDSL is designed to make machine learning code easier to read and write. For learning more about Tile eDSL see : [Tile eDSL](https://plaidml.readthedocs.io/en/latest/usage/edsl.html#how-to-write-tile-code)

## ONNX Layers supported using PlaidML

The table below shows the ONNX layers created using Tile eDSL wrapper functions that translate onnx nodes into eDSL and generate MLIR code for the specified device, to provide the most efficient execution possible. PlaidML also provides an op library which has been built on top of Tile eDSL to facilitate developers, this library provides access to eDSL implementations of commonly used ML operations with an easy to use fluent API. The table below lists supported layers and corresponding eDSL op library functions employed. 

TODO: distinguish between edsl, oplib and wrapper edsl functions 

| **ONNX Layers** | **eDSL Layers** | **CPU** |
| --- | --- | --- | --- | --- | --- |
| Abs | abs | Yes 
| Acos | acos | Yes 
| Acosh | acosh | Yes 
| Add | + | Yes 
| And | && | Yes 
| ArgMax | argmax | Yes |
| ArgMin | argmax | Yes |
| Asin | asin | Yes |
| Asinh | - | No |
| Atan | atan | Yes |
| Atanh | - | No |
| AveragePool | pool | No |
| BatchNormalization | - | No |
| BitShift | - | No |
| Cast | cast | Yes | Yes | No | No
| Ceil | ceil | Yes |
| Clip | clip | Yes |
| Compress | - | No |
| Concat | concat | Yes | Yes | No | No
| ConcatFromSquence | - | No |
| Constant | - | No |
| ConstantOfShape | - | No |
| Conv | convolution | Yes |
| ConvInteger | - | No |
| ConvTranspose | - | No |
| Cos | cos | Yes |
| Cosh | cosh | Yes |
| Cumsum | cumsum | Yes |
| DepthToSpace | - | No |
| DequantizeLinear | - | No |
| Det | - | No |
| Div | / | Yes |
| Dropout | - | No |
| Einsum | - | No |
| Elu | - | Yes |
| Equal | == | Yes |
| Erf | erf | Yes |
| Exp | exp | Yes |
| Expand | - | No |
| EyeLike | - | Yes |
| Flatten | - | Yes |
| Floor | floor | Yes |
| GRU | - | No |
| Gather | - | No |
| GatherElements | - | No |
| GatherND | - | No |
| Gemm | - | No |
| GlobalAveragePool | - | No |
| GlobalLpPool | - | No |
| GlobalMaxPool | - | No |
| Greater | - | No |
| HardSigmoid | - | Yes |
| HardMax | - | No |
| Identity | - | Yes |
| If | - | No |
| InstantNormalization | - | No |
| IsInf | - | No |
| IsNaN | - | No |
| LRN | lrn | Yes |
| LeakyRelu | relu | Yes |
| Less | < | Yes |
| Log | log | Yes |
| LogSoftmax | - | Yes |
| Loop | - | No |
| LpNormalization | - | No |
| LpPool | - | No |
| MatMul | - | No |
| MatMulInteger | - | No |
| Max | maximum | Yes |
| MaxPool | pool | Yes |
| MaxRoiPool | - | No |
| MaxUnpool | - | No |
| Mean | - | Yes |
| Min | minimum | Yes |
| Mod | % | Yes |
| Mul | * | Yes |
| Multinomial | - | No |
| Neg | - | Yes |
| NonMaxSuppression | - | No |
| NonZero | - | No |
| Not | - | Yes |
| OneHot | - | No |
| Or | - | No |
| PRelu | relu | Yes |
| Pad | - | No |
| Pow | pow | Yes |
| QLinearConv | - | No |
| QLinearMatmul | - | No |
| QuantizeLinear | - | No |
| RNN | - | No |
| RandomNormal | - | No |
| RandomNormalLike | - | No |
| RandomUniform | - | No |
| RandomUniformLike | - | No |
| Reciprocal | - | No |
| ReduceL1 | - | No |
| ReduceL2 | - | No |
| ReduceLogSum | - | No |
| ReduceLogSumExp | - | No |
| ReduceMax | max | Yes |
| ReduceMean | mean | Yes |
| ReduceMin | min | Yes |
| ReduceProd | prod | Yes |
| ReduceSum | sum | Yes |
| ReduceSumSquare | - | No |
| Relu | relu | Yes |
| Reshape | reshape | Yes |
| Resize | - | No |
| ReverseSequence | - | No |
| RoiAlign | - | No |
| Round | - | No |
| Scan | - | No |
| Scatter | - | No |
| ScatterElements | - | No |
| ScatterND | - | No |
| Selu | - | Yes |
| SequenceAt | - | No |
| SequenceConstruct | - | No |
| SequenceEmpty | - | No |
| SequenceErase | - | No |
| SequenceInsert | - | No |
| SequenceLength | - | No |
| Shape | - | No |
| Shrink | - | No |
| Sigmoid | sigmoid | Yes |
| Sign | - | Yes |
| Sin | sin | Yes |
| Sinh | sinh | Yes |
| Size | - | No |
| Slice | - | No |
| Softmax | - | No |
| SoftPlus | - | No |
| SoftSign | - | No |
| SpaceToDepth | - | No |
| Split | - | No |
| SplitToSequence | - | No |
| Sqrt | sqrt | Yes |
| Squeeze | squeeze | Yes |
| StringNormalizer | - | No |
| Sub | - | Yes |
| Sum | + | Yes |
| Tan | tan | Yes |
| Tanh | tanh | Yes |
| TfdfVectorizer | - | No |
| ThresholdRelu | relu | Yes |
| Tile | tile | Tes | No | No | No
| TopK | - | No |
| Transpose | transpose | Yes |
| Unique | - | No |
| UnSqueeze | unsqueeze | Yes |
| UpSample | - | No |
| Where | - | Yes |
| Xor | ^ | Yes |

## Topology Support

TODO: add topology support here 

## How to run the ONNX node/model tests 

TODO: add instructions on running onnx_

## Sample Inference Resnet50

TODO: