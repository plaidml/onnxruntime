#!/bin/bash

# This is a temporary file created for a Github pipeline
# This file will be deleted once this branch is finalized 

# This file contains commands that build onnxruntime with plaidml execution 
# provider support and runs internal node and inference tests 


#Start the time
START=$(date +%s)

# Download plaidml-v1 from source and build it 
# TODO: PlaidML - this should be done in .gitmodules and the build instructions 
# added to onnxruntime OR post plaidml-v1 release the user will be instructed to
# install plaidml-v1 through pip    

git clone --recursive --branch plaidml-v1 https://github.com/plaidml/plaidml.git ./build/plaidml
cd build/plaidml/

# Set environment variables so that onnxruntime can find plaidml 
if [[ "$OSTYPE" == "linux-gnu"* ]]; 
then
    ./configure
    conda activate .cenv/
    bazelisk build //plaidml:plaidml
    conda deactivate
    export TODO_TEMP_PLAIDML_DIR=$PWD
    export TODO_TEMP_PLAIDML_LIB_DIR=$PWD/bazel-bin/plaidml/libplaidml.so
else
# TODO (PlaidML): dylib is no longer produced for plaidml through the bazel 
# build setup ('bazelisk build //plaidml:plaidml' previously shlib) (see pull 1335)
# fix required to get this working on mac again 
# to run locally on mac use plaidml-v1 commit #(236c0c44d985d320e36f208c8e5f3c671996b27a)
    echo "Error: This buld requires Linux"
fi
cd ../../
# Build onnxruntime with plaidml execution provider support 
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_plaidml


#install onnx 
pip install onnx 

#grab the node tests
python3 -m onnx.backend.test.cmd_tools generate-data -o ./build/pipeline_test_data

#run the node tests using plaidml execution provider
./build/Linux/RelWithDebInfo/onnx_test_runner -e plaidml ./build/pipeline_test_data

# download and run resnet50
# TODO: PalidMl this should use https://github.com/onnx/models#gitlfs- to download specific models
# cd build/
# mkdir pipeline_model_data
# cd pipeline_model_data
# wget https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v1-7.tar.gz 

# cd build/pipeline_model_data/
# tar -xvf resnet50-v1-7.tar.gz

# cd ../../
# ./build/Linux/RelWithDebInfo/onnx_test_runner -e plaidml ./build/pipeline_model_data/resnet50v2
# Build complete -> print time
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "-----------------------------------"
echo "This build took $DIFF seconds"
echo "-----------------------------------"


