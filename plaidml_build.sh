#!/bin/bash

# This is a temporary file created for a Github pipeline
# This file will be deleted once this branch is finalized 

# This file contains commands that build onnxruntime with plaidml execution 
# provider support and runs internal node and inference tests 


#Start the time
START=$(date +%s)

# Download plaidml-v1 from source and build it 
# TODO (PlaidML): - this should be done in .gitmodules and the build instructions 
# added to onnxruntime OR post plaidml-v1 release the user will be instructed to
# install plaidml-v1 through pip    
git clone --recursive --branch plaidml-v1 https://github.com/plaidml/plaidml.git ./build/plaidml
cd build/plaidml/
# TODO (PlaidML): update all ops to new eDSL API
#git checkout f912a9007e7750c8328b5a9fcb0609848610de13 
./configure
conda activate .cenv/
bazelisk build //plaidml:plaidml
conda deactivate
# Set environment variables so that onnxruntime can find plaidml 
export TODO_TEMP_PLAIDML_DIR=$PWD
export TODO_TEMP_PLAIDML_LIB_DIR=$PWD/bazel-bin/plaidml/libplaidml.so
# TODO: (PlaidML) temp fix for local build on mac -> place libplaidml.so into ~/lib to get past 
# dyld: library not loaded error. This needs to be investigated and fixed when mac machines are 
# added to pipeline

# Build onnxruntime with plaidml execution provider support 
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_plaidml

# Build complete -> print time
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "-----------------------------------"
echo "This build took $DIFF seconds"
echo "-----------------------------------"