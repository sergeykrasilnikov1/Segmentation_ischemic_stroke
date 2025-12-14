#!/bin/bash

ONNX_MODEL=${1:-"models/model.onnx"}
OUTPUT_PATH=${2:-"models/model.trt"}
PRECISION=${3:-"fp16"}
WORKSPACE_SIZE=${4:-4096}

if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: ONNX model not found: $ONNX_MODEL"
    exit 1
fi

trtexec \
    --onnx=$ONNX_MODEL \
    --saveEngine=$OUTPUT_PATH \
    --$PRECISION \
    --workspace=$WORKSPACE_SIZE

if [ $? -eq 0 ]; then
    echo "TensorRT conversion successful. Engine saved to $OUTPUT_PATH"
else
    echo "TensorRT conversion failed"
    exit 1
fi
