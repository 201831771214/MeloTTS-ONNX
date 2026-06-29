# !/bin/bash
CUR_PATH=$(realpath $(dirname $0))
EXPORT_SCRIPT=${CUR_PATH}/export_melo.py
COMPILE_SCRIPT=${CUR_PATH}/compile.py


# export melo model to onnx
python ${EXPORT_SCRIPT} -sl 512 -mmf 1024 --qnn --opset 16

# compile melo onnx to precompiled_qnn_onnx
python ${COMPILE_SCRIPT}