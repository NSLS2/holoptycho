#!/usr/bin/env bash

# Host-side base directory that is mounted as /models inside the container
MODELS_HOST_DIR="/nsls2/data2/hxn/legacy/home/home/tshimamur/02Pixi/ptycho_gui/holoscan-framework/models"
MODELS_CONTAINER_DIR="/models"

# --- Prompt for ONNX file path ---
read -p "Enter path to ONNX file: " ONNX_HOST_PATH

# Strip trailing whitespace/newline
ONNX_HOST_PATH="${ONNX_HOST_PATH%"${ONNX_HOST_PATH##*[![:space:]]}"}"

# Validate the file exists
if [[ ! -f "$ONNX_HOST_PATH" ]]; then
    echo "Error: ONNX file not found at '$ONNX_HOST_PATH'"
    exit 1
fi

# --- Derive default engine output path (replace .onnx with .engine) ---
ONNX_BASENAME="$(basename "$ONNX_HOST_PATH")"
DERIVED_ENGINE_BASENAME="${ONNX_BASENAME%.onnx}.engine"
DERIVED_ENGINE_HOST_PATH="$(dirname "$ONNX_HOST_PATH")/${DERIVED_ENGINE_BASENAME}"

echo ""
echo "Output engine path will be: ${DERIVED_ENGINE_HOST_PATH}"
read -p "Press Enter to confirm or type a different path: " USER_ENGINE_PATH

if [[ -n "$USER_ENGINE_PATH" ]]; then
    ENGINE_HOST_PATH="${USER_ENGINE_PATH%"${USER_ENGINE_PATH##*[![:space:]]}"}"
else
    ENGINE_HOST_PATH="$DERIVED_ENGINE_HOST_PATH"
fi

# --- Translate host paths to container-internal /models paths ---
ONNX_CONTAINER_PATH="${MODELS_CONTAINER_DIR}/$(realpath --relative-to="$MODELS_HOST_DIR" "$ONNX_HOST_PATH")"
ENGINE_CONTAINER_PATH="${MODELS_CONTAINER_DIR}/$(realpath --relative-to="$MODELS_HOST_DIR" "$ENGINE_HOST_PATH")"

echo ""
echo "Running TensorRT engine build..."
echo "  ONNX input  : ${ONNX_HOST_PATH}"
echo "  Engine output will be saved to: ${ENGINE_HOST_PATH}"
echo ""

podman run --rm --device nvidia.com/gpu=all \
  -v "${MODELS_HOST_DIR}:${MODELS_CONTAINER_DIR}" \
  -v /nsls2/data2/hxn/legacy/home/home/tshimamur/01Holoscan/edgePtychoViT:/edgePtychoViT \
  hxn-ptycho-holoscan \
  pixi run python /edgePtychoViT/build_trt_engine.py \
        --onnx "${ONNX_CONTAINER_PATH}" \
        --engine "${ENGINE_CONTAINER_PATH}" \
        --fp16

echo ""
echo "Done. Engine file saved to: ${ENGINE_HOST_PATH}"
