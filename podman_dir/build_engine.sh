podman run --rm --device nvidia.com/gpu=all \
-v /nsls2/data2/hxn/legacy/home/home/tshimamur/02Pixi/ptycho_gui/holoscan-framework/models:/models \
-v /nsls2/data2/hxn/legacy/home/home/tshimamur/HoloScanContainerAIInference/holoscan-framework/Holoptycho/edgePtychoViT:/edgePtychoViT \
hxn-ptycho-holoscan \
pixi run python /edgePtychoViT/build_trt_engine.py \
      --onnx /models/model_406149_epoch025_wprobe.onnx \
      --engine /models/model_406149_epoch025_wprobe_srv6.engine \
      --fp16

