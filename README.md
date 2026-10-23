# ONNX Object Detection UI

A local Streamlit app that runs an object detector you provide against an image you upload. It preprocesses the image, runs ONNX Runtime on CPU, maps post-NMS boxes back to the original image, and lets you download the annotation and detection JSON.

The repository contains no fake detections, pretrained weights, or example images. The app makes no external inference calls; uploaded images are processed by the machine serving Streamlit.

## Requirements

- Python 3.10 or newer
- A compatible ONNX object-detection model

Install the CPU runtime and UI dependencies:

```bash
python -m pip install -r requirements.txt
```

The application uses the ONNX Runtime Python `InferenceSession` API with its CPU execution provider. For a different hardware provider, install its matching ONNX Runtime package and adjust the provider configuration for that environment.

## Model contract

To keep output decoding explicit, the app accepts a model with:

- One `float32` input shaped `[1, 3, H, W]` in RGB/NCHW order, with pixel values normalized to `[0, 1]`.
- One post-NMS output shaped `[1, detections, 6]` or `[detections, 6]`.
- Output rows `[x1, y1, x2, y2, confidence, class_id]`, with pixel coordinates in the model input image.

The app letterboxes images with RGB fill value 114 and reverses that scale/padding for the displayed boxes. Static model dimensions are detected automatically; dynamic models use the input-size setting in the sidebar. Set class names in the UI if they are not embedded in the model.

Exporters use different ONNX input and output conventions. For an Ultralytics model, export with embedded NMS where supported and verify that the resulting model matches this contract; see the [Ultralytics export documentation](https://docs.ultralytics.com/modes/export). The runtime does not import Ultralytics or download weights.

## Run locally

Set the model path in the sidebar or through `ONNX_MODEL_PATH`, then start Streamlit:

```bash
# macOS/Linux
export ONNX_MODEL_PATH=/absolute/path/to/detector.onnx
streamlit run yolo_app.py --server.address 127.0.0.1
```

```powershell
$env:ONNX_MODEL_PATH = "C:\models\detector.onnx"
streamlit run yolo_app.py --server.address 127.0.0.1
```

Choose an image, set the confidence threshold and class names, then select **Run detection**. The app displays boxes and provides PNG and JSON downloads. It accepts one image at a time; it does not capture live video or load raw YOLO prediction tensors without NMS.

## Docker

Build the CPU app image and mount a model file. The port mapping below binds the UI to your local machine:

```bash
docker build -t onnx-object-detection-ui .
docker run --rm -p 127.0.0.1:8501:8501 -e ONNX_MODEL_PATH=/models/detector.onnx -v "$PWD/models:/models:ro" onnx-object-detection-ui
```

## Model licensing

No model weights are included. Check the license for each model you use. Ultralytics documents YOLOv8 model licensing under AGPL-3.0 and Enterprise options in its [model documentation](https://docs.ultralytics.com/models/yolov8); this app does not change those model terms.
