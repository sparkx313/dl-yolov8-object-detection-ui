"""Run a local Streamlit UI for compatible, post-NMS ONNX object detectors."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw


@st.cache_resource(show_spinner=False)
def load_session(model_path: str, modified_ns: int, file_size: int) -> ort.InferenceSession:
    del modified_ns, file_size  # Included in Streamlit's cache key to reload replaced model files.
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def fixed_dimension(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, np.integer)) and int(value) > 0 else None


def validate_session(session: ort.InferenceSession) -> tuple[Any, Any, int | None, int | None]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Model must expose exactly one input and one detection output.")
    model_input = inputs[0]
    if model_input.type != "tensor(float)" or len(model_input.shape) != 4:
        raise ValueError("Model input must be a float32 NCHW image tensor.")
    if fixed_dimension(model_input.shape[0]) not in (None, 1):
        raise ValueError("Model input batch dimension must be dynamic or 1.")
    if fixed_dimension(model_input.shape[1]) != 3:
        raise ValueError("Model input must have three RGB channels in NCHW order.")
    return model_input, outputs[0], fixed_dimension(model_input.shape[2]), fixed_dimension(model_input.shape[3])


def letterbox(image: Image.Image, height: int, width: int) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    original_width, original_height = image.size
    scale = min(width / original_width, height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(resized, (left, top))
    values = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(values, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
    return tensor, (resized_width / original_width, resized_height / original_height, left, top)


def parse_detections(
    output: np.ndarray,
    original_size: tuple[int, int],
    transform: tuple[float, float, int, int],
    threshold: float,
) -> list[dict[str, Any]]:
    rows = np.asarray(output)
    if rows.ndim == 3:
        if rows.shape[0] != 1:
            raise ValueError("Detection output must have batch size 1.")
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(
            "Detection output must be a single post-NMS tensor shaped [1, detections, 6] "
            "with rows [x1, y1, x2, y2, confidence, class_id]."
        )
    if not np.issubdtype(rows.dtype, np.number):
        raise ValueError("Detection output must contain numeric values.")

    scale_x, scale_y, pad_left, pad_top = transform
    original_width, original_height = original_size
    detections: list[dict[str, Any]] = []
    for row in rows.astype(np.float64, copy=False):
        x1, y1, x2, y2, confidence, class_value = row.tolist()
        if (
            not np.isfinite(row).all()
            or confidence < threshold
            or confidence > 1.0
            or class_value < 0
        ):
            continue
        class_id = int(round(class_value))
        if not np.isclose(class_value, class_id, atol=1e-3):
            continue
        x1 = (x1 - pad_left) / scale_x
        x2 = (x2 - pad_left) / scale_x
        y1 = (y1 - pad_top) / scale_y
        y2 = (y2 - pad_top) / scale_y
        x1 = min(max(x1, 0.0), float(original_width))
        x2 = min(max(x2, 0.0), float(original_width))
        y1 = min(max(y1, 0.0), float(original_height))
        y2 = min(max(y2, 0.0), float(original_height))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "class_id": class_id,
                "confidence": round(float(confidence), 6),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )
    return detections


def annotate(image: Image.Image, detections: list[dict[str, Any]], class_names: list[str]) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for item in detections:
        class_id = item["class_id"]
        label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        color = ((37 * class_id + 40) % 220 + 35, (83 * class_id + 90) % 220 + 35, (131 * class_id + 15) % 220 + 35)
        box = (item["x1"], item["y1"], item["x2"], item["y2"])
        draw.rectangle(box, outline=color, width=max(2, rendered.width // 300))
        caption = f"{label} {item['confidence']:.2f}"
        bounds = draw.textbbox((0, 0), caption)
        caption_width = bounds[2] - bounds[0]
        caption_height = bounds[3] - bounds[1]
        caption_top = max(0, int(item["y1"]) - caption_height - 6)
        draw.rectangle(
            (int(item["x1"]), caption_top, int(item["x1"]) + caption_width + 8, caption_top + caption_height + 6),
            fill=color,
        )
        draw.text((int(item["x1"]) + 4, caption_top + 3), caption, fill=(255, 255, 255))
    return rendered


def main() -> None:
    st.set_page_config(page_title="ONNX Object Detection", page_icon="🎯", layout="wide")
    st.title("Object Detection")
    st.caption("Run a compatible ONNX detector on an image selected from your device.")

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("ONNX model path", value=os.environ.get("ONNX_MODEL_PATH", ""))
        st.caption("Use a trusted local model file. Model weights are not downloaded by this app.")
        confidence = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        dynamic_size = st.selectbox("Input size for dynamic models", options=(320, 416, 512, 640, 768, 960), index=3)
        class_text = st.text_input("Class names, comma separated (optional)", value="")

    uploaded = st.file_uploader("Choose an image", type=("jpg", "jpeg", "png", "bmp", "webp"))
    if not model_path:
        st.info("Enter the path to a compatible ONNX model to begin.")
        return
    checkpoint = Path(model_path).expanduser()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".onnx":
        st.error("Model path must point to an existing .onnx file available to this app process.")
        return
    if uploaded is None:
        st.info("Select an image to run inference.")
        return
    image_bytes = uploaded.getvalue()
    if len(image_bytes) > 25 * 1024 * 1024:
        st.error("Image must be 25 MB or smaller.")
        return
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")
    except OSError as exc:
        st.error(f"Could not read image: {exc}")
        return

    if st.button("Run detection", type="primary"):
        try:
            session = load_session(str(checkpoint.resolve()), checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
            model_input, model_output, fixed_height, fixed_width = validate_session(session)
            height = fixed_height or dynamic_size
            width = fixed_width or dynamic_size
            tensor, transform = letterbox(image, height, width)
            output = session.run([model_output.name], {model_input.name: tensor})[0]
            detections = parse_detections(
                output, image.size, transform, confidence
            )
            class_names = [name.strip() for name in class_text.split(",") if name.strip()]
            annotated = annotate(image, detections, class_names)

            left, right = st.columns(2)
            with left:
                st.image(image, caption="Input", use_container_width=True)
            with right:
                st.image(annotated, caption=f"{len(detections)} detection(s)", use_container_width=True)
            st.json(detections)

            image_buffer = io.BytesIO()
            annotated.save(image_buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=image_buffer.getvalue(),
                file_name=f"{Path(uploaded.name).stem}_detections.png",
                mime="image/png",
            )
            report = json.dumps(
                {"image": uploaded.name, "detections": detections}, indent=2
            )
            st.download_button(
                "Download detection JSON",
                data=report,
                file_name=f"{Path(uploaded.name).stem}_detections.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
