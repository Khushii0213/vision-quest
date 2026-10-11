from __future__ import annotations

import streamlit as st

from ..config import MAX_UPLOAD_SIDE, TOP_K
from ..imageio import InvalidImageError, fit_longest_side, load_image
from ..model import InferenceError
from .. import services
from . import components as ui


def _capture_again() -> None:
    st.session_state.cam_n = st.session_state.get("cam_n", 0) + 1  # new widget key clears the photo


def render() -> None:
    st.markdown('<h1 class="vq-title" style="font-size:2.6rem">Webcam Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="vq-lede">Take a photo with your webcam, or upload one, and see the top five '
                'ImageNet classes MobileNetV3-Small predicts.</p>', unsafe_allow_html=True)
    mode = st.radio("Image source", ["Camera", "Upload an image"], horizontal=True, key="cam_mode")
    data = None
    if mode == "Camera":
        shot = st.camera_input("Take a photo", key=f"cam_{st.session_state.get('cam_n', 0)}")
        if shot is not None:
            data = shot.getvalue()
        st.button("Capture again", key="cam_again", on_click=_capture_again)
        with st.expander("Camera not working?"):
            st.markdown(
                "- Allow camera access when the browser asks. On macOS also open System Settings, Privacy & Security, "
                "Camera, and switch on your browser.\n"
                "- Use Safari or Chrome at `http://localhost:8501`; other addresses may block the camera.\n"
                "- Close other apps that use the camera (Zoom, FaceTime, Photo Booth), then reload the page.\n"
                "- No camera at all? Choose **Upload an image** above. Nothing else is needed.")
    else:
        up = st.file_uploader("Upload JPG, PNG, WebP or BMP", type=["png", "jpg", "jpeg", "webp", "bmp"], key="cam_upload")
        if up is not None:
            data = up.getvalue()
    if data is None:
        st.markdown('<p class="vq-note">No image yet.</p>', unsafe_allow_html=True)
        return
    try:
        img = fit_longest_side(load_image(data), MAX_UPLOAD_SIDE)
    except InvalidImageError as exc:
        ui.panel("error", "That image cannot be used", str(exc))
        return

    left, right = st.columns([1, 1], gap="large")
    left.markdown(ui.figure_html(img, "Input image"), unsafe_allow_html=True)
    with right:
        clf, err = services.classifier_or_error()
        if clf is None:
            ui.model_error(err)
            return
        try:
            preds, ms = ui.timed_predict(clf, img, TOP_K)
        except InferenceError as exc:
            ui.panel("error", "Inference failed", str(exc))
            return
        st.markdown('<p class="vq-h">Top 5 predictions</p>' + ui.pred_rows_html(preds), unsafe_allow_html=True)
        st.markdown(f'<p class="vq-note">Inference took {ms:.0f} ms on this computer (CPU, measured live). '
                    'Labels come from the 1,000 ImageNet classes. The model can only name those classes, and a high '
                    'confidence does not guarantee a correct answer.</p>', unsafe_allow_html=True)
