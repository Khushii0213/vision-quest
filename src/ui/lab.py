from __future__ import annotations

import random

import streamlit as st

from ..augmentation import AugmentParams, apply_augmentations
from ..compare import compare_predictions
from ..config import MAX_UPLOAD_SIDE, NOISE_SEED, TOP_K, WORK_SIZE
from ..imageio import InvalidImageError, fit_longest_side, load_image, upscale_square
from ..model import InferenceError
from .. import services
from . import components as ui
from .setup import render_setup

DEFAULTS = {"lab_brightness": 1.0, "lab_rotation": 0, "lab_blur": 0.0, "lab_contrast": 1.0,
            "lab_noise": 0, "lab_gamma": 1.0}


def _reset() -> None:
    st.session_state.update(DEFAULTS)


def _random_sample(count: int) -> None:
    st.session_state.lab_sample = random.randint(1, count)


def _pick_source(status):
    """Returns (original image, description) or (None, None)."""
    options = ["Built-in dataset", "Upload an image"] if status.dataset else ["Upload an image"]
    source = st.radio("Image source", options, horizontal=True, key="lab_source")
    if source == "Built-in dataset":
        ds = status.dataset
        c1, c2, c3 = st.columns([2, 2, 1])
        cls_name = c1.selectbox("Class", ds.class_names, key="lab_class")
        pool = ds.index_by_class()[ds.class_names.index(cls_name)]
        st.session_state.lab_sample = min(max(int(st.session_state.get("lab_sample", 1)), 1), len(pool))
        n = c2.number_input("Sample number", 1, len(pool), key="lab_sample")
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c3.button("Random", key="lab_random", on_click=_random_sample, args=(len(pool),))
        idx = pool[int(n) - 1]
        return upscale_square(ds.image(idx), WORK_SIZE), f"CIFAR-10 test image, ground-truth label: {ds.label_name(idx)}"
    if status.needs_download:
        render_setup(status)
    up = st.file_uploader("Upload JPG, PNG, WebP or BMP", type=["png", "jpg", "jpeg", "webp", "bmp"], key="lab_upload")
    if up is None:
        st.markdown('<p class="vq-note">Choose an image to begin.</p>', unsafe_allow_html=True)
        return None, None
    try:
        return fit_longest_side(load_image(up.getvalue()), MAX_UPLOAD_SIDE), f"Uploaded file: {up.name}"
    except InvalidImageError as exc:
        ui.panel("error", "That file cannot be used", str(exc))
        return None, None


def render() -> None:
    st.markdown('<h1 class="vq-title" style="font-size:2.6rem">AI Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="vq-lede">Distort an image and compare what a pretrained MobileNetV3-Small predicts '
                'before and after.</p>', unsafe_allow_html=True)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    status = services.dataset_status()
    original, desc = _pick_source(status)
    if original is None:
        return

    ctrl, main = st.columns([1, 3], gap="large")
    with ctrl:
        st.markdown('<p class="vq-h">Augmentation</p>', unsafe_allow_html=True)
        st.slider("Brightness", 0.2, 2.0, step=0.05, key="lab_brightness")
        st.slider("Rotation (degrees)", -180, 180, step=1, key="lab_rotation")
        st.slider("Blur radius", 0.0, 6.0, step=0.1, key="lab_blur")
        st.slider("Contrast", 0.2, 2.0, step=0.05, key="lab_contrast")
        st.slider("Noise (std-dev)", 0, 60, step=1, key="lab_noise")
        st.slider("Gamma (above 1 = darker)", 0.4, 3.0, step=0.05, key="lab_gamma")
        st.button("Reset experiment", key="lab_reset", on_click=_reset)
    params = AugmentParams(brightness=st.session_state.lab_brightness, rotation=st.session_state.lab_rotation,
                           blur=st.session_state.lab_blur, contrast=st.session_state.lab_contrast,
                           noise=st.session_state.lab_noise, gamma=st.session_state.lab_gamma)
    augmented = apply_augmentations(original, params, seed=NOISE_SEED)

    with main:
        a, b = st.columns(2)
        a.markdown(ui.figure_html(original, "Original"), unsafe_allow_html=True)
        changed = ", ".join(params.changed()) or "nothing changed"
        b.markdown(ui.figure_html(augmented, f"Transformed ({changed})"), unsafe_allow_html=True)
        st.markdown(f'<p class="vq-note">{ui.esc(desc)}. The model predicts ImageNet-1K classes, a different label '
                    'set from the dataset label, so its answers are not scored against it.</p>', unsafe_allow_html=True)

        clf, err = services.classifier_or_error()
        if clf is None:
            ui.model_error(err)
        else:
            try:
                p0, t0 = ui.timed_predict(clf, original, TOP_K)
                p1, t1 = ui.timed_predict(clf, augmented, TOP_K)
            except InferenceError as exc:
                ui.panel("error", "Inference failed", str(exc))
            else:
                cmp = compare_predictions(p0, p1)
                s0 = {r.class_index: r.status for r in cmp.rows if r.original_rank}
                s1 = {r.class_index: r.status for r in cmp.rows if r.augmented_rank}
                c1, c2 = st.columns(2)
                c1.markdown('<p class="vq-h">Original: top 5</p>' + ui.pred_rows_html(p0, s0), unsafe_allow_html=True)
                c2.markdown('<p class="vq-h">Transformed: top 5</p>' + ui.pred_rows_html(p1, s1), unsafe_allow_html=True)
                st.markdown(f'<p class="vq-note">Inference took {t0:.0f} ms and {t1:.0f} ms on this computer (CPU, measured live). '
                            'Orange bars entered the top 5, grey bars changed rank.</p>', unsafe_allow_html=True)
                _comparison(cmp, params)

    with st.expander("How the prediction is made"):
        st.markdown(_TECH)
    with st.expander("Concepts: classification, detection, augmentation, transfer learning"):
        st.markdown(_CONCEPTS)


def _comparison(cmp, params) -> None:
    def rank(r):
        return str(r) if r else "not in top 5"

    def pct(v):
        return f"{v:.1%}" if v is not None else "-"

    rows = "".join(
        f'<tr><td>{ui.esc(r.label)}</td><td class="num">{rank(r.original_rank)}</td><td class="num">{pct(r.original_conf)}</td>'
        f'<td class="num">{rank(r.augmented_rank)}</td><td class="num">{pct(r.augmented_conf)}</td>'
        f'<td class="num">{f"{r.delta * 100:+.1f} pp" if r.delta is not None else "-"}</td>'
        f'<td><span class="chip {r.status}">{r.status}</span></td></tr>' for r in cmp.rows)
    st.markdown('<p class="vq-h" style="margin-top:.8rem">What changed</p><table class="vq-table"><tr><th>Class</th>'
                '<th class="num">Rank before</th><th class="num">Confidence before</th><th class="num">Rank after</th>'
                '<th class="num">Confidence after</th><th class="num">Change</th><th>Status</th></tr>'
                f'{rows}</table>', unsafe_allow_html=True)
    note = "No transformation is active, so both columns should match." if params.is_neutral else cmp.describe()
    st.markdown(f'<div class="vq-panel" style="margin-top:.8rem"><p class="t">What this shows</p><p>{ui.esc(note)}</p>'
                '<p>Confidence is the softmax probability the model assigns to a class. It is not accuracy. '
                'A confidence that rises or falls after a transformation shows how the output moved, not whether the '
                'prediction became more or less correct. This lab has no ImageNet ground truth for these images, '
                'so it makes no accuracy claim.</p></div>', unsafe_allow_html=True)


_TECH = """
**1. Preprocessing.** The image is converted to RGB, resized to 256 px (bicubic), centre-cropped to 224 x 224,
scaled from 0-255 to 0-1 and normalised with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation
(0.229, 0.224, 0.225). These are the transforms shipped with the pretrained weights, so the network sees the
same kind of input it was trained on.

**2. Inference.** One forward pass of MobileNetV3-Small (about 2.5 million parameters) with the model in
`eval()` mode and under `torch.inference_mode()`: no gradients are tracked and batch-norm layers use their stored
statistics. Nothing is trained or fine-tuned here.

**3. Softmax.** The network outputs 1,000 raw scores (logits) `z`. Softmax turns them into probabilities
`p_i = exp(z_i) / sum_j exp(z_j)`, which are positive and sum to 1. The five largest are shown.

**4. Confidence.** The confidence is just the probability of a class. Neural networks can be confidently wrong,
and confidence is not calibrated accuracy.

**Low-resolution note.** CIFAR-10 images are 32 x 32. They are enlarged to 256 x 256 for display and for the model,
which adds no detail, so predictions on them are often weak. That is a property of the data, not a bug.
"""

_CONCEPTS = """
- **Image classification** assigns one label (or a ranked list) to a whole image. This app does classification.
- **Object detection** finds *where* objects are, returning boxes plus labels. This app does not do detection.
- **Data augmentation** applies transformations (rotation, blur, noise, ...) to images. It is normally used to enlarge
  training sets; here it is used to probe how the model reacts.
- **Transfer learning** reuses a network trained on one large task (ImageNet) for another task, usually by fine-tuning.
  We use the pretrained network as-is, with no fine-tuning.
- **Inference** is running a trained model on new input, as opposed to training it.
- **Confidence versus accuracy.** Confidence is the probability the model reports. Accuracy is how often its answers match
  ground truth over many examples. One image's confidence says nothing certain about either.
"""
