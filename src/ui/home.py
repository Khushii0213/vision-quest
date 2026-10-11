from __future__ import annotations

import random

import streamlit as st

from ..augmentation import apply_augmentations, random_params
from ..config import TILE_SIZE
from ..game import DIFFICULTIES
from ..imageio import upscale_square
from .. import services
from . import components as ui
from . import state
from .setup import render_setup


def _mosaic(ds) -> str:
    """Decorative contact sheet; built once per session from a fixed seed."""
    if "mosaic" not in st.session_state:
        rng = random.Random(11)
        idx = rng.sample(range(len(ds)), 12)
        tiles = []
        for n, i in enumerate(idx):
            img = upscale_square(ds.image(i), TILE_SIZE)
            img = apply_augmentations(img, random_params(rng, 0.45 if n % 3 else 0.0), seed=n)
            ref = f"{chr(65 + n // 4)}{n % 4 + 1}"
            tiles.append(ui.tile_html(ui.data_uri(img), ref, "is-target" if n == 6 else ""))
        st.session_state.mosaic = f'<div class="vq-mosaic">{"".join(tiles)}</div>', ds.label_name(idx[6])
    return st.session_state.mosaic


def render() -> None:
    status = services.dataset_status()
    data = services.get_storage().load()
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown('<h1 class="vq-title">Find the match before the clock does.</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="vq-lede">Each round names an object. Pick the one tile that shows it from a mosaic of '
            'look-alikes that are rotated, darkened, blurred or noisy. The AI Lab then shows what a pretrained '
            'network makes of the same distortions.</p>', unsafe_allow_html=True)
        st.radio("Difficulty", ["Easy", "Normal", "Hard"], index=1, horizontal=True, key="difficulty_label")
        cfg = DIFFICULTIES[st.session_state.difficulty_label.lower()]
        st.markdown(
            f'<p class="vq-note">{cfg.rounds} rounds, {cfg.lives} lives, {cfg.skips} skips. Time falls from '
            f'{cfg.time_start:g} s to {cfg.time_floor:g} s and the grid grows from {cfg.tiles_start} to '
            f'{cfg.tiles_max} tiles. Score multiplier x{cfg.multiplier_pct / 100:g}.</p>', unsafe_allow_html=True)
        st.button("Start run", key="start_run", type="primary", on_click=state.start_run,
                  disabled=status.dataset is None)
        c1, c2 = st.columns(2)
        c1.button("Open AI Lab", key="home_lab", on_click=state.set_screen, args=("lab",))
        c2.button("Open Webcam Lab", key="home_webcam", on_click=state.set_screen, args=("webcam",))
        st.markdown(
            '<p class="vq-note">Scoring: a correct pick earns 100, plus up to 50 for speed, plus 20 for each '
            'step of your streak (up to 100), times the difficulty multiplier. Wrong, late or skipped rounds '
            'earn 0 and reset the streak.</p>', unsafe_allow_html=True)
    with right:
        if status.dataset is not None:
            html, target = _mosaic(status.dataset)
            st.markdown(f'<div class="vq-sheet">{html}<p class="vq-note" style="margin:.6rem 0 0">'
                        f'Sample mosaic. The outlined tile is a {ui.esc(target)}. Tile labels like B3 are grid references.</p></div>',
                        unsafe_allow_html=True)
        else:
            render_setup(status)
        st.markdown(ui.strip_html([("Best score", data["best_score"]), ("Rounds played", data["rounds_played"]),
                                   ("Best streak", data["best_streak"])]), unsafe_allow_html=True)
