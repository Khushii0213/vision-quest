"""Shared HTML snippets and small UI helpers. Every dynamic string is HTML-escaped."""
from __future__ import annotations

import base64
import html
import io
import time
from functools import lru_cache

import streamlit as st
from PIL import Image

from ..config import ASSETS_DIR
from . import state

NAV = (("home", "Arcade"), ("lab", "AI Lab"), ("webcam", "Webcam Lab"), ("results", "Results"))


@lru_cache(maxsize=1)
def _css() -> str:
    return (ASSETS_DIR / "styles.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _logo() -> str:
    return (ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")


def inject_css() -> None:
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def render_header(active: str) -> None:
    active = "home" if active == "play" else active
    with st.container(key="nav"):
        cols = st.columns([3, 1, 1, 1.2, 1])
        cols[0].markdown(
            f'<div class="vq-brand">{_logo()}<div><b>Vision Quest</b>'
            '<small>Computer vision arcade, UCS668 Data Centre Vision</small></div></div>',
            unsafe_allow_html=True)
        for col, (key, label) in zip(cols[1:], NAV):
            col.button(label, key=f"nav_{key}", type="primary" if key == active else "secondary",
                       on_click=state.set_screen, args=(key,))
    st.markdown('<hr class="vq-rule">', unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def data_uri(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def tile_html(uri: str, ref: str, state_cls: str = "", caption: str = "") -> str:
    cap = f'<div class="cap">{esc(caption)}</div>' if caption else ""
    return (f'<div class="vq-tile {state_cls}"><img src="{uri}" alt="Tile {esc(ref)}">'
            f'<span class="ref">{esc(ref)}</span>{cap}</div>')


def panel(kind: str, title: str, body: str) -> None:
    cls = {"error": "err", "ok": "ok"}.get(kind, "")
    st.markdown(f'<div class="vq-panel {cls}"><p class="t">{esc(title)}</p><p>{esc(body)}</p></div>',
                unsafe_allow_html=True)


def strip_html(items: list[tuple[str, object]]) -> str:
    cells = "".join(f"<div><b>{esc(v)}</b><span>{esc(k)}</span></div>" for k, v in items)
    return f'<div class="vq-strip">{cells}</div>'


def pred_rows_html(preds, statuses: dict[int, str] | None = None) -> str:
    """Horizontal bars for a top-k list. ``statuses`` maps class_index -> css class."""
    rows = []
    for rank, p in enumerate(preds, 1):
        cls = (statuses or {}).get(p.class_index, "")
        rows.append(
            f'<div class="vq-pred {cls}"><div class="lab" title="{esc(p.label)}"><small>{rank}</small>{esc(p.label)}</div>'
            f'<div class="track"><div class="fill" style="width:{max(p.confidence, 0) * 100:.1f}%"></div></div>'
            f'<div class="val">{p.confidence:.1%}</div></div>')
    return "".join(rows)


def model_error(message: str) -> None:
    panel("error", "The AI model is not available", message + " No predictions are shown, and none are faked.")
    st.button("Try again", key="retry_model")


def figure_html(img: Image.Image, caption: str, max_side: int = 512) -> str:
    shown = img.copy()
    shown.thumbnail((max_side, max_side))
    return f'<figure class="vq-fig"><img src="{data_uri(shown)}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def timed_predict(classifier, img: Image.Image, top_k: int = 5):
    """Run the real model and return (predictions, milliseconds measured on this machine)."""
    t0 = time.perf_counter()
    preds = classifier.predict(img, top_k=top_k)
    return preds, (time.perf_counter() - t0) * 1000.0
