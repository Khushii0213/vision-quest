"""First-run gate: shown when the dataset has not been downloaded yet."""
from __future__ import annotations

import streamlit as st

from . import components as ui
from . import state


def render_setup(status) -> None:
    if status.error:
        ui.panel("error", "The dataset could not be prepared", status.error)
        st.button("Try the download again", key="retry_dataset", type="primary")
        return
    ui.panel("info", "First launch: one download needed",
             "Vision Quest plays with CIFAR-10 (60 MB of images, about 170 MB archive). It is downloaded once "
             "into the project's data folder and reused offline afterwards. The AI Lab and Webcam Lab also "
             "download a 10 MB model on first use. An internet connection is required for this step only.")
    st.button("Download dataset", key="download_dataset", type="primary", on_click=state.request_download)
