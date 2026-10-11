"""Vision Quest: entry point.  Run with:  streamlit run app.py"""
import streamlit as st

from src.config import ASSETS_DIR

st.set_page_config(page_title="Vision Quest", page_icon=str(ASSETS_DIR / "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")

from src.ui import arcade, components as ui, home, lab, results, webcam  # noqa: E402

SCREENS = {"home": home.render, "play": arcade.render, "lab": lab.render,
           "webcam": webcam.render, "results": results.render}


def main() -> None:
    st.session_state.setdefault("screen", "home")
    ui.inject_css()
    screen = st.session_state.screen if st.session_state.screen in SCREENS else "home"
    ui.render_header(screen)
    SCREENS[screen]()


main()
