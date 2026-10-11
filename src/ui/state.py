"""Session-state actions used as button callbacks."""
from __future__ import annotations

import time

import streamlit as st

from ..game import DIFFICULTIES, Game, GameError, Phase
from .. import services


def set_screen(name: str) -> None:
    game = st.session_state.get("game")
    if game is not None and name != "play" and game.phase is not Phase.OVER:
        game.abandon()  # leaving mid-run abandons it (timers cannot run in the background)
    st.session_state.screen = name


def start_run() -> None:
    status = services.dataset_status()
    if status.dataset is None:
        return
    label = st.session_state.get("difficulty_label", "Normal")
    cfg = DIFFICULTIES.get(str(label).lower(), DIFFICULTIES["normal"])
    st.session_state.game = Game(cfg, status.dataset.index_by_class(), clock=time.monotonic)
    st.session_state.pop("assets", None)
    st.session_state.pop("last_record", None)
    st.session_state.screen = "play"


def pick_tile(position: int) -> None:
    game = st.session_state.get("game")
    if game is None:
        return
    try:
        game.submit(position)
    except GameError:
        pass  # stale/double click after the round already ended: state is unchanged, nothing is scored


def skip_round() -> None:
    game = st.session_state.get("game")
    if game is not None:
        try:
            game.skip()
        except GameError:
            pass


def next_round() -> None:
    game = st.session_state.get("game")
    if game is not None:
        try:
            game.next_round()
        except GameError:
            pass


def request_download() -> None:
    st.session_state.download_requested = True
