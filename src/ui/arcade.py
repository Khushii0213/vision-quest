from __future__ import annotations

import streamlit as st

from ..augmentation import apply_augmentations
from ..config import TILE_SIZE
from ..game import Outcome, Phase, grid_columns
from ..imageio import upscale_square
from ..storage import StorageError
from .. import services
from . import components as ui
from . import state
from .setup import render_setup


def _ref(pos: int, cols: int) -> str:
    return f"{chr(65 + pos // cols)}{pos % cols + 1}"


def _ensure_assets(game, ds) -> None:
    key = (id(game), game.round.number)
    if st.session_state.get("assets_key") == key:
        return
    r = game.round
    tiles = [ui.data_uri(apply_augmentations(upscale_square(ds.image(t.dataset_index), TILE_SIZE), t.params, seed=t.seed))
             for t in r.tiles]
    ref = ui.data_uri(upscale_square(ds.image(r.reference_index), TILE_SIZE))
    st.session_state.assets = {"tiles": tiles, "ref": ref}
    st.session_state.assets_key = key


@st.fragment(run_every=0.25)
def _timer() -> None:
    game = st.session_state.get("game")
    if game is None or game.phase is not Phase.PLAYING:
        return
    if game.expire() is not None:  # time is up: end the round and redraw the whole page
        st.rerun()
    left, limit = game.remaining(), game.round.time_limit
    low = "low" if left <= limit * 0.3 else ""
    st.markdown(f'<div class="vq-timer"><span class="t">{left:4.1f}</span><span class="vq-note">seconds left</span></div>'
                f'<div class="vq-meter {low}"><i style="width:{left / limit * 100:.1f}%"></i></div>',
                unsafe_allow_html=True)


def _result_banner(game, ds, cols: int) -> None:
    res, r = game.last_result, game.round
    target = ds.class_names[r.target_label]
    where = f"{_ref(r.correct_position, cols)}, a {ui.esc(target)}"
    if res.outcome is Outcome.CORRECT:
        cls, title = "good", f"Correct, +{res.score.total}"
        body = f"{res.score.describe()}. Answered in {res.response_time:.1f} s."
    elif res.outcome is Outcome.WRONG:
        cls, title, body = "bad", "Not that one", f"The match was tile {where}. You lost a life."
    elif res.outcome is Outcome.TIMEOUT:
        cls, title, body = "bad", "Time ran out", f"The match was tile {where}. You lost a life."
    else:
        cls, title, body = "neutral", "Round skipped", f"The match was tile {where}."
    st.markdown(f'<div class="vq-result {cls}"><p class="t">{title}</p><p>{body}</p></div>', unsafe_allow_html=True)


def _side(game, ds, cols: int) -> None:
    d, r = game.difficulty, game.round
    st.markdown(f'<p class="vq-sub">Round {r.number} of {d.rounds}, {d.label}</p>', unsafe_allow_html=True)
    st.markdown(ui.strip_html([("Score", game.score), ("Streak", game.streak), ("Lives", game.lives),
                               ("Skips", game.skips_left)]).replace("vq-strip", "vq-hud"), unsafe_allow_html=True)
    st.markdown(f'<div class="vq-target"><img src="{st.session_state.assets["ref"]}" alt="Reference photo">'
                f'<div><p class="vq-note">Find the</p><p class="name">{ui.esc(ds.class_names[r.target_label])}</p>'
                '<p class="vq-note">Reference photo, unaltered</p></div></div>', unsafe_allow_html=True)
    if game.phase is Phase.PLAYING:
        _timer()
        st.button(f"Skip round ({game.skips_left} left)", key="skip", on_click=state.skip_round,
                  disabled=game.skips_left <= 0)
    else:
        _result_banner(game, ds, cols)
        last = game.lives <= 0 or r.number >= d.rounds
        st.button("See results" if last else "Next round", key="next", type="primary", on_click=state.next_round)
    st.button("Restart run", key="restart", on_click=state.start_run)


def _grid(game, ds, cols: int) -> None:
    r, reveal = game.round, game.phase is Phase.REVEAL
    chosen = game.last_result.chosen_position if reveal else None
    uris = st.session_state.assets["tiles"]
    for start in range(0, len(r.tiles), cols):
        for col, pos in zip(st.columns(cols, gap="small"), range(start, min(start + cols, len(r.tiles)))):
            ref, cls, cap = _ref(pos, cols), "", ""
            if reveal:
                name = ds.class_names[r.tiles[pos].label]
                if pos == r.correct_position:
                    cls, cap = "is-correct", f"{name}, the match"
                elif pos == chosen:
                    cls, cap = "is-wrong", f"{name}, your pick"
                else:
                    cap = name
            with col:
                st.markdown(ui.tile_html(uris[pos], ref, cls, cap), unsafe_allow_html=True)
                st.button(f"Pick {ref}", key=f"tile_{r.number}_{pos}", on_click=state.pick_tile, args=(pos,),
                          disabled=reveal)


def _over(game) -> None:
    if not game.recorded:
        try:
            st.session_state.last_record = services.get_storage().record_run(game.summary())
        except StorageError as exc:
            st.session_state.last_record = None
            st.warning(f"This run could not be saved: {exc}")
        game.recorded = True
    rec = st.session_state.get("last_record") or {}
    s = game.summary()
    title = {"completed": "Run complete", "out_of_lives": "Out of lives"}.get(game.over_reason, "Run ended")
    st.markdown(f'<h1 class="vq-title" style="font-size:2.6rem">{title}</h1>', unsafe_allow_html=True)
    if rec.get("new_best"):
        ui.panel("ok", "New best score", f"{s['score']} points beats the previous best of {rec['previous_best']}.")
    avg = f"{s['avg_response_s']} s" if s["avg_response_s"] is not None else "none"
    st.markdown(ui.strip_html([("Score", s["score"]), ("Correct", s["correct"]), ("Wrong", s["wrong"]),
                               ("Timed out", s["timeouts"]), ("Skipped", s["skipped"]),
                               ("Avg answer time", avg), ("Best streak", s["best_streak"])]), unsafe_allow_html=True)
    c = st.columns(3)
    c[0].button("Play again", key="again", type="primary", on_click=state.start_run)
    c[1].button("See results", key="over_results", on_click=state.set_screen, args=("results",))
    c[2].button("Back to arcade", key="over_home", on_click=state.set_screen, args=("home",))


def render() -> None:
    status = services.dataset_status()
    if status.dataset is None:
        render_setup(status)
        return
    ds, game = status.dataset, st.session_state.get("game")
    if game is None:
        st.markdown('<p class="vq-lede">No run in progress.</p>', unsafe_allow_html=True)
        st.button("Start run", key="start_run_empty", type="primary", on_click=state.start_run)
        return
    if game.phase is Phase.OVER:
        _over(game)
        return
    _ensure_assets(game, ds)
    cols = grid_columns(len(game.round.tiles))
    left, right = st.columns([4, 9], gap="large")
    with left:
        _side(game, ds, cols)
    with right:
        _grid(game, ds, cols)
