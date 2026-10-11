from __future__ import annotations

import streamlit as st

from ..game import Phase
from ..storage import Storage, StorageError
from .. import services
from . import components as ui
from . import state


def _history_svg(runs: list[dict]) -> str:
    runs = runs[-20:]
    if not runs:
        return ""
    top = max(max(r["score"] for r in runs), 1)
    w, h, gap = 640, 120, 6
    bw = (w - gap * (len(runs) - 1)) / len(runs)
    best = max(r["score"] for r in runs)
    bars = []
    for i, r in enumerate(runs):
        bh = max(2, r["score"] / top * (h - 18))
        fill = "#C7D45A" if r["score"] == best else "#477C78"
        x = i * (bw + gap)
        bars.append(f'<rect x="{x:.1f}" y="{h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{fill}" stroke="#171B24" stroke-width="1.5"/>'
                    f'<title>Run {i + 1}: {r["score"]} points</title>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Score history, oldest to newest">'
            f'<line x1="0" y1="{h}" x2="{w}" y2="{h}" stroke="#171B24" stroke-width="1.5"/>{"".join(bars)}</svg>')


def _reset_progress() -> None:
    try:
        services.get_storage().reset()
        st.session_state.reset_done = True
    except StorageError as exc:
        st.session_state.reset_error = str(exc)


def render() -> None:
    data = services.get_storage().load()
    game = st.session_state.get("game")
    live = game is not None and game.phase is not Phase.OVER
    last = data["runs"][-1] if data["runs"] else None
    st.markdown('<h1 class="vq-title" style="font-size:2.6rem">Results</h1>', unsafe_allow_html=True)

    if live:
        st.markdown(f'<p class="vq-note">A run is in progress with {game.score} points. Leaving the arcade abandons it.</p>',
                    unsafe_allow_html=True)
    current = (f"{last['score']} points on {last['difficulty']}" if last else "no finished run yet")
    avg = Storage.average_response(data)
    st.markdown(ui.strip_html([
        ("Last finished run", current), ("Best score", data["best_score"]), ("Best streak", data["best_streak"]),
        ("Average answer time", f"{avg} s" if avg is not None else "none yet")]), unsafe_allow_html=True)
    st.markdown(ui.strip_html([
        ("Rounds played", data["rounds_played"]), ("Correct", data["correct_total"]),
        ("Wrong", data["wrong_total"]), ("Runs saved", len(data["runs"]))]), unsafe_allow_html=True)

    st.markdown('<p class="vq-h" style="margin-top:1rem">Score history</p>', unsafe_allow_html=True)
    if data["runs"]:
        st.markdown(_history_svg(data["runs"]), unsafe_allow_html=True)
        st.markdown('<p class="vq-note">Last 20 runs, oldest on the left. The tallest bar is outlined in chartreuse.</p>',
                    unsafe_allow_html=True)
        rows = "".join(
            f'<tr><td>{ui.esc(r["timestamp"][:16].replace("T", " "))} UTC</td><td>{ui.esc(r["difficulty"])}</td>'
            f'<td class="num">{r["score"]}</td><td class="num">{r["correct"]} / {r["wrong"]}</td>'
            f'<td class="num">{r["best_streak"]}</td>'
            f'<td class="num">{ui.esc(r["avg_response_s"]) if r["avg_response_s"] is not None else "none"}</td></tr>'
            for r in reversed(data["runs"][-10:]))
        st.markdown('<table class="vq-table"><tr><th>Finished</th><th>Level</th><th class="num">Score</th>'
                    '<th class="num">Right / wrong</th><th class="num">Streak</th><th class="num">Avg time (s)</th></tr>'
                    f'{rows}</table>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="vq-lede">Nothing here yet. Finish a run and it will appear.</p>', unsafe_allow_html=True)
        st.button("Start a run", key="results_start", type="primary", on_click=state.start_run)

    with st.expander("Reset saved progress"):
        st.markdown('<p class="vq-note">Deletes best score, totals and history stored on this computer.</p>',
                    unsafe_allow_html=True)
        sure = st.checkbox("I want to erase all saved progress", key="confirm_reset")
        st.button("Reset progress", key="reset_progress", disabled=not sure, on_click=_reset_progress)
        if st.session_state.pop("reset_done", False):
            st.success("Progress erased.")
        if err := st.session_state.pop("reset_error", None):
            st.error(err)
