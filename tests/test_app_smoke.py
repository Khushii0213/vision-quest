"""Headless UI smoke tests using Streamlit's AppTest.

The dataset and classifier are replaced by test-only fakes (tests/fakes.py) because the real ones
need downloads. These tests check wiring and state handling, not model accuracy.
"""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import services  # noqa: E402
from src.game import Phase  # noqa: E402
from src.model import ModelUnavailableError  # noqa: E402
from tests.fakes import FakeClassifier, fake_dataset  # noqa: E402

APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    ds = fake_dataset()
    monkeypatch.setattr(services, "load_dataset", lambda: ds)
    monkeypatch.setattr(services, "load_classifier", lambda: FakeClassifier())
    monkeypatch.setattr("src.services.STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr("src.services.CifarDataset.is_downloaded", staticmethod(lambda root: True))
    return ds


def run():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at, key):
    at.button(key=key).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home_renders(env):
    at = run()
    assert at.session_state.screen == "home"
    assert any("Find the match" in m.value for m in at.markdown)
    assert at.button(key="start_run") is not None


def test_play_pick_next_flow_and_no_double_score(env):
    at = click(run(), "start_run")
    game = at.session_state.game
    assert at.session_state.screen == "play" and game.phase is Phase.PLAYING
    pos = game.round.correct_position
    click(at, f"tile_1_{pos}")
    game = at.session_state.game
    assert game.phase is Phase.REVEAL and game.score > 0
    score = game.score
    at.run()  # plain rerun must not score again
    assert at.session_state.game.score == score and len(at.session_state.game.results) == 1
    click(at, "next")
    assert at.session_state.game.round.number == 2


def test_wrong_pick_and_skip(env):
    at = click(run(), "start_run")
    g = at.session_state.game
    wrong = (g.round.correct_position + 1) % len(g.round.tiles)
    click(at, f"tile_1_{wrong}")
    assert at.session_state.game.score == 0 and at.session_state.game.lives == g.difficulty.lives - 1
    click(at, "next")
    click(at, "skip")
    assert at.session_state.game.skips_left == g.difficulty.skips - 1


def test_run_to_game_over_records_once(env, tmp_path):
    at = click(run(), "start_run")
    g = at.session_state.game
    for _ in range(g.difficulty.rounds):
        g = at.session_state.game
        click(at, f"tile_{g.round.number}_{g.round.correct_position}")
        click(at, "next")
    assert at.session_state.game.phase is Phase.OVER
    at.run()
    at.run()
    from src.storage import Storage
    data = Storage(tmp_path / "state.json").load()
    assert len(data["runs"]) == 1 and data["best_score"] == at.session_state.game.score


def test_leaving_run_abandons_without_saving(env, tmp_path):
    at = click(run(), "start_run")
    click(at, "nav_lab")
    assert at.session_state.game.over_reason == "abandoned"
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("screen", ["nav_lab", "nav_webcam", "nav_results", "nav_home"])
def test_every_screen_renders(env, screen):
    at = click(run(), screen)
    assert at.session_state.screen == screen.replace("nav_", "")


def test_lab_predictions_and_reset(env):
    at = click(run(), "nav_lab")
    assert any("Original: top 5" in m.value for m in at.markdown)
    at.slider(key="lab_blur").set_value(2.0).run()
    assert not at.exception and any("What changed" in m.value for m in at.markdown)
    click(at, "lab_reset")
    assert at.session_state.lab_blur == 0.0


def test_lab_model_unavailable_is_reported_not_faked(env, monkeypatch):
    def boom():
        raise ModelUnavailableError("weights could not be downloaded")
    monkeypatch.setattr(services, "load_classifier", boom)
    at = click(run(), "nav_lab")
    text = " ".join(m.value for m in at.markdown)
    assert "not available" in text and "none are faked" in text
    assert "Original: top 5" not in text


def test_first_run_gate_when_dataset_missing(env, monkeypatch):
    monkeypatch.setattr("src.services.CifarDataset.is_downloaded", staticmethod(lambda root: False))
    at = run()
    assert at.button(key="download_dataset") is not None
    assert at.button(key="start_run").disabled


def test_results_empty_state_and_reset_requires_confirmation(env):
    at = click(run(), "nav_results")
    assert at.button(key="reset_progress").disabled


def test_timer_fragment_expires_round_in_ui(env):
    import time
    at = click(run(), "start_run")
    at.session_state.game._clock = lambda: time.monotonic() + 1000  # simulate the deadline passing
    at.run()
    assert not at.exception
    g = at.session_state.game
    assert g.phase is Phase.REVEAL and g.last_result.outcome.value == "timeout" and g.score == 0
