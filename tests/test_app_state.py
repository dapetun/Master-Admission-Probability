"""Ключ параметров анализа и клиентский фильтр отчёта."""

from __future__ import annotations

from app import (
    FOCUS_ALL,
    analysis_input_key,
    analysis_ui_state_key,
    custom_probability_cache_key,
    report_probability_for_selected_scenario,
    resolve_focus_program_for_analysis,
    resolve_selected_scenario,
    selected_overlap_programs,
    stored_analysis_matches,
    threats_filter_selection,
)


def _key(**overrides):
    base = dict(
        my_code=123,
        data_dir="data/raw",
        seats_path="seats.yaml",
        include_pending=True,
        campus=None,
        monte_carlo=1000,
        focus_program=None,
        data_mtime=1.0,
        seats_mtime=2.0,
    )
    base.update(overrides)
    return analysis_input_key(**base)


def test_analysis_input_key_stable_for_same_analysis_params() -> None:
    assert _key() == _key()


def test_analysis_input_key_changes_with_analysis_params() -> None:
    original = _key()
    assert _key(my_code=456) != original
    assert _key(include_pending=False) != original
    assert _key(monte_carlo=500) != original
    assert _key(seats_path="other.yaml") != original
    assert _key(campus="Москва") != original
    assert _key(data_mtime=9.0) != original
    assert _key(focus_program="A") != original


def test_analysis_input_key_not_affected_by_selected_scenario() -> None:
    heavy_before = _key(focus_program="A")
    ui_auto = analysis_ui_state_key(
        selected_scenario="auto",
        custom_prob_enabled=False,
        custom_prob_value=0.35,
    )
    ui_balanced = analysis_ui_state_key(
        selected_scenario="balanced",
        custom_prob_enabled=False,
        custom_prob_value=0.35,
    )
    assert heavy_before == _key(focus_program="A")
    assert ui_auto != ui_balanced


def test_stored_analysis_matches() -> None:
    key = _key()
    assert stored_analysis_matches(key, key)
    assert not stored_analysis_matches(None, key)
    assert not stored_analysis_matches(key, None)
    assert not stored_analysis_matches(key, _key(my_code=456))


def test_selected_overlap_programs_keeps_user_order_of_my_programs() -> None:
    my_programs = ["A", "B", "C"]
    assert selected_overlap_programs(["C", "A", "ghost"], my_programs) == ["A", "C"]
    assert selected_overlap_programs([], my_programs) == []
    assert selected_overlap_programs(None, my_programs) == []
    assert selected_overlap_programs(my_programs, my_programs) == my_programs


def test_threats_filter_selection_resets_when_program_keys_change() -> None:
    my_programs = [
        "Прикладные модели искусственного интеллекта (Москва)",
        "Финтех (НН)",
    ]
    stale = ["Прикладные модели искусственного интеллекта"]
    assert threats_filter_selection(
        my_code=1,
        my_programs=my_programs,
        stored_code=1,
        stored_options=("Прикладные модели искусственного интеллекта", "Финтех"),
        stored_selected=stale,
    ) == my_programs
    assert threats_filter_selection(
        my_code=1,
        my_programs=my_programs,
        stored_code=1,
        stored_options=tuple(my_programs),
        stored_selected=[my_programs[0]],
    ) == [my_programs[0]]
    assert threats_filter_selection(
        my_code=1,
        my_programs=my_programs,
        stored_code=1,
        stored_options=tuple(my_programs),
        stored_selected=[],
    ) == []
    assert threats_filter_selection(
        my_code=2,
        my_programs=my_programs,
        stored_code=1,
        stored_options=tuple(my_programs),
        stored_selected=[my_programs[0]],
    ) == my_programs


def test_resolve_selected_scenario_prefers_valid_or_auto() -> None:
    probs = {"auto": object(), "balanced": object()}
    assert resolve_selected_scenario(probs, "balanced") == "balanced"
    assert resolve_selected_scenario(probs, "missing") == "auto"
    assert resolve_selected_scenario({"balanced": object()}, "missing") == "balanced"
    assert resolve_selected_scenario(None, None) == "auto"


def test_report_probability_for_selected_scenario_uses_active_or_fallback() -> None:
    auto = object()
    balanced = object()
    probs = {"auto": auto, "balanced": balanced}
    assert report_probability_for_selected_scenario(probs, "balanced") is balanced
    assert report_probability_for_selected_scenario(probs, "missing") is auto
    assert report_probability_for_selected_scenario({}, "auto") is None
    assert report_probability_for_selected_scenario(None, None) is None


def test_custom_probability_cache_key_changes_with_inputs() -> None:
    key = custom_probability_cache_key(
        selected_scenario="auto",
        consent_p=0.35,
        n_simulations=1000,
        focus_program="A",
    )
    assert key == ("auto", 0.35, 1000, "A")
    assert custom_probability_cache_key(
        selected_scenario="balanced",
        consent_p=0.35,
        n_simulations=1000,
        focus_program="A",
    ) != key
    assert custom_probability_cache_key(
        selected_scenario="auto",
        consent_p=0.4,
        n_simulations=1000,
        focus_program="A",
    ) != key


def test_resolve_focus_program_for_analysis_uses_stable_widget_choice() -> None:
    assert resolve_focus_program_for_analysis(
        monte_carlo=1000,
        focus_program_choice="A",
    ) == "A"
    assert resolve_focus_program_for_analysis(
        monte_carlo=1000,
        focus_program_choice=FOCUS_ALL,
    ) is None
    assert resolve_focus_program_for_analysis(
        monte_carlo=0,
        focus_program_choice="A",
    ) is None
