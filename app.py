"""Простой веб-интерфейс: ввод кода → отчёт модели."""

from __future__ import annotations

import html
import math
import os
from pathlib import Path
from statistics import median

import streamlit as st
from streamlit import config as st_config

from admission_sim.load import load_dataset
from admission_sim.pipeline import (
    PERSONAL_MC_MAX_COMPETITORS,
    PERSONAL_MC_MAX_TOTAL_RUNS,
    run_analysis_all_scenarios,
)
from admission_sim.report import (
    CONSENT_BANDS_CAPTION,
    MC_SCENARIO_UNCERTAINTY_NOTE,
    PENDING_SCORE_CAPTION,
    SEAT_FILL_CAPTION,
    THREATS_SELECT_PROGRAM_CAPTION,
    build_focus_hero_data,
    build_markdown_report,
    consent_band_chart,
    consent_band_rows,
    filter_counterfactuals_by_overlap,
    focus_hero_html,
    has_consent_bands,
    mc_ci95_note,
    my_program_names,
    other_programs_note,
    pending_ahead_rows,
    prepare_threats_view,
    probability_chart_rows,
    program_fill_stats,
    resolve_hero_program,
    short_program_label,
    seat_fill_rows,
    focus_competitor_rows,
    focus_competitor_table_rows,
    threat_table_rows,
    _fmt_dest,
)

from admission_sim.pipeline import estimate_probabilities_for_applicants
from admission_sim.scenarios import SCENARIO_LABELS, ProbabilityEstimate, estimate_probability
from admission_sim.simulate import subgraph_for_program


DISCLAIMER = (
    "**Дисклеймер.** Неофициальный инструмент для личного анализа публичных "
    "обезличенных конкурсных списков. Не является расчётом приёмной комиссии, "
    "вуза или Госуслуг и не гарантирует результат зачисления."
)
LICENSE_NOTE = "Лицензия: MIT."
FOCUS_ALL = "Все мои программы"
ANALYSIS_RESULT_KEY = "analysis_result"
ANALYSIS_HEAVY_KEY_KEY = "analysis_heavy_key"
ANALYSIS_KEY_KEY = ANALYSIS_HEAVY_KEY_KEY
ANALYSIS_UI_KEY_KEY = "analysis_ui_key"
THREATS_OVERLAP_KEY = "threats_overlap_programs"
THREATS_OVERLAP_CODE_KEY = "threats_overlap_for_code"
THREATS_OVERLAP_OPTIONS_KEY = "threats_overlap_options"
COMPETITOR_PROB_CACHE_KEY = "competitor_probabilities_cache"
SELECTED_SCENARIO_KEY = "selected_scenario"
FOCUS_PROGRAM_CHOICE_KEY = "focus_program_choice"
CUSTOM_PROB_ENABLED_KEY = "user_custom_probability_enabled"
CUSTOM_PROB_VALUE_KEY = "user_custom_probability_p"
CUSTOM_PROB_CACHE_KEY = "user_custom_probability_cache_key"
CUSTOM_PROB_CACHE_EST_KEY = "user_custom_probability_est_cached"
ALLOW_UNSAFE_EXTERNAL_ENV = "ADMISSION_SIM_ALLOW_UNSAFE_EXTERNAL"
SCENARIO_ICONS = {
    "auto": "◉",
    "balanced": "◍",
    "optimistic": "▲",
    "pessimistic": "▼",
}
UI_PALETTE = {
    "bg_soft": "#F8FAFC",
    "bg_muted": "#EEF2F6",
    "bg_blue": "#EAF1FF",
    "bg_green": "#E8F5EE",
    "bg_rose": "#FCEDEF",
    "text_main": "#2F3B4A",
    "text_muted": "#5F6B7A",
    "border_neutral": "#D6DFE8",
    "border_blue": "#8AA8D8",
    "border_green": "#7FAF97",
    "border_rose": "#C99595",
    "accent_blue": "#476FA6",
}


def external_access_blocked(server_address: str | None, env_value: str | None) -> bool:
    """True, если запуск потенциально внешний и нет явного unsafe-override."""
    address = (server_address or "").strip().lower()
    unsafe_external_allowed = (env_value or "").strip() == "1"
    potentially_external = address not in {"", "localhost", "127.0.0.1", "::1"}
    return potentially_external and not unsafe_external_allowed


def _palette_css_vars() -> str:
    """CSS-переменные палитры интерфейса."""
    return "\n".join(f"    --{name}: {value};" for name, value in UI_PALETTE.items())


def _section_header(title: str, *, tone: str = "blue") -> str:
    """Возвращает цветной заголовок секции с читаемым контрастом."""
    safe_title = html.escape(title)
    return (
        f'<div class="section-title section-title--{tone}">'
        f'<span class="section-title__label">{safe_title}</span>'
        "</div>"
    )


def _scenario_badge_line(scenario: str, external: float, none: float) -> str:
    """Строка статуса выбранного сценария: цвет + текстовые маркеры."""
    safe_label = html.escape(SCENARIO_LABELS.get(scenario, scenario))
    safe_scenario = html.escape(scenario)
    icon = SCENARIO_ICONS.get(scenario, "•")
    return (
        '<div class="scenario-status">'
        f'<span class="scenario-badge scenario-badge--{safe_scenario}">{icon} {safe_label}</span>'
        f'<span class="scenario-status__meta">Не в загруженных '
        f'(или данные неполные): {external:.1%}</span>'
        f'<span class="scenario-status__meta">Без зачисления: {none:.1%}</span>'
        "</div>"
    )


def app_styles() -> str:
    """Единый CSS приложения: отступы, hero-блок, мягкие callout."""
    return (
        "<style>\n:root {\n"
        + _palette_css_vars()
        + "\n}\n"
        + """
.block-container {
    padding-top: 1.35rem;
}
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border_neutral);
    background: linear-gradient(180deg, #f9fbfd 0%, #f3f7fb 100%);
}
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text_main);
}
[data-testid="stMetric"] {
    background: var(--bg_soft);
    border: 1px solid var(--border_neutral);
    border-left: 4px solid var(--border_blue);
    border-radius: 10px;
    padding: 0.65rem 0.75rem;
    box-shadow: 0 1px 2px rgba(69, 67, 64, 0.04);
    min-width: 0;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 2px 6px rgba(70, 95, 130, 0.12);
}
[data-testid="stMetricValue"] {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
}
.section-title {
    margin: 0.35rem 0 0.55rem;
}
.section-title__label {
    display: inline-block;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    padding: 0.38rem 0.72rem;
    border-radius: 999px;
    border: 1px solid;
}
.section-title--blue .section-title__label {
    background: var(--bg_blue);
    border-color: var(--border_blue);
    color: #2f4f7f;
}
.section-title--green .section-title__label {
    background: var(--bg_green);
    border-color: var(--border_green);
    color: #2f6650;
}
.section-title--rose .section-title__label {
    background: var(--bg_rose);
    border-color: var(--border_rose);
    color: #7a4a4a;
}
.focus-hero {
    background: linear-gradient(180deg, #eef4fb 0%, #f5f8fb 100%);
    border: 1px solid var(--border_blue);
    border-radius: 12px;
    padding: 1.35rem 1.5rem 1.15rem;
    margin: 0.35rem 0 1rem;
    box-shadow: 0 1px 3px rgba(69, 67, 64, 0.05);
}
.focus-hero__title {
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1.25;
    color: var(--text_main);
    margin: 0 0 0.45rem;
}
.focus-hero__meta {
    color: var(--text_muted);
    font-size: 0.92rem;
    line-height: 1.45;
    margin: 0 0 0.95rem;
}
.focus-hero__callout {
    background: var(--bg_blue);
    border: 1px solid #bdd1eb;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 1.1rem;
    color: #5A6570;
    font-size: 0.92rem;
    line-height: 1.45;
}
.focus-hero__callout-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.focus-hero__metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
}
@media (max-width: 900px) {
    .focus-hero__metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
.focus-hero__metric {
    background: var(--bg_soft);
    border: 1px solid var(--border_neutral);
    border-top: 3px solid #b8c7da;
    border-radius: 10px;
    padding: 0.85rem 0.65rem 0.75rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(69, 67, 64, 0.03);
}
.focus-hero__metric--active {
    border-color: var(--border_blue);
    border-top-color: var(--accent_blue);
    box-shadow: inset 0 0 0 1px #c8d9ee;
    transform: translateY(-1px);
}
.focus-hero__metric-value {
    font-size: 1.85rem;
    font-weight: 600;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}
.focus-hero__metric-label {
    color: var(--text_muted);
    font-size: 0.82rem;
    line-height: 1.25;
}
.st-key-hero_scenario_control {
    background: var(--bg_muted);
    border: 1px solid var(--border_neutral);
    border-radius: 12px;
    padding: 0.5rem;
    margin-bottom: 0.55rem;
}
.st-key-hero_scenario_control [role="radiogroup"] button[aria-pressed="true"] {
    border: 1px solid var(--border_blue) !important;
    background: var(--bg_blue) !important;
    color: #2d4f81 !important;
    font-weight: 700 !important;
}
.st-key-threats_filters_tint,
.st-key-competitors_filters_tint,
.st-key-custom_prob_tint {
    background: linear-gradient(180deg, #f8fbff 0%, #f3f8ff 100%);
    border: 1px solid #d8e4f4;
    border-radius: 12px;
    padding: 0.4rem 0.7rem 0.15rem;
    margin: 0.35rem 0 0.5rem;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border_neutral);
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    background: #edf3fb !important;
    color: #2e4e7b !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"] [role="row"]:nth-child(odd) [role="gridcell"] {
    background: #f9fbfe !important;
}
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
    background: #eef4ff !important;
}
.scenario-status {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.45rem;
}
.scenario-badge {
    display: inline-flex;
    align-items: center;
    border: 1px solid;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.86rem;
    font-weight: 700;
}
.scenario-badge--auto {
    border-color: var(--border_blue);
    background: var(--bg_blue);
    color: #2f4f7f;
}
.scenario-badge--balanced {
    border-color: #9ab6cf;
    background: #eef5ff;
    color: #395977;
}
.scenario-badge--optimistic {
    border-color: var(--border_green);
    background: var(--bg_green);
    color: #2d664f;
}
.scenario-badge--pessimistic {
    border-color: var(--border_rose);
    background: var(--bg_rose);
    color: #784a4a;
}
.scenario-status__meta {
    color: #4f5f73;
    background: #f5f8fc;
    border: 1px solid #d8e1eb;
    border-radius: 8px;
    padding: 0.14rem 0.45rem;
    font-size: 0.82rem;
    font-weight: 600;
}
.score-range-indicator {
    margin: 0.2rem 0 0.45rem;
}
.score-range-indicator__track {
    width: 100%;
    height: 0.35rem;
    border-radius: 999px;
    background: linear-gradient(
        to right,
        #D1D5DB 0%,
        #D1D5DB var(--inactive-width, 0%),
        #C7CDD4 var(--inactive-width, 0%),
        #C7CDD4 100%
    );
}
</style>
"""
    )


def metric_destination_text(dest: str | None) -> str:
    """Короткий текст для st.metric с обрезкой длинных названий программ."""
    full_text = _fmt_dest(dest)
    if dest is None:
        return full_text
    return short_program_label(full_text)


def analysis_input_key(
    *,
    my_code: int,
    data_dir: str,
    seats_path: str,
    include_pending: bool,
    campus: str | None,
    monte_carlo: int,
    focus_program: str | None,
    data_mtime: float,
    seats_mtime: float,
) -> tuple:
    """Параметры, от которых зависит расчёт (не фильтры отображения отчёта)."""
    return (
        my_code,
        data_dir,
        seats_path,
        include_pending,
        campus,
        int(monte_carlo),
        focus_program,
        data_mtime,
        seats_mtime,
    )


def stored_analysis_matches(
    stored_key: tuple | None,
    current_key: tuple | None,
) -> bool:
    """Есть ли сохранённый результат для текущих параметров анализа."""
    return (
        stored_key is not None
        and current_key is not None
        and stored_key == current_key
    )


def analysis_ui_state_key(
    *,
    selected_scenario: str | None,
    custom_prob_enabled: bool,
    custom_prob_value: float,
) -> tuple[str | None, bool, float]:
    """UI-ключ отчёта: влияет на представление, но не на тяжёлый расчёт."""
    return (
        selected_scenario,
        bool(custom_prob_enabled),
        round(float(custom_prob_value), 6),
    )


def resolve_focus_program_for_analysis(
    *,
    monte_carlo: int,
    focus_program_choice: str | None,
) -> str | None:
    """Нормализует выбор focus-программы для ключа и запуска MC."""
    if int(monte_carlo) <= 0:
        return None
    if not focus_program_choice or focus_program_choice == FOCUS_ALL:
        return None
    return str(focus_program_choice)


def selected_overlap_programs(
    selected: list[str] | None,
    my_programs: list[str],
) -> list[str]:
    """Оставляет выбранные программы пересечения, которые есть у поступающего."""
    if not selected:
        return []
    allowed = set(selected)
    return [name for name in my_programs if name in allowed]


def threats_filter_selection(
    *,
    my_code: int,
    my_programs: list[str],
    stored_code: int | None,
    stored_options: tuple[str, ...] | None,
    stored_selected: list[str] | None,
) -> list[str]:
    """Выбор фильтра угроз: сброс, если сменился код или ключи программ."""
    if stored_code != my_code or stored_options != tuple(my_programs):
        return list(my_programs)
    return selected_overlap_programs(stored_selected, my_programs)


def competitors_score_slider_config(
    score_values: list[float],
    me_score: float | None,
) -> dict[str, object]:
    """Готовит безопасный диапазон и подписи для фильтра баллов конкурентов."""
    valid_scores: list[float] = []
    for score in score_values:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid_scores.append(value)
    if not valid_scores:
        return {
            "min_value": 0.0,
            "max_value": 0.0,
            "value": (0.0, 0.0),
            "disabled": True,
            "caption": "Нет данных по баллам: фильтр отключён.",
            "inactive_share": 0.0,
        }

    real_min = min(valid_scores)
    real_max = max(valid_scores)
    min_selectable = real_min
    caption: str | None = None

    if me_score is not None and math.isfinite(float(me_score)):
        clamped_me = min(max(float(me_score), real_min), real_max)
        min_selectable = max(real_min, clamped_me)
        if float(me_score) > real_max:
            caption = "Ваш балл выше максимума конкурентов; доступна только верхняя точка диапазона."
        elif min_selectable > real_min:
            caption = f"Ниже вашего балла ({float(me_score):g}) диапазон недоступен."

    span = real_max - real_min
    inactive_share = 0.0 if span <= 0 else max(0.0, min(1.0, (min_selectable - real_min) / span))
    return {
        "min_value": float(min_selectable),
        "max_value": float(real_max),
        "value": (float(min_selectable), float(real_max)),
        "disabled": False,
        "caption": caption,
        "inactive_share": float(inactive_share),
    }


def _dir_mtime(path: Path) -> float:
    """Максимальный mtime файлов в каталоге (инвалидация кэша Streamlit)."""
    if not path.is_dir():
        return 0.0
    files = list(path.glob("*_Budget.xlsx")) + list(path.glob("*.csv"))
    if not files:
        return 0.0
    return max(f.stat().st_mtime for f in files)


@st.cache_resource(max_entries=8, show_spinner="Загрузка списков…")
def _load_dataset_cached(
    data_dir_str: str,
    seats_path_str: str,
    include_pending: bool,
    campus: str | None,
    data_mtime: float,
    seats_mtime: float,
):
    """Общий Dataset: симуляция его не меняет, копировать через cache_data дорого."""
    return load_dataset(
        Path(data_dir_str),
        Path(seats_path_str),
        include_pending=include_pending,
        campus=campus,
    )


@st.fragment
def _render_threats_section() -> None:
    """Фильтр угроз без полного пересчёта страницы (результат — в session_state)."""
    result = st.session_state[ANALYSIS_RESULT_KEY]
    dest = result.vpp.destination(result.my_code)
    my_programs = my_program_names(result.dataset, result.my_code)
    threats_view = prepare_threats_view(result.counterfactuals, dest)
    st.markdown(_section_header(threats_view.title, tone="rose"), unsafe_allow_html=True)
    skip_threats_table = False
    if result.counterfactuals and my_programs:
        next_selection = threats_filter_selection(
            my_code=result.my_code,
            my_programs=my_programs,
            stored_code=st.session_state.get(THREATS_OVERLAP_CODE_KEY),
            stored_options=st.session_state.get(THREATS_OVERLAP_OPTIONS_KEY),
            stored_selected=st.session_state.get(THREATS_OVERLAP_KEY),
        )
        if st.session_state.get(THREATS_OVERLAP_KEY) != next_selection:
            st.session_state[THREATS_OVERLAP_KEY] = next_selection
        st.session_state[THREATS_OVERLAP_CODE_KEY] = result.my_code
        st.session_state[THREATS_OVERLAP_OPTIONS_KEY] = tuple(my_programs)
        with st.container(key="threats_filters_tint"):
            selected_overlap = selected_overlap_programs(
                st.multiselect(
                    "Программы пересечения",
                    options=my_programs,
                    placeholder=THREATS_SELECT_PROGRAM_CAPTION,
                    help="Показываем только тех, кто пересекается с выбранными программами.",
                    key=THREATS_OVERLAP_KEY,
                ),
                my_programs,
            )
        if not selected_overlap:
            st.caption(THREATS_SELECT_PROGRAM_CAPTION)
            skip_threats_table = True
        else:
            threats_view = prepare_threats_view(
                filter_counterfactuals_by_overlap(
                    result.counterfactuals,
                    set(selected_overlap),
                ),
                dest,
                filtered=True,
            )
    if skip_threats_table:
        return
    if threats_view.caption:
        for paragraph in threats_view.caption.split("\n\n"):
            st.caption(paragraph)
    if threats_view.empty_message:
        st.write(threats_view.empty_message)
    elif threats_view.shown:
        st.dataframe(threat_table_rows(threats_view), hide_index=True)
    if threats_view.count_caption:
        st.caption(threats_view.count_caption)


def _hero_scenario_shares(
    result,
    focus_program: str,
) -> dict[str, float]:
    """Доли прогонов по сценариям для hero берём из уже посчитанных вероятностей."""
    probs = getattr(result, "probabilities_by_scenario", {}) or {}
    out: dict[str, float] = {}
    for scenario in SCENARIO_LABELS:
        prob = probs.get(scenario)
        out[scenario] = prob.by_program.get(focus_program, 0.0) if prob else 0.0
    return out


def resolve_selected_scenario(
    probabilities_by_scenario: dict[str, ProbabilityEstimate] | None,
    selected: str | None,
) -> str:
    """Возвращает валидный сценарий выбора для UI (с дефолтом auto)."""
    probs = probabilities_by_scenario or {}
    if selected in probs:
        return str(selected)
    if "auto" in probs:
        return "auto"
    return next(iter(probs), "auto")


def _render_scenario_selector(
    *,
    scenario_shares: dict[str, float],
    selected_scenario: str,
) -> str:
    """Нативный выбор сценария плашками через segmented_control."""
    options = [scenario for scenario in SCENARIO_LABELS if scenario in scenario_shares]
    if not options:
        return selected_scenario

    labels = {
        scenario: (
            f"{SCENARIO_ICONS.get(scenario, '•')} "
            f"{SCENARIO_LABELS.get(scenario, scenario)} · "
            f"{scenario_shares.get(scenario, 0.0):.1%}"
        )
        for scenario in options
    }
    # Один источник истины для выбранного сценария: widget key в session_state.
    # Это исключает ручной double-rerun и "прыжки" страницы.
    if st.session_state.get(SELECTED_SCENARIO_KEY) not in options:
        st.session_state[SELECTED_SCENARIO_KEY] = (
            selected_scenario if selected_scenario in options else options[0]
        )

    selected = st.segmented_control(
        "Сценарий в деталях",
        options=options,
        selection_mode="single",
        default=st.session_state.get(SELECTED_SCENARIO_KEY, options[0]),
        format_func=lambda scenario: labels.get(scenario, str(scenario)),
        key=SELECTED_SCENARIO_KEY,
    )
    return str(selected) if selected else options[0]


def report_probability_for_selected_scenario(
    probabilities_by_scenario: dict[str, ProbabilityEstimate] | None,
    selected_scenario: str | None,
) -> ProbabilityEstimate | None:
    """Возвращает вероятность для активного сценария в report."""
    probs = probabilities_by_scenario or {}
    if not probs:
        return None
    selected = resolve_selected_scenario(probs, selected_scenario)
    return probs.get(selected) or next(iter(probs.values()))


def custom_probability_cache_key(
    *,
    selected_scenario: str,
    consent_p: float,
    n_simulations: int,
    focus_program: str | None,
) -> tuple[str, float, int, str | None]:
    """Ключ кэша пользовательской вероятности для report."""
    return (selected_scenario, round(float(consent_p), 6), int(n_simulations), focus_program)


def _estimate_user_probability_for_report(
    result,
    *,
    selected_scenario: str,
    consent_p: float,
) -> ProbabilityEstimate | None:
    """Считает/кэширует пользовательскую вероятность для выбранного сценария."""
    selected_prob = report_probability_for_selected_scenario(
        getattr(result, "probabilities_by_scenario", None),
        selected_scenario,
    )
    if selected_prob is None or selected_prob.n_simulations <= 0:
        return None

    cache_key = custom_probability_cache_key(
        selected_scenario=selected_prob.scenario,
        consent_p=consent_p,
        n_simulations=int(selected_prob.n_simulations),
        focus_program=selected_prob.focus_program,
    )
    cached_key = st.session_state.get(CUSTOM_PROB_CACHE_KEY)
    cached_prob = st.session_state.get(CUSTOM_PROB_CACHE_EST_KEY)
    if cached_prob is not None and cached_key == cache_key:
        return cached_prob

    workers = min(4, os.cpu_count() or 1)
    user_probability = estimate_probability(
        result.dataset.applicants,
        result.dataset.seats,
        result.my_code,
        n_simulations=int(selected_prob.n_simulations),
        scenario=selected_prob.scenario,
        consent_probability=float(consent_p),
        seed=42,
        n_workers=workers,
        focus_program=selected_prob.focus_program,
    )
    st.session_state[CUSTOM_PROB_CACHE_KEY] = cache_key
    st.session_state[CUSTOM_PROB_CACHE_EST_KEY] = user_probability
    return user_probability


def _render_custom_probability_controls(
    result,
    *,
    selected_scenario: str,
) -> None:
    """Рендерит пользовательскую вероятность рядом с выбором сценария."""
    selected_prob = report_probability_for_selected_scenario(
        getattr(result, "probabilities_by_scenario", None),
        selected_scenario,
    )
    allow_custom = selected_prob is not None and selected_prob.n_simulations > 0
    if not allow_custom:
        return

    with st.container(border=True, key="custom_prob_tint"):
        st.caption("Report: пользовательская вероятность")
        enabled = st.checkbox("Задать вручную", key=CUSTOM_PROB_ENABLED_KEY)
        if enabled:
            st.slider(
                "Вероятность согласия",
                0.0,
                1.0,
                0.35,
                0.05,
                key=CUSTOM_PROB_VALUE_KEY,
            )


def _render_focus_hero(
    result,
    *,
    active_scenario: str,
) -> dict[str, float]:
    """Hero-блок фокусной программы со сравнением сценариев."""
    probs = getattr(result, "probabilities_by_scenario", {}) or {}
    if not probs:
        return {}

    p = probs.get(active_scenario) or next(iter(probs.values()))
    if p is None:
        return {}

    focus_program = resolve_hero_program(my_program_names(result.dataset, result.my_code), p)
    if focus_program is None:
        return {}

    scenario_shares = _hero_scenario_shares(result, focus_program)
    hero = build_focus_hero_data(
        result.dataset,
        result.my_code,
        p,
        scenario_shares=scenario_shares,  # type: ignore[arg-type]
        active_scenario=active_scenario,  # type: ignore[arg-type]
        manual_consent=False,
    )
    if hero is None:
        return {}
    st.markdown(focus_hero_html(hero), unsafe_allow_html=True)
    return scenario_shares


@st.fragment
def _render_report_download() -> None:
    """Скачивание и Markdown: тяжёлый рендер только при открытом блоке."""
    result = st.session_state[ANALYSIS_RESULT_KEY]

    selected_scenario = resolve_selected_scenario(
        getattr(result, "probabilities_by_scenario", None),
        st.session_state.get(SELECTED_SCENARIO_KEY),
    )
    user_probability = None
    if st.session_state.get(CUSTOM_PROB_ENABLED_KEY, False):
        consent_p = float(st.session_state.get(CUSTOM_PROB_VALUE_KEY, 0.35))
        user_probability = _estimate_user_probability_for_report(
            result,
            selected_scenario=selected_scenario,
            consent_p=consent_p,
        )

    md = build_markdown_report(
        result.dataset,
        result.my_code,
        vpp=result.vpp,
        vpp_if_consent=result.vpp_if_consent,
        ovp=result.ovp,
        counterfactuals=result.counterfactuals,
        probability=report_probability_for_selected_scenario(
            result.probabilities_by_scenario,
            selected_scenario,
        ),
        include_pending=result.include_pending,
        consent_model=result.consent_models_by_scenario,
        user_probability=user_probability,
    )
    st.download_button("Скачать report.md", md, file_name="report.md")
    md_exp = st.expander("Полный Markdown", on_change="rerun")
    if md_exp.open:
        with md_exp:
            st.markdown(md)


def _probability_top_program(probability: ProbabilityEstimate) -> tuple[str, float]:
    """Топ-направление по MC для одной персоны."""
    if probability.by_program:
        program, share = max(probability.by_program.items(), key=lambda item: item[1])
        return program, float(share)
    if probability.external >= probability.none:
        return "не в загруженных (или данные неполные)", float(probability.external)
    return "не зачислен", float(probability.none)


def _render_focus_competitors_section(
    result,
    *,
    selected_scenario: str,
) -> None:
    """Таблица конкурентов на выбранной программе + опциональный персональный MC."""
    my_programs = my_program_names(result.dataset, result.my_code)
    if not my_programs:
        return

    focus_program = st.selectbox(
        "Программа для секции конкурентов",
        options=my_programs,
        key="competitors_focus_program",
    )
    base_rows = focus_competitor_rows(
        result.dataset,
        result.my_code,
        focus_program,
        vpp=result.vpp,
        ovp=result.ovp,
    )
    if not base_rows:
        st.info("На этой программе нет других абитуриентов в загруженных списках.")
        return

    with st.container(key="competitors_filters_tint"):
        only_above = st.checkbox("Только выше меня", value=True, key="competitors_only_above")
        only_consent = st.checkbox(
            "Только с поданным согласием",
            value=False,
            key="competitors_only_consent",
        )
        code_query = st.text_input(
            "Поиск по коду",
            value="",
            placeholder="например 123456",
            key="competitors_code_query",
        ).strip()

        me_profile = result.dataset.applicants.get(result.my_code)
        me_app = me_profile.application_for(focus_program) if me_profile is not None else None
        me_score = float(me_app.score) if me_app is not None else None
        slider_config = competitors_score_slider_config(
            [float(row.score) for row in base_rows],
            me_score,
        )
        inactive_share = float(slider_config["inactive_share"])
        if inactive_share > 0:
            st.markdown(
                (
                    '<div class="score-range-indicator">'
                    f'<div class="score-range-indicator__track" style="--inactive-width:{inactive_share * 100:.1f}%"></div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        caption_text = slider_config["caption"]
        if isinstance(caption_text, str) and caption_text:
            st.caption(caption_text)
        min_value = float(slider_config["min_value"])
        max_value = float(slider_config["max_value"])
        default_range = slider_config["value"]
        stored_range = st.session_state.get("competitors_score_range")
        if (
            isinstance(stored_range, tuple)
            and len(stored_range) == 2
            and all(isinstance(item, (int, float)) for item in stored_range)
        ):
            left = min(max(float(stored_range[0]), min_value), max_value)
            right = min(max(float(stored_range[1]), min_value), max_value)
            normalized_range = (min(left, right), max(left, right))
        else:
            normalized_range = default_range
        st.session_state["competitors_score_range"] = normalized_range
        score_range = st.slider(
            "Диапазон баллов",
            min_value=min_value,
            max_value=max_value,
            value=normalized_range,
            step=0.1,
            disabled=bool(slider_config["disabled"]),
            key="competitors_score_range",
        )

    filtered = [
        row
        for row in base_rows
        if (not only_above or row.above_me)
        and (not only_consent or row.consent)
        and score_range[0] <= row.score <= score_range[1]
        and (not code_query or code_query in str(row.code))
    ]
    if not filtered:
        st.info("По текущим фильтрам конкурентов не найдено.")
        return

    above_rows = [row for row in filtered if row.above_me]
    above_consent = [row for row in above_rows if row.consent]
    pri1_on_focus = [row for row in above_rows if row.priority_on_focus == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Выше меня", len(above_rows))
    c2.metric("Выше и с согласием", len(above_consent))
    c3.metric(
        "Медианный балл",
        f"{median([row.score for row in above_rows]):g}"
        if above_rows
        else "—",
    )
    c4.metric("Выше с pri1", len(pri1_on_focus))
    if above_rows:
        mean_score = sum(row.score for row in above_rows) / len(above_rows)
        st.caption(f"Средний балл выше вас: {mean_score:.2f}")

    table_rows = focus_competitor_table_rows(filtered)

    calc_prob = st.checkbox(
        "Посчитать персональные MC-вероятности «куда поступит»",
        value=False,
        key="competitors_calc_prob",
        help="Считает для каждого конкурента его шанс поступления по текущим данным.",
    )
    if calc_prob:
        default_n = 500
        if result.probabilities_by_scenario:
            sample = result.probabilities_by_scenario.get(selected_scenario) or next(
                iter(result.probabilities_by_scenario.values())
            )
            default_n = int(sample.n_simulations)
        n_sim = st.slider(
            "Прогонов на одного конкурента",
            min_value=100,
            max_value=2000,
            value=min(1000, max(100, default_n)),
            step=100,
            key="competitors_prob_n_sim",
        )
        codes = sorted({row.code for row in filtered})
        codes_total = len(codes)
        capped_codes = codes[:PERSONAL_MC_MAX_COMPETITORS]
        if codes_total > len(capped_codes):
            st.warning(
                f"Персональный MC ограничен: считаю первые {len(capped_codes)} "
                f"конкурентов из {codes_total} по текущей сортировке."
            )
        effective_runs = min(
            int(n_sim),
            max(1, PERSONAL_MC_MAX_TOTAL_RUNS // len(capped_codes)),
        )
        if effective_runs < int(n_sim):
            st.warning(
                f"Сработал лимит вычислительного бюджета: {len(capped_codes)} "
                f"конкурентов × {effective_runs} прогонов "
                f"(вместо {n_sim})."
            )
        cache = st.session_state.setdefault(COMPETITOR_PROB_CACHE_KEY, {})
        cache_key = (
            st.session_state.get(ANALYSIS_KEY_KEY),
            focus_program,
            tuple(capped_codes),
            int(effective_runs),
        )
        estimates = cache.get(cache_key)
        if estimates is None:
            with st.spinner("Считаю персональные вероятности конкурентов…"):
                estimates = estimate_probabilities_for_applicants(
                    result.dataset,
                    capped_codes,
                    n_simulations=int(effective_runs),
                    scenario=selected_scenario,
                    seed=42,
                    consent_model=result.consent_models_by_scenario.get(selected_scenario),
                )
            cache[cache_key] = estimates
        for row in table_rows:
            estimate = estimates.get(int(row["Код"]))
            if estimate is None:
                row["Шанс на программу, %"] = "—"
                row["Шанс на любую, %"] = "—"
                row["Топ-назначение"] = "—"
                continue
            top_program, top_share = _probability_top_program(estimate)
            row["Шанс на программу, %"] = round(
                estimate.by_program.get(focus_program, 0.0) * 100, 1
            )
            row["Шанс на любую, %"] = round(estimate.any_loaded * 100, 1)
            row["Топ-назначение"] = f"{top_program} ({top_share:.0%})"

        code_for_details = st.selectbox(
            "Быстрый просмотр распределения по программам",
            options=capped_codes,
            key="competitors_prob_preview_code",
        )
        picked = estimates.get(int(code_for_details))
        if picked is not None:
            detail_rows = [
                {"Программа": program, "Доля прогонов": f"{share:.1%}"}
                for program, share in picked.by_program.items()
            ]
            detail_rows.extend(
                [
                    {
                        "Программа": "не в загруженных (или данные неполные)",
                        "Доля прогонов": f"{picked.external:.1%}",
                    },
                    {"Программа": "не зачислен", "Доля прогонов": f"{picked.none:.1%}"},
                ]
            )
            st.dataframe(detail_rows, hide_index=True)
            st.caption(mc_ci95_note(picked))

    st.dataframe(
        table_rows,
        hide_index=True,
        column_config={"_above_me": None},
    )


def main() -> None:
    st.set_page_config(page_title="Симулятор поступления", layout="wide")
    address = st_config.get_option("server.address")
    if external_access_blocked(address, os.environ.get(ALLOW_UNSAFE_EXTERNAL_ENV)):
        st.error(
            "Внешний доступ отключён по умолчанию для безопасности. "
            f"Сейчас server.address={address!r}. "
            f"Для осознанного небезопасного запуска установите "
            f"{ALLOW_UNSAFE_EXTERNAL_ENV}=1."
        )
        st.stop()

    st.markdown(app_styles(), unsafe_allow_html=True)
    st.session_state.setdefault(ANALYSIS_RESULT_KEY, None)
    st.session_state.setdefault(ANALYSIS_KEY_KEY, None)
    st.session_state.setdefault(ANALYSIS_UI_KEY_KEY, None)
    st.session_state.setdefault(SELECTED_SCENARIO_KEY, "auto")
    st.session_state.setdefault(FOCUS_PROGRAM_CHOICE_KEY, FOCUS_ALL)
    st.session_state.setdefault(CUSTOM_PROB_ENABLED_KEY, False)
    st.session_state.setdefault(CUSTOM_PROB_VALUE_KEY, 0.35)
    st.title("Симулятор приоритетов магистратуры")
    st.info(DISCLAIMER)
    st.caption(LICENSE_NOTE)

    default_data = Path("data/raw")
    if not (default_data.is_dir() and any(default_data.glob("*.csv"))):
        if list(Path(".").glob("*.csv")):
            default_data = Path(".")

    with st.sidebar:
        st.header("Параметры")
        st.caption("Неофициально · не приёмная комиссия · MIT")
        my_code_raw = st.text_input(
            "Код поступающего",
            value="",
            placeholder="ваш код из списка",
        )
        data_dir = Path(
            st.text_input("Каталог CSV", value=str(default_data))
        )
        seats_path = Path(st.text_input("Файл мест", value="seats.yaml"))
        campus = st.text_input(
            "Кампус (пусто = весь вуз)",
            value="",
        )
        include_pending = st.checkbox(
            "Учитывать ожидание результатов вступительных испытаний",
            value=True,
            help="Если выключить, заявки без итоговых баллов не попадут в расчёт.",
        )
        monte_carlo = st.slider(
            "Случайных прогонов",
            0,
            5000,
            1000,
            100,
            help="Сколько раз повторять случайный сценарий согласий. 0 — выключено.",
        )
        focus_slot = st.container()
        run = st.button("Рассчитать", type="primary", width="stretch")

        campus_val_preview = campus.strip() or None
        user_programs: list[str] = []
        preview_dataset = None
        focus_program_preview = None
        code_preview = my_code_raw.strip()
        if code_preview.isdigit() and data_dir.is_dir():
            with focus_slot.skeleton():
                try:
                    preview_dataset = _load_dataset_cached(
                        str(data_dir),
                        str(seats_path),
                        include_pending,
                        campus_val_preview,
                        _dir_mtime(data_dir),
                        seats_path.stat().st_mtime if seats_path.is_file() else 0.0,
                    )
                    uid = int(code_preview)
                    if uid in preview_dataset.applicants:
                        user_programs = my_program_names(preview_dataset, uid)
                except (FileNotFoundError, ValueError, OSError):
                    preview_dataset = None
        with focus_slot:
            if user_programs and monte_carlo > 0:
                focus_choice = st.selectbox(
                    "Программа для случайных прогонов",
                    options=[FOCUS_ALL, *user_programs],
                    index=0,
                    key=FOCUS_PROGRAM_CHOICE_KEY,
                    help="Одна программа — быстрее. «Все мои программы» — полный расчёт.",
                )
                if focus_choice != FOCUS_ALL:
                    focus_program_preview = focus_choice
                    if preview_dataset is not None:
                        reduced, red_seats = subgraph_for_program(
                            preview_dataset.applicants,
                            preview_dataset.seats,
                            focus_choice,
                        )
                        st.caption(
                            f"В случайные прогоны войдут {len(red_seats)} конкурсов "
                            f"из {len(preview_dataset.programs)} и "
                            f"{len(reduced)} абитуриентов "
                            f"из {len(preview_dataset.applicants)}."
                        )
            if monte_carlo >= 3000:
                campus_hint = (
                    " Укажите кампус — расчёт быстрее."
                    if not campus.strip()
                    else ""
                )
                focus_hint = (
                    " Или выберите одну программу выше."
                    if not focus_program_preview
                    else ""
                )
                st.warning(
                    f"При {monte_carlo} прогонах на полных данных ожидайте "
                    f"примерно {max(1, monte_carlo // 40)}–{max(2, monte_carlo // 25)} с "
                    f"(4 ядра). "
                    f"Для быстрой оценки достаточно 500–1000.{campus_hint}{focus_hint}"
                )

    code_text = my_code_raw.strip()
    campus_val = campus.strip() or None
    focus_program_for_analysis = resolve_focus_program_for_analysis(
        monte_carlo=int(monte_carlo),
        focus_program_choice=st.session_state.get(FOCUS_PROGRAM_CHOICE_KEY),
    )
    current_ui_key = analysis_ui_state_key(
        selected_scenario=st.session_state.get(SELECTED_SCENARIO_KEY),
        custom_prob_enabled=bool(st.session_state.get(CUSTOM_PROB_ENABLED_KEY, False)),
        custom_prob_value=float(st.session_state.get(CUSTOM_PROB_VALUE_KEY, 0.35)),
    )
    data_mtime = _dir_mtime(data_dir)
    seats_mtime = seats_path.stat().st_mtime if seats_path.is_file() else 0.0
    current_key = (
        analysis_input_key(
            my_code=int(code_text),
            data_dir=str(data_dir),
            seats_path=str(seats_path),
            include_pending=include_pending,
            campus=campus_val,
            monte_carlo=int(monte_carlo),
            focus_program=focus_program_for_analysis,
            data_mtime=data_mtime,
            seats_mtime=seats_mtime,
        )
        if code_text.isdigit()
        else None
    )

    result = None
    if run:
        if current_key is None:
            st.error("Укажите числовой код поступающего.")
            return
        my_code = int(code_text)
        try:
            dataset = _load_dataset_cached(
                str(data_dir),
                str(seats_path),
                include_pending,
                campus_val,
                data_mtime,
                seats_mtime,
            )
            with st.spinner("Расчёт…"):
                result = run_analysis_all_scenarios(
                    data_dir,
                    seats_path,
                    my_code,
                    include_pending=include_pending,
                    campus=campus_val,
                    monte_carlo=int(monte_carlo),
                    dataset=dataset,
                    focus_program=focus_program_for_analysis,
                )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            st.error(str(exc))
            return
        st.session_state[ANALYSIS_RESULT_KEY] = result
        st.session_state[ANALYSIS_KEY_KEY] = current_key
        st.session_state[ANALYSIS_UI_KEY_KEY] = current_ui_key
        # Пользовательская вероятность привязана к текущему датасету/MC,
        # поэтому чистим кэш при пересчёте.
        for k in (
            CUSTOM_PROB_CACHE_KEY,
            CUSTOM_PROB_CACHE_EST_KEY,
        ):
            st.session_state.pop(k, None)
    elif stored_analysis_matches(st.session_state[ANALYSIS_KEY_KEY], current_key):
        result = st.session_state[ANALYSIS_RESULT_KEY]
        st.session_state[ANALYSIS_UI_KEY_KEY] = current_ui_key
    else:
        if st.session_state[ANALYSIS_RESULT_KEY] is not None:
            st.info(
                "Параметры расчёта изменились. "
                "Нажмите «Рассчитать», чтобы обновить отчёт."
            )
        else:
            st.info("Введите код слева и нажмите «Рассчитать».")
        st.markdown(
            "Сначала модель смотрит **зачисление только среди уже подавших согласие** "
            "(снимок «как сейчас»). Отдельно считает крайний случай "
            "**если согласие подадут все**, и **случайные прогоны**: много раз "
            "случайно решает, кто из неопределившихся согласится."
        )
        st.markdown("---")
        st.caption(f"{DISCLAIMER} {LICENSE_NOTE}")
        return

    if result is None:
        return

    me = result.dataset.applicants[result.my_code]
    my_programs = my_program_names(result.dataset, result.my_code)
    my_set = set(my_programs)
    my_zero = [p for p in result.zero_seat_programs if p in my_set]
    my_unknown = [p for p in result.unknown_seat_programs if p in my_set]
    other_unknown = len(result.unknown_seat_programs) - len(my_unknown)

    if my_zero:
        st.warning("Нулевые места у ваших программ: " + ", ".join(my_zero))
    if my_unknown:
        st.warning(
            "Нет числа бюджетных мест у ваших программ: "
            + ", ".join(my_unknown)
            + " (модель считает: не зачислен в загруженные программы "
            + "или данные неполные)."
        )
    elif other_unknown:
        st.info(
            f"Нет числа бюджетных мест у прочих конкурсов: {other_unknown} "
            "(на ваш паспорт не влияет напрямую). "
            "При необходимости дополните seats.yaml."
        )

    st.caption(
        "Зачисление только среди уже подавших согласие — снимок «как сейчас»: "
        "кто ещё не согласился, в зачисление не входит. "
        "Крайний случай «если согласие подадут все» показывает, есть ли место "
        "даже при полной конкуренции."
    )
    current_text = metric_destination_text(result.vpp.destination(result.my_code))
    consent_text = metric_destination_text(
        result.vpp_if_consent.destination(result.my_code)
    )
    all_text = metric_destination_text(result.ovp.destination(result.my_code))

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Сейчас, с текущими согласиями",
        current_text,
    )
    c2.metric(
        "Если только вы подадите согласие",
        consent_text,
    )
    c3.metric(
        "Если согласие подадут все",
        all_text,
    )

    if me.missing_higher_priority:
        present = sorted({a.priority for a in me.applications})
        missing = [p for p in range(1, max(present) + 1) if p not in present]
        if missing:
            st.warning(
                f"В загруженных списках есть приоритеты {present}, "
                f"нет {missing}. При согласии модель считает: не зачислен "
                "в загруженные программы или данные неполные."
            )

    st.markdown(_section_header("Ваши программы", tone="blue"), unsafe_allow_html=True)
    pending_by_program = {
        row.program: row
        for row in pending_ahead_rows(result.dataset.applicants, result.my_code)
    }
    st.dataframe(
        [
            {
                "Программа": app.program,
                "Приоритет": app.priority,
                "Баллы": app.score,
                "Место": app.rank,
                "Статус": app.status,
                "Согласие": "да" if app.consent else "нет",
                "Выше вас ждут экзамен": (
                    f"{pending_by_program[app.program].pending_ahead} из "
                    f"{pending_by_program[app.program].pending_on_program}"
                    if app.program in pending_by_program
                    else "—"
                ),
            }
            for app in me.sorted_applications()
        ],
        hide_index=True,
    )
    if result.include_pending:
        st.caption(PENDING_SCORE_CAPTION)

    model = result.consent_models_by_scenario.get("auto") or next(
        iter(result.consent_models_by_scenario.values()), None
    )
    if has_consent_bands(model) and model is not None:
        st.markdown(
            _section_header("Согласия по месту в списке", tone="green"),
            unsafe_allow_html=True,
        )
        st.altair_chart(consent_band_chart(model.bands), use_container_width=True)
        st.dataframe(
            consent_band_rows(model.bands),
            hide_index=True,
        )
        st.caption(CONSENT_BANDS_CAPTION)
        if not result.probabilities_by_scenario:
            st.caption(model.description)

    if result.probabilities_by_scenario:
        selected_scenario = resolve_selected_scenario(
            result.probabilities_by_scenario,
            st.session_state.get(SELECTED_SCENARIO_KEY),
        )
        if st.session_state.get(SELECTED_SCENARIO_KEY) != selected_scenario:
            st.session_state[SELECTED_SCENARIO_KEY] = selected_scenario

        st.markdown(_section_header("Случайные прогоны", tone="blue"), unsafe_allow_html=True)
        p = result.probabilities_by_scenario.get(selected_scenario) or next(
            iter(result.probabilities_by_scenario.values())
        )
        scenario_shares = _render_focus_hero(
            result,
            active_scenario=p.scenario,
        )
        selected_scenario = _render_scenario_selector(
            scenario_shares=scenario_shares,
            selected_scenario=selected_scenario,
        )

        p = result.probabilities_by_scenario.get(selected_scenario) or p
        _render_custom_probability_controls(result, selected_scenario=selected_scenario)
        st.markdown(
            _scenario_badge_line(
                p.scenario,
                p.external,
                p.none,
            ),
            unsafe_allow_html=True,
        )
        if p.by_program:
            st.bar_chart(
                probability_chart_rows(p.by_program),
                x="Программа",
                y="Доля прогонов, %",
                horizontal=True,
                sort=False,
                x_label="Доля прогонов, %",
                color="primary",
                height=max(220, 52 * len(p.by_program)),
            )
            st.caption(
                "Доля прогонов с зачислением на программу, % для выбранного сценария. "
                "В одном прогоне — одна программа; сумма ≈ шанс любой загруженной. "
                "Сравнение четырёх сценариев на одном графике не строится: "
                "каждый сценарий — отдельный расчёт."
            )
        st.caption(p.consent_model_description)
        st.caption(
            "Много раз случайно решаем, кто из неопределившихся подаст согласие "
            "(по сценарию выше). Это не крайний случай «согласие подадут все»: "
            "здесь не предполагается, что согласятся все."
        )
        st.caption(mc_ci95_note(p))
        st.caption(MC_SCENARIO_UNCERTAINTY_NOTE)
        if p.focus_program:
            st.caption(
                f"Расчёт для программы **{p.focus_program}**. "
                "Другие конкурсы включены, только если по более высокому "
                "приоритету с них могут уйти люди, которые претендуют и сюда."
            )
        st.dataframe(
            [
                {
                    "Программа": name,
                    "Доля прогонов": f"{share:.1%}",
                }
                for name, share in p.by_program.items()
            ],
            hide_index=True,
        )

    st.markdown(
        _section_header("Заполнение мест (сейчас, с текущими согласиями)", tone="green"),
        unsafe_allow_html=True,
    )
    st.caption(SEAT_FILL_CAPTION)
    fill_stats = [
        program_fill_stats(result.dataset, result.vpp, program, result.my_code)
        for program in my_programs
    ]
    st.dataframe(seat_fill_rows(fill_stats), hide_index=True)
    st.caption(
        "Для вашего шанса смотрите места − «выше вас с согласием», а не «свободно». "
        "«Зачислено сейчас» включает людей ниже вас на остатке мест — "
        "они вас не обгоняют, если вы согласитесь."
    )

    other_note = other_programs_note(
        sum(1 for p in result.dataset.programs if p not in my_set)
    )
    if other_note:
        st.caption(other_note)

    _render_threats_section()
    st.markdown(_section_header("Конкуренты выше меня", tone="rose"), unsafe_allow_html=True)
    selected_scenario = resolve_selected_scenario(
        result.probabilities_by_scenario,
        st.session_state.get(SELECTED_SCENARIO_KEY),
    )
    _render_focus_competitors_section(
        result,
        selected_scenario=selected_scenario,
    )
    _render_report_download()

    st.markdown("---")
    st.caption(f"{DISCLAIMER} {LICENSE_NOTE}")


if __name__ == "__main__":
    main()
