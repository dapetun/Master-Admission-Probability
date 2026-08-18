"""Простой веб-интерфейс: ввод кода → отчёт модели."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from admission_sim.load import load_dataset
from admission_sim.pipeline import run_analysis
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
    focus_program_shares_by_scenario,
    has_consent_bands,
    mc_ci95_note,
    my_program_names,
    other_programs_note,
    pending_ahead_rows,
    prepare_threats_view,
    probability_chart_rows,
    program_fill_stats,
    resolve_hero_program,
    seat_fill_rows,
    threat_table_rows,
    _fmt_dest,
)
from admission_sim.scenarios import SCENARIO_LABELS
from admission_sim.simulate import subgraph_for_program


DISCLAIMER = (
    "**Дисклеймер.** Неофициальный инструмент для личного анализа публичных "
    "обезличенных конкурсных списков. Не является расчётом приёмной комиссии, "
    "вуза или Госуслуг и не гарантирует результат зачисления."
)
LICENSE_NOTE = "Лицензия: MIT."
FOCUS_ALL = "Все мои программы"
ANALYSIS_RESULT_KEY = "analysis_result"
ANALYSIS_KEY_KEY = "analysis_key"
THREATS_OVERLAP_KEY = "threats_overlap_programs"
THREATS_OVERLAP_CODE_KEY = "threats_overlap_for_code"
THREATS_OVERLAP_OPTIONS_KEY = "threats_overlap_options"


def app_styles() -> str:
    """Единый CSS приложения: отступы, hero-блок, мягкие callout."""
    return """
<style>
.block-container {
    padding-top: 1.35rem;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #D8DEE4;
}
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #454340;
}
[data-testid="stMetric"] {
    background: #FAFBFC;
    border: 1px solid #D8DEE4;
    border-radius: 10px;
    padding: 0.65rem 0.75rem;
    box-shadow: 0 1px 2px rgba(69, 67, 64, 0.04);
}
.focus-hero {
    background: #EEF1F4;
    border: 1px solid #D8DEE4;
    border-radius: 12px;
    padding: 1.35rem 1.5rem 1.15rem;
    margin: 0.35rem 0 1rem;
    box-shadow: 0 1px 3px rgba(69, 67, 64, 0.05);
}
.focus-hero__title {
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1.25;
    color: #454340;
    margin: 0 0 0.45rem;
}
.focus-hero__meta {
    color: #6B7280;
    font-size: 0.92rem;
    line-height: 1.45;
    margin: 0 0 0.95rem;
}
.focus-hero__callout {
    background: #EEF2F6;
    border: 1px solid #D5DCE4;
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
    background: #FAFBFC;
    border: 1px solid #D8DEE4;
    border-radius: 10px;
    padding: 0.85rem 0.65rem 0.75rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(69, 67, 64, 0.03);
}
.focus-hero__metric--active {
    border-color: #A8B0BA;
    box-shadow: inset 0 0 0 1px #C5CDD5;
}
.focus-hero__metric-value {
    font-size: 1.85rem;
    font-weight: 600;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}
.focus-hero__metric-label {
    color: #6B7280;
    font-size: 0.82rem;
    line-height: 1.25;
}
</style>
"""


def analysis_input_key(
    *,
    my_code: int,
    data_dir: str,
    seats_path: str,
    include_pending: bool,
    campus: str | None,
    monte_carlo: int,
    scenario: str,
    consent_p: float | None,
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
        scenario,
        consent_p,
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
    st.subheader(threats_view.title)
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
        selected_overlap = selected_overlap_programs(
            st.multiselect(
                "Программы пересечения",
                options=my_programs,
                placeholder=THREATS_SELECT_PROGRAM_CAPTION,
                help=(
                    "В таблице остаются люди, которые выше вас без согласия "
                    "на выбранных программах и при согласии занимают именно её. "
                    "По умолчанию — все ваши. Кто ушёл бы на другую программу "
                    "или вовне, скрыт. Список для каждой программы считается "
                    "отдельно, фильтр только прячет строки."
                ),
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


@st.cache_data(show_spinner="Сценарии…", max_entries=8)
def _cached_hero_scenario_shares(
    data_dir_str: str,
    seats_path_str: str,
    include_pending: bool,
    campus: str | None,
    my_code: int,
    monte_carlo: int,
    focus_program: str,
    active_scenario: str,
    active_share: float,
    active_focus_program: str | None,
    data_mtime: float,
    seats_mtime: float,
) -> dict[str, float]:
    """Кэш сравнения сценариев для hero-блока (дороже одного прогона MC)."""
    dataset = _load_dataset_cached(
        data_dir_str,
        seats_path_str,
        include_pending,
        campus,
        data_mtime,
        seats_mtime,
    )
    shares = focus_program_shares_by_scenario(
        dataset.applicants,
        dataset.seats,
        my_code,
        focus_program=focus_program,
        n_simulations=monte_carlo,
        current_scenario=active_scenario,  # type: ignore[arg-type]
        current_share=active_share,
        current_focus_program=active_focus_program,
    )
    return dict(shares)


def _hero_scenario_shares(
    result,
    focus_program: str,
    *,
    consent_p: float | None,
    data_dir: Path,
    seats_path: Path,
    campus_val: str | None,
    data_mtime: float,
    seats_mtime: float,
) -> dict[str, float]:
    """Доли прогонов по сценариям для hero; при ручном p — один сценарий."""
    p = result.probability
    assert p is not None
    active_share = p.by_program.get(focus_program, 0.0)
    if consent_p is not None:
        return {scenario: active_share for scenario in SCENARIO_LABELS}
    return _cached_hero_scenario_shares(
        str(data_dir),
        str(seats_path),
        result.include_pending,
        campus_val,
        result.my_code,
        p.n_simulations,
        focus_program,
        p.scenario,
        active_share,
        p.focus_program,
        data_mtime,
        seats_mtime,
    )


def _render_focus_hero(
    result,
    *,
    consent_p: float | None,
    active_scenario: str,
    data_dir: Path,
    seats_path: Path,
    campus_val: str | None,
    data_mtime: float,
    seats_mtime: float,
) -> None:
    """Hero-блок фокусной программы со сравнением сценариев."""
    p = result.probability
    if p is None:
        return
    focus_program = resolve_hero_program(
        my_program_names(result.dataset, result.my_code),
        p,
    )
    if focus_program is None:
        return
    scenario_shares = _hero_scenario_shares(
        result,
        focus_program,
        consent_p=consent_p,
        data_dir=data_dir,
        seats_path=seats_path,
        campus_val=campus_val,
        data_mtime=data_mtime,
        seats_mtime=seats_mtime,
    )
    hero = build_focus_hero_data(
        result.dataset,
        result.my_code,
        p,
        scenario_shares=scenario_shares,  # type: ignore[arg-type]
        active_scenario=active_scenario,  # type: ignore[arg-type]
        manual_consent=consent_p is not None,
    )
    if hero is None:
        return
    st.markdown(focus_hero_html(hero), unsafe_allow_html=True)


@st.fragment
def _render_report_download() -> None:
    """Скачивание и Markdown: тяжёлый рендер только при открытом блоке."""
    result = st.session_state[ANALYSIS_RESULT_KEY]
    md = build_markdown_report(
        result.dataset,
        result.my_code,
        vpp=result.vpp,
        vpp_if_consent=result.vpp_if_consent,
        ovp=result.ovp,
        counterfactuals=result.counterfactuals,
        probability=result.probability,
        include_pending=result.include_pending,
        consent_model=result.consent_model,
    )
    st.download_button("Скачать report.md", md, file_name="report.md")
    md_exp = st.expander("Полный Markdown", on_change="rerun")
    if md_exp.open:
        with md_exp:
            st.markdown(md)


def main() -> None:
    st.set_page_config(page_title="Симулятор поступления", layout="wide")
    st.markdown(app_styles(), unsafe_allow_html=True)
    st.session_state.setdefault(ANALYSIS_RESULT_KEY, None)
    st.session_state.setdefault(ANALYSIS_KEY_KEY, None)
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
            help="Например Москва / Санкт-Петербург. Пусто — все кампусы.",
        )
        include_pending = st.checkbox(
            "Учитывать ожидание результатов вступительных испытаний",
            value=True,
            help=(
                "Без этого заявки, где ещё нет баллов за экзамен, "
                "пропадут из паспорта, случайных прогонов и заполнения мест. "
                "Баллы из файла не дополняются: часто это 0, будущие результаты "
                "модель не угадывает."
            ),
        )
        monte_carlo = st.slider(
            "Случайных прогонов",
            0,
            5000,
            1000,
            100,
            help=(
                "Много раз случайно решаем, кто из неопределившихся подаст "
                "согласие. 0 — не считать."
            ),
        )
        focus_slot = st.container()
        scenario_label = st.radio(
            "Сценарий согласий прочих",
            options=list(SCENARIO_LABELS.values()),
            index=0,
            key="scenario_label",
            help=(
                "Авто — доли согласия, уже видные в списках, отдельно "
                "для верхней, средней и нижней трети конкурентов по месту. "
                "Не «согласятся все». "
                "Гарантия «пройду, даже если согласие подадут все» — отдельный "
                "крайний случай в результатах, не случайные прогоны. "
                "Оптимистичный — конкуренты реже соглашаются. "
                "Пессимистичный — конкуренты чаще, но не все. "
                "Сбалансированный — одна общая доля для всех без согласия."
            ),
        )
        scenario = next(
            key for key, label in SCENARIO_LABELS.items() if label == scenario_label
        )
        with st.expander("Дополнительно"):
            override = st.checkbox(
                "Задать одну вероятность вручную (вместо сценария)",
                value=False,
            )
            consent_p = None
            if override:
                consent_p = st.slider(
                    "Общая вероятность согласия у прочих",
                    0.0,
                    1.0,
                    0.35,
                    0.05,
                )
        run = st.button("Рассчитать", type="primary", width="stretch")

        campus_val_preview = campus.strip() or None
        user_programs: list[str] = []
        preview_dataset = None
        focus_program = None
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
                    key="focus_program_choice",
                    help=(
                        "Одна программа ускоряет расчёт: не считаем все конкурсы, "
                        "но оставляем те, куда люди могут уйти по более высокому "
                        "приоритету. «Все мои программы» — полный расчёт, как раньше."
                    ),
                )
                if focus_choice != FOCUS_ALL:
                    focus_program = focus_choice
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
                    if not focus_program
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
            scenario=scenario,
            consent_p=consent_p,
            focus_program=focus_program,
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
                result = run_analysis(
                    data_dir,
                    seats_path,
                    my_code,
                    include_pending=include_pending,
                    campus=campus_val,
                    monte_carlo=int(monte_carlo),
                    scenario=scenario,
                    consent_p=consent_p,
                    dataset=dataset,
                    focus_program=focus_program,
                )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            st.error(str(exc))
            return
        st.session_state[ANALYSIS_RESULT_KEY] = result
        st.session_state[ANALYSIS_KEY_KEY] = current_key
    elif stored_analysis_matches(st.session_state[ANALYSIS_KEY_KEY], current_key):
        result = st.session_state[ANALYSIS_RESULT_KEY]
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
            + " (модель считает уход вне загруженных программ)."
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
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Сейчас, с текущими согласиями",
        _fmt_dest(result.vpp.destination(result.my_code)),
        help=(
            "Зачисление только среди уже подавших согласие. "
            "Это текущая картина по спискам."
        ),
    )
    c2.metric(
        "Если вы подадите согласие",
        _fmt_dest(result.vpp_if_consent.destination(result.my_code)),
        help="То же, но вы точно среди согласившихся.",
    )
    c3.metric(
        "Если согласие подадут все",
        _fmt_dest(result.ovp.destination(result.my_code)),
        help=(
            "Крайний случай: согласие как будто у всех в загруженных списках. "
            "Если здесь вы зачислены — место есть, даже когда соглашается каждый. "
            "Авто / пессимистичный так не делают."
        ),
    )

    if me.missing_higher_priority:
        present = sorted({a.priority for a in me.applications})
        missing = [p for p in range(1, max(present) + 1) if p not in present]
        if missing:
            st.warning(
                f"В загруженных списках есть приоритеты {present}, "
                f"нет {missing}. При согласии модель считает уход на внешнюю программу."
            )

    st.subheader("Ваши программы")
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

    model = result.consent_model
    if has_consent_bands(model) and model is not None:
        st.subheader("Согласия по месту в списке")
        st.altair_chart(consent_band_chart(model.bands), use_container_width=False)
        st.dataframe(
            consent_band_rows(model.bands),
            hide_index=True,
        )
        st.caption(CONSENT_BANDS_CAPTION)
        if result.probability is None:
            st.caption(model.description)

    if result.probability is not None:
        st.subheader("Случайные прогоны")
        p = result.probability
        _render_focus_hero(
            result,
            consent_p=consent_p,
            active_scenario=p.scenario,
            data_dir=data_dir,
            seats_path=seats_path,
            campus_val=campus_val,
            data_mtime=data_mtime,
            seats_mtime=seats_mtime,
        )
        st.write(
            f"Сценарий: **{SCENARIO_LABELS.get(p.scenario, p.scenario)}** · "
            f"шанс ухода вне загруженных программ = {p.external:.1%} · "
            f"шанс без зачисления = {p.none:.1%}"
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
                "Доля прогонов с зачислением на программу, %. "
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

    st.subheader("Заполнение мест (сейчас, с текущими согласиями)")
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
    _render_report_download()

    st.markdown("---")
    st.caption(f"{DISCLAIMER} {LICENSE_NOTE}")


if __name__ == "__main__":
    main()
