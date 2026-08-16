"""Простой веб-интерфейс: ввод кода → отчёт модели."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from admission_sim.model import EXTERNAL
from admission_sim.pipeline import run_analysis
from admission_sim.report import (
    build_markdown_report,
    my_program_names,
    program_fill_stats,
)
from admission_sim.scenarios import SCENARIO_LABELS


def _fmt_dest(dest: str | None) -> str:
    if dest is None:
        return "не зачислен"
    if dest == EXTERNAL:
        return "вне загруженных программ"
    return dest


DISCLAIMER = (
    "**Дисклеймер.** Неофициальный инструмент для личного анализа публичных "
    "обезличенных конкурсных списков. Не является расчётом приёмной комиссии, "
    "вуза или Госуслуг и не гарантирует результат зачисления."
)
LICENSE_NOTE = "Лицензия: MIT."


def main() -> None:
    st.set_page_config(page_title="Симулятор поступления", layout="wide")
    st.title("Симулятор приоритетов магистратуры")
    st.info(DISCLAIMER)
    st.caption(LICENSE_NOTE)

    default_data = Path("data/raw")
    if not (default_data.is_dir() and any(default_data.glob("*.csv"))):
        if list(Path(".").glob("*.csv")):
            default_data = Path(".")

    with st.sidebar:
        st.header("Параметры")
        st.caption("Неофициально · не ПК · MIT")
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
            "Учитывать «Ожидание результатов ВИ» / pending",
            value=True,
            help=(
                "Без этого заявки только с pending (часто балл 0 до ВИ) "
                "пропадут из паспорта, Monte Carlo и заполнения мест."
            ),
        )
        monte_carlo = st.slider("Monte Carlo прогонов", 0, 5000, 500, 100)
        scenario_label = st.radio(
            "Сценарий согласий прочих",
            options=list(SCENARIO_LABELS.values()),
            index=0,
            help=(
                "Авто — из текущих долей согласий. "
                "Оптимистичный — конкуренты реже соглашаются. "
                "Пессимистичный — чаще. "
                "Сбалансированный — одна общая доля для всех."
            ),
        )
        scenario = next(
            key for key, label in SCENARIO_LABELS.items() if label == scenario_label
        )
        with st.expander("Дополнительно"):
            override = st.checkbox("Задать одну P вручную (вместо сценария)", value=False)
            consent_p = None
            if override:
                consent_p = st.slider("Общая P(согласие у прочих)", 0.0, 1.0, 0.35, 0.05)
        run = st.button("Рассчитать", type="primary", use_container_width=True)

    if not run:
        st.info("Введите код слева и нажмите «Рассчитать».")
        st.markdown("---")
        st.caption(f"{DISCLAIMER} {LICENSE_NOTE}")
        return

    code_text = my_code_raw.strip()
    if not code_text.isdigit():
        st.error("Укажите числовой код поступающего.")
        return
    my_code = int(code_text)

    try:
        result = run_analysis(
            data_dir,
            seats_path,
            my_code,
            include_pending=include_pending,
            campus=campus.strip() or None,
            monte_carlo=int(monte_carlo),
            scenario=scenario,
            consent_p=consent_p,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(str(exc))
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
            "Без КЦП у ваших программ: "
            + ", ".join(my_unknown)
            + " (поглощаются как EXTERNAL)."
        )
    elif other_unknown:
        st.info(
            f"Без КЦП у прочих конкурсов: {other_unknown} "
            "(на ваш паспорт не влияет напрямую). "
            "При необходимости дополните seats.yaml."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("ВПП сейчас", _fmt_dest(result.vpp.destination(result.my_code)))
    c2.metric(
        "ВПП + ваше согласие",
        _fmt_dest(result.vpp_if_consent.destination(result.my_code)),
    )
    c3.metric("ОВП (все согласны)", _fmt_dest(result.ovp.destination(result.my_code)))

    if me.missing_higher_priority:
        present = sorted({a.priority for a in me.applications})
        missing = [p for p in range(1, max(present) + 1) if p not in present]
        if missing:
            st.warning(
                f"В загруженных списках есть приоритеты {present}, "
                f"нет {missing}. При согласии модель считает уход на внешнюю программу."
            )

    st.subheader("Ваши программы")
    st.dataframe(
        [
            {
                "Программа": app.program,
                "Приоритет": app.priority,
                "Баллы": app.score,
                "Место": app.rank,
                "Статус": app.status,
                "Согласие": "да" if app.consent else "нет",
            }
            for app in me.sorted_applications()
        ],
        use_container_width=True,
        hide_index=True,
    )

    if result.probability is not None:
        st.subheader("Monte Carlo")
        p = result.probability
        st.write(
            f"Сценарий: **{SCENARIO_LABELS.get(p.scenario, p.scenario)}** · "
            f"P(зачисление на загруженную программу) = **{p.any_loaded:.1%}** · "
            f"P(вовне) = {p.external:.1%} · P(нет) = {p.none:.1%}"
        )
        st.caption(p.consent_model_description)
        st.caption(
            "Только ваши программы: доля прогонов с зачислением именно сюда. "
            "В одном прогоне — одна программа; сумма строк ~ P(любая загруженная)."
        )
        st.dataframe(
            [
                {
                    "Программа": name,
                    "P": f"{p.by_program.get(name, 0.0):.1%}",
                }
                for name in my_programs
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Заполнение мест (ВПП сейчас)")
    seat_rows = []
    for program in my_programs:
        stats = program_fill_stats(result.dataset, result.vpp, program)
        seat_rows.append(
            {
                "Программа": stats.program,
                "K": "?" if stats.seats is None else stats.seats,
                "Зачислено": stats.enrolled,
                "Проходной": stats.cutoff,
                "Приоритет 1": stats.priority1,
            }
        )
    st.dataframe(seat_rows, use_container_width=True, hide_index=True)

    other = [p for p in result.dataset.programs if p not in my_set]
    if other:
        known = 0
        total_seats = 0
        total_enrolled = 0
        unknown = 0
        for program in other:
            stats = program_fill_stats(result.dataset, result.vpp, program)
            total_enrolled += stats.enrolled
            if stats.seats is None:
                unknown += 1
            else:
                known += 1
                total_seats += int(stats.seats)
        st.caption(
            f"Остальные конкурсы: {len(other)} "
            f"(с КЦП: {known}, мест {total_seats}, зачислено {total_enrolled}; "
            f"без КЦП: {unknown})"
        )

    if result.counterfactuals:
        st.subheader("Контрфакты (угрозы)")
        st.dataframe(
            [
                {
                    "Код": item.code,
                    "Куда": _fmt_dest(item.destination),
                    "Вытесняет": "да" if item.displaces_me else "нет",
                    "Вы до": _fmt_dest(item.my_destination_before),
                    "Вы после": _fmt_dest(item.my_destination_after),
                }
                for item in result.counterfactuals
            ],
            use_container_width=True,
            hide_index=True,
        )

    md = build_markdown_report(
        result.dataset,
        result.my_code,
        vpp=result.vpp,
        vpp_if_consent=result.vpp_if_consent,
        ovp=result.ovp,
        counterfactuals=result.counterfactuals,
        probability=result.probability,
        include_pending=result.include_pending,
    )
    st.download_button("Скачать report.md", md, file_name="report.md")
    with st.expander("Полный Markdown"):
        st.markdown(md)

    st.markdown("---")
    st.caption(f"{DISCLAIMER} {LICENSE_NOTE}")


if __name__ == "__main__":
    main()
