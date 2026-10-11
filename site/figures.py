"""Draw the website's figures from the rules document, so every number on them is a
config number: the temperature and humidity rules to scale, and how a controller is judged.

``python site/build_rules.py`` writes them next to ``rules.json``; the colours are fixed
so the figures read the same on any background the site uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = ROOT / "site" / "content" / "figures"
INK, MUTED, LINE = "#1f2a44", "#5b6470", "#cfd4db"
GREEN, GREEN_SOFT = "#2c6a4c", "#dcebe2"
HEAT, HEAT_SOFT = "#b2432e", "#f3d9d2"
WATER, WATER_SOFT = "#2b6c97", "#d6e6f1"
WARM = "#b97c12"
BOX_CHARS = 23  # characters per line inside a stage box of the judged figure
FONT = "IBM Plex Sans, system-ui, sans-serif"
# Range of the light-following 6-day target on the held-out springs (2022 to 2025):
# min and max over the four seasons of the 6-day mean of (base + per_1000_j_cm2 * daily
# radiation), from the Amsterdam_OM weather files whose SHA-256 are in the run record of
# results/2026-10-10-run-2. Stated in crop-rules.md and docs/kpis.md as well.
TARGET_RANGE_HELD_OUT = (18.3, 20.7)
# Humidity in the Het Nieuwe Telen Tomaat 2010 spring trial, where grey mould stayed
# controllable: mean RH and daily peak RH (de Gelder et al. 2010).
HNT_MEAN_RH, HNT_PEAK_RH = (85, 87), (93, 94)


def _rules(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["key"]: r["params"] for c in doc["gated_classes"] for r in c["rules"]}


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{title}" '
        f'font-family="{FONT}" font-size="12">'
    )
    return "\n".join(
        [head, f"  <title>{title}</title>", *("  " + line for line in body), "</svg>", ""]
    )


def _text(
    x: float, y: float, s: str, *, fill: str = MUTED, anchor: str = "start", bold: bool = False
) -> str:
    weight = ' font-weight="600"' if bold else ""
    return f'<text x="{x:g}" y="{y:g}" fill="{fill}" text-anchor="{anchor}"{weight}>{s}</text>'


def _rect(x: float, y: float, w: float, h: float, fill: str, *, extra: str = "") -> str:
    return f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{fill}"{extra}/>'


def _gradient(name: str, start: str, end: str) -> str:
    return (
        f'<linearGradient id="{name}" x1="0" x2="1">'
        f'<stop offset="0" stop-color="{WARM}" stop-opacity="{start}"/>'
        f'<stop offset="1" stop-color="{WARM}" stop-opacity="{end}"/></linearGradient>'
    )


def _vline(x: float, y0: float, y1: float, stroke: str) -> str:
    return (
        f'<line x1="{x:g}" y1="{y0:g}" x2="{x:g}" y2="{y1:g}" stroke="{stroke}" stroke-width="2"/>'
    )


def _arrow(x: float, y: float, length: float) -> str:
    return (
        f'<path d="M{x:g} {y:g} h{length:g} m-6 -5 l6 5 l-6 5" fill="none" '
        f'stroke="{MUTED}" stroke-width="1.5"/>'
    )


def _axis(x0: float, x1: float, y: float, ticks: list[tuple[float, str]]) -> list[str]:
    out = [f'<line x1="{x0:g}" y1="{y}" x2="{x1:g}" y2="{y}" stroke="{LINE}"/>']
    for x, label in ticks:
        out.append(f'<line x1="{x:g}" y1="{y}" x2="{x:g}" y2="{y + 5}" stroke="{MUTED}"/>')
        out.append(_text(x, y + 18, label, anchor="middle"))
    return out


def temperature_figure(doc: dict[str, Any]) -> str:
    """Temperature rules on one scale, 6 to 36 C."""
    p = _rules(doc)
    lo, hi = 6, 36

    def x(t: float) -> float:
        return 50 + (t - lo) * 22

    peak, dose, mean24, mean6 = p["peak"], p["dose"], p["mean_24h"], p["mean_6d"]
    t0, t1 = TARGET_RANGE_HELD_OUT
    band = mean6["band"]
    body = [
        "<defs>",
        _gradient("cold", ".9", ".12"),
        _gradient("heat", ".12", ".9"),
        '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="{GREEN_SOFT}"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{GREEN}" stroke-width="1.5" '
        'stroke-opacity=".55"/>'
        "</pattern>",
        "</defs>",
        _text(50, 18, "absolute limits", fill=INK, bold=True),
        _rect(x(lo), 24, x(peak["low"]) - x(lo), 16, HEAT),
        _rect(x(peak["high"]), 24, x(hi) - x(peak["high"]), 16, HEAT),
        _text(x(peak["low"]) + 4, 36, f"fails below {peak['low']:g}"),
        _text(x(peak["high"]) - 4, 36, f"fails above {peak['high']:g}", anchor="end"),
        _text(50, 62, "dose builds", fill=INK, bold=True),
        _rect(
            x(dose["limit_low"]), 68, x(dose["onset_low"]) - x(dose["limit_low"]), 16, "url(#cold)"
        ),
        _rect(
            x(dose["onset_high"]),
            68,
            x(dose["limit_high"]) - x(dose["onset_high"]),
            16,
            "url(#heat)",
        ),
        _text(x(dose["onset_low"]) + 4, 80, f"from {dose['onset_low']:g}"),
        _text(x(dose["onset_high"]) - 4, 80, f"from {dose['onset_high']:g}", anchor="end"),
        _text(50, 106, "6-day mean band", fill=INK, bold=True),
        _rect(x(t0 - band), 112, x(t1 - band) - x(t0 - band), 16, "url(#hatch)"),
        _rect(
            x(t1 - band),
            112,
            x(t0 + band) - x(t1 - band),
            16,
            GREEN_SOFT,
            extra=f' stroke="{GREEN}"',
        ),
        _rect(x(t0 + band), 112, x(t1 + band) - x(t0 + band), 16, "url(#hatch)"),
        _rect(x(t0), 117, x(t1) - x(t0), 6, GREEN),
        _text(x(t1 + band) + 4, 124, f"target {t0:g} to {t1:g} with the light, band ±{band:g}"),
        _text(50, 150, "24-hour mean cap", fill=INK, bold=True),
        _vline(x(mean24["high"]), 154, 174, HEAT),
        _rect(x(mean24["high"]), 158, 88, 12, HEAT_SOFT),
        _text(x(mean24["high"]) + 4, 168, f"fails above {mean24['high']:g}", fill=HEAT),
        *_axis(
            x(lo),
            x(hi),
            190,
            [(x(t), f"{t}" if t < hi else f"{t} C") for t in range(lo, hi + 1, 2)],
        ),
    ]
    return _svg(760, 222, "Temperature rules on one scale from 6 to 36 C", body)


def humidity_figure(doc: dict[str, Any]) -> str:
    """Humidity rules on one scale, 80 to 100% RH."""
    p = _rules(doc)
    lo, hi = 80, 100

    def x(rh: float) -> float:
        return 50 + (rh - lo) * 33

    hours, run = p["humid_hours"], p["longest_run"]
    body = [
        _text(50, 18, "humid hours", fill=INK, bold=True),
        _rect(
            x(hours["threshold"]),
            24,
            x(hi) - x(hours["threshold"]),
            16,
            WATER_SOFT,
            extra=f' stroke="{WATER}"',
        ),
        _text(
            x(hours["threshold"]) + 4,
            36,
            f"above {hours['threshold']:g}: counts; {hours['window_days']}-day mean at most "
            f"{hours['max_hours_per_day']:g} h/day",
            fill=WATER,
        ),
        _text(50, 62, "longest run", fill=INK, bold=True),
        _rect(
            x(run["threshold"]),
            68,
            x(hi) - x(run["threshold"]),
            16,
            WATER,
            extra=' fill-opacity=".75"',
        ),
        _text(
            x(run["threshold"]) - 4,
            80,
            f"above {run['threshold']:g}: no run over {run['max_hours']:g} h",
            anchor="end",
        ),
        _text(50, 104, "grower practice", fill=INK, bold=True),
        _rect(
            x(HNT_MEAN_RH[0]),
            110,
            x(HNT_MEAN_RH[1]) - x(HNT_MEAN_RH[0]),
            10,
            GREEN,
            extra=' fill-opacity=".6"',
        ),
        _rect(
            x(HNT_PEAK_RH[0]),
            110,
            x(HNT_PEAK_RH[1]) - x(HNT_PEAK_RH[0]),
            10,
            GREEN,
            extra=' fill-opacity=".6"',
        ),
        _text(x(HNT_MEAN_RH[1]) + 4, 119, f"mean RH {HNT_MEAN_RH[0]} to {HNT_MEAN_RH[1]}"),
        _text(x(HNT_PEAK_RH[1]) + 4, 119, f"daily peak {HNT_PEAK_RH[0]} to {HNT_PEAK_RH[1]}"),
        *_axis(
            x(lo),
            x(hi),
            140,
            [(x(r), f"{r}" if r < hi else f"{r} %RH") for r in range(lo, hi + 1, 2)],
        ),
    ]
    return _svg(760, 168, "Humidity rules on one scale from 80 to 100 percent RH", body)


def judged_figure(doc: dict[str, Any]) -> str:
    """How a controller is judged: four stages left to right."""
    gate = doc["gate"]
    conf = f"{gate['confidence']:.0%}"
    share = f"{gate['max_failing_share']:.0%}"
    stages = [
        ("Same seasons", "one simulated greenhouse, held-out weather years", GREEN_SOFT),
        ("Model error on purpose", "deliberately wrong model parameters", GREEN_SOFT),
        (
            "Crop-safety gate",
            f"at most {share} of seasons break a rule ({conf} confidence)",
            HEAT_SOFT,
        ),
        ("Economics", "net value per m2 and decision time", WATER_SOFT),
    ]
    body = []
    w, gap, top, h = 172, 18, 14, 86
    for i, (head, sub, fill) in enumerate(stages):
        x0 = 14 + i * (w + gap)
        body.append(_rect(x0, top, w, h, fill, extra=f' rx="6" stroke="{LINE}"'))
        body.append(_text(x0 + 12, top + 24, head, fill=INK, bold=True))
        words, line, lines = sub.split(), "", []
        for word in words:
            if len(line) + len(word) > BOX_CHARS and line:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        for j, text in enumerate(lines[:3]):
            body.append(_text(x0 + 12, top + 44 + j * 15, text))
        if i < len(stages) - 1:
            ax = x0 + w + 4
            body.append(_arrow(ax, top + h / 2, gap - 10))
    body.append(
        _text(
            14,
            top + h + 24,
            "A controller that fails the gate is reported but never ranked. "
            "All results are simulated.",
        )
    )
    return _svg(760, 140, "How Phytotron judges a controller, in four stages", body)


FIGURES = {
    "temperature-rules.svg": temperature_figure,
    "humidity-rules.svg": humidity_figure,
    "how-judged.svg": judged_figure,
}


def render_figures(doc: dict[str, Any]) -> dict[str, str]:
    return {name: draw(doc) for name, draw in FIGURES.items()}
