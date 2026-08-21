"""One chart primitive, emitted as SVG, in two modes.

Geometry is computed here; appearance lives in CSS. On the page that means the
chart inherits the palette and ``prefers-color-scheme`` repaints it with
everything else - a raster image cannot do that, which is why the published PNG
this replaced had to go.

The README needs the same picture through ``<img src>``, where the page's CSS
does not reach, so a standalone mode bakes one theme's colours into the file and
the README references two of them from a ``<picture>``. Same coordinates, two
wrappers - stated because it is the kind of thing that is discovered after the
first one renders unstyled.

All three findings in this release share one shape: an ordered count on x, a
naive curve climbing away from the nominal alpha, and a corrected curve at or
below it. So there is one primitive, not three drawings.
"""

from __future__ import annotations

from dataclasses import dataclass

from .record import Finding
from .theme import DARK, LIGHT

WIDTH = 760
HEIGHT = 400
MARGIN_LEFT = 62
MARGIN_RIGHT = 104
MARGIN_TOP = 20
MARGIN_BOTTOM = 54

#: Series are told apart by colour *and* dash *and* marker shape, so the chart
#: survives greyscale printing and colour-vision deficiency.
_MARKERS = {"naive": "circle", "corrected": "square"}


@dataclass(frozen=True)
class _Scale:
    """Maps data coordinates onto the drawing area."""

    x_max: float
    y_max: float

    def x(self, value: float) -> float:
        span = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
        return MARGIN_LEFT + span * (value / self.x_max)

    def y(self, value: float) -> float:
        span = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
        return HEIGHT - MARGIN_BOTTOM - span * (value / self.y_max)


def _axis_top(highest_rate: float) -> tuple[float, float]:
    """A y axis that ends on a round number just above the data."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2):
        top = step
        while top < highest_rate * 1.12:
            top += step
        if top / step <= 6.0:
            return top, step
    return 1.0, 0.2


def _number(value: float) -> str:
    """Trim float noise so the same input always produces the same bytes."""
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def _marker(shape: str, x: float, y: float, css: str) -> str:
    if shape == "square":
        return (
            f'<rect class="{css}" x="{_number(x - 3.6)}" y="{_number(y - 3.6)}" '
            f'width="7.2" height="7.2"/>'
        )
    return f'<circle class="{css}" cx="{_number(x)}" cy="{_number(y)}" r="4"/>'


def _series_paths(finding: Finding, scale: _Scale) -> list[str]:
    parts: list[str] = []
    for series in finding.series:
        points = [
            (scale.x(cell.x), scale.y(cell.summary.rejection_rate)) for cell in series.cells
        ]
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{_number(x)} {_number(y)}"
            for index, (x, y) in enumerate(points)
        )
        parts.append(f'<path class="series-{series.role}" d="{path}"/>')
        parts.extend(
            _marker(_MARKERS[series.role], x, y, f"marker-{series.role}") for x, y in points
        )
        last_x, last_y = points[-1]
        parts.append(
            f'<text class="series-label label-{series.role}" x="{_number(last_x + 9)}" '
            f'y="{_number(last_y + 4)}">{_short_name(series.name)}</text>'
        )
    return parts


def _short_name(name: str) -> str:
    """The legend is inline beside the line, so it has to be short."""
    return name.split(" (")[0]


def _grid_and_axes(finding: Finding, scale: _Scale, step: float) -> list[str]:
    parts: list[str] = []
    ticks = int(round(scale.y_max / step))
    for index in range(ticks + 1):
        value = step * index
        y = scale.y(value)
        parts.append(
            f'<line class="grid" x1="{MARGIN_LEFT}" y1="{_number(y)}" '
            f'x2="{WIDTH - MARGIN_RIGHT}" y2="{_number(y)}"/>'
        )
        parts.append(
            f'<text class="tick" x="{MARGIN_LEFT - 10}" y="{_number(y + 4)}" '
            f'text-anchor="end">{value:.0%}</text>'
        )

    for value in finding.x_values:
        x = scale.x(value)
        parts.append(
            f'<text class="tick" x="{_number(x)}" y="{HEIGHT - MARGIN_BOTTOM + 20}" '
            f'text-anchor="middle">{value:g}</text>'
        )

    parts.append(
        f'<line class="axis" x1="{MARGIN_LEFT}" y1="{HEIGHT - MARGIN_BOTTOM}" '
        f'x2="{WIDTH - MARGIN_RIGHT}" y2="{HEIGHT - MARGIN_BOTTOM}"/>'
    )
    parts.append(
        f'<text class="axis-label" x="{(MARGIN_LEFT + WIDTH - MARGIN_RIGHT) / 2:.0f}" '
        f'y="{HEIGHT - 12}" text-anchor="middle">{finding.x_label}</text>'
    )

    nominal_y = scale.y(finding.nominal)
    parts.append(
        f'<line class="nominal" x1="{MARGIN_LEFT}" y1="{_number(nominal_y)}" '
        f'x2="{WIDTH - MARGIN_RIGHT}" y2="{_number(nominal_y)}"/>'
    )
    parts.append(
        f'<text class="nominal-label" x="{WIDTH - MARGIN_RIGHT + 9}" '
        f'y="{_number(nominal_y + 4)}">nominal {finding.nominal:.0%}</text>'
    )
    return parts


def _description(finding: Finding) -> str:
    naive = finding.series_by_role("naive")
    corrected = finding.series_by_role("corrected")
    return (
        f"{_short_name(naive.name)} rises from {naive.rates[0]:.1%} to "
        f"{max(naive.rates):.1%} as the x axis grows, while "
        f"{_short_name(corrected.name)} stays at or below the nominal "
        f"{finding.nominal:.0%}. The same numbers are in the table beside this chart."
    )


def chart(finding: Finding, theme: str | None = None) -> str:
    """SVG for one finding.

    Args:
        theme: ``None`` for the page, where the palette comes from the
            stylesheet; ``"light"`` or ``"dark"`` for a standalone file that
            has to carry its own colours.

    The chart is ``role="img"`` with a title and a description, and the table
    printed next to it is the accessible form of the same data - a chart is not
    an accessible way to publish eight numbers, it is a fast way to see a shape.
    """
    top, step = _axis_top(finding.worst_rate)
    scale = _Scale(x_max=max(finding.x_values) * 1.02, y_max=top)

    body = "\n  ".join(_grid_and_axes(finding, scale, step) + _series_paths(finding, scale))
    title = f"{finding.title} - {finding.x_label.lower()} against the false positive rate"
    opening = (
        f'<svg class="chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg"'
    )
    if theme is None:
        return (
            f"{opening}>\n  <title>{title}</title>\n"
            f"  <desc>{_description(finding)}</desc>\n  {body}\n</svg>"
        )

    palette = LIGHT if theme == "light" else DARK
    return (
        f'{opening} width="{WIDTH}" height="{HEIGHT}">\n'
        f"  <title>{title}</title>\n  <desc>{_description(finding)}</desc>\n"
        f"  <style>{_standalone_style(palette)}</style>\n"
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{palette["bg"]}"/>\n  {body}\n</svg>'
    )


def _standalone_style(palette: dict[str, str]) -> str:
    """The page's chart rules with the variables resolved.

    GitHub does not apply a host page's CSS to a referenced SVG, so a file meant
    for the README carries the one theme it was built for.
    """
    return (
        f'.axis{{stroke:{palette["border"]};stroke-width:1}}'
        f'.grid{{stroke:{palette["border"]};stroke-width:1;stroke-dasharray:2 4}}'
        f'.tick,.axis-label,.nominal-label{{fill:{palette["muted"]};font-size:12px;'
        f"font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif}}"
        f'.nominal{{stroke:{palette["muted"]};stroke-width:1.4;stroke-dasharray:6 4}}'
        f'.series-naive{{stroke:{palette["danger"]};fill:none;stroke-width:2.4}}'
        f'.series-corrected{{stroke:{palette["accent"]};fill:none;stroke-width:2.4;'
        f"stroke-dasharray:7 4}}"
        f'.marker-naive{{fill:{palette["danger"]}}}'
        f'.marker-corrected{{fill:{palette["accent"]}}}'
        f".series-label{{font-size:12px;font-weight:600;"
        f"font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif}}"
        f'.label-naive{{fill:{palette["danger"]}}}'
        f'.label-corrected{{fill:{palette["accent"]}}}'
    )
