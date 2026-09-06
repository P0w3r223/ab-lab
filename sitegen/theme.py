"""The portfolio's page standard: the head block and the colour palette.

Vendored rather than shared, and that is a recorded debt (ADR 0007 D7). The
presentation audit's Phase 1 was meant to produce one head block and one palette
for every repository to emit verbatim; it does not exist, and four pages in the
portfolio already carry their own copy.

**Two values here deliberately differ from the page this was vendored from**,
because two sibling repositories had already corrected them and left the
measurement in their CSS. Copying "faithfully" would have shipped two known
contrast defects onto a page whose argument is that the craft is checkable:

* ``--positive`` is ``#047857`` rather than the obvious ``#059669``, which
  measures 3.5:1 on ``--surface`` - below AA for text this size.
* ``--accent-soft`` is ``#5b93e4`` rather than ``#93c5fd``, which sat at 1.8:1
  against the page, under the 3:1 a non-text graphic needs.

``--danger`` is not new either: it is taken from the one page in the portfolio
that already needed a red. Inventing a second one is exactly the divergence this
module exists to record.
"""

from __future__ import annotations

LIGHT = {
    "bg": "#ffffff",
    "surface": "#f6f8fa",
    "border": "#e3e7ee",
    "text": "#1c2430",
    "muted": "#5b6472",
    "accent": "#2563eb",
    "accent-soft": "#5b93e4",
    "positive": "#047857",
    "warn": "#b45309",
    "danger": "#b42318",
}

DARK = {
    "bg": "#0f1319",
    "surface": "#161c25",
    "border": "#263041",
    "text": "#e6eaf2",
    "muted": "#98a3b6",
    "accent": "#6ea8fe",
    "accent-soft": "#2c4a7c",
    "positive": "#34d399",
    "warn": "#fbbf24",
    "danger": "#f87171",
}

SITE_URL = "https://p0w3r223.github.io/ab-lab/"
REPO_URL = "https://github.com/P0w3r223/ab-lab"
#: `0007` §5 clause 6 — the one link back to the profile, whose README is the index.
#: `0003` §7 settled hub-and-spoke over a mesh: one target, one string per repository,
#: so cutting a project changes that README and no page.
PROFILE_URL = "https://github.com/P0w3r223"

#: The one sentence every layer of the presentation chain quotes.
CLAIM = (
    "A 5% test is only 5% if you look once, count each user once, "
    "and test one metric"
)


def _variables(palette: dict[str, str], indent: str) -> str:
    return "\n".join(f"{indent}--{name}: {value};" for name, value in palette.items())


def head(title: str, description: str, og_description: str) -> str:
    """Everything above ``<body>``, with zero external-origin resources.

    No font is requested over the network: the stack is what the reader's device
    already has. The icon is an inline ``data:`` SVG for the same reason - the
    audit's criterion is zero external origins, and a favicon is a request like
    any other.
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:type" content="website">
<meta property="og:title" content="{CLAIM}">
<meta property="og:description" content="{og_description}">
<meta property="og:url" content="{SITE_URL}">
<meta name="twitter:card" content="summary">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 \
viewBox=%220 0 16 16%22><text y=%2213%22 font-size=%2213%22>&#128200;</text></svg>">
<style>
/* Both colour schemes are first-class: the palette is a set of variables and dark mode swaps
   the variables rather than restating the design. The charts inherit them, which is why they
   are inline SVG and not images - a raster cannot follow a theme. */
:root {{
  color-scheme: light dark;
{_variables(LIGHT, "  ")}
  --radius: 10px;
}}

@media (prefers-color-scheme: dark) {{
  :root {{
{_variables(DARK, "    ")}
  }}
}}

* {{ box-sizing: border-box; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
    Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  max-width: 60rem;
  margin: 0 auto;
  padding: 2.5rem 1.25rem 4rem;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}

a {{ color: var(--accent); }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

.eyebrow {{
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--muted);
  margin: 0 0 0.5rem;
}}

h1 {{ font-size: 1.9rem; line-height: 1.25; letter-spacing: -0.01em; margin: 0 0 0.75rem; }}
h2 {{ font-size: 1.25rem; margin: 0 0 0.6rem; }}
.lead {{ color: var(--muted); font-size: 1.05rem; margin: 0 0 2rem; }}

section {{
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.4rem;
  margin: 1.5rem 0;
  background: var(--surface);
}}

.tiles {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0 0 1.5rem; padding: 0; }}
.tile {{
  flex: 1 1 12rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.9rem 1rem;
  background: var(--surface);
}}
.tile .value {{ font-size: 1.6rem; font-weight: 700; line-height: 1.1; }}
.tile .value.bad {{ color: var(--danger); }}
.tile .value.good {{ color: var(--positive); }}
.tile .caption {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; }}

.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 0.4rem 0.65rem; border-bottom: 1px solid var(--border); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; font-size: 0.85rem; }}
td.bad {{ color: var(--danger); font-weight: 600; }}
td.good {{ color: var(--positive); }}

code {{
  background: var(--bg);
  border: 1px solid var(--border);
  padding: 0.05rem 0.3rem;
  border-radius: 4px;
  font-size: 0.88em;
}}

figure {{ margin: 1.2rem 0 0; }}
figcaption {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.4rem; }}

/* Chart appearance lives here, geometry lives in Python. That split is what lets
   prefers-color-scheme repaint the chart with the page and nothing be re-rendered. */
.chart {{ width: 100%; height: auto; }}
.chart .axis {{ stroke: var(--border); stroke-width: 1; }}
.chart .grid {{ stroke: var(--border); stroke-width: 1; stroke-dasharray: 2 4; }}
.chart .tick {{ fill: var(--muted); font-size: 12px; }}
.chart .axis-label {{ fill: var(--muted); font-size: 12px; }}
.chart .nominal {{ stroke: var(--muted); stroke-width: 1.4; stroke-dasharray: 6 4; }}
.chart .nominal-label {{ fill: var(--muted); font-size: 12px; }}
.chart .series-naive {{ stroke: var(--danger); fill: none; stroke-width: 2.4; }}
.chart .series-corrected {{
  stroke: var(--accent); fill: none; stroke-width: 2.4; stroke-dasharray: 7 4;
}}
.chart .marker-naive {{ fill: var(--danger); }}
.chart .marker-corrected {{ fill: var(--accent); }}
.chart .series-label {{ font-size: 12px; font-weight: 600; }}
.chart .label-naive {{ fill: var(--danger); }}
.chart .label-corrected {{ fill: var(--accent); }}

footer {{ color: var(--muted); font-size: 0.85rem; margin-top: 2.5rem; }}

@media (max-width: 34rem) {{
  h1 {{ font-size: 1.5rem; }}
  section {{ padding: 1rem; }}
}}
</style>
</head>
<body>
"""
