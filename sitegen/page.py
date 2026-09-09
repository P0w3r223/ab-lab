"""The published page.

Numbers are generated, prose is written. Every figure below is interpolated from
the committed record; every paragraph is authored here and reconciled by hand
with the README, because "the four renderings of a claim say the same thing" is
the one criterion in the presentation standard that cannot be scripted.
"""

from __future__ import annotations

from .charts import chart
from .markdown import finding_table, validation_table
from .numbers import integer, percent, rate_with_error
from .record import Finding, Record, Validation
from .theme import CLAIM, PROFILE_URL, REPO_URL, head

ADR_LINKS = [
    ("0001", "statsmodels is a test oracle, not a dependency"),
    ("0003", "mSPRT rather than group-sequential boundaries"),
    ("0006", "one claim measured three ways"),
    ("0008", "cluster-robust variance"),
    ("0009", "three corrections, two guarantees"),
    ("0007", "the page is generated from a committed record"),
]

ADR_FILES = {
    "0001": "0001-no-statsmodels-at-runtime.md",
    "0003": "0003-msprt-over-group-sequential.md",
    "0006": "0006-three-mechanisms-of-alpha-inflation.md",
    "0008": "0008-cluster-robust-variance.md",
    "0009": "0009-multiplicity-control.md",
    "0007": "0007-the-page-is-generated.md",
}


def _table(markdown: str, extra_class: str = "") -> str:
    """Render a Markdown table as HTML, so both surfaces share one builder."""
    lines = [line for line in markdown.splitlines() if not line.startswith("|---")]
    rows = []
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        tag = "th" if index == 0 else "td"
        rendered = "".join(f"<{tag}>{_inline(cell)}</{tag}>" for cell in cells)
        rows.append(f"    <tr>{rendered}</tr>")
    body = "\n".join(rows)
    classes = f' class="{extra_class}"' if extra_class else ""
    return f'<div class="table-wrap">\n  <table{classes}>\n{body}\n  </table>\n</div>'


def _inline(text: str) -> str:
    """The only Markdown the tables use is bold, and it marks the headline cell."""
    while "**" in text:
        text = text.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    return text


def _tiles(findings: list[Finding], validation: Validation | None) -> str:
    """One number per finding: the worst the naive procedure gets, and where."""
    tiles = []
    for finding in findings:
        naive = finding.series_by_role("naive")
        worst_index = naive.rates.index(max(naive.rates))
        where = f"{finding.x_values[worst_index]:g} {finding.x_label.lower()}"
        tiles.append(
            (
                "bad",
                percent(naive.rates[worst_index]),
                f"of A/A experiments declared a winner at {where}, against a "
                f"nominal {percent(finding.nominal, 0)} - and "
                f"{percent(finding.series_by_role('corrected').rates[worst_index])} "
                f"once corrected",
            )
        )
    if validation is not None:
        passing = sum(1 for row in validation.rows if row.passed)
        tiles.append(
            (
                "good" if passing == len(validation.rows) else "bad",
                f"{passing}/{len(validation.rows)}",
                "claims this package makes about itself, checked against a "
                "simulated world where the answer is known",
            )
        )
    rendered = "\n".join(
        f'  <li class="kpi"><div class="value {kind}">{value}</div>'
        f'<div class="caption">{caption}</div></li>'
        for kind, value, caption in tiles
    )
    return f'<ul class="tiles">\n{rendered}\n</ul>'


def _winners_curse(finding: Finding) -> str:
    """Only the peeking record carries the stopped-effect figures, so this
    paragraph appears where the data supports it and nowhere else."""
    naive = finding.series_by_role("naive")
    honest = naive.cells[0].summary.mean_absolute_estimate_when_stopped
    worst_cell = naive.cells[naive.rates.index(max(naive.rates))]
    inflated = worst_cell.summary.mean_absolute_estimate_when_stopped
    if inflated is None or honest is None or finding.key != "peeking":
        return ""
    return (
        f"<p>The p-value is not the only casualty. Among the experiments that "
        f"<em>were</em> stopped as significant, the reported effect is inflated too, "
        f"because stopping happens precisely on the noisy excursions: a mean absolute "
        f"effect of {inflated:.3f} when peeking against {honest:.3f} with a single look, "
        f"in a world where the true effect is exactly zero. Both are false positives; the "
        f"peeked one claims to be more than twice as large.</p>"
    )


def _finding_section(finding: Finding) -> str:
    """One finding, rendered from the record and its own labels.

    Nothing here knows which mechanism it is describing. The three findings share
    a shape - an ordered count on x, a curve that climbs away from the nominal
    rate, a corrected curve that does not - so they share one renderer, and the
    prose that differs between them lives in the record as ``question``.
    """
    naive = finding.series_by_role("naive")
    corrected = finding.series_by_role("corrected")
    worst = max(naive.rates)
    worst_at = finding.x_values[naive.rates.index(worst)]
    unit = finding.x_label.lower()

    return f"""<section>
  <h2>{finding.title}</h2>
  <p>{finding.question}</p>
  {_table(finding_table(finding))}
  <p>The first row is the control case, where there is nothing to correct and the two
  rules should agree: {rate_with_error(naive.cells[0].summary)} against a nominal
  {percent(finding.nominal, 0)}. By {worst_at:g} the naive rule reaches
  {percent(worst)} - <strong>one A/A experiment in {round(1 / worst)} declared a
  winner</strong> - while {_short_series(corrected.name).lower()} never exceeds
  {percent(max(corrected.rates))} anywhere in the range. Every cell carries its Monte
  Carlo error, and at these run counts a single cell two or three sigmas from its target
  is what sampling looks like rather than a finding.</p>
  {_winners_curse(finding)}
  <figure>
    {chart(finding)}
    <figcaption>Same data, same alpha, same {unit}. The only difference is the decision
    rule: {_short_series(naive.name).lower()} against
    {_short_series(corrected.name).lower()}.</figcaption>
  </figure>
</section>"""


def _short_series(name: str) -> str:
    return name.split(" (")[0]


#: The order the claim names them in, so the page reads as one sentence rather
#: than as whatever order a dictionary happened to produce.
FINDING_ORDER = ("peeking", "clustering", "multiplicity")


def _lead(findings: list[Finding]) -> str:
    """The one paragraph under the claim, honest about how much is on the page."""
    counted = (
        "Each one is measured below"
        if len(findings) == len(FINDING_ORDER)
        else f"{len(findings)} of the three are measured below"
    )
    return (
        f"Three ways an A/B experiment quietly stops being the test it claims to be. "
        f"{counted}, on experiments with <em>no true effect at all</em>, and each is paired "
        f"with the procedure that puts the rate back. The statistics are implemented in this "
        f"package; <code>statsmodels</code> appears only in the test suite, as an oracle."
    )


def ordered_findings(record: Record) -> list[Finding]:
    """The findings in the order the claim names them."""
    return _ordered(record.findings)


def _ordered(findings: dict[str, Finding]) -> list[Finding]:
    known = [findings[key] for key in FINDING_ORDER if key in findings]
    extra = sorted(
        (finding for key, finding in findings.items() if key not in FINDING_ORDER),
        key=lambda finding: finding.key,
    )
    return known + extra


def _validation_section(validation: Validation) -> str:
    design = validation.design
    return f"""<section>
  <h2>How do I know this code is right?</h2>
  <p>Every method is checked twice: against a reference implementation or hand arithmetic in
  the test suite, and against a simulated world where the truth is known. The second is the
  one that matters, because it tests the choice of formula and not just its transcription.
  {integer(design["n_experiments"])} simulated experiments per row, seed {design["seed"]};
  "MC error" is the standard error of the empirical rate - the noise floor of the run
  itself.</p>
  {_table(validation_table(validation))}
  <p>Note the claim in the last row. A fixed-horizon test promises its false positive rate
  <em>equals</em> alpha; an anytime-valid test promises only that it stays <em>at most</em>
  alpha, and the mSPRT is measurably conservative. The first version of the validation
  script judged it by the equality criterion and reported correct behaviour as a failure -
  the fix was to make the claim explicit per row, and the rule deciding it now lives on the
  result object so this page, the script and the test suite cannot disagree about what
  "pass" means.</p>
</section>"""


def _limits_section() -> str:
    return """<section>
  <h2>What this package will not do for you</h2>
  <p><strong>Independence is optional, but you have to ask for it.</strong> The default
  tests assume one observation per unit, and nothing detects when that is false.
  <code>ab_lab.cluster</code> corrects it when you say so - and its own estimator is
  anti-conservative below about forty units per arm, which the result reports rather than
  hides.</p>
  <p><strong>The mSPRT is conservative.</strong> Its measured false positive rate under ten
  looks is well under the nominal 5%. Validity is bought with power, and a correctly
  executed group-sequential design would stop sooner - the reasoning for choosing it anyway
  is in <a href="decisions/0003-msprt-over-group-sequential.md">ADR 0003</a>.</p>
  <p><strong>The bootstrap's p-value has a floor</strong> of <code>2/(n_resamples+1)</code>.
  A "p &lt; 0.001" read off a 1\u202f000-resample bootstrap is an artefact.</p>
  <p><strong>Normal approximations are used for proportions</strong> and are unreliable at
  very low rates with small samples - at least ten successes and ten failures expected per
  arm is the rule of thumb.</p>
  <p><strong>Nothing decides what "the family" is.</strong> <code>ab_lab.multiplicity</code>
  corrects across a set of metrics, but which metrics belong in one family is a judgement,
  not a computation - and correcting a subset while reading the rest raw controls nothing
  at all. The library makes you name the members; it cannot make that the right list.</p>
  <p><strong>Sample sizes may differ by a few percent from an online calculator</strong>,
  which usually applies the absolute-difference formula rather than Cohen's h. For the same
  reason a one-point drop and a one-point lift are not the same experiment, so a guardrail
  metric has to be sized in the direction it can actually move.</p>
</section>"""


def _provenance_rows(record: Record) -> str:
    """One row per script that contributed figures, with its own seed.

    An earlier version took `next(iter(record.findings.values()))` and printed
    that one section's script, seed and package version as though it were the
    whole page's. Since `load()` sorts the findings, the first value is
    *clustering* - so the published page attributed the peeking chart to
    `three_inflations.py` at seed 20260821, when it came from
    `peeking_pitfalls.py` at seed 20260721, and named a package version that no
    longer existed. The byte-equality guard could not catch it: the generator
    faithfully produced the wrong bytes.
    """
    sources: dict[tuple[str, str, str, str], list[str]] = {}
    for finding in _ordered(record.findings):
        key = (
            finding.recorded["script"],
            str(finding.design["seed"]),
            finding.recorded["recorded_on"],
            finding.recorded["ab_lab_version"],
        )
        sources.setdefault(key, []).append(finding.title.lower())
    if record.validation is not None:
        recorded = record.validation.recorded
        key = (
            recorded["script"],
            str(record.validation.design["seed"]),
            recorded["recorded_on"],
            recorded["ab_lab_version"],
        )
        sources.setdefault(key, []).append("the validation table")

    rows = []
    for (script, seed, recorded_on, version), sections in sources.items():
        rows.append(
            f"    <tr><td><code>{script}</code></td><td>{seed}</td>"
            f"<td>{recorded_on}</td><td>{version}</td>"
            f"<td>{'; '.join(sections)}</td></tr>"
        )
    header = (
        "    <tr><th>Script</th><th>Seed</th><th>Recorded</th><th>ab-lab</th>"
        "<th>What it produced</th></tr>"
    )
    body = "\n".join([header, *rows])
    return f'<div class="table-wrap">\n  <table>\n{body}\n  </table>\n</div>'


def _footer(record: Record) -> str:
    links = " · ".join(
        f'<a href="decisions/{ADR_FILES[number]}">ADR {number}</a> {title}'
        for number, title in ADR_LINKS
    )
    return f"""<footer>
  <p>Every figure on this page is interpolated from
  <a href="data/findings.json">docs/data/findings.json</a>. Nothing here was typed by
  hand, and a test compares the committed bytes of this page against what the generator
  produces. The figures come from more than one run, so each one says which:</p>
  {_provenance_rows(record)}
  <p>{links}</p>
  <p>Portfolio project · <a href="{REPO_URL}">source on GitHub</a> ·
  <a href="{PROFILE_URL}">the rest of the portfolio</a></p>
</footer>
</body>
</html>
"""


def render(record: Record) -> str:
    """The whole page, from the record and nothing else."""
    if not record.findings:
        raise ValueError("the record contains no findings; run the examples with --record")
    findings = _ordered(record.findings)
    peeking = record.findings["peeking"]
    worst = max(peeking.series_by_role("naive").rates)
    worst_of_all = max(max(f.series_by_role("naive").rates) for f in findings)

    description = (
        f"Look twice, count a user twice, or measure a second metric, and a "
        f"{percent(peeking.nominal, 0)} test stops being one - up to "
        f"{percent(worst_of_all)} on experiments where there is no effect to find. "
        f"Measured here, with the correction for each."
    )
    og_description = (
        f"One A/A experiment in {round(1 / worst)} is declared a winner after "
        f"{peeking.x_values[-1]:g} looks. Measured, with the correction."
    )

    sections = [_tiles(findings, record.validation)]
    sections.extend(_finding_section(finding) for finding in findings)
    if record.validation is not None:
        sections.append(_validation_section(record.validation))
    sections.append(_limits_section())

    return (
        head(f"{CLAIM} — ab-lab", description, og_description)
        + f"""<p class="eyebrow">applied statistics · no A/B library underneath</p>
<h1>{CLAIM}.</h1>
<p class="lead">{_lead(findings)}</p>
"""
        + "\n\n".join(section for section in sections if section)
        + "\n\n"
        + _footer(record)
    )
