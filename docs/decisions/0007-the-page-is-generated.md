# ADR 0007 — The page is generated from a committed record, not from a rerun and not from typing

Date: 2026-08-21
Status: accepted
Author: Piotr Cząstkiewicz
Related to: [0006](0006-three-mechanisms-of-alpha-inflation.md), portfolio presentation audit v2
Amends: [ADR 0004](0004-scripts-instead-of-notebooks.md) (where the formatter lives), and the committed-artefact rule in `CLAUDE.md`

---

## Context

`docs/index.html` is published by GitHub Pages at `https://p0w3r223.github.io/ab-lab/`.
It is hand-written, and in a package whose README claims "every table above is
regenerable with one command" that is not a stylistic problem. It has already
failed:

| Surface | The same table row |
|---|---|
| `examples/validation_table.py` (`f"...{design.per_group:,}/arm"`) | `n = 14,745/arm` |
| `README.md:69` | `n = 14 745/arm` |
| `docs/index.html:63` | `n = 14,745/arm`, and the row is worded differently again |

The script's own output matches neither downstream copy. Somebody edited a
transcription by hand after pasting it, which is the failure mode the whole
package argues against.

Six further defects, all verified rather than inferred:

* **The README never links the live page.** `grep -c github.io README.md` returns
  `0` across 206 lines. The page is reachable only from an index README and the
  About field.
* **The page drops the Monte Carlo error the README carries.** `docs/index.html`
  prints the peeking rates bare; `README.md` prints them with `±`. The project's
  first code rule is that assumptions ship with the estimate, and the surface
  that is read most is the one that dropped them.
* **Three external-origin elements over two origins** — two `preconnect` hints
  and a stylesheet — which then pull font files. The criterion is the audit's
  "zero external-origin resources"; the count is worth stating precisely because
  it is the kind of number that gets quoted back.
* **A raster chart in the report body.** A PNG cannot follow the colour scheme,
  so the chart and dark mode are one change, not two.
* **`<title>` is a method list.** It reads `ab-lab — A/B experiment statistics`,
  not the bare repo name an earlier draft claimed — but it still fails the
  audit's "states a finding, not the repo name", and it tells a reader nothing
  they did not know from the link they clicked.
* **A hand-typed date** in the footer, and a claim that ADRs exist without
  linking them — they are published one directory away, because `docs/` is the
  Pages root.

Underneath all of it, the same claim exists in five renderings — page, README,
GitHub About, profile README, portfolio index — and all five are *lists of
methods*. None of them states a finding. They do not contradict each other; they
fail to say anything a reader would repeat.

---

## Decision

### D1 — The generator cannot recompute

`doc-extract` is the portfolio's reference implementation of a generated page,
and its generator recomputes every figure at build time. That works because
scoring its committed predictions takes milliseconds.

ab-lab's figures are Monte Carlo. `examples/peeking_pitfalls.py` runs
`1+2+3+5+7+10+14+20 = 62` looks × 4 000 experiments × 2 decision rules, roughly
half a million p-value evaluations; `examples/validation_table.py` runs five rows
of 10 000 experiments, one of which draws 14 745 units per arm. A generator that
recomputes turns every pull request into a multi-minute run on two Python
versions, and makes the staleness test cost the same again.

**The evidence is committed as a record, and the page is a pure function of it.**

```
examples/*.py --record  ─▶  docs/data/findings.json  ─▶  sitegen/  ─┬─▶ docs/index.html
   (minutes, run by hand)      (committed record)     (milliseconds) ├─▶ README tables
                                                                     └─▶ stdout
```

The record stores **counts, never rates**: `n_experiments`, `n_rejections`,
`mean_estimate`, and the design block (seed, sample sizes, look schedule) that
produced them. The renderer rehydrates a `SimulationSummary` and asks *it* for
`rejection_rate` and `monte_carlo_error`. Neither the page nor the README
computes either, so neither can disagree about them.

A consequence worth stating because it is a strict improvement on the reference
implementation: `build()` touches no clock, no `git`, no network and no RNG. The
provenance comes out of the record. `doc-extract`'s staleness test must strip a
region containing a commit hash before comparing; ab-lab's is **one byte-equality
assertion with no exemptions**.

### D2 — The generator lives in `sitegen/`, not under `docs/`

The project rule that rendering belongs in `examples/` has as its subject
`src/ab_lab/` — its purpose is that every number the library produces can be
asserted in a test, which is why the library returns dataclasses. Where the
*renderer* lives was never actually decided.

`docs/build_index.py` would follow doc-extract, but that file is 1 977 lines,
two and a half times this project's own size ceiling, and ab-lab needs three
findings, three charts and a Markdown renderer.

An earlier draft added a second argument — that it would make `docs/` hold a
third category beside the site and the design record. That argument is not
available, because this ADR itself adds `docs/data/`, and the audit names a
published dataset there as a legitimate category. What survives is narrower and
still decisive: `docs/` is the *published* tree, and build **source** is the one
thing in that list a reader never wants served. Output belongs under `docs/`;
the program that writes it does not.

`examples/` was rejected too: it means "how a user would use this library", and a
portfolio-page builder is not that. A reader running every example should not get
a website.

`sitegen/{build,record,charts,markdown,page}.py` keeps source with source and
output with output, keeps every file inside the size rule, publishes nothing
extra, and moves no URL.

**The import budget is enforced, not documented.** `sitegen/` may import
`ab_lab.results`, `power`, `srm` and `sequential` — closed-form, microseconds.
It may not call `run_experiments`, `run_with_peeking` or `peeking_curve`, nor
construct a generator. Cheap closed-form facts are computed at build time;
everything Monte Carlo comes from the record.

**How the guard actually works, because the obvious version does not.** An
earlier draft said to monkeypatch those three names plus
`numpy.random.Generator` and call `build()`, and claimed that a successful build
*proves* nothing simulated. Both halves are wrong, and both were checked:

* Replacing `numpy.random.Generator` with a class that raises leaves
  `np.random.default_rng(0)` working — it still returns a
  `numpy.random._generator.Generator` and still draws numbers, because
  `default_rng` constructs the C-level type directly and never looks up the
  module attribute. Patching `numpy.random.default_rng` does bite. The guard as
  first written was a no-op.
* `monkeypatch.setattr("ab_lab.simulate.run_experiments", ...)` only bites if
  `sitegen` resolves the name through the module at call time. A future
  `from ab_lab.simulate import run_experiments` at module top binds before the
  fixture runs and escapes the patch entirely.

The guard is therefore: import `sitegen.build` in a **subprocess**, run it, and
assert `"ab_lab.simulate" not in sys.modules`. That catches the import form the
monkeypatch misses and the RNG path at the same time, so it subsumes both
problems instead of patching around them. "Provably" is then earned rather than
claimed.

**And the guard needs `sitegen` to be importable at all, which today it is not.**
`pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, there
is no root `conftest.py`, and under pytest's default import mode the repository
root never reaches `sys.path` — verified by writing a probe test that does
`import examples.validation_table` and watching it fail. As specified, G1, G2 and
G3 would all have failed on their first run, and G2 needs `examples/` importable
as well as `sitegen/`. The fix is decided here rather than at the keyboard:
`pythonpath = ["src", "."]`. The reference implementation hit the same wall and
worked around it in test code, with a comment saying the generator "is not a
package and not on the path" — config is the better place for it.

### D3 — One set of numbers, three renderings

Today three implementations format the same tables: two example scripts and the
hand-typed HTML. The fix removes code rather than adding a layer. `examples/`
keeps the computation and stops *owning* its formatting; its job becomes run the
simulation, write the record, then render through `sitegen`.
`sitegen/markdown.py` owns Markdown, `sitegen/page.py` owns HTML, and both call
one function for the number itself.

**This amends ADR 0004, which has to be said out loud.** That ADR's decision text
is "Scripts …, *which print markdown-ready tables* and optionally write the
chart", and both `README.md` and `CLAUDE.md` tell a reader to run them and get a
table. An earlier draft of this section said `examples/` "loses its printers",
which would have silently falsified a prior decision and two documents — in the
release whose entire purpose is removing that kind of drift.

So: the scripts keep printing. `--record` writes the evidence, `--print` renders
the table to stdout, and the default with no flag stays what it is today, so
every command already documented keeps working. What changes is that the printing
goes through the same renderer as the page instead of a second implementation of
it. ADR 0004's decision survives; only the location of the formatter moves, and
this ADR is listed as amending it.

README tables live between `<!-- generated:peeking -->` fences and are rewritten
by the generator, guarded by the same test that guards the page.

**The boundary, stated so that nobody builds a prose templating engine: numbers
are generated, prose is written.** The page's paragraphs and the README's
paragraphs are authored independently and reconciled by a human reading them side
by side. The audit is explicit that this criterion cannot be scripted.

### D4 — Four guards at four different costs

| | Catches | Cost |
|---|---|---|
| G1 | the committed page or README fences differ from what the generator produces | reads one JSON and formats strings — under a second, or the design failed |
| G2 | someone changed `N_EXPERIMENTS` or the look schedule and did not re-record | ~0 ms |
| G3 | the generator started simulating (D2's monkeypatch) | ~0 ms |
| G4 | a **gross** behaviour change: each cell replayed at ~1% of its recorded n, recorded rate within k combined Monte Carlo sigmas | ~1% of a full run |
| G5 | slow drift and dependency-driven change: a weekly workflow reruns both scripts in full and compares rates within Monte Carlo error | minutes, `schedule` + `workflow_dispatch` |

**G4 is sold honestly or not at all.** At n ≈ 300 per cell its own Monte Carlo
error near p = 0.25 is about 0.025, so a 3-sigma band is roughly ±0.08. It
catches "the mSPRT stopped being anytime-valid". It does not catch a two-point
drift. **G5 is the real check**, and writing that down is better than implying
G4 is a proof.

G5 compares rates within Monte Carlo error rather than byte-exactly, on purpose.
The RNG stream is deterministic here — the looks consume no randomness — but
numpy guarantees stream stability for `RandomState`, not for `Generator`. A
byte-exact nightly would go red on a legitimate numpy release. Comparing within
Monte Carlo error survives upgrades while still catching real change.

The scheduled job does **not** publish, so the audit's trap about schedule-only
workflows does not apply, and its rule forbidding bare figures downstream of a
scheduled rebuild does not bind this repo. The page is rebuilt on demand, by a
human, when the numbers change.

Rejected: hashing the source of the modules a record depends on. It trips on a
docstring edit and teaches everyone to re-record in order to silence it, which is
worse than no guard.

### D5 — The chart

Inline SVG emitted by `sitegen/charts.py`: geometry in Python, appearance in
CSS classes, so `prefers-color-scheme` repaints the chart with the page and there
is nothing to re-render. All three findings share one shape — an ordered count on
x, a naive curve climbing away from nominal alpha, a corrected curve at or below
it — so they share **one primitive**, not three bespoke drawings.

Constraints for the implementing commit: x on a linear scale, because
`[1, 2, 3, 5, 7, 10, 14, 20]` is not evenly spaced and categorical spacing would
flatter the curve; series distinguished by colour **and** marker shape **and**
dash, so the chart survives greyscale and colour-vision deficiency; the nominal
alpha as a dashed rule labelled inline rather than in a legend; `role="img"` with
`<title>` and `<desc>`, the adjacent table being the accessible form of the same
data.

The README gets two generated SVGs and a `<picture>` element switching on
`prefers-color-scheme` — GitHub's documented mechanism. `docs/images/peeking.png`
is deleted, and with it **`matplotlib` leaves the `dev` extra**, whose only
justification in `pyproject.toml` is the `--plot` flag. Every CI install shrinks.
A published asset URL goes 404; nothing in this repository references it, and
nothing outside it is *believed* to — which is an assumption, not a check, and is
accepted as such.

Two consequences an implementer would otherwise discover the hard way:

* **One geometry function, two emission modes.** The page's chart is inline SVG
  styled by the page's own CSS classes; the README's two files are referenced
  with `<img src>`, where page CSS does not reach, so each standalone file must
  carry its own `<style>` with the light or dark values baked in. Same
  coordinates, different wrapper.
* **Removing `matplotlib` falsifies a documented command.** `--plot` appears in
  `README.md` and in `CLAUDE.md`'s Commands block. Both edits belong to the same
  commit as the removal.

### D6 — A page of three findings

One `Finding` schema — `{x_label, nominal, series[]}` — with three instances,
rather than three bespoke sections. The three findings *are* one claim measured
three ways, so one schema, one chart primitive and one table renderer is the
architecture stating that. Adding a fourth finding becomes data rather than code.
The escape hatch is written down: a finding that does not fit gets its own
renderer rather than bending the schema.

Page structure: header carrying the claim; a row of one number per finding;
three finding sections, each prose then table then chart; the validation table
with verdicts from the shared rule of ADR 0006; "what this will not do for you";
a footer stating when the numbers were recorded, from which script, at which
seed, and linking the ADRs.

**The ADR links are to files, not to the directory.** `docs/` is the Pages root
and `docs/decisions/` contains only Markdown — no `index.html`, no `.nojekyll` —
and Pages serves no directory listing, so a link to `decisions/` 404s while each
file inside it resolves. An earlier draft linked the directory.

### D7 — The portfolio standard, and a debt recorded rather than hidden

The audit's Phase 1 was supposed to produce a shared head-meta block and palette
so that each repository would not implement them independently and diverge. That
artifact does not exist. Four pages in the portfolio already carry their own
`:root` and their own head block — car-price-ml, doc-extract, it-job-radar and
pl-review-sense — so ab-lab's would be the **fifth implementation**, and the
audit's own baseline of "two repos with full meta, six with none" is itself now
stale at four of eight.

Decided: vendor the block and the palette, and record the debt here rather than
let the audit's prediction come true silently.

**But "copy doc-extract faithfully" turned out not to be a well-defined
instruction, and this is the strongest evidence yet for the debt.** The audit
names *it-job-radar* as the reference, not doc-extract. Their `:root` blocks are
byte-identical today, so the outcome happens to coincide — but two sibling repos
have already deviated **on purpose**, and left the reason in the CSS:

* car-price-ml raises `--positive` from `#059669` to `#047857`, commented
  *"Darker than the obvious #059669, which measures 3.5:1 on --surface — below AA
  for text this size."*
* pl-review-sense raises `--accent-soft` from `#93c5fd` to `#5b93e4`, commented
  *"at #93c5fd this sat at 1.8:1 against the page, under the 3:1 a non-text
  graphic needs."*

doc-extract carries the uncorrected values of both. A faithful copy would
therefore ship two documented contrast defects onto a page whose entire argument
is that the craft is checkable. There is no single faithful source — which is a
better argument for Phase 1 than any copy-count.

So: adopt the **corrected** values, and record the two deltas as part of what
gets proposed upstream. Likewise `--danger`, which an earlier draft called "one
new variable required" and proposed to invent: **it already exists** in
car-price-ml as `#b42318` light and `#f87171` dark, with a comment explaining why
it is distinct from `--warn`. Inventing a second red is exactly the divergence
this section exists to prevent. Adopt those values and cite the source.

**Revisit when** the Phase 1 artifact lands: the vendored palette is replaced by
it, and this ADR is superseded in that respect only.

### D8 — The claim, and its five renderings

> **A 5% test is only 5% if you look once, count each user once, and test one
> metric.**

It is a finding rather than a list of methods; it covers all three results; it
fits a `<title>`, an About field and a table cell; and it does not need recutting
when a fourth finding lands. The alternative — "checked twenty times, one A/A
experiment in four is declared a winner" — is more memorable but describes
finding one only, and would have to be rewritten as soon as clustering and
multiplicity ship, which is the drift this work exists to remove.

**Rule for digits downstream.** The exact figure appears only on the page, where
it is generated. Every other layer quotes the robust rounding — "one A/A
experiment in four" — which survives a re-record moving 25.3% to 24.8%. A digit
downstream is permitted here because nothing rebuilds on a schedule; a *fragile*
digit is not.

| Layer | What changes |
|---|---|
| **L4** page | `<title>` = the claim + " — ab-lab"; `meta description`, `og:title`/`description`/`url`/`type`, `twitter:card`, inline `data:` SVG icon; `h1` = the claim; every figure interpolated from the record, none typed |
| **L3** README | the claim as the first non-badge line, the live link immediately after it — inside the standard's fifteen-line window and fixing its total absence; tables generated between fences; limitation bullets 1 and 3 and roadmap items 1 and 4 struck when ADR 0006 lands |
| **L2** About | description replaced by the text below; topics gain `monte-carlo`; website and MIT licence already correct. **A manual step — no pull request can carry it.** |
| **L0** profile README | the ab-lab row rewritten to quote the claim; it currently reads as a method list, and ab-lab appears in two tables there |
| **L1** portfolio index | the listing collapsed — ab-lab appears **three** times in `current_projects/README.md`, at lines 19, 52 and 59 ("Pinned on profile", where it is reduced to "applied statistics"), each with different wording. Fixing two of three leaves the third free to drift, which is how this started. The method list survives *below* the claim, not instead of it |

The About description, written out so that its length is checkable now rather
than at the keyboard — **281 characters against the audit's 350 limit**, counted,
not estimated:

> A 5% test is only 5% if you look once, count each user once, and test one
> metric. Three ways an A/B experiment stops being the test it claims to be —
> peeking, clustered users, many metrics — each measured on simulated experiments
> where the truth is known, each with its correction.

Order of execution is page → README → About → profile → index, because the title
is what the other four quote.

### D9 — Line endings, before the first generated commit

The working machine is Windows, CI is Ubuntu. A generator writing with Python's
default newline translation produces CRLF locally and LF in CI, and a byte-
equality test then fails for a reason having nothing to do with the numbers.
Confirmed rather than assumed: `core.autocrlf` is `true` here, and `README.md`
(206 CRLF lines, zero LF), `docs/index.html` and `pyproject.toml` are CRLF in the
worktree while every file under `src/` is LF. The generated files land on the
wrong side of that split.

`newline="\n"` on every write, plus `.gitattributes` marking `docs/index.html`,
`docs/data/*.json`, `README.md` **and the two generated README SVGs** as
`text eol=lf`. And because two of those files already exist as CRLF, adding
`.gitattributes` is not sufficient on its own: without `git add --renormalize .`
in the same commit they show up as wholly modified at the next touch. This lands
in the first commit of the work, not after the first red build.

---

## The project rule this amends

> Do not commit generated artefacts other than `docs/images/`.

becomes

> Do not commit generated artefacts other than `docs/images/`, `docs/index.html`
> and `docs/data/`, each of which is guarded by a test asserting that the
> committed bytes are what the generator produces.

The rule was written to keep *unverifiable* artefacts out of the repository. The
record is its opposite: it is what makes the page verifiable, and it replaces
`docs/images/peeking.png` — the one committed artefact today whose numbers
nothing can check.

### Every `CLAUDE.md` edit this release needs, and who owns it

Both ADRs amend rules in prose, and neither originally gave those edits a home in
a commit. A rule amended only in an ADR is a rule that still reads the old way in
the file everyone actually loads.

| Edit | Owner |
|---|---|
| the artefact rule above | PR 2, first commit |
| Commands block: drop `--plot`, add `--record` / `--print` and the site build | PR 2, with the `matplotlib` removal |
| architecture block: `sitegen/`, `docs/data/` | PR 2 |
| architecture block: `cluster.py`, `multiplicity.py`, `_validation.py` | PR 3 and PR 5, each with its module |
| the rendering rule, which currently says rendering belongs in `examples/` | PR 2 — it belongs in `examples/` **and** `sitegen/`, and the rule's real subject is that `src/ab_lab/` never renders |

---

## Consequences

* `docs/index.html`, `docs/data/findings.json` and the README's fenced regions
  become build output under test. Hand-editing any of them turns the pull request
  red. Existing CI already runs `pytest` on every pull request, so no new gating
  job is needed.
* The page can no longer show a number the repository does not contain. It can
  still show a *stale* one — that is what G4 and G5 are for, and their limits are
  stated rather than implied.
* `docs/decisions/` URLs do not move. `docs/images/peeking.png` 404s, accepted.
* `matplotlib` leaves the `dev` extra.
* **Revisit when** a finding needs a shape the `Finding` schema cannot express,
  or the record grows past a few hundred kilobytes and has to leave the published
  tree, or the Phase 1 artifact lands.
