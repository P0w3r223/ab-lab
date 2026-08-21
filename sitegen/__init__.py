"""Generator for the published page and the README's tables.

Not part of the installed package: it lives beside `src/`, not inside it,
because `docs/` is the published tree and build source is the one thing in it a
reader never wants served (ADR 0007 D2). It reads the committed evidence in
`docs/data/findings.json` and renders it; it never simulates.
"""
