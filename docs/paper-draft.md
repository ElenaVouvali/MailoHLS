# Historical manuscript notes

These notes describe the previous paper-source layout. The manuscript build
files have been removed from the current development branch; they remain in
Git history, including the initial `26db8ec` snapshot. Evaluation kernels live in
`evaluation_set/`, and measured result tables live in `paper_plots/`.

## SALT-HLS / MailoHLS ASPLOS 2027 September-Cycle Draft

This project is intentionally **MICRO-anchored**. It preserves the organization,
core prose, and figure inventory of the rejected MICRO'26 submission and adds
only the methodology clarifications and experimental slots needed for the
ASPLOS'27 revision.

## Paper organization

The body follows the MICRO order:

1. Introduction
2. Background & Related Work
3. Motivation
4. SALT-HLS: Design Overview
5. SALT-HLS: Detailed Design
6. Evaluation Setup
7. Experimental Results
8. Conclusion

The ASPLOS-only changes are integrated *inside* those sections instead of
introducing a new paper structure. Reproducibility details are in an appendix.

## Internal TODO mode

`Preamble.tex` currently contains:

```latex
\internaldrafttrue
```

This renders red ASPLOS TODOs, result placeholders, and legacy-review markers.
Before a submission build switch it to:

```latex
\internaldraftfalse
```

Do not remove TODOs until the corresponding claim is backed by a frozen
checkpoint/result.

## Build

The official ASPLOS class is already used:

```latex
\documentclass[sigplan,anonymous,review,nonacm]{acmart}
```

Typical local build:

```bash
pdflatex main.tex
bibtex main        # or bibtex8 main if the local bibtex wrapper is unavailable
pdflatex main.tex
pdflatex main.tex
```

No page-geometry or manual vertical-spacing hacks are used. Do not add `\vspace`
or packages that compress ACM spacing.

## Naming / anonymity

The public project name `MailoHLS` is renamed to `SALT-HLS` in the anonymous
submission, following ASPLOS double-blind guidance for already-public systems.
Do not link the public MailoHLS repository from the anonymous manuscript. Use an
anonymized artifact if code is exposed to reviewers.

## Private control files

The following are working documents and should **not** be included in the PDF:

- `PRIVATE_MICRO_reviews.md`
- `PRIVATE_Future_Directions_for_MailoHLS.md`
- `ASPLOS_COMPLIANCE_CHECKLIST.md`
- `PAPER_EDITING_GUIDE.md`
- `FIGURE_UPDATE_PLAN.md`
- `ASPLOS_RESUBMISSION_NOTE_DRAFT.tex`
