# Publication verification

Checks completed on 14 October 2026:

- Eight automated tests passed: frozen dataset adapter, validation-only threshold selection, archived evidence integrity, figure links and attribution, manuscript corrections, and exact-hash frozen-view reconstruction.
- Saved decisions reproduce all 15 acoustic test-run macro F1 scores.
- Visual split membership is 728/200/219 with no exact-image hash or source-group crossing.
- Acoustic split membership is 504/168/168; test labels are 96 normal and 72 defective.
- All eight accepted figures are present; their image files and the eight numerical tables remain unchanged in the local revised Word package.
- The revised full Markdown paper includes the eight figures, eight tables and mathematical expressions.
- A location-only scan found no matching credential strings, credential-bearing URLs, private keys, signed links or private user paths in the exported text files. This is a pattern scan, not a guarantee against every possible secret.
- Original local drafts remain preserved. The revised Word version contains 63 paragraph/heading/caption/citation-format changes and formats 64 reference entries.

Word layout verification is complete. A user-approved task-local LibreOffice 26.8.1 engine was extracted from the official package after its checksum and The Document Foundation signature were verified. No system-wide installation was performed. The canonical document renderer produced 25 pages at 150 dpi. All 25 pages were inspected; after minor wording and web-reference italic corrections, six changed pages were inspected again and 19 pages remained pixel-identical. No clipping, overlap, split table, missing equation glyph or detached caption was found. The accepted image bytes, numerical table XML and native equation XML remain unchanged. The verified Word file is included in `paper/`.

No new model training was performed during the publication step. No physical printer, synchronized fusion or manufacturing-agent acceptance result is added by these software and editorial checks.
