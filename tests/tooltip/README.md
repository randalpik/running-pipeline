# Tooltip layout tests

Headless regression net for the shared cursor tooltip
(`src/plotting/_scaffold/base.css` + `cursor_tooltip.js`). Catches the recurring
width bugs — lines clipped by the box edge, and the box sitting wider than its
content — in a real CSS layout engine, so they fail here instead of shipping.

The test loads the **real** scaffold files (no copies) into a blank page, calls
the exposed `window.__RP_TT_MEASURE(html)` for a battery of payloads, renders
the result exactly as the live tooltip does, and asserts three invariants per
payload: **no clip**, **box hugs its widest line**, **within [min, max] width**.
Add a payload to `FIXTURES` in `tooltip.spec.ts` whenever a new tooltip shape
appears.

## Run

```bash
cd tests/tooltip
npm install
npx playwright install chromium   # first run only
npm test
```

## What "verifiable" means here

Tooltip layout depends on real text wrapping and flexbox sizing — jsdom has no
layout engine, so only a browser can validate it. These tests are the contract:
if `measure()` in `cursor_tooltip.js` or the `.tt-inner` rules in `base.css`
regress, a fixture fails. To confirm the net actually works, revert the
`measure()` change to the old `ghost.getBoundingClientRect()` and watch the
long-decomposition / wrapped-list fixtures fail.

## zoom.spec.ts

Regression net for the Misc Trends drag-to-zoom (the `__rpZoomDragging`
tooltip-suppression guard + the drag/relayout/reset flow). Drives the REAL
built page, so it needs `python src/plots/plot_qualitative_trends.py` to have
run first; the spec skips (not fails) when `output/qualitative_trends.html`
is absent.
