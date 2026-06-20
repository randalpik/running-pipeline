// Layout-invariant guard for the shared cursor tooltip.
//
// These tests are the regression net for the recurring tooltip-width bugs:
// lines clipped by the box edge, and the box sitting wider than its content
// (dead space right of the right-justified rows). They load the REAL scaffold
// — src/plotting/_scaffold/base.css + cursor_tooltip.js — into a blank page,
// call the exposed `window.__RP_TT_MEASURE(html)` for a battery of payloads
// (including the originally-reported Training/Workouts cases and pathological
// strings), render the result exactly as the live tooltip does, and assert in
// a real layout engine that:
//
//   1. No clip   — inner content never overflows its box.
//   2. Box hugs  — the box is no wider than its widest rendered line (or it's
//                  sitting on the min-width floor).
//   3. In bounds — min-width <= width <= max-width.
//
// If any of these regress, the matching CSS/JS change fails here instead of
// shipping. See ../../src/plotting/_scaffold/cursor_tooltip.js `measure()`.

import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const SCAFFOLD = path.resolve(here, '../../src/plotting/_scaffold');
const CSS_PATH = path.join(SCAFFOLD, 'base.css');
const JS_PATH = path.join(SCAFFOLD, 'cursor_tooltip.js');

type Fixture = { name: string; html: string };

// Payloads mirror the structure the plot hover-builders emit: <br>-joined
// text lines for snap tooltips, `.tt-row` flex rows for smooth tooltips,
// `.tt-date` / `.tt-section` chrome, and the watch `·`-separated rep lists.
const FIXTURES: Fixture[] = [
  {
    name: 'short — no wrap needed',
    html: '<div class="tt-date">2026-06-15</div>'
      + '<div class="tt-row"><span>5K fitness</span><b>5:01/mi</b></div>',
  },
  {
    // The reported Training 2026-06-15 case: a long decomposition line that
    // previously got clipped by the box edge (no wrapper span).
    name: 'training — long decomposition must wrap, not clip',
    html: '<b>Intervals</b><br>'
      + '2400 + 3 × 1200 + 1600m @ 3:15/mi, rest 2:00/mi between reps<br>'
      + '<b>Temp:</b> 18°C<br>'
      + '<b>5K-equiv:</b> 4:50/mi   <b>5K fitness:</b> 5:01/mi',
  },
  {
    // The reported Workouts case: wraps correctly, but the box used to stay
    // pinned to the full cap (dead space right of the rows).
    name: 'workouts — long watch rep list + rows hugs',
    html: '<b>Workout</b><br>'
      + '8 × 800m @ 2:40/mi, rest 2:00/mi<br>'
      + '<b>Watch:</b> 800@2:38 · 800@2:39 · 800@2:40 · 800@2:41 · '
      + '800@2:38 · 800@2:39 · 800@2:40 · 800@2:41<br>'
      + '<b>Temp:</b> 12°C   <b>5K fitness:</b> 4:58/mi',
  },
  {
    name: 'pathological — single unbreakable token',
    html: '<b>Workout</b><br>'
      + 'Supercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercali',
  },
  {
    name: 'smooth — rows only',
    html: '<div class="tt-date">2024-09-22</div>'
      + '<div class="tt-row"><span>5K fitness</span><b>4:58/mi</b></div>'
      + '<div class="tt-row"><span>Training quality</span><b>5:04/mi</b></div>'
      + '<div class="tt-row"><span>Diff</span><b>+6 sec/mi</b></div>',
  },
  {
    name: 'mixed — sections, rows, and a wrapped body line',
    html: '<div class="tt-date">2025-03-10</div>'
      + '<div class="tt-section"><div class="tt-section-title">Session</div>'
      + 'A long continuous fartlek with surges every few minutes across rolling terrain<br>'
      + '<b>Watch:</b> 1600@5:30 · 1600@5:31 · 1600@5:29 · 1600@5:32</div>'
      + '<div class="tt-section"><div class="tt-row"><span>5K fitness</span><b>5:01/mi</b></div></div>',
  },
];

test.beforeEach(async ({ page }) => {
  await page.setContent(
    '<!doctype html><html><head></head><body>'
    + '<div class="rp-tooltip"></div><div class="rp-spike"></div>'
    + '</body></html>',
  );
  await page.addStyleTag({ path: CSS_PATH });
  await page.addScriptTag({ path: JS_PATH });
  await page.waitForFunction(
    () => typeof (window as any).__RP_TT_MEASURE === 'function',
  );
});

for (const fx of FIXTURES) {
  test(`tooltip layout — ${fx.name}`, async ({ page }) => {
    const r = await page.evaluate((html) => {
      const measure = (window as any).__RP_TT_MEASURE as (h: string) => any;
      const m = measure(html);
      // Render the visible tooltip exactly as cursor_tooltip.js `paint()` does.
      const tt = document.querySelector('.rp-tooltip') as HTMLElement;
      tt.innerHTML =
        '<div class="tt-inner" style="width:' + m.innerW + 'px">' + html + '</div>';
      tt.style.width = m.w + 'px';
      tt.style.height = m.h + 'px';
      tt.style.display = 'block';
      const inner = tt.querySelector('.tt-inner') as HTMLElement;
      return { m, scrollW: inner.scrollWidth, clientW: inner.clientWidth };
    }, fx.html);

    // 1. No clip — content fits inside its box (1px sub-pixel tolerance).
    expect(
      r.scrollW,
      `content overflows box (scrollW=${r.scrollW} > clientW=${r.clientW})`,
    ).toBeLessThanOrEqual(r.clientW + 1);

    // 2. Box hugs the widest rendered line — unless it's on the min-width floor.
    const innerContent = r.m.innerW - r.m.pad;
    const hugs = innerContent <= r.m.content + 1 || r.m.innerW <= r.m.min + 1;
    expect(
      hugs,
      `box not hugging (innerContent=${innerContent}, widestLine=${r.m.content}, min=${r.m.min})`,
    ).toBeTruthy();

    // 3. Within [min-width, max-width].
    expect(r.m.innerW).toBeGreaterThanOrEqual(Math.floor(r.m.min));
    expect(r.m.innerW).toBeLessThanOrEqual(Math.ceil(r.m.cap));
  });
}
