// Layout-invariant guard for the shared cursor tooltip.
//
// These tests are the regression net for the recurring tooltip-width bugs:
// lines clipped by the box edge, the box sitting wider than its content (dead
// space right of the right-justified rows), and the box sitting narrower than
// its content (premature wrapping / clipped rows whose value is a bare text
// node). They load the REAL scaffold — src/plotting/_scaffold/base.css +
// cursor_tooltip.js — into a blank page, call the exposed
// `window.__RP_TT_MEASURE(html)` for a battery of payloads (including the
// originally-reported Training/Workouts/Fitness cases and pathological
// strings), render the result exactly as the live tooltip does, and assert in
// a real layout engine that:
//
//   1. No clip      — inner content never overflows its box.
//   2. In bounds    — min-width <= width <= max-width.
//   3. Right-sized  — when the content fits unwrapped (natural width <= cap),
//                     the box equals that natural width: not narrower (which
//                     would wrap/clip lines) and not wider (dead space). When
//                     it must wrap, the box hugs the widest rendered line.
//
// The `natural` width is measured independently here (max-content, no cap) so
// the assertions don't just echo measure()'s own math. If any of this
// regresses, the matching CSS/JS change fails here instead of shipping.
// See ../../src/plotting/_scaffold/cursor_tooltip.js `measure()`.

import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const SCAFFOLD = path.resolve(here, '../../src/plotting/_scaffold');
const CSS_PATH = path.join(SCAFFOLD, 'base.css');
const JS_PATH = path.join(SCAFFOLD, 'cursor_tooltip.js');

type Fixture = { name: string; html: string };

// Payloads mirror the structure the plot hover-builders emit: <br>-joined
// text lines for snap tooltips, `.tt-row` flex rows for smooth tooltips
// (some with bare text-node values, like the Fitness frontier rows),
// multi-fragment inline lines (text + <b> + <span>, like the race header),
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
    // The reported Fitness case: rows whose VALUE is a bare text node (not an
    // element). The box must be wide enough that these nowrap rows don't clip.
    name: 'fitness — rows with bare text-node values must not clip',
    html: '<div class="tt-date">2021-06-22 (Tue)</div>'
      + '<div class="tt-section">'
      + '<div class="tt-row"><span>5K frontier</span><b>5:06/mi</b></div>'
      + '<div class="tt-row"><span>95% band</span>5:01–5:11/mi</div>'
      + '<div class="tt-row"><span>Projected Critical Speed</span>5:09/mi</div>'
      + '</div>',
  },
  {
    // Fitness race detail: single visual lines split across text + <span>/<b>
    // fragments — must be measured as whole lines, not under-counted per node.
    name: 'fitness — multi-fragment race header line',
    html: '<div class="tt-date">2017-11-04 (Sat)</div>'
      + '<div class="tt-section">'
      + '<div class="tt-row"><span>5K frontier</span><b>5:08/mi</b></div>'
      + '<div class="tt-row"><span>Projected Critical Speed</span>5:12/mi</div>'
      + '</div>'
      + '<div class="tt-section">'
      + "<div>Redmond &amp; Lake Washington @ Bellevue <span class=\"tt-mute\">(XC)</span></div>"
      + '<div>5000m in <b>15:58</b> <span class="tt-mute">(5:08/mi)</span></div>'
      + '</div>',
  },
  {
    name: 'fitness — race with elevation audit line (Aug 2026)',
    html: '<div class="tt-date">2026-05-31 (Sun)</div>'
      + '<div class="tt-section">'
      + '<div><b>North Shore Classic</b> <span class="tt-mute">(Road)</span></div>'
      + '<div>21098m in <b>1:09:52</b> <span class="tt-mute">(5:20/mi)</span></div>'
      + '<div>Elevation: <b>+538 ft (+1:07) / −542 ft (−0:25)</b></div>'
      + '<div>Course correction: <b>1:09:10</b> <span class="tt-mute">(5:17/mi)</span></div>'
      + '<div>5K equivalent: <b>14:48</b> <span class="tt-mute">(4:46/mi)</span></div>'
      + '</div>',
  },
  {
    name: 'recovery — elevation kv line wraps, not clips',
    html: '<div class="tt-date">2024-03-27</div>'
      + '<div class="tt-section">'
      + '<b>Pace:</b> 5:43/mi  (8.6 mi) <span style="color:#888">[watch-measured]</span><br>'
      + '<b>Elevation:</b> +392 ft (+1:58) / −391 ft (−1:02)<br>'
      + '<b>Temp:</b> 12°C<br><i>Sammamish River Trail</i></div>',
  },
  {
    name: 'pathological — single unbreakable token',
    html: '<b>Workout</b><br>'
      + 'Supercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercali',
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
      const tt = document.querySelector('.rp-tooltip') as HTMLElement;
      tt.style.display = 'block';

      // Independent natural-width probe: max-content, no cap. This is the width
      // every line needs to render unwrapped — measured here, NOT taken from
      // measure(), so the assertions are a real cross-check.
      tt.innerHTML =
        '<div class="tt-inner" style="width:max-content;max-width:none">' + html + '</div>';
      const natural = Math.ceil(
        (tt.querySelector('.tt-inner') as HTMLElement).getBoundingClientRect().width,
      );

      // The real thing.
      const m = measure(html);
      tt.innerHTML =
        '<div class="tt-inner" style="width:' + m.innerW + 'px">' + html + '</div>';
      const inner = tt.querySelector('.tt-inner') as HTMLElement;
      let rowClip = false;
      inner.querySelectorAll('.tt-row').forEach((el) => {
        if ((el as HTMLElement).scrollWidth > (el as HTMLElement).clientWidth + 1) rowClip = true;
      });
      return { m, natural, scrollW: inner.scrollWidth, clientW: inner.clientWidth, rowClip };
    }, fx.html);

    const { m, natural } = r;
    const clampedNatural = Math.min(Math.max(natural, m.min), m.cap);

    // 1. No clip — content (incl. nowrap rows) fits inside its box.
    expect(r.scrollW, `content overflows box (scrollW=${r.scrollW} > clientW=${r.clientW})`)
      .toBeLessThanOrEqual(r.clientW + 1);
    expect(r.rowClip, 'a .tt-row clips its value').toBeFalsy();

    // 2. Within [min-width, max-width].
    expect(m.innerW).toBeGreaterThanOrEqual(Math.floor(m.min) - 1);
    expect(m.innerW).toBeLessThanOrEqual(Math.ceil(m.cap));

    // 3. Right-sized.
    if (natural <= m.cap) {
      // Fits unwrapped → box equals the natural width (not narrower = no
      // premature wrap/clip; not wider = no dead space).
      expect(
        Math.abs(m.innerW - clampedNatural),
        `box should equal natural width (innerW=${m.innerW}, natural=${natural}, clamped=${clampedNatural})`,
      ).toBeLessThanOrEqual(2);
    } else {
      // Must wrap → box hugs the widest rendered line (no dead space), and is
      // no wider than the cap.
      const innerContent = m.innerW - m.pad;
      expect(
        innerContent <= m.content + 1 || m.innerW <= m.min + 1,
        `box not hugging wrapped content (innerContent=${innerContent}, widestLine=${m.content})`,
      ).toBeTruthy();
    }
  });
}
