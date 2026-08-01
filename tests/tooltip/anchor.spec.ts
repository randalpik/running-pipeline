// Regression net for the legend-anchored sidebar clamp
// (src/plotting/_scaffold/overlay_anchor.js): on short viewports the box is
// pulled back on-screen and given an inline max-height so its own
// overflow-y:auto engages; on normal desktop viewports the clamp is a
// provable no-op (top still sits exactly below the legend). Also guards the
// desktop no-regression side of the mobile work: no scroll mode, native
// dragmode, desktop grid.

import { test, expect, Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(here, '../../output');

// Runs only under the chromium project (the mobile project's testMatch is
// touch.spec.ts alone); tests set their own viewports.

async function ready(page: Page, file: string) {
  await page.goto('file://' + file);
  await page.waitForFunction(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    return gd && gd._fullLayout && gd._fullLayout._size;
  });
  await page.waitForTimeout(500);
}

function boxState(page: Page, id: string) {
  return page.evaluate((boxId) => {
    const el = document.getElementById(boxId);
    if (!el) return null;
    const b = el.getBoundingClientRect();
    const legend = document.querySelector(
      '.plotly-graph-div .infolayer > .legend');
    const bg = legend && legend.querySelector('.bg');
    const lb = (bg || legend) && (bg || legend)!.getBoundingClientRect().bottom;
    return {
      top: b.top, bottom: b.bottom, h: b.height,
      maxH: el.style.maxHeight,
      scrolls: el.scrollHeight > el.clientHeight + 1,
      legendBottom: lb, vh: window.innerHeight,
    };
  }, id);
}

const CASES = [
  { file: 'training_quality.html', box: 'tq-routes' },
  { file: 'long_runs.html', box: 'lr-gradient' },
];

for (const { file, box } of CASES) {
  const FILE = path.join(OUT, file);

  test.describe(`${box} clamp`, () => {
    test.skip(!fs.existsSync(FILE), `${file} not built`);

    test('short viewport: box reachable, scrolls internally, never buries the legend',
        async ({ page }) => {
      await page.setViewportSize({ width: 880, height: 400 });
      await ready(page, FILE);
      const s = await boxState(page, box);
      test.skip(!s, `#${box} not rendered on this profile`);
      expect(s!.top).toBeGreaterThanOrEqual(68);
      expect(s!.bottom).toBeLessThanOrEqual(s!.vh - 8);
      // At least the guaranteed strip is visible...
      expect(s!.h).toBeGreaterThanOrEqual(89);
      // ...but the box never rises past what that strip requires — i.e. it
      // sits below the legend, or overlaps at most its last rows.
      expect(s!.top).toBeGreaterThanOrEqual(
        Math.min(s!.legendBottom! + 12, s!.vh - 12 - 90) - 1.5);
      expect(s!.maxH).not.toBe('');
    });

    test('desktop viewport: clamp is a no-op (top = legend bottom + gap)',
        async ({ page }) => {
      await page.setViewportSize({ width: 1440, height: 900 });
      await ready(page, FILE);
      const s = await boxState(page, box);
      test.skip(!s, `#${box} not rendered on this profile`);
      expect(Math.abs(s!.top - (s!.legendBottom! + 12))).toBeLessThanOrEqual(1.5);
      expect(s!.scrolls).toBe(false);   // natural height fits, no inner scroll
    });
  });
}

test.describe('desktop no-regression on the reshaped race page', () => {
  const FILE = path.join(OUT, 'race_pace_by_distance.html');
  test.skip(!fs.existsSync(FILE), 'race_pace_by_distance.html not built');
  test.use({ viewport: { width: 1440, height: 900 } });

  test('no scroll mode, dragmode zoom, desktop grid', async ({ page }) => {
    await ready(page, FILE);
    const r = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      return {
        scroll: document.body.classList.contains('rp-scroll'),
        dragmode: gd._fullLayout.dragmode,
        // Desktop 2x4: xaxis3 sits to the RIGHT of xaxis2 in row 1.
        x3RightOfX2: gd.layout.xaxis3.domain[0] > gd.layout.xaxis2.domain[1],
        height: gd.clientHeight,
      };
    });
    expect(r.scroll).toBe(false);
    expect(r.dragmode).toBe('zoom');
    expect(r.x3RightOfX2).toBe(true);
    expect(Math.abs(r.height - (900 - 60))).toBeLessThanOrEqual(2);
  });
});
