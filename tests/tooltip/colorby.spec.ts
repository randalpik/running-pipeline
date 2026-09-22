// Regression net for the colour-by dropdown shared by Recovery and Long Runs
// (src/plotting/color_by.py + src/plotting/_scaffold/color_by.js +
// the .rp-gradient / .rp-select rules in _scaffold/base.css).
//
// Drives the REAL built pages — restyling and strip geometry are meaningless
// without live Plotly traces — so it needs the plot scripts to have run. Each
// page's block skips (not fails) when its HTML is absent.
//
// Invariants:
//   1. Every variable recolours every marker trace tagged meta.color_by, and
//      all of them get the SAME array (Recovery draws the same runs in two
//      panels, which must stay in lockstep).
//   2. The legend strip's tick labels stay inside the gradient bar for every
//      variable, in the STATIC html and after the JS re-render. The end labels
//      carry data-at and no inline `left` precisely because an inline
//      `left: 100%` beats the stylesheet rule and pushes the label off the
//      sidebar's clipped right edge — which shipped once ("14d+" -> "14d").
//   3. Changing colour never disturbs the per-point marker.opacity array the
//      Recovery hide-filters drive, nor the y values any toggle has set.

import { test, expect, Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGES = {
  recovery: path.resolve(here, '../../output/recovery_pace.html'),
  longRuns: path.resolve(here, '../../output/long_runs.html'),
};

async function ready(page: Page, file: string) {
  await page.goto('file://' + file);
  await page.waitForSelector('.plotly-graph-div .scatterlayer');
  await page.waitForFunction(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    return gd && gd.data && gd.data.some((t: any) => t.meta && t.meta.color_by);
  });
}

function keys(page: Page) {
  return page.evaluate(() =>
    [...document.querySelectorAll('select[data-colorby] option')]
      .map(o => (o as HTMLOptionElement).value));
}

// Tick labels must not spill past either end of the bar they annotate.
//
// Measures the TEXT via a Range, not the span's border box. When an
// absolutely-positioned span ends up with both `left` and `right` resolved
// (exactly what the inline-style bug produced: inline `left: 100%` beating the
// stylesheet's `left: auto` while its `right: 0` still applied), the box
// collapses to zero width at the bar's edge and the nowrap text overflows the
// box rather than the bar. A border-box measurement reports no overflow and
// the guard silently passes — verified by mutation.
function tickOverflow(page: Page) {
  return page.evaluate(() => {
    const bar = document.getElementById('cb-strip-bar')!.getBoundingClientRect();
    return [...document.querySelectorAll('#cb-strip-ticks span')].map(s => {
      const r = document.createRange();
      r.selectNodeContents(s);
      const b = r.getBoundingClientRect();
      return { text: s.textContent || '',
               left: bar.left - b.left, right: b.right - bar.right };
    });
  });
}

function traceState(page: Page) {
  return page.evaluate(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    const idx: number[] = [];
    gd.data.forEach((t: any, i: number) => { if (t.meta && t.meta.color_by) idx.push(i); });
    const first = gd.data[idx[0]].marker.color;
    return {
      nTraces: idx.length,
      isArray: Array.isArray(first),
      n: first.length,
      head: first.slice(0, 40).join(','),
      allTracesMatch: idx.every((i: number) => {
        const c = gd.data[i].marker.color;
        return Array.isArray(c) && c.length === first.length
            && c.every((v: string, j: number) => v === first[j]);
      }),
      opacityIsArray: Array.isArray(gd.data[idx[0]].marker.opacity),
      nHidden: Array.isArray(gd.data[idx[0]].marker.opacity)
        ? gd.data[idx[0]].marker.opacity.filter((v: number) => v === 0).length : 0,
      yHead: Array.from(gd.data[idx[0]].y).slice(0, 40).join(','),
    };
  });
}

for (const [name, file] of Object.entries(PAGES)) {
  test.describe(`colour-by — ${name}`, () => {
    test.skip(!fs.existsSync(file), `${file} not built — run its plot script first`);

    test('every variable recolours all tagged traces, in lockstep', async ({ page }) => {
      await ready(page, file);
      const ks = await keys(page);
      expect(ks.length).toBe(8);

      const seen = new Set<string>();
      for (const k of ks) {
        await page.selectOption('select[data-colorby]', k);
        const st = await traceState(page);
        expect(st.isArray, `${k}: marker.color must be a per-point array`).toBe(true);
        expect(st.allTracesMatch, `${k}: tagged traces must share one array`).toBe(true);
        expect(st.n).toBeGreaterThan(0);
        seen.add(st.head);
      }
      // Distinct encodings must not all collapse to the same colouring.
      expect(seen.size).toBeGreaterThan(4);
    });

    test('tick labels stay inside the gradient bar for every variable', async ({ page }) => {
      await ready(page, file);

      // The AS-SHIPPED html first, before any selectOption — the JS
      // re-renders the ticks on every change, so looping straight into the
      // dropdown would only ever exercise renderTicks() and would silently
      // pass on a broken widgets.gradient_strip.
      const onLoad = await tickOverflow(page);
      expect(onLoad.length).toBeGreaterThan(0);
      for (const t of onLoad) {
        expect(t.left, `static: "${t.text}" spills left`).toBeLessThanOrEqual(0.6);
        expect(t.right, `static: "${t.text}" spills right`).toBeLessThanOrEqual(0.6);
      }

      // Then the JS path, for every variable.
      for (const k of await keys(page)) {
        await page.selectOption('select[data-colorby]', k);
        for (const t of await tickOverflow(page)) {
          expect(t.left, `${k}: "${t.text}" spills left`).toBeLessThanOrEqual(0.6);
          expect(t.right, `${k}: "${t.text}" spills right`).toBeLessThanOrEqual(0.6);
        }
      }
    });
  });
}

test.describe('colour-by — recovery hide-filters survive a recolour', () => {
  test.skip(!fs.existsSync(PAGES.recovery), 'output/recovery_pace.html not built');

  test('marker.opacity and y are untouched by marker.color restyles', async ({ page }) => {
    await ready(page, PAGES.recovery);
    const cb = page.locator('#norm-filter input[data-mode="filter"]').first();
    await cb.click();
    const before = await traceState(page);
    expect(before.opacityIsArray).toBe(true);
    expect(before.nHidden).toBeGreaterThan(0);

    await page.selectOption('select[data-colorby]', 'altitude');
    const after = await traceState(page);
    expect(after.nHidden).toBe(before.nHidden);
    expect(after.yHead).toBe(before.yHead);
    expect(after.head).not.toBe(before.head);   // colour really did change
  });
});
