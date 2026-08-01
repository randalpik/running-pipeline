// Geography (Locations) x-zoom correctness.
//
// The bar snapping and the custom tooltip both used to derive the bin pitch
// as plotWidth / binCount, which is only the true pitch when the whole range
// is on screen. Zooming therefore re-snapped bars onto the unzoomed grid
// (huge gaps, mangled bars) and made the tooltip resolve the wrong bin.
// A second trap: the axis TYPE differs per mode — yearly is a `category`
// axis, monthly x-values ('2016-01') make Plotly infer a `date` axis — so
// bin centres must be resolved per type.
//
// Ground truth here is each bar's own Plotly datum (`__data__.p`) resolved
// through the axis's category list — independent of the bin math under test.
//
// The axis is pinned `category` in BOTH modes (see applyMode in
// make_geography_plot.js). On a date axis Plotly spaces bars by real month
// length while giving them one width, so zooming magnified the 0-3 day
// remainder into ragged 1-7px gaps; categories are uniform by construction.

import { test, expect, Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = path.resolve(here, '../../output/mileage_by_geography.html');

const BAR_RE = String.raw`^M([-\d.]+),([-\d.]+)V([-\d.]+)H([-\d.]+)V([-\d.]+)Z$`;

async function ready(page: Page, mode: 'year' | 'month') {
  await page.goto('file://' + PAGE);
  await page.waitForFunction(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    return gd && gd._fullLayout && gd._fullLayout._size;
  });
  await page.waitForTimeout(900);
  if (mode === 'month') {
    await page.click('#geo-toggle .rp-btn-pill[data-value="month"]');
    await page.waitForTimeout(900);
  }
}

// Visible bar columns: centre in client px, the bar's own datum, and the
// category name that datum indexes.
function columns(page: Page) {
  return page.evaluate((reSrc) => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    const r = gd.getBoundingClientRect(), s = gd._fullLayout._size;
    const cats = gd._fullLayout.xaxis._categories || [];
    const re = new RegExp(reSrc);
    const seen = new Map<number, any>();
    gd.querySelectorAll('.barlayer .point path').forEach((pt: any) => {
      const m = (pt.getAttribute('d') || '').match(re);
      if (!m || Math.abs(parseFloat(m[2]) - parseFloat(m[3])) < 0.5) return;
      const l = parseFloat(m[1]), rr = parseFloat(m[4]);
      const cx = (l + rr) / 2;
      if (cx < 0 || cx > s.w) return;
      const p = (pt.__data__ || {}).p;
      if (!seen.has(p)) {
        seen.set(p, { p, bin: cats[p], w: rr - l, left: l, right: rr,
                      cx: r.left + s.l + cx, cy: r.top + s.t + s.h * 0.92 });
      }
    });
    return Array.from(seen.values()).sort((a, b) => a.cx - b.cx);
  }, BAR_RE);
}

// Rendered slot width, and the gap the JS should choose for it. Mirrors
// GAP_STEPS / GAP_WIDE in make_geography_plot.js — keep in sync.
function pitchOf(page: Page) {
  return page.evaluate(() => {
    const xa = (document.querySelector('.plotly-graph-div') as any)
      ._fullLayout.xaxis;
    return Math.abs(xa.c2p(1) - xa.c2p(0));
  });
}
const expectedGap = (pitch: number) => (pitch < 30 ? 1 : pitch < 100 ? 2 : 4);

// '2016-01' -> 'Jan 2016'; '2016' -> '2016' (what the tooltip renders).
function binLabel(bin: string) {
  if (/^\d{4}$/.test(bin)) return bin;
  return new Date(bin + '-01T00:00:00Z').toLocaleString('en-US',
    { month: 'short', year: 'numeric', timeZone: 'UTC' });
}

async function zoomMiddle(page: Page) {
  const g = await page.evaluate(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    const r = gd.getBoundingClientRect(), s = gd._fullLayout._size;
    return { l: r.left + s.l, t: r.top + s.t, w: s.w, h: s.h };
  });
  await page.mouse.move(g.l + g.w * 0.35, g.t + g.h * 0.5);
  await page.mouse.down();
  await page.mouse.move(g.l + g.w * 0.62, g.t + g.h * 0.5, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(800);
}

test.describe('geography x-zoom', () => {
  test.skip(!fs.existsSync(PAGE), 'mileage_by_geography.html not built');

  // The gap is derived from the rendered slot width, one rule for both modes
  // and every zoom level — not a per-mode constant as it used to be.
  for (const mode of ['year', 'month'] as const) {
    test(`${mode}: full-range gaps are uniform and match the slot width`,
        async ({ page }) => {
      await ready(page, mode);
      const cols = await columns(page);
      const sorted = [...cols].sort((a, b) => a.left - b.left);
      const gaps: number[] = [];
      for (let i = 1; i < sorted.length; i++) {
        if (sorted[i].p - sorted[i - 1].p === 1) {
          gaps.push(Math.round(sorted[i].left - sorted[i - 1].right));
        }
      }
      expect(gaps.length).toBeGreaterThan(0);
      expect([...new Set(gaps)]).toHaveLength(1);
      expect(gaps[0]).toBe(expectedGap(await pitchOf(page)));
    });
  }

  for (const mode of ['year', 'month'] as const) {
    test(`${mode}: zoomed bars stay well-formed and the tooltip follows`,
        async ({ page }) => {
      await ready(page, mode);
      const fullX = await page.evaluate(() =>
        (document.querySelector('.plotly-graph-div') as any)
          ._fullLayout.xaxis.range.join('|'));
      const yBefore = await page.evaluate(() =>
        (document.querySelector('.plotly-graph-div') as any)
          ._fullLayout.yaxis.range.join('|'));

      await zoomMiddle(page);
      const cols = await columns(page);
      expect(cols.length).toBeGreaterThanOrEqual(3);

      // Bars must be wider than at full range and consistently sized —
      // the mangling showed up as wildly mixed widths.
      const widths = cols.map(c => c.w);
      expect(Math.min(...widths)).toBeGreaterThan(8);
      expect(Math.max(...widths) - Math.min(...widths)).toBeLessThanOrEqual(2);
      // ...and must not overlap each other.
      const sorted = [...cols].sort((a, b) => a.left - b.left);
      for (let i = 1; i < sorted.length; i++) {
        expect(sorted[i].left).toBeGreaterThanOrEqual(sorted[i - 1].right - 0.01);
      }
      // The reported symptom: gaps between ADJACENT bins must all be the
      // same. On the old date axis these came out 1-7px when zoomed.
      const gaps: number[] = [];
      for (let i = 1; i < sorted.length; i++) {
        if (sorted[i].p - sorted[i - 1].p === 1) {
          gaps.push(Math.round(sorted[i].left - sorted[i - 1].right));
        }
      }
      expect(gaps.length).toBeGreaterThan(0);
      expect([...new Set(gaps)]).toHaveLength(1);
      expect(gaps[0]).toBe(expectedGap(await pitchOf(page)));

      // Hovering a bar shows THAT bar, and the spike lands on it.
      for (const i of [0, Math.floor(cols.length / 2), cols.length - 1]) {
        const col = cols[i];
        await page.mouse.move(col.cx, col.cy);
        await page.waitForTimeout(140);
        const got = await page.evaluate(() => {
          const t = document.getElementById('geo-tooltip')!;
          const sp = document.getElementById('geo-spike')!;
          const m = (sp.style.transform || '').match(/translateX\(([-\d.]+)px\)/);
          return { label: (t.querySelector('.hov-day')?.textContent || '').trim(),
                   shown: t.style.display === 'block',
                   spikeX: m ? parseFloat(m[1]) : null };
        });
        expect(got.shown).toBe(true);
        expect(Math.abs(got.spikeX! - col.cx)).toBeLessThanOrEqual(3);
        expect(got.label).toBe(binLabel(col.bin));
      }

      // y is pinned (horizontal-only zoom); double-click restores the range.
      const yAfter = await page.evaluate(() =>
        (document.querySelector('.plotly-graph-div') as any)
          ._fullLayout.yaxis.range.join('|'));
      expect(yAfter).toBe(yBefore);
      await page.mouse.dblclick(cols[1].cx, cols[1].cy);
      await page.waitForTimeout(800);
      expect(await page.evaluate(() =>
        (document.querySelector('.plotly-graph-div') as any)
          ._fullLayout.xaxis.range.join('|'))).toBe(fullX);
    });
  }
});
